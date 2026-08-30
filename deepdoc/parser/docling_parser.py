#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from __future__ import annotations

import logging
import re
import base64
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pdfplumber
import requests
from PIL import Image

from common.constants import MAXIMUM_PAGE_NUMBER

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except Exception:
    DocumentConverter = None
    PdfFormatOption = None
    InputFormat = None
    PdfPipelineOptions = None

try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except Exception:

    class RAGFlowPdfParser:
        pass


from deepdoc.parser.utils import extract_pdf_outlines


class DoclingContentType(str, Enum):
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    EQUATION = "equation"


@dataclass
class _BBox:
    page_no: int
    x0: float
    y0: float
    x1: float
    y1: float


def _extract_bbox_from_prov(item, prov_attr: str = "prov") -> Optional[_BBox]:
    prov = getattr(item, prov_attr, None)
    if not prov:
        return None

    prov_item = prov[0] if isinstance(prov, list) else prov
    pn = getattr(prov_item, "page_no", None)
    bb = getattr(prov_item, "bbox", None)
    if pn is None or bb is None:
        return None

    coords = [getattr(bb, attr) for attr in ("l", "t", "r", "b")]
    if None in coords:
        return None

    return _BBox(page_no=int(pn), x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


class DoclingParser(RAGFlowPdfParser):
    def __init__(self, docling_server_url: str = "", request_timeout: int = 600):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.page_images: list[Image.Image] = []
        self.page_heights: dict[int, float] = {}
        self.page_from = 0
        self.page_to = 10_000
        self.outlines = []
        self.docling_server_url = (docling_server_url or "").rstrip("/")
        self.request_timeout = request_timeout

    def _effective_server_url(self, docling_server_url: Optional[str] = None) -> str:
        return (docling_server_url or self.docling_server_url or "").rstrip("/") or (os.environ.get("DOCLING_SERVER_URL", "").rstrip("/"))

    @staticmethod
    def _is_http_endpoint_valid(url: str, timeout: int = 5) -> bool:
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code in [200, 301, 302, 307, 308]
        except Exception:
            try:
                response = requests.get(url, timeout=timeout, allow_redirects=True)
                return response.status_code in [200, 301, 302, 307, 308]
            except Exception:
                return False

    def check_installation(self, docling_server_url: Optional[str] = None) -> bool:
        server_url = self._effective_server_url(docling_server_url)
        if server_url:
            for path in ("/openapi.json", "/docs", "/v1/convert/source"):
                if self._is_http_endpoint_valid(f"{server_url}{path}", timeout=5):
                    return True
            self.logger.warning(f"[Docling] external server not reachable: {server_url}")
            return False

        if DocumentConverter is None:
            self.logger.warning("[Docling] 'docling' is not importable, please: pip install docling")
            return False
        try:
            _ = DocumentConverter()
            return True
        except Exception as e:
            self.logger.error(f"[Docling] init DocumentConverter failed: {e}")
            return False

    @staticmethod
    def _resolve_page_range(page_from: int, page_to: int) -> Optional[tuple[int, int]]:
        """Translate RAGFlow's task page range into Docling's ``page_range``.

        RAGFlow is 0-based with ``page_to`` exclusive (Python slice stop); Docling
        is 1-based with both bounds inclusive and rejects a range whose start is
        below 1 or whose end precedes its start. Bounds are clamped first, since
        ``parse_pdf`` is public and its callers are not required to pre-clamp.

        Returns ``None`` for a range that covers the whole document, and for an
        empty one; both mean "convert everything".
        """
        start = max(0, page_from)
        end = min(page_to, MAXIMUM_PAGE_NUMBER)
        if start == 0 and end >= MAXIMUM_PAGE_NUMBER:
            return None
        if end <= start:
            return None
        return (start + 1, end)

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        self.page_from = page_from
        self.page_to = page_to
        bytes_io = None
        try:
            if not isinstance(fnm, (str, PathLike)):
                bytes_io = BytesIO(fnm)

            opener = pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(bytes_io)
            with opener as pdf:
                pages = pdf.pages[page_from:page_to]
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for p in pages]
        except Exception as e:
            self.page_images = []
            self.logger.exception(e)
        finally:
            if bytes_io:
                bytes_io.close()

    def _make_line_tag(self, bbox: _BBox) -> str:
        if bbox is None:
            return ""
        x0, x1, top, bott = bbox.x0, bbox.x1, bbox.y0, bbox.y1
        # Docling numbers pages from the start of the document while page_images
        # only holds the rendered window, and crop() adds page_from back when it
        # turns a tag into a position. Emit window-relative page numbers, like
        # cropout_docling_table already indexes by.
        page_no = bbox.page_no - getattr(self, "page_from", 0)
        page_height = None
        if hasattr(self, "page_images") and self.page_images and 0 < page_no <= len(self.page_images):
            page_height = self.page_images[page_no - 1].size[1]
        elif getattr(self, "page_heights", None):
            # Docling reports the page height from the start of the document, so
            # never offset it: resolve against the absolute page number.
            page_height = self.page_heights.get(bbox.page_no)
        if page_height is not None:
            top, bott = page_height - top, page_height - bott
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(page_no, x0, x1, top, bott)

    @staticmethod
    def extract_positions(txt: str) -> list[tuple[list[int], float, float, float, float]]:
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text: str, ZM: int = 1, need_position: bool = False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            return (None, None) if need_position else None

        # a position tag is emitted even when page rendering failed (__images__ leaves
        # page_images empty), and a tag can name a page beyond the rendered range, so
        # indexing page_images below would raise IndexError. mirror
        # cropout_docling_table and the sibling parsers: bail out when there are no
        # page images and drop positions that fall outside them. When only the
        # images are missing the positions still flow, so click-to-highlight keeps
        # working on remote conversions whose page rasterisation failed.
        if not getattr(self, "page_images", None):
            self.logger.warning("[Docling] crop called without page images; returning positions only.")
            if need_position:
                positions = [
                    (pns[0] + self.page_from, int(left), int(right), int(top), int(bottom))
                    for pns, left, right, top, bottom in poss
                    if pns
                ]
                if positions:
                    return None, positions
            return (None, None) if need_position else None

        page_count = len(self.page_images)
        valid_poss = []
        for p in poss:
            if p[0] and all(0 <= pn < page_count for pn in p[0]):
                valid_poss.append(p)
            else:
                self.logger.warning(f"[Docling] Position on pages {p[0]} is out of range for {page_count} rendered page(s); skipping it.")
        poss = valid_poss
        if not poss:
            return (None, None) if need_position else None

        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1], pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1], pos[4] + 120)))
        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if bottom <= top:
                bottom = top + 4
            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))

            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))
            remain_bottom = bottom - img0.size[1]
            for pn in pns[1:]:
                if remain_bottom <= 0:
                    break
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(remain_bottom, page.size[1]))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                remain_bottom -= page.size[1]

        if not imgs:
            return (None, None) if need_position else None

        height = sum(i.size[1] + GAP for i in imgs)
        width = max(i.size[0] for i in imgs)
        pic = Image.new("RGB", (width, int(height)), (245, 245, 245))
        h = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(h)))
            h += img.size[1] + GAP

        return (pic, positions) if need_position else pic

    def _iter_doc_items(self, doc) -> Iterable[tuple[str, Any, Optional[_BBox]]]:
        for t in getattr(doc, "texts", []):
            label = getattr(t, "label", "")
            if label in ("formula",):
                text = getattr(t, "text", "") or getattr(t, "orig", "")
                bbox = _extract_bbox_from_prov(t)
                yield (DoclingContentType.EQUATION.value, text, bbox)
                continue

            parent = getattr(t, "parent", "")
            ref = getattr(parent, "cref", "")
            if (label in ("section_header", "text") and ref in ("#/body",)) or label in ("list_item",):
                text = getattr(t, "text", "") or ""
                bbox = _extract_bbox_from_prov(t)
                yield (DoclingContentType.TEXT.value, text, bbox)

    def _transfer_to_sections(self, doc, parse_method: str) -> list[tuple[str, ...]]:
        sections: list[tuple[str, ...]] = []
        for typ, payload, bbox in self._iter_doc_items(doc):
            if typ == DoclingContentType.TEXT.value:
                section = payload.strip()
                if not section:
                    continue
            elif typ == DoclingContentType.EQUATION.value:
                section = payload.strip()
                if not section:
                    continue
            else:
                continue

            tag = self._make_line_tag(bbox) if isinstance(bbox, _BBox) else ""
            if parse_method in {"manual", "pipeline"}:
                sections.append((section, typ, tag))
            elif parse_method == "paper":
                sections.append((section + tag, typ))
            else:
                sections.append((section, tag))
        return sections

    def cropout_docling_table(self, page_no: int, bbox: tuple[float, float, float, float], zoomin: int = 1):
        if not getattr(self, "page_images", None):
            return None, ""

        idx = (page_no - 1) - getattr(self, "page_from", 0)
        if idx < 0 or idx >= len(self.page_images):
            return None, ""

        page_img = self.page_images[idx]
        W, H = page_img.size
        left, top, right, bott = bbox

        x0 = float(left)
        y0 = float(H - top)
        x1 = float(right)
        y1 = float(H - bott)

        x0, y0 = max(0.0, min(x0, W - 1)), max(0.0, min(y0, H - 1))
        x1, y1 = max(x0 + 1.0, min(x1, W)), max(y0 + 1.0, min(y1, H))

        try:
            crop = page_img.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
        except Exception:
            return None, ""

        pos = (page_no - 1 if page_no > 0 else 0, x0, x1, y0, y1)
        return crop, [pos]

    def _transfer_to_tables(self, doc):
        tables = []
        for tab in getattr(doc, "tables", []):
            img = None
            positions = ""
            bbox = _extract_bbox_from_prov(tab)
            if bbox:
                img, positions = self.cropout_docling_table(bbox.page_no, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            html = ""
            try:
                html = tab.export_to_html(doc=doc)
            except Exception:
                pass
            tables.append(((img, html), positions if positions else ""))
        for pic in getattr(doc, "pictures", []):
            img = None
            positions = ""
            bbox = _extract_bbox_from_prov(pic)
            if bbox:
                img, positions = self.cropout_docling_table(bbox.page_no, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            captions = ""
            try:
                captions = pic.caption_text(doc=doc)
            except Exception:
                pass
            tables.append(((img, [captions]), positions if positions else ""))
        return tables

    def _set_page_heights(self, doc: dict) -> None:
        """Record page heights (points) from a converted Docling JSON document.

        Used as a drop-in source of page dimensions when ``page_images`` is
        unavailable, so position tags can still be emitted with TOP-origin
        coordinates (the y-flip needs the page height).
        """
        self.page_heights = {}
        pages = doc.get("pages")
        if not isinstance(pages, dict):
            return
        for page_no, page in pages.items():
            if not isinstance(page, dict):
                continue
            size = page.get("size")
            if not isinstance(size, dict) or size.get("height") is None:
                continue
            try:
                self.page_heights[int(page_no)] = float(size["height"])
            except (TypeError, ValueError):
                continue

    @staticmethod
    def _picture_description_text(pic: dict) -> str:
        """Extract the vision-model description from a Docling JSON picture item.

        With ``do_picture_description`` enabled, docling-serve records the
        generated description both on the picture's ``meta.description.text`` and
        as a ``description`` annotation; prefer the metadata, falling back to the
        first ``description``-kind annotation. Returns "" when the picture has no
        description so callers can skip it.
        """
        if not isinstance(pic, dict):
            return ""
        meta = pic.get("meta")
        if isinstance(meta, dict):
            desc = meta.get("description")
            if isinstance(desc, dict) and isinstance(desc.get("text"), str):
                text = desc["text"].strip()
                if text:
                    return text
        for annotation in pic.get("annotations") or []:
            if not isinstance(annotation, dict):
                continue
            if annotation.get("kind") not in (None, "description"):
                continue
            text = (annotation.get("text") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _json_bbox(item: dict) -> Optional[_BBox]:
        """Extract the first ``prov`` bounding box from a Docling JSON item."""
        prov = item.get("prov")
        if not isinstance(prov, list) or not prov or not isinstance(prov[0], dict):
            return None
        prov_item = prov[0]
        bb = prov_item.get("bbox")
        if not isinstance(bb, dict):
            return None
        try:
            coords = [float(bb[k]) for k in ("l", "t", "r", "b")]
        except (KeyError, TypeError, ValueError):
            return None
        page_no = prov_item.get("page_no")
        if page_no is None:
            return None
        try:
            page_no = int(page_no)
        except (TypeError, ValueError):
            return None
        return _BBox(page_no=page_no, x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])

    def _json_to_sections(self, doc: dict, parse_method: str) -> list[tuple[str, ...]]:
        """Build tagged sections from the ``json_content`` of a remote conversion,
        mirroring ``_transfer_to_sections`` on the object model.

        When the export carries a ``body`` tree (docling-serve does), its
        ``children`` is the authoritative reading order and sections follow it.
        Otherwise the flat ``texts`` list is used, in array order. Figure
        descriptions are NOT emitted here: they ride inside the figure's chunk,
        paired with its caption by ``_json_to_tables``.
        """
        sections: list[tuple[str, ...]] = []
        body = doc.get("body")
        if isinstance(body, dict):
            children = body.get("children")
            if isinstance(children, list) and children:
                self._json_walk_children(doc, children, sections, parse_method)
                return sections

        texts = doc.get("texts")
        if isinstance(texts, list):
            for item in texts:
                if not isinstance(item, dict):
                    continue
                self._json_append_text_section(item, sections, parse_method)
        return sections

    def _json_walk_children(
        self,
        doc: dict,
        children: list,
        sections: list[tuple[str, ...]],
        parse_method: str,
    ) -> None:
        """Emit snippets in the order Docling's ``body.children`` tree lists them:
        body text/formulas via the text filter, and picture descriptions at the
        picture's own position. Tables and key-value items are handled separately
        by ``_json_to_tables``."""
        for child in children:
            if not isinstance(child, dict):
                continue
            ref = child.get("$ref")
            if not isinstance(ref, str) or not ref.startswith("#/"):
                continue
            parts = ref.split("/")
            if len(parts) != 3:
                continue
            kind, index_str = parts[1], parts[2]
            try:
                index = int(index_str)
            except (TypeError, ValueError):
                continue
            if kind == "texts":
                texts = doc.get("texts")
                if isinstance(texts, list) and index < len(texts) and isinstance(texts[index], dict):
                    self._json_append_text_section(texts[index], sections, parse_method)
            elif kind == "groups":
                group = None
                groups = doc.get("groups")
                if isinstance(groups, list) and index < len(groups):
                    group = groups[index]
                elif isinstance(groups, dict):
                    group = groups.get(index_str)
                if isinstance(group, dict) and isinstance(group.get("children"), list):
                    self._json_walk_children(doc, group["children"], sections, parse_method)

    def _json_append_text_section(
        self,
        item: dict,
        sections: list[tuple[str, ...]],
        parse_method: str,
    ) -> None:
        """Append a single body text/formula item, honouring the same label filter
        used before the body-tree ordering existed."""
        label = (item.get("label") or "").strip()
        typ = None
        payload = None
        if label == "formula":
            typ = DoclingContentType.EQUATION.value
            payload = (item.get("text") or item.get("orig") or "").strip()
        elif label == "list_item":
            typ = DoclingContentType.TEXT.value
            payload = (item.get("text") or "").strip()
        elif label in ("section_header", "text"):
            parent = item.get("parent")
            ref = parent.get("$ref") if isinstance(parent, dict) else ""
            if ref == "#/body":
                typ = DoclingContentType.TEXT.value
                payload = (item.get("text") or "").strip()
        if typ is None or not payload:
            return
        bbox = self._json_bbox(item)
        tag = self._make_line_tag(bbox) if isinstance(bbox, _BBox) else ""
        self._append_json_section(sections, payload, typ, tag, parse_method)

    @staticmethod
    def _append_json_section(
        sections: list[tuple[str, ...]],
        payload: str,
        typ: str,
        tag: str,
        parse_method: str,
    ) -> None:
        """Append a section in the tuple shape dictated by ``parse_method``."""
        if parse_method in {"manual", "pipeline"}:
            sections.append((payload, typ, tag))
        elif parse_method == "paper":
            sections.append((payload + tag, typ))
        else:
            sections.append((payload, tag))

    @staticmethod
    def _table_html_from_json(tab: dict) -> str:
        """Rebuild the table markup from ``data.table_cells`` (row/col offsets).

        This server's JSON export carries no ``table_html``/``text`` fields, only
        per-cell text with offset bounds, so the HTML is synthesised from those.
        Rows/columns larger than any cell's bounds are dropped.
        """
        data = tab.get("data")
        if not isinstance(data, dict):
            return ""
        cells = data.get("table_cells")
        if not isinstance(cells, list):
            return ""
        specs = []
        max_row = 0
        max_col = 0
        for c in cells:
            if not isinstance(c, dict) or not isinstance(c.get("text"), str):
                continue
            try:
                r0 = int(c.get("start_row_offset_idx") or 0)
                c0 = int(c.get("start_col_offset_idx") or 0)
                r1 = max(int(c.get("end_row_offset_idx") or (r0 + 1)), r0 + 1)
                c1 = max(int(c.get("end_col_offset_idx") or (c0 + 1)), c0 + 1)
            except (TypeError, ValueError):
                continue
            max_row = max(max_row, r1)
            max_col = max(max_col, c1)
            specs.append({
                "r0": r0, "c0": c0, "rowspan": r1 - r0, "colspan": c1 - c0,
                "text": c.get("text"), "header": bool(c.get("column_header")),
            })
        if not specs or max_row <= 0 or max_col <= 0:
            return ""
        anchor = {(s["r0"], s["c0"]): s for s in specs}
        out = ["<table>"]
        for r in range(max_row):
            row_cells = []
            c = 0
            while c < max_col:
                spec = anchor.get((r, c))
                if spec is None:
                    c += 1
                    continue
                tag = "th" if spec["header"] else "td"
                attrs = ""
                if spec["rowspan"] > 1:
                    attrs += f' rowspan="{spec["rowspan"]}"'
                if spec["colspan"] > 1:
                    attrs += f' colspan="{spec["colspan"]}"'
                row_cells.append(f"<{tag}{attrs}>{spec['text']}</{tag}>")
                c += spec["colspan"] if spec["colspan"] > 0 else 1
            out.append("<tr>" + "".join(row_cells) + "</tr>")
        out.append("</table>")
        return "".join(out)

    def _json_to_tables(self, doc: dict):
        """Mirror ``_transfer_to_tables`` on a converted Docling JSON document."""
        tables = []
        ref_to_text = {}
        texts = doc.get("texts")
        if isinstance(texts, list):
            for item in texts:
                if isinstance(item, dict) and item.get("self_ref"):
                    ref_to_text[item["self_ref"]] = (item.get("text") or "").strip()

        for tab in doc.get("tables") or []:
            if not isinstance(tab, dict):
                continue
            img = None
            positions = ""
            bbox = self._json_bbox(tab)
            if bbox:
                img, positions = self.cropout_docling_table(bbox.page_no, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
                if not positions and bbox.page_no in self.page_heights:
                    height = self.page_heights[bbox.page_no]
                    x0, y0 = int(bbox.x0), int(height - bbox.y0)
                    x1, y1 = int(bbox.x1), int(height - bbox.y1)
                    positions = [(bbox.page_no - 1, x0, x1, y0, y1)]
            html = self._table_html_from_json(tab)
            if not html:
                html = (tab.get("text") or "").strip()
            tables.append(((img, html), positions if positions else ""))

        for pic in doc.get("pictures") or []:
            if not isinstance(pic, dict):
                continue
            img = None
            positions = ""
            bbox = self._json_bbox(pic)
            if bbox:
                img, positions = self.cropout_docling_table(bbox.page_no, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
                if not positions and bbox.page_no in self.page_heights:
                    height = self.page_heights[bbox.page_no]
                    x0, y0 = int(bbox.x0), int(height - bbox.y0)
                    x1, y1 = int(bbox.x1), int(height - bbox.y1)
                    positions = [(bbox.page_no - 1, x0, x1, y0, y1)]
            captions = []
            for cap in pic.get("captions") or []:
                ref = cap.get("$ref") if isinstance(cap, dict) else ""
                caption = ref_to_text.get(ref, "")
                if caption:
                    captions.append(caption)
            description = self._picture_description_text(pic)
            if description:
                captions.append(description)
            tables.append(((img, captions), positions if positions else ""))
        return tables

    @staticmethod
    def _sections_from_remote_text(text: str, parse_method: str) -> list[tuple[str, ...]]:
        txt = (text or "").strip()
        if not txt:
            return []
        if parse_method in {"manual", "pipeline"}:
            return [(txt, DoclingContentType.TEXT.value, "")]
        if parse_method == "paper":
            return [(txt, DoclingContentType.TEXT.value)]
        return [(txt, "")]

    @staticmethod
    def _extract_remote_document_entries(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("document"), dict):
            return [payload["document"]]
        if isinstance(payload.get("documents"), list):
            return [d for d in payload["documents"] if isinstance(d, dict)]
        if isinstance(payload.get("results"), list):
            docs = []
            for it in payload["results"]:
                if isinstance(it, dict):
                    if isinstance(it.get("document"), dict):
                        docs.append(it["document"])
                    elif isinstance(it.get("result"), dict):
                        docs.append(it["result"])
                    else:
                        docs.append(it)
            return docs
        return []

    @staticmethod
    def _looks_like_chunk_response(payload: Any) -> bool:
        """Return True iff ``payload`` looks like a chunking endpoint response.

        A chunk response is either a non-empty top-level list or a dict that
        carries a non-empty ``results`` or ``chunks`` list. A standard
        conversion response (``{"document": ..., "status": ...}``) does not
        match, so a server that silently ignored the ``do_chunking`` flag is
        correctly classified as standard even when the request payload asked
        for chunking.
        """
        if isinstance(payload, list):
            return bool(payload)
        if isinstance(payload, dict):
            for key in ("results", "chunks"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    return True
        return False

    def _parse_pdf_remote(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        parse_method: str = "raw",
        docling_server_url: Optional[str] = None,
        request_timeout: Optional[int] = None,
        page_range: Optional[tuple[int, int]] = None,
    ):
        """
        Parses a PDF document using a remote Docling server.

        Sends the document with chunking options first, then falls back to a
        standard conversion payload if the server rejects the chunking parameters.
        The chunked-vs-standard parsing decision is made from the **response
        shape**, not the request shape: Docling Serve silently drops unknown
        fields such as ``do_chunking`` and returns a standard conversion
        response, so the response is treated as standard even when chunking
        was requested.
        """
        server_url = self._effective_server_url(docling_server_url)
        if not server_url:
            raise RuntimeError("[Docling] DOCLING_SERVER_URL is not configured.")

        timeout = request_timeout or self.request_timeout
        if binary is not None:
            if isinstance(binary, (bytes, bytearray)):
                pdf_bytes = bytes(binary)
            else:
                pdf_bytes = bytes(binary.getbuffer())
        else:
            src_path = Path(filepath)
            if not src_path.exists():
                raise FileNotFoundError(f"PDF not found: {src_path}")
            with open(src_path, "rb") as f:
                pdf_bytes = f.read()

        if callback:
            callback(0.2, f"[Docling] Requesting external server: {server_url}")

        filename = Path(filepath).name or "input.pdf"
        b64 = base64.b64encode(pdf_bytes).decode("ascii")

        # docling-serve's ConvertDocumentsOptions.page_range is Docling's own field
        # (docling.datamodel.service.options): 1-based, both bounds inclusive.
        # Docling still numbers the pages it returns from the start of the
        # document, so nothing downstream needs rebasing. A task covering the
        # whole document omits the field entirely.
        range_opt = {"page_range": list(page_range)} if page_range else {}

        # Shared conversion options. image_export_mode is "placeholder" so page
        # images are emitted as <!-- image --> markers instead of inline base64
        # data URIs: binary image payloads must never reach the embedding model
        # as supposedly-"textual" content. do_ocr enables OCR of scanned pages.
        base_opts = {
            "from_formats": ["pdf"],
            "to_formats": ["json", "md", "text"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            **range_opt,
        }

        # Picture description: replace every image with a description produced by
        # the preset's vision model (external_vlm, by default). Only sent when the
        # switch is on, so an older docling-serve deployment that drops unknown
        # options sees no change. The server emits the same value it needs to
        # include, so images are never sent as base64 to the embedding model.
        if os.environ.get("DOCLING_PICTURE_DESCRIPTION", "0").strip().lower() in ("1", "true", "yes", "on"):
            picture_opts = {
                "do_picture_description": True,
                "picture_description_preset": os.environ.get("DOCLING_PICTURE_DESCRIPTION_PRESET", "external_vlm").strip() or "external_vlm",
                "picture_description_area_threshold": float(os.environ.get("DOCLING_PICTURE_DESCRIPTION_AREA_THRESHOLD", "0.01") or 0.01),
            }
            base_opts.update(picture_opts)
            self.logger.info(
                f"[Docling] Picture description enabled via preset "
                f"{picture_opts['picture_description_preset']} "
                f"(area_threshold={picture_opts['picture_description_area_threshold']})."
            )

        # Standard payloads
        # Standard fallback payloads (no chunking)
        v1_payload_standard = {
            "options": dict(base_opts),
            "sources": [{"kind": "file", "filename": filename, "base64_string": b64}],
        }
        v1alpha_payload_standard = {
            "options": dict(base_opts),
            "file_sources": [{"filename": filename, "base64_string": b64}],
        }

        # --- NEW: Correct API Contract for Chunking ---
        chunking_opts = {
            **base_opts,
            "do_chunking": True,
            "chunking_options": {
                "max_tokens": 512,
                "overlap": 50,
                "tokenizer": "sentencepiece",  # Required by Docling contract
            },
        }
        v1_payload_chunked = {
            "options": chunking_opts,
            "sources": [{"kind": "file", "filename": filename, "base64_string": b64}],
        }
        v1alpha_payload_chunked = {
            "options": chunking_opts,
            "file_sources": [{"filename": filename, "base64_string": b64}],
        }

        errors = []
        response_json = None
        is_chunked_response = False

        # Try chunked endpoints first, then fall back to standard if the server is older
        for endpoint, payload, chunk_flag in (
            ("/v1/convert/source", v1_payload_chunked, True),
            ("/v1alpha/convert/source", v1alpha_payload_chunked, True),
            ("/v1/convert/source", v1_payload_standard, False),
            ("/v1alpha/convert/source", v1alpha_payload_standard, False),
        ):
            try:
                resp = requests.post(
                    f"{server_url}{endpoint}",
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code < 300:
                    response_json = resp.json()
                    response_is_chunk = self._looks_like_chunk_response(response_json)
                    is_chunked_response = chunk_flag and response_is_chunk

                    if chunk_flag and response_is_chunk:
                        self.logger.info(f"[Docling] Successfully used native chunking on: {endpoint}")
                    elif chunk_flag:
                        self.logger.warning(f"[Docling] Server ignored chunking request on {endpoint}; treating response as standard conversion.")
                    else:
                        self.logger.info(f"[Docling] Chunking unavailable, fell back to standard: {endpoint}")
                    break

                # If chunking request is rejected (e.g., 422 Unprocessable Entity on older servers),
                # log it and let the loop naturally fall back to the standard payload.
                if chunk_flag:
                    self.logger.warning(f"[Docling] Server rejected chunking parameters: HTTP {resp.status_code}")
                    continue

                errors.append(f"{endpoint}: HTTP {resp.status_code} {resp.text[:300]}")

            except Exception as exc:
                self.logger.error(f"[Docling] Request error on {endpoint}: {exc}")
                errors.append(f"{endpoint}: {exc}")

        if response_json is None:
            raise RuntimeError("[Docling] remote convert failed: " + " | ".join(errors))

        sections: list[tuple[str, ...]] = []
        tables = []

        # --- NEW: Handle Native Chunked Response ---
        if is_chunked_response:
            # The chunking endpoint returns an array of chunk items
            chunks = response_json if isinstance(response_json, list) else response_json.get("results", [])
            for chunk_data in chunks:
                if not isinstance(chunk_data, dict):
                    continue
                # Depending on the exact docling-serve spec, the text might be nested
                chunk_text = chunk_data.get("text", "")
                if not chunk_text and isinstance(chunk_data.get("chunk"), dict):
                    chunk_text = chunk_data["chunk"].get("text", "")

                if isinstance(chunk_text, str) and chunk_text.strip():
                    # Feed the pre-sliced chunks directly into RAGFlow's expected format
                    sections.extend(self._sections_from_remote_text(chunk_text, parse_method=parse_method))

            if callback:
                callback(0.95, f"[Docling] Native chunks received: {len(sections)}")
            if sections:
                return sections, tables

            self.logger.warning("[Docling] Native chunking returned no usable chunks; trying standard response parsing.")

        # --- FALLBACK: Standard RAGFlow parsing for older docling servers ---
        docs = self._extract_remote_document_entries(response_json)
        if not docs:
            raise RuntimeError("[Docling] remote response does not contain parsed documents.")

        for doc in docs:
            json_content = doc.get("json_content")
            if isinstance(json_content, dict):
                # Prefer the JSON export: it carries per-item page/bbox geometry,
                # which is what powers click-to-highlight and chunk preview images
                # (tags + page_images). Fall back to the flat markdown/text export
                # for servers that return a JSON-less document.
                self._set_page_heights(json_content)
                json_sections = self._json_to_sections(json_content, parse_method=parse_method)
                if json_sections:
                    sections.extend(json_sections)
                    json_tables = self._json_to_tables(json_content)
                    if json_tables:
                        tables.extend(json_tables)
                    continue

            md = doc.get("md_content")
            txt = doc.get("text_content")
            if isinstance(md, str) and md.strip():
                sections.extend(self._sections_from_remote_text(md, parse_method=parse_method))
            elif isinstance(txt, str) and txt.strip():
                sections.extend(self._sections_from_remote_text(txt, parse_method=parse_method))

        if callback:
            callback(0.95, f"[Docling] Remote sections: {len(sections)}, tables: {len(tables)}")
        return sections, tables

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        method: str = "auto",
        delete_output: bool = True,
        parse_method: str = "raw",
        docling_server_url: Optional[str] = None,
        request_timeout: Optional[int] = None,
        page_from: int = 0,
        page_to: int = MAXIMUM_PAGE_NUMBER,
    ):
        self.outlines = extract_pdf_outlines(binary if binary is not None else filepath)

        if not self.check_installation(docling_server_url=docling_server_url):
            raise RuntimeError("Docling not available, please install `docling`")

        # RAGFlow splits a large PDF into several page-range tasks and calls this
        # once per range. Without forwarding the range every task converts the
        # whole document, so the work is repeated N times and each request is as
        # slow and as memory-hungry as the entire file.
        page_range = self._resolve_page_range(page_from, page_to)
        if page_range:
            self.logger.info(f"[Docling] resolved page range {page_range} (from page_from={page_from} page_to={page_to})")

        server_url = self._effective_server_url(docling_server_url)
        if server_url:
            # Render the task's page window so the shared crop()/cropout_docling_table
            # machinery can attach chunk preview images and highlight rects, mirroring
            # the local docling and deepdoc paths. Previews are rasterised locally
            # (pdfplumber) from the PDF, never fetched from the server, so the
            # image_export_mode setting has no bearing on them.
            try:
                if page_range:
                    self.__images__(binary if binary is not None else filepath, zoomin=1, page_from=page_range[0] - 1, page_to=page_range[1])
                else:
                    self.__images__(binary if binary is not None else filepath, zoomin=1)
            except Exception as e:
                self.logger.warning(f"[Docling] render pages failed: {e}")
            return self._parse_pdf_remote(
                filepath=filepath,
                binary=binary,
                callback=callback,
                parse_method=parse_method,
                docling_server_url=server_url,
                request_timeout=request_timeout,
                page_range=page_range,
            )

        if binary is not None:
            tmpdir = Path(output_dir) if output_dir else Path.cwd() / ".docling_tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            name = Path(filepath).name or "input.pdf"
            tmp_pdf = tmpdir / name
            with open(tmp_pdf, "wb") as f:
                if isinstance(binary, (bytes, bytearray)):
                    f.write(binary)
                else:
                    f.write(binary.getbuffer())
            src_path = tmp_pdf
        else:
            src_path = Path(filepath)
            if not src_path.exists():
                raise FileNotFoundError(f"PDF not found: {src_path}")

        if callback:
            callback(0.1, f"[Docling] Converting: {src_path}")

        try:
            # Render the same window the conversion covers, so a page-split task
            # does not pay the whole document's rasterisation cost, and so
            # page_from matches the images the position logic indexes into.
            if page_range:
                self.__images__(str(src_path), zoomin=1, page_from=page_range[0] - 1, page_to=page_range[1])
            else:
                self.__images__(str(src_path), zoomin=1)
        except Exception as e:
            self.logger.warning(f"[Docling] render pages failed: {e}")

        do_formula_enrichment = os.environ.get("DOCLING_FORMULA_ENRICHMENT", "0").strip().lower() in ("1", "true", "yes", "on")
        self.logger.info(f"[Docling] Local conversion (formula_enrichment={do_formula_enrichment}): {src_path}")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_formula_enrichment = do_formula_enrichment
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        conv_res = conv.convert(str(src_path), page_range=page_range) if page_range else conv.convert(str(src_path))
        doc = conv_res.document
        if callback:
            callback(0.7, f"[Docling] Parsed doc: {getattr(doc, 'num_pages', 'n/a')} pages")

        sections = self._transfer_to_sections(doc, parse_method=parse_method)
        tables = self._transfer_to_tables(doc)

        if callback:
            callback(0.95, f"[Docling] Sections: {len(sections)}, Tables: {len(tables)}")

        if binary is not None and delete_output:
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                pass

        if callback:
            callback(1.0, "[Docling] Done.")
        return sections, tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = DoclingParser()
    print("Docling available:", parser.check_installation())
    sections, tables = parser.parse_pdf(filepath="test_docling/toc.pdf", binary=None)
    print(len(sections), len(tables))
