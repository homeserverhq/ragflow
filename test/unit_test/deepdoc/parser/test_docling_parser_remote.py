from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeImage:
    """Stands in for a rendered page; only its height is ever read."""

    def __init__(self, height: int, width: int = 600):
        self.size = (width, height)


def _load_docling_parser(monkeypatch):
    common_pkg = types.ModuleType("common")
    constants_mod = types.ModuleType("common.constants")
    constants_mod.MAXIMUM_PAGE_NUMBER = 1000

    deepdoc_pkg = types.ModuleType("deepdoc")
    parser_pkg = types.ModuleType("deepdoc.parser")
    parser_pkg.__path__ = []
    utils_mod = types.ModuleType("deepdoc.parser.utils")
    utils_mod.extract_pdf_outlines = lambda _source: []

    pil_pkg = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.Image = object
    pil_pkg.Image = image_mod

    monkeypatch.setitem(sys.modules, "common", common_pkg)
    monkeypatch.setitem(sys.modules, "common.constants", constants_mod)
    monkeypatch.setitem(sys.modules, "deepdoc", deepdoc_pkg)
    monkeypatch.setitem(sys.modules, "deepdoc.parser", parser_pkg)
    monkeypatch.setitem(sys.modules, "deepdoc.parser.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "pdfplumber", types.ModuleType("pdfplumber"))
    monkeypatch.setitem(sys.modules, "PIL", pil_pkg)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)

    spec = importlib.util.spec_from_file_location(
        "_docling_parser_under_test",
        ROOT / "deepdoc" / "parser" / "docling_parser.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p2
def test_remote_chunked_200_standard_payload_falls_back(monkeypatch):
    module = _load_docling_parser(monkeypatch)
    calls = []

    def fake_post(_url, json, timeout):
        calls.append((json, timeout))
        return _Response({"document": {"md_content": "# Parsed\n\nbody"}})

    monkeypatch.setattr(module.requests, "post", fake_post)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    sections, tables = parser._parse_pdf_remote("sample.pdf", binary=b"%PDF", parse_method="raw")

    assert sections == [("# Parsed\n\nbody", "")]
    assert tables == []
    assert calls[0][0]["options"]["do_chunking"] is True


@pytest.mark.p2
def test_chunk_shape_helper_recognises_chunk_payloads(monkeypatch):
    """A response that is chunk-shaped (list, or dict with non-empty results/chunks)
    is classified as chunked regardless of which payload was sent."""
    module = _load_docling_parser(monkeypatch)
    assert module.DoclingParser._looks_like_chunk_response([{"text": "chunk-1"}]) is True
    assert module.DoclingParser._looks_like_chunk_response({"results": [{"text": "chunk-1"}, {"text": "chunk-2"}]}) is True
    assert module.DoclingParser._looks_like_chunk_response({"chunks": [{"text": "chunk-1"}]}) is True


@pytest.mark.p2
def test_chunk_shape_helper_rejects_standard_payloads(monkeypatch):
    """A standard conversion response, empty containers, and non-payload types
    are correctly classified as not-chunked."""
    module = _load_docling_parser(monkeypatch)
    standard = {"document": {"md_content": "body"}, "status": "success"}
    assert module.DoclingParser._looks_like_chunk_response(standard) is False
    assert module.DoclingParser._looks_like_chunk_response({}) is False
    assert module.DoclingParser._looks_like_chunk_response({"results": []}) is False
    assert module.DoclingParser._looks_like_chunk_response({"chunks": []}) is False
    assert module.DoclingParser._looks_like_chunk_response([]) is False
    assert module.DoclingParser._looks_like_chunk_response("not-a-payload") is False
    assert module.DoclingParser._looks_like_chunk_response(None) is False
    assert module.DoclingParser._looks_like_chunk_response(42) is False


@pytest.mark.p2
def test_remote_chunked_request_with_results_list_is_treated_as_chunked(monkeypatch):
    """A server that returns a ``results`` list (Docling Serve's native chunk
    shape) is treated as chunked and each chunk becomes a section."""
    module = _load_docling_parser(monkeypatch)

    def fake_post(_url, json, timeout):
        return _Response({"results": [{"text": "alpha"}, {"text": "beta"}]})

    monkeypatch.setattr(module.requests, "post", fake_post)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    sections, tables = parser._parse_pdf_remote("sample.pdf", binary=b"%PDF", parse_method="raw")

    assert sections == [("alpha", ""), ("beta", "")]
    assert tables == []


@pytest.mark.p2
def test_remote_top_level_list_response_is_treated_as_chunked(monkeypatch):
    """A server that returns a top-level JSON array of chunks is treated
    as chunked (matches the existing implicit assumption in the code)."""
    module = _load_docling_parser(monkeypatch)

    def fake_post(_url, json, timeout):
        return _Response([{"text": "first"}, {"text": "second"}])

    monkeypatch.setattr(module.requests, "post", fake_post)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    sections, _ = parser._parse_pdf_remote("sample.pdf", binary=b"%PDF", parse_method="raw")

    assert sections == [("first", ""), ("second", "")]


@pytest.mark.p2
def test_remote_chunked_request_with_ignored_flag_does_not_log_success(monkeypatch, caplog):
    """When Docling Serve silently drops the ``do_chunking`` flag and returns
    a standard conversion response, RAGFlow must not log a chunking-success
    message and must log a warning instead."""
    module = _load_docling_parser(monkeypatch)

    def fake_post(_url, json, timeout):
        return _Response({"document": {"md_content": "real content"}, "status": "success"})

    monkeypatch.setattr(module.requests, "post", fake_post)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    with caplog.at_level(logging.DEBUG, logger="DoclingParser"):
        sections, _ = parser._parse_pdf_remote("sample.pdf", binary=b"%PDF", parse_method="raw")

    assert sections == [("real content", "")]
    flat = " ".join(record.getMessage() for record in caplog.records)
    assert "Successfully used native chunking" not in flat
    assert "Server ignored chunking request" in flat


def _capture_remote_payloads(monkeypatch, module, reject_chunking: bool = False):
    """Fake out the remote server and the installation probe so ``parse_pdf`` can
    be driven end to end. Returns the list of payloads it posts; with
    ``reject_chunking`` the chunked attempts 422 so the standard fallback
    payloads are exercised too."""
    payloads = []

    def fake_post(_url, json, timeout):
        payloads.append(json)
        if reject_chunking and json["options"].get("do_chunking"):
            return _Response(None, status_code=422)
        return _Response({"document": {"md_content": "body"}})

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.DoclingParser, "check_installation", lambda _self, **_kw: True)
    return payloads


@pytest.mark.p2
def test_resolve_page_range_translates_ragflow_convention(monkeypatch):
    """RAGFlow's 0-based/exclusive task range becomes Docling's 1-based/inclusive
    one; an un-narrowed or empty range asks for the whole document instead."""
    module = _load_docling_parser(monkeypatch)
    resolve = module.DoclingParser._resolve_page_range

    assert resolve(0, 13) == (1, 13)
    assert resolve(144, 157) == (145, 157)
    assert resolve(0, module.MAXIMUM_PAGE_NUMBER) is None
    # Docling rejects end < start, so a degenerate range must not be sent.
    assert resolve(12, 12) is None


@pytest.mark.p2
def test_resolve_page_range_clamps_out_of_bounds_input(monkeypatch):
    """``parse_pdf`` is public, so out-of-range bounds must be clamped rather
    than passed on: Docling rejects a start below 1 outright."""
    module = _load_docling_parser(monkeypatch)
    resolve = module.DoclingParser._resolve_page_range
    maximum = module.MAXIMUM_PAGE_NUMBER

    assert resolve(-1, 5) == (1, 5)
    assert resolve(-10, maximum + 100) is None
    assert resolve(5, maximum + 100) == (6, maximum)

    for page_from, page_to in ((-1, 5), (5, maximum + 100), (0, 13), (144, 157)):
        start, end = resolve(page_from, page_to)
        assert start >= 1
        assert end >= start


@pytest.mark.p2
def test_line_tag_is_relative_to_the_rendered_window(monkeypatch):
    """Docling numbers pages from the start of the document, but page_images only
    holds the rendered window and ``crop`` adds ``page_from`` back — so a tag must
    name the page relative to that window."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    parser.page_from = 144
    parser.page_images = [_FakeImage(800), _FakeImage(800)]

    # absolute page 145 is the first page of a window starting at page_from=144
    bbox = module._BBox(page_no=145, x0=1.0, y0=10.0, x1=2.0, y1=20.0)
    tag = parser._make_line_tag(bbox)

    assert tag.startswith("@@1\t")
    # y coordinates are flipped against the height of the page actually rendered
    assert "790.0\t780.0" in tag
    # and crop() maps it back onto the absolute page it came from
    pages, *_ = module.DoclingParser.extract_positions(tag)[0]
    assert pages[0] + parser.page_from == 144


@pytest.mark.p2
def test_page_range_is_threaded_into_remote_payload(monkeypatch):
    """A task that owns pages 145-157 must ask Docling Serve for exactly that
    window instead of converting the whole document — on every payload variant
    the fallback chain can reach, not just the first one."""
    module = _load_docling_parser(monkeypatch)
    payloads = _capture_remote_payloads(monkeypatch, module, reject_chunking=True)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.parse_pdf("sample.pdf", binary=b"%PDF", page_from=144, page_to=157)

    # two rejected chunked attempts, then the standard payload that succeeds
    assert len(payloads) == 3
    assert all(payload["options"]["page_range"] == [145, 157] for payload in payloads)


@pytest.mark.p2
def test_full_document_request_omits_page_range(monkeypatch):
    """A task covering the whole document omits ``page_range`` entirely, so the
    server converts everything."""
    module = _load_docling_parser(monkeypatch)
    payloads = _capture_remote_payloads(monkeypatch, module, reject_chunking=True)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.parse_pdf("sample.pdf", binary=b"%PDF")

    assert len(payloads) == 3
    assert all("page_range" not in payload["options"] for payload in payloads)


def _drive_local_conversion(monkeypatch, module, tmp_path, **parse_kwargs):
    """Run ``parse_pdf``'s local branch against stubbed docling classes. Returns
    the page bounds ``__images__`` was rendered with and the range handed to
    ``DocumentConverter.convert``."""
    captured = {}

    class _FakeConverter:
        def __init__(self, *_a, **_kw):
            pass

        def convert(self, _source, page_range=None):
            captured["convert_range"] = page_range
            return types.SimpleNamespace(document=types.SimpleNamespace(texts=[], tables=[], pictures=[]))

    def fake_images(_self, _fnm, zoomin=1, page_from=0, page_to=module.MAXIMUM_PAGE_NUMBER, callback=None):
        captured["rendered"] = (page_from, page_to)

    monkeypatch.setattr(module, "DocumentConverter", _FakeConverter)
    monkeypatch.setattr(module, "PdfPipelineOptions", lambda: types.SimpleNamespace(do_formula_enrichment=False))
    monkeypatch.setattr(module, "PdfFormatOption", lambda **_kw: object())
    monkeypatch.setattr(module, "InputFormat", types.SimpleNamespace(PDF="pdf"))
    monkeypatch.setattr(module.DoclingParser, "__images__", fake_images)
    monkeypatch.setattr(module.DoclingParser, "check_installation", lambda _self, **_kw: True)
    monkeypatch.setattr(module.DoclingParser, "_effective_server_url", lambda _self, *_a, **_kw: "")

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    parser = module.DoclingParser()
    parser.parse_pdf(str(pdf), **parse_kwargs)
    return captured


@pytest.mark.p2
def test_local_conversion_renders_only_the_selected_pages(monkeypatch, tmp_path):
    """Rasterising is as expensive as converting, so a page-split task must render
    its own window only — and page_from must match it, since the position logic
    indexes page_images relative to the window."""
    module = _load_docling_parser(monkeypatch)
    captured = _drive_local_conversion(monkeypatch, module, tmp_path, page_from=144, page_to=157)

    assert captured["convert_range"] == (145, 157)
    # __images__ is 0-based with an exclusive stop, so the same 13 pages
    assert captured["rendered"] == (144, 157)


@pytest.mark.p2
def test_local_conversion_of_whole_document_renders_every_page(monkeypatch, tmp_path):
    """A task that owns the whole document renders it whole and converts it whole."""
    module = _load_docling_parser(monkeypatch)
    captured = _drive_local_conversion(monkeypatch, module, tmp_path)

    assert captured["convert_range"] is None
    assert captured["rendered"] == (0, module.MAXIMUM_PAGE_NUMBER)


@pytest.mark.p2
def test_crop_without_page_images_returns_positions_only(monkeypatch):
    """When page rendering failed, ``crop`` returns positions from the tags (so
    click-to-highlight keeps working) without generating any preview image."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.page_images = []

    tag = "@@1\t1.0\t2.0\t3.0\t4.0##"
    pic, positions = parser.crop(tag, need_position=True)
    assert pic is None
    assert positions == [(0, 1, 2, 3, 4)]
    assert parser.crop(tag) is None


@pytest.mark.p2
def test_crop_drops_positions_beyond_rendered_pages(monkeypatch):
    """A tag naming a page past the rendered range must be filtered out rather
    than indexed, mirroring the range check in cropout_docling_table."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser(docling_server_url="http://docling.local")
    # a single rendered page; the sentinel is never indexed because the
    # out-of-range position is dropped first.
    parser.page_images = [object()]

    tag = "@@5\t1.0\t2.0\t3.0\t4.0##"
    assert parser.crop(tag, need_position=True) == (None, None)
    assert parser.crop(tag) is None


class _FakePageImg:
    """Stands in for a rendered page image; only its size and crop are used."""

    def __init__(self, width: int, height: int):
        self.size = (width, height)

    def crop(self, _box):
        return self

    def convert(self, _mode):
        return self


def _fake_docling_json():
    """A DoclingDocument-shaped dict matching the docling-serve JSON export:
    flat ``texts`` list, ``pages`` keyed by page number, and tables carrying
    per-cell offsets under ``data.table_cells``."""
    return {
        "pages": {
            "1": {"size": {"width": 612.0, "height": 792.0}},
            "5": {"size": {"width": 612.0, "height": 792.0}},
        },
        "texts": [
            {
                "self_ref": "#/texts/0", "label": "section_header", "parent": {"$ref": "#/body"},
                "text": "Title",
                "prov": [{"page_no": 1, "bbox": {"l": 63.595, "t": 693.014878, "r": 548.410483, "b": 680.1176442, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/1", "label": "text", "parent": {"$ref": "#/body"},
                "text": "A body paragraph.",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 656.0, "r": 548.4, "b": 640.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/2", "label": "list_item",
                "text": "- item",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 620.0, "r": 300.0, "b": 610.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/3", "label": "formula",
                "text": "E=mc^2",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 600.0, "r": 200.0, "b": 590.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/4", "label": "text", "parent": {"$ref": "#/not-body"},
                "text": "Ignored, not a body text.",
                "prov": [{"page_no": 1, "bbox": {"l": 0.0, "t": 0.0, "r": 10.0, "b": 10.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/5", "label": "caption",
                "text": "Figure: overview",
                "prov": [{"page_no": 1, "bbox": {"l": 10.0, "t": 10.0, "r": 20.0, "b": 20.0, "coord_origin": "BOTTOMLEFT"}}],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0", "label": "table",
                "prov": [{"page_no": 5, "bbox": {"l": 155.5, "t": 717.4, "r": 456.4, "b": 637.4, "coord_origin": "BOTTOMLEFT"}}],
                "data": {
                    "num_rows": 2, "num_cols": 2,
                    "table_cells": [
                        {"text": "Asset", "start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "column_header": True},
                        {"text": "Version", "start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "column_header": True},
                        {"text": "Docling", "start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "column_header": False},
                        {"text": "2.5.2", "start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "column_header": False},
                    ],
                },
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0", "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 92.1, "t": 737.1, "r": 514.7, "b": 538.6, "coord_origin": "BOTTOMLEFT"}}],
                "captions": [{"$ref": "#/texts/5"}],
            }
        ],
    }


@pytest.mark.p2
def test_json_bbox_extracts_first_prov(monkeypatch):
    module = _load_docling_parser(monkeypatch)
    item = {"prov": [{"page_no": 5, "bbox": {"l": 155.5, "t": 717.4, "r": 456.4, "b": 637.4, "coord_origin": "BOTTOMLEFT"}}]}
    bbox = module.DoclingParser._json_bbox(item)
    assert (bbox.page_no, bbox.x0, bbox.y0, bbox.x1, bbox.y1) == (5, 155.5, 717.4, 456.4, 637.4)
    assert module.DoclingParser._json_bbox({}) is None
    assert module.DoclingParser._json_bbox({"prov": []}) is None


@pytest.mark.p2
def test_make_line_tag_flips_with_page_heights_when_no_images(monkeypatch):
    """Without rendered page images the tag still flips y to TOP-origin, using
    the page height recorded from the JSON export (keyed by absolute page)."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    parser.page_images = []
    parser.page_heights = {145: 800}
    parser.page_from = 144

    bbox = module._BBox(page_no=145, x0=1.0, y0=10.0, x1=2.0, y1=20.0)
    assert parser._make_line_tag(bbox) == "@@1\t1.0\t2.0\t790.0\t780.0##"


@pytest.mark.p2
def test_json_to_sections_builds_tagged_sections(monkeypatch):
    """Text/headers/lists/equations from the JSON export become sections carrying
    position tags (top-origin coordinates), exactly like the local path."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _fake_docling_json()
    parser._set_page_heights(doc)
    sections = parser._json_to_sections(doc, parse_method="raw")

    assert sections == [
        ("Title", "@@1\t63.6\t548.4\t99.0\t111.9##"),
        ("A body paragraph.", "@@1\t63.6\t548.4\t136.0\t152.0##"),
        ("- item", "@@1\t63.6\t300.0\t172.0\t182.0##"),
        ("E=mc^2", "@@1\t63.6\t200.0\t192.0\t202.0##"),
    ]


@pytest.mark.p2
def test_json_to_sections_parse_method_shapes(monkeypatch):
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _fake_docling_json()
    parser._set_page_heights(doc)

    manual = parser._json_to_sections(doc, parse_method="manual")
    assert manual[0] == ("Title", "text", "@@1\t63.6\t548.4\t99.0\t111.9##")

    paper = parser._json_to_sections(doc, parse_method="paper")
    assert paper[0] == ("Title@@1\t63.6\t548.4\t99.0\t111.9##", "text")


@pytest.mark.p2
def test_json_to_tables_positions_only_without_images(monkeypatch):
    """Table and figure items carry highlight positions computed from their
    bounding box even when no page images were rendered."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _fake_docling_json()
    parser._set_page_heights(doc)

    tables = parser._json_to_tables(doc)
    assert len(tables) == 2

    (tab_img, tab_html), tab_pos = tables[0]
    assert tab_img is None
    assert "<table>" in tab_html
    assert "Asset" in tab_html and "Docling" in tab_html
    assert tab_pos == [(4, 155, 456, 74, 154)]

    (pic_img, captions), pic_pos = tables[1]
    assert pic_img is None
    assert captions == ["Figure: overview"]
    assert pic_pos == [(0, 92, 514, 54, 253)]


@pytest.mark.p2
def test_json_to_tables_uses_page_images_for_crop(monkeypatch):
    """When the task window is rendered, tables get a cropped preview image plus
    its highlight positions, matching the deepdoc/local-docling table path."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    parser.page_images = [_FakePageImg(612, 792)]
    doc = _fake_docling_json()
    # move the fake table onto the rendered page so the crop path is exercised
    doc["tables"][0]["prov"] = [{"page_no": 1, "bbox": {"l": 155.0, "t": 717.0, "r": 456.0, "b": 637.0, "coord_origin": "BOTTOMLEFT"}}]
    parser._set_page_heights(doc)

    (tab_img, tab_html), tab_pos = parser._json_to_tables(doc)[0]
    assert tab_img is not None
    assert tab_pos == [(0, 155.0, 456.0, 75.0, 155.0)]


@pytest.mark.p2
def test_remote_branch_renders_the_task_window(monkeypatch):
    """A page-split task renders only its own page window on the remote path,
    matching the local branch, so crop() can build previews and rects over the
    exact pages Docling was asked to convert."""
    module = _load_docling_parser(monkeypatch)
    captured = {}

    def fake_images(_self, _fnm, zoomin=1, page_from=0, page_to=module.MAXIMUM_PAGE_NUMBER, callback=None):
        captured["rendered"] = (page_from, page_to)

    def fake_post(_url, json, timeout):
        return _Response({"document": {"json_content": _fake_docling_json(), "md_content": "flat fallback"}})

    monkeypatch.setattr(module.DoclingParser, "__images__", fake_images)
    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.DoclingParser, "check_installation", lambda _self, **_kw: True)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.parse_pdf("sample.pdf", binary=b"%PDF", page_from=144, page_to=157)

    assert captured["rendered"] == (144, 157)


@pytest.mark.p2
def test_standard_response_prefers_json_content(monkeypatch):
    """The standard-conversion response is parsed from ``json_content`` into
    tagged sections (+ tables) instead of a single plain markdown blob."""
    module = _load_docling_parser(monkeypatch)

    def fake_post(_url, json, timeout):
        return _Response({"document": {"json_content": _fake_docling_json(), "md_content": "flat fallback"}})

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.DoclingParser, "check_installation", lambda _self, **_kw: True)
    monkeypatch.setattr(module.DoclingParser, "__images__", lambda _self, *_a, **_k: None)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    sections, tables = parser.parse_pdf("sample.pdf", binary=b"%PDF", parse_method="raw")

    assert sections == [
        ("Title", "@@1\t63.6\t548.4\t99.0\t111.9##"),
        ("A body paragraph.", "@@1\t63.6\t548.4\t136.0\t152.0##"),
        ("- item", "@@1\t63.6\t300.0\t172.0\t182.0##"),
        ("E=mc^2", "@@1\t63.6\t200.0\t192.0\t202.0##"),
    ]
    assert tables and "Asset" in tables[0][0][1]


@pytest.mark.p2
def test_picture_description_options_omitted_by_default(monkeypatch):
    """With DOCLING_PICTURE_DESCRIPTION unset the picture-description options
    must not be sent, so older docling-serve deployments see no change."""
    module = _load_docling_parser(monkeypatch)
    payloads = _capture_remote_payloads(monkeypatch, module, reject_chunking=True)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.parse_pdf("sample.pdf", binary=b"%PDF")

    assert len(payloads) == 3
    for payload in payloads:
        options = payload["options"]
        assert "do_picture_description" not in options
        assert "picture_description_preset" not in options
        assert "picture_description_area_threshold" not in options


@pytest.mark.p2
def test_picture_description_options_added_when_enabled(monkeypatch):
    """DOCLING_PICTURE_DESCRIPTION=1 adds the picture-description options to every
    payload variant with the configured preset and area threshold."""
    monkeypatch.setenv("DOCLING_PICTURE_DESCRIPTION", "1")
    monkeypatch.setenv("DOCLING_PICTURE_DESCRIPTION_PRESET", "external_vlm")
    monkeypatch.setenv("DOCLING_PICTURE_DESCRIPTION_AREA_THRESHOLD", "0.01")
    module = _load_docling_parser(monkeypatch)
    payloads = _capture_remote_payloads(monkeypatch, module, reject_chunking=True)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.parse_pdf("sample.pdf", binary=b"%PDF")

    assert len(payloads) == 3
    for payload in payloads:
        options = payload["options"]
        assert options["do_picture_description"] is True
        assert options["picture_description_preset"] == "external_vlm"
        assert isinstance(options["picture_description_area_threshold"], float)
        assert abs(options["picture_description_area_threshold"] - 0.01) < 1e-9


@pytest.mark.p2
def test_picture_description_options_use_env_values(monkeypatch):
    """The picture-description preset and threshold honor the env vars, and the
    threshold is parsed as a float."""
    monkeypatch.setenv("DOCLING_PICTURE_DESCRIPTION", "true")
    monkeypatch.setenv("DOCLING_PICTURE_DESCRIPTION_PRESET", "granite_vision")
    monkeypatch.setenv("DOCLING_PICTURE_DESCRIPTION_AREA_THRESHOLD", "0.05")
    module = _load_docling_parser(monkeypatch)
    payloads = _capture_remote_payloads(monkeypatch, module, reject_chunking=True)

    parser = module.DoclingParser(docling_server_url="http://docling.local")
    parser.parse_pdf("sample.pdf", binary=b"%PDF")

    options = payloads[0]["options"]
    assert options["picture_description_preset"] == "granite_vision"
    assert abs(options["picture_description_area_threshold"] - 0.05) < 1e-9


def _picture_description_json():
    """A json_content export whose pictures carry vision-model descriptions."""
    return {
        "pages": {"1": {"size": {"width": 612.0, "height": 792.0}}},
        "texts": [
            {
                "self_ref": "#/texts/0", "label": "text", "parent": {"$ref": "#/body"},
                "text": "A body paragraph.",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 656.0, "r": 548.4, "b": 640.0, "coord_origin": "BOTTOMLEFT"}}],
            },
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0", "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 92.1, "t": 737.1, "r": 514.7, "b": 538.6, "coord_origin": "BOTTOMLEFT"}}],
                "meta": {"description": {"text": "Image Description (g34b): a car"}},
            },
            {
                "self_ref": "#/pictures/1", "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 100.0, "t": 400.0, "r": 200.0, "b": 300.0, "coord_origin": "BOTTOMLEFT"}}],
                "meta": {"description": {"text": "   "}},
                "annotations": [{"kind": "description", "text": "Image Description (g34b): a diagram"}],
            },
            {
                "self_ref": "#/pictures/2", "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 300.0, "t": 500.0, "r": 400.0, "b": 400.0, "coord_origin": "BOTTOMLEFT"}}],
                "meta": {},
            },
        ],
    }


@pytest.mark.p2
def test_json_to_sections_ignores_picture_descriptions(monkeypatch):
    """Picture descriptions must NOT become text sections: they ride inside the
    figure chunk paired with its caption by ``_json_to_tables`` instead."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _picture_description_json()
    parser._set_page_heights(doc)
    sections = parser._json_to_sections(doc, parse_method="raw")

    assert sections == [("A body paragraph.", "@@1\t63.6\t548.4\t136.0\t152.0##")]


@pytest.mark.p2
def test_json_to_tables_pairs_description_with_caption(monkeypatch):
    """A figure's chunk carries its caption followed by the VLM description, so
    both are embedded and retrievable together in one chunk."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _picture_description_json()
    # give the first picture a caption ref so the pairing is exercised
    doc["pictures"][0]["captions"] = [{"$ref": "#/texts/4"}]
    caption_item = {
        "self_ref": "#/texts/4", "label": "caption",
        "text": "Figure X: a shiny car",
        "prov": [{"page_no": 1, "bbox": {"l": 92.1, "t": 730.0, "r": 514.7, "b": 700.0, "coord_origin": "BOTTOMLEFT"}}],
    }
    doc["texts"].append(caption_item)
    # the second picture has no caption but a description (annotation fallback)
    parser._set_page_heights(doc)

    tables = parser._json_to_tables(doc)
    pic_rows = [rows for (img, rows), pos in tables if pos]
    assert any("Figure X: a shiny car" in rows[0] for rows in pic_rows)
    described = [rows for rows in pic_rows if any("Image Description" in r for r in rows)]
    assert len(described) == 2, described
    for rows in described:
        assert rows[-1].startswith("Image Description (g34b):")
    # caption-first ordering when the figure has one
    paired = [rows for rows in described if rows[0].startswith("Figure X:")]
    assert paired == [["Figure X: a shiny car", "Image Description (g34b): a car"]]


@pytest.mark.p2
def test_json_to_tables_description_without_caption_is_not_dropped(monkeypatch):
    """A described figure without a caption still becomes a chunk: the description
    alone makes the rows list non-empty for tokenize_table."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _picture_description_json()
    doc["pictures"][2] = {
        "self_ref": "#/pictures/2", "label": "picture",
        "prov": [{"page_no": 1, "bbox": {"l": 300.0, "t": 500.0, "r": 400.0, "b": 400.0, "coord_origin": "BOTTOMLEFT"}}],
        "meta": {"description": {"text": "Image Description (g34b): a lonely chart"}},
    }
    parser._set_page_heights(doc)

    tables = parser._json_to_tables(doc)
    pic_rows = [rows for (img, rows), pos in tables if isinstance(rows, list)]
    assert any(rows == ["Image Description (g34b): a lonely chart"] for rows in pic_rows)


def _body_tree_json():
    """A json_content export carrying a ``body.children`` tree (the authoritative
    reading order): text, a picture with a description, text, a group holding a
    list_item, then a trailing described picture."""
    return {
        "pages": {"1": {"size": {"width": 612.0, "height": 792.0}}},
        "texts": [
            {
                "self_ref": "#/texts/0", "label": "text", "parent": {"$ref": "#/body"},
                "text": "Intro paragraph.",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 656.0, "r": 548.4, "b": 640.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/1", "label": "text", "parent": {"$ref": "#/body"},
                "text": "Middle paragraph.",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 500.0, "r": 548.4, "b": 484.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/2", "label": "list_item", "parent": {"$ref": "#/groups/0"},
                "text": "- list item",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 300.0, "r": 300.0, "b": 292.0, "coord_origin": "BOTTOMLEFT"}}],
            },
            {
                "self_ref": "#/texts/3", "label": "text", "parent": {"$ref": "#/body"},
                "text": "Closing paragraph.",
                "prov": [{"page_no": 1, "bbox": {"l": 63.6, "t": 100.0, "r": 548.4, "b": 84.0, "coord_origin": "BOTTOMLEFT"}}],
            },
        ],
        "groups": [
            {
                "self_ref": "#/groups/0", "label": "list", "subtype": "list",
                "parent": {"$ref": "#/body"},
                "children": [{"$ref": "#/texts/2"}],
            },
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0", "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 92.1, "t": 600.0, "r": 514.7, "b": 501.6, "coord_origin": "BOTTOMLEFT"}}],
                "meta": {"description": {"text": "Image Description (g34b): a car"}},
            },
            {
                "self_ref": "#/pictures/1", "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 92.1, "t": 200.0, "r": 514.7, "b": 101.6, "coord_origin": "BOTTOMLEFT"}}],
                "meta": {"description": {"text": "Image Description (g34b): a diagram"}},
            },
        ],
        "body": {
            "self_ref": "#/body", "parent": None, "label": "body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/groups/0"},
                {"$ref": "#/texts/3"},
                {"$ref": "#/pictures/1"},
            ],
        },
    }


@pytest.mark.p2
def test_json_to_sections_body_order_skips_picture_descriptions(monkeypatch):
    """Sections follow the ``body.children`` reading order but never include
    picture descriptions -- those belong to the figure chunks."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()
    doc = _body_tree_json()
    parser._set_page_heights(doc)
    sections = parser._json_to_sections(doc, parse_method="raw")

    assert sections == [
        ("Intro paragraph.", "@@1\t63.6\t548.4\t136.0\t152.0##"),
        ("Middle paragraph.", "@@1\t63.6\t548.4\t292.0\t308.0##"),
        ("- list item", "@@1\t63.6\t300.0\t492.0\t500.0##"),
        ("Closing paragraph.", "@@1\t63.6\t548.4\t692.0\t708.0##"),
    ]


@pytest.mark.p2
def test_json_to_sections_body_tree_descends_into_groups_and_falls_back(monkeypatch):
    """Group children are expanded in place, and a document without a body tree
    still works through the flat-lists fallback."""
    module = _load_docling_parser(monkeypatch)
    parser = module.DoclingParser()

    doc = _body_tree_json()
    parser._set_page_heights(doc)
    raw = parser._json_to_sections(doc, parse_method="raw")
    assert any(s[0] == "- list item" for s in raw)

    # drop the body tree: fallback path emits only text items (no descriptions)
    no_body = {k: v for k, v in _body_tree_json().items() if k != "body"}
    parser._set_page_heights(no_body)
    fallback = parser._json_to_sections(no_body, parse_method="raw")
    assert fallback == [
        ("Intro paragraph.", "@@1\t63.6\t548.4\t136.0\t152.0##"),
        ("Middle paragraph.", "@@1\t63.6\t548.4\t292.0\t308.0##"),
        ("- list item", "@@1\t63.6\t300.0\t492.0\t500.0##"),
        ("Closing paragraph.", "@@1\t63.6\t548.4\t692.0\t708.0##"),
    ]


@pytest.mark.p2
def test_picture_description_text_prefers_meta_then_annotation(monkeypatch):
    """meta.description.text wins; a blank metadata value falls back to the
    description annotation; other annotation kinds are ignored."""
    module = _load_docling_parser(monkeypatch)
    extract = module.DoclingParser._picture_description_text

    assert extract({"meta": {"description": {"text": "  from meta  "}}}) == "from meta"
    assert extract({"meta": {"description": {"text": " "}}, "annotations": [{"kind": "description", "text": "from annotation"}]}) == "from annotation"
    assert extract({"meta": {}, "annotations": [{"kind": "note", "text": "ignored"}, {"kind": "description", "text": "the description"}]}) == "the description"
    assert extract({"meta": None, "annotations": []}) == ""
    assert extract({}) == ""
    assert extract(None) == ""
