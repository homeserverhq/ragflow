#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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

from types import SimpleNamespace

import pytest

import common.browser_utils as browser_utils


def _make_executable(path):
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _clear_path_overrides(monkeypatch):
    monkeypatch.delenv(browser_utils.CHROMIUM_BINARY_ENV, raising=False)
    monkeypatch.delenv(browser_utils.CHROMEDRIVER_BINARY_ENV, raising=False)


@pytest.fixture
def crawl4ai_manager():
    pytest.importorskip("crawl4ai")
    from crawl4ai.browser_manager import BrowserManager

    original_build_args = BrowserManager._build_browser_args
    had_patch_marker = hasattr(BrowserManager, "_ragflow_system_chromium")
    patch_marker = getattr(BrowserManager, "_ragflow_system_chromium", None)
    was_patched = browser_utils._CRAWL4AI_PATCHED

    yield BrowserManager

    BrowserManager._build_browser_args = original_build_args
    if had_patch_marker:
        BrowserManager._ragflow_system_chromium = patch_marker
    elif hasattr(BrowserManager, "_ragflow_system_chromium"):
        delattr(BrowserManager, "_ragflow_system_chromium")
    browser_utils._CRAWL4AI_PATCHED = was_patched


def test_resolve_chromium_prefers_env_override(monkeypatch, tmp_path):
    executable = _make_executable(tmp_path / "chromium-custom")
    monkeypatch.setenv(browser_utils.CHROMIUM_BINARY_ENV, str(executable))

    assert browser_utils.resolve_chromium_binary() == str(executable)


def test_resolve_chromium_uses_known_path(monkeypatch, tmp_path):
    _clear_path_overrides(monkeypatch)
    executable = _make_executable(tmp_path / "chromium")
    monkeypatch.setattr(browser_utils, "CHROMIUM_CANDIDATES", (str(executable),))
    monkeypatch.setattr(browser_utils, "CHROMIUM_COMMANDS", ())

    assert browser_utils.resolve_chromium_binary() == str(executable)


def test_resolve_chromium_missing_returns_empty(monkeypatch):
    _clear_path_overrides(monkeypatch)
    monkeypatch.setattr(browser_utils, "CHROMIUM_CANDIDATES", ())
    monkeypatch.setattr(browser_utils, "CHROMIUM_COMMANDS", ())

    assert browser_utils.resolve_chromium_binary() == ""


def test_resolve_chromedriver_prefers_env_override(monkeypatch, tmp_path):
    executable = _make_executable(tmp_path / "chromedriver-custom")
    monkeypatch.setenv(browser_utils.CHROMEDRIVER_BINARY_ENV, str(executable))

    assert browser_utils.resolve_chromedriver() == str(executable)


def test_resolve_chromedriver_missing_returns_empty(monkeypatch):
    _clear_path_overrides(monkeypatch)
    monkeypatch.setattr(browser_utils, "CHROMEDRIVER_CANDIDATES", ())
    monkeypatch.setattr(browser_utils, "CHROMEDRIVER_COMMANDS", ())

    assert browser_utils.resolve_chromedriver() == ""


def test_crawl4ai_patch_injects_system_chromium(monkeypatch, tmp_path, crawl4ai_manager):
    executable = _make_executable(tmp_path / "chromium")
    monkeypatch.setattr(browser_utils, "CHROMIUM_CANDIDATES", (str(executable),))
    monkeypatch.setattr(browser_utils, "CHROMIUM_COMMANDS", ())

    def fake_original_build_args(self):
        return {"headless": True}

    crawl4ai_manager._build_browser_args = fake_original_build_args
    browser_utils._CRAWL4AI_PATCHED = False

    browser_utils.ensure_crawl4ai_uses_system_chromium()

    manager = SimpleNamespace(config=SimpleNamespace(browser_type="chromium"))
    args = crawl4ai_manager._build_browser_args(manager)

    assert args["executable_path"] == str(executable)


def test_crawl4ai_patch_is_idempotent(monkeypatch, tmp_path, crawl4ai_manager):
    executable = _make_executable(tmp_path / "chromium")
    monkeypatch.setattr(browser_utils, "CHROMIUM_CANDIDATES", (str(executable),))
    monkeypatch.setattr(browser_utils, "CHROMIUM_COMMANDS", ())
    browser_utils._CRAWL4AI_PATCHED = False

    browser_utils.ensure_crawl4ai_uses_system_chromium()
    first_patch = crawl4ai_manager._build_browser_args
    browser_utils.ensure_crawl4ai_uses_system_chromium()

    assert crawl4ai_manager._build_browser_args is first_patch


def test_crawl4ai_patch_leaves_other_browsers_alone(monkeypatch, tmp_path, crawl4ai_manager):
    executable = _make_executable(tmp_path / "chromium")
    monkeypatch.setattr(browser_utils, "CHROMIUM_CANDIDATES", (str(executable),))
    monkeypatch.setattr(browser_utils, "CHROMIUM_COMMANDS", ())

    def fake_original_build_args(self):
        return {"headless": True}

    crawl4ai_manager._build_browser_args = fake_original_build_args
    browser_utils._CRAWL4AI_PATCHED = False

    browser_utils.ensure_crawl4ai_uses_system_chromium()

    manager = SimpleNamespace(config=SimpleNamespace(browser_type="firefox"))
    args = crawl4ai_manager._build_browser_args(manager)

    assert "executable_path" not in args


def test_crawl4ai_patch_requires_chromium(monkeypatch):
    _clear_path_overrides(monkeypatch)
    monkeypatch.setattr(browser_utils, "CHROMIUM_CANDIDATES", ())
    monkeypatch.setattr(browser_utils, "CHROMIUM_COMMANDS", ())
    browser_utils._CRAWL4AI_PATCHED = False

    with pytest.raises(RuntimeError, match="No Chromium executable found"):
        browser_utils.ensure_crawl4ai_uses_system_chromium()
