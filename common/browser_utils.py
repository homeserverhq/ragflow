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
"""Shared Chromium discovery and Crawl4AI wiring.

The release image installs Chromium and chromedriver from the distro packages.
Every browser consumer must use those binaries instead of falling back to a
runtime Playwright/Chrome-for-Testing download.
"""

import logging
import os
import shutil
import threading

logger = logging.getLogger(__name__)

CHROMIUM_BINARY_ENV = "RAGFLOW_CHROMIUM_BINARY"
CHROMEDRIVER_BINARY_ENV = "RAGFLOW_CHROMEDRIVER"

CHROMIUM_CANDIDATES = (
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/opt/chrome/chrome",
    "/usr/local/bin/chrome",
    "/usr/local/bin/google-chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
)
CHROMIUM_COMMANDS = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
)
CHROMEDRIVER_CANDIDATES = (
    "/usr/bin/chromedriver",
    "/usr/local/bin/chromedriver",
)
CHROMEDRIVER_COMMANDS = ("chromedriver",)

_CRAWL4AI_PATCH_LOCK = threading.Lock()
_CRAWL4AI_PATCHED = False


def _is_executable(path: str | None) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def resolve_chromium_binary() -> str:
    """Return the Chromium executable to use, or an empty string if none exists."""
    explicit = os.getenv(CHROMIUM_BINARY_ENV, "").strip()
    if explicit:
        if _is_executable(explicit):
            return explicit
        logger.warning("%s=%s is not an executable file", CHROMIUM_BINARY_ENV, explicit)

    for path in CHROMIUM_CANDIDATES:
        if _is_executable(path):
            return path
    for command in CHROMIUM_COMMANDS:
        path = shutil.which(command)
        if path and _is_executable(path):
            return path
    return ""


def resolve_chromedriver() -> str:
    """Return the chromedriver executable to use, or an empty string if none exists."""
    explicit = os.getenv(CHROMEDRIVER_BINARY_ENV, "").strip()
    if explicit:
        if _is_executable(explicit):
            return explicit
        logger.warning("%s=%s is not an executable file", CHROMEDRIVER_BINARY_ENV, explicit)

    for path in CHROMEDRIVER_CANDIDATES:
        if _is_executable(path):
            return path
    for command in CHROMEDRIVER_COMMANDS:
        path = shutil.which(command)
        if path and _is_executable(path):
            return path
    return ""


def ensure_crawl4ai_uses_system_chromium() -> str:
    """Make Crawl4AI launch the system Chromium instead of a bundled browser.

    Crawl4AI 0.9.x does not expose ``executable_path`` in ``BrowserConfig``; its
    Playwright launch therefore uses the bundled browser (Chrome for Testing in
    Playwright 1.57+). Patch the launch-argument builder once so it points at the
    distro-installed Chromium.

    Returns the resolved Chromium path.
    """
    global _CRAWL4AI_PATCHED

    executable = resolve_chromium_binary()
    if not executable:
        raise RuntimeError(f"No Chromium executable found. Install chromium or set {CHROMIUM_BINARY_ENV}; refusing to fall back to a runtime browser download.")

    with _CRAWL4AI_PATCH_LOCK:
        if _CRAWL4AI_PATCHED:
            return executable

        from crawl4ai.browser_manager import BrowserManager

        if getattr(BrowserManager, "_ragflow_system_chromium", False):
            _CRAWL4AI_PATCHED = True
            return executable

        original_build_args = BrowserManager._build_browser_args

        def _build_browser_args_with_system_chromium(self):
            args = original_build_args(self)
            if getattr(self.config, "browser_type", "chromium") == "chromium":
                browser_path = resolve_chromium_binary()
                if not browser_path:
                    raise RuntimeError(f"No Chromium executable found. Install chromium or set {CHROMIUM_BINARY_ENV}; refusing to fall back to a runtime browser download.")
                args["executable_path"] = browser_path
            return args

        BrowserManager._build_browser_args = _build_browser_args_with_system_chromium
        BrowserManager._ragflow_system_chromium = True
        _CRAWL4AI_PATCHED = True

    logger.info("Crawl4AI will use system Chromium at %s", executable)
    return executable
