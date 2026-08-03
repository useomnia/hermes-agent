"""Tests that browser_vision can read a browser_preview without weakening SSRF.

``browser_vision`` re-checks ``window.location.href`` before capturing, and
``_is_safe_url`` accepts only http/https. A previewed workspace file is written
into a blank page over CDP, so its URL stays "about:blank" — which failed that
check, making vision refuse every preview it was ever pointed at.

"about:blank" is exempt because it has no fetched origin to protect. Everything
the guard exists for — private addresses, the cloud metadata endpoint, non-http
schemes — must still be blocked, which is what the second class below pins.
"""

import json

import pytest

from tools import browser_tool


PREVIEW_URL = "about:blank"
PRIVATE_URL = "http://127.0.0.1:8080/secret"
METADATA_URL = "http://169.254.169.254/latest/meta-data/"
FILE_URL = "file:///etc/passwd"


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda key: key)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)


def _report_url(monkeypatch, url):
    """Make the guard's location.href probe report `url`."""

    def _run(session_key, command, args, timeout=None, _engine_override=None):
        if command == "eval":
            return {"success": True, "data": {"result": json.dumps(url)}}
        raise AssertionError(f"unexpected browser command: {command}")

    monkeypatch.setattr(browser_tool, "_run_browser_command", _run)


def _blocked_error(monkeypatch, url):
    """Return the guard's refusal for `url`, or None when it let the capture run.

    Capture is stubbed to raise, so reaching it surfaces as "not blocked"
    without needing a real Chrome.
    """
    _report_url(monkeypatch, url)
    monkeypatch.setattr(
        browser_tool,
        "_is_safe_url",
        lambda candidate: candidate.startswith("https://"),
    )
    try:
        result = browser_tool.browser_vision("what is on screen?", task_id="test")
    except Exception:
        return None
    if not isinstance(result, str):
        return None
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return None
    if payload.get("success") is False and "private or internal" in payload.get("error", ""):
        return payload["error"]
    return None


class TestPreviewIsReadable:
    def test_should_not_block_vision_on_a_previewed_page(self, monkeypatch):
        assert _blocked_error(monkeypatch, PREVIEW_URL) is None

    def test_should_not_block_when_the_url_probe_reports_nothing(self, monkeypatch):
        assert _blocked_error(monkeypatch, "") is None


class TestGuardStillHolds:
    @pytest.mark.parametrize(
        "url",
        [PRIVATE_URL, METADATA_URL, FILE_URL, "http://10.0.0.5/admin", "about:blank#"],
        ids=["loopback", "metadata", "file-scheme", "private-range", "not-exactly-blank"],
    )
    def test_should_still_block_an_unsafe_page_url(self, monkeypatch, url):
        error = _blocked_error(monkeypatch, url)
        assert error is not None, f"{url} must stay blocked"
        assert url in error
