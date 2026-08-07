"""fetch_file tool: env-gated availability and hook round-trip contract.

The tool restores a durably-stored Omnio file onto disk by POSTing to the
in-sprite proxy's fetch hook (OMNIO_FILE_FETCH_HOOK). These tests pin the
wire contract (payload fields, auth header, outcome handling) with the hook
fully mocked — no live network.
"""

import io
import json
import urllib.error
import urllib.request

import pytest

from tools import file_tools


_HOOK_URL = "http://127.0.0.1:8642/internal/fetch-file"


def _ok_response(body: dict) -> io.BytesIO:
    raw = io.BytesIO(json.dumps(body).encode("utf-8"))
    raw.__enter__ = lambda *a: raw  # type: ignore[attr-defined]
    raw.__exit__ = lambda *a: False  # type: ignore[attr-defined]
    return raw


def _http_error(code: int, detail: dict | None = None) -> urllib.error.HTTPError:
    body = json.dumps({"detail": detail} if detail is not None else {}).encode("utf-8")
    return urllib.error.HTTPError(_HOOK_URL, code, "err", hdrs=None, fp=io.BytesIO(body))


@pytest.fixture
def hook_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(file_tools.FETCH_FILE_HOOK_ENV, _HOOK_URL)
    monkeypatch.setenv("OMNIO_TOOLBOX_BRAND", "brand-uuid-1")
    monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "svc-token")


def test_check_fn_is_false_without_hook_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(file_tools.FETCH_FILE_HOOK_ENV, raising=False)
    assert file_tools._check_fetch_file_reqs() is False


def test_check_fn_delegates_to_file_reqs_when_env_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(file_tools.FETCH_FILE_HOOK_ENV, _HOOK_URL)
    monkeypatch.setattr(file_tools, "_check_file_reqs", lambda: True)
    assert file_tools._check_fetch_file_reqs() is True
    monkeypatch.setattr(file_tools, "_check_file_reqs", lambda: False)
    assert file_tools._check_fetch_file_reqs() is False


def test_missing_path_is_a_tool_error(hook_env):
    result = file_tools._handle_fetch_file({})
    assert json.loads(result)["error"].startswith("fetch_file: missing required field 'path'")


def test_unconfigured_hook_is_a_tool_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(file_tools.FETCH_FILE_HOOK_ENV, raising=False)
    result = file_tools._handle_fetch_file({"path": "~/report.xlsx"})
    assert "not configured" in json.loads(result)["error"]


def test_restored_outcome_reports_path_size_and_type(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _ok_response(
            {
                "path": "/home/report.xlsx",
                "filename": "report.xlsx",
                "size": 2 * 1024 * 1024,
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "outcome": "restored",
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = file_tools._handle_fetch_file({"path": "~/report.xlsx"})

    assert result.startswith("Restored /home/report.xlsx (2.0MB, ")
    assert captured["url"] == _HOOK_URL
    assert captured["payload"] == {"path": "~/report.xlsx", "brand": "brand-uuid-1"}
    header_keys = {key.lower(): value for key, value in captured["headers"].items()}
    assert header_keys["x-omnio-service-token"] == "svc-token"


def test_already_present_outcome_leaves_disk_untouched_message(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _ok_response(
            {
                "path": "/brand/assets/logo.png",
                "size": 512,
                "content_type": "image/png",
                "outcome": "already_present",
            }
        ),
    )
    result = file_tools._handle_fetch_file({"path": "/brand/assets/logo.png"})
    assert "already exists on disk" in result
    assert "nothing was fetched" in result


def test_version_is_forwarded_and_reported(hook_env, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _ok_response(
            {
                "path": "/brand/report.pdf",
                "filename": "report.pdf",
                "size": 512,
                "content_type": "application/pdf",
                "version": 2,
                "outcome": "restored",
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = file_tools._handle_fetch_file({"path": "/brand/report.pdf", "version": 2})

    assert captured["payload"] == {
        "path": "/brand/report.pdf",
        "brand": "brand-uuid-1",
        "version": 2,
    }
    assert "version 2" in result


def test_invalid_version_is_a_tool_error(hook_env):
    for bad in (0, -1, "two", 1.5):
        result = file_tools._handle_fetch_file({"path": "/brand/report.pdf", "version": bad})
        assert "'version' must be a positive integer" in result


def test_404_with_suggestions_lists_similar_stored_paths(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    error = _http_error(
        404,
        {
            "message": "No stored version of this path",
            "suggestions": [
                {"path": "/brand/reports/q2-performance.pdf", "version": 2},
                {"path": "/brand/reports/q2-summary.pdf", "version": 1},
            ],
        },
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(error)
    )
    result = file_tools._handle_fetch_file({"path": "/brand/reports/q2.pdf"})

    assert "these stored paths look similar" in result
    assert "/brand/reports/q2-performance.pdf (versions up to 2)" in result
    assert "/brand/reports/q2-summary.pdf" in result


def test_versioned_404_reports_the_latest_existing_version(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    error = _http_error(404, {"message": "No stored copy of this version", "latest_version": 3})
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(error)
    )
    result = file_tools._handle_fetch_file({"path": "/brand/report.pdf", "version": 9})

    assert "versions go up to 3" in result


def test_versioned_404_explains_versioning(hook_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(404))
    )
    result = file_tools._handle_fetch_file({"path": "/brand/report.pdf", "version": 9})
    assert "No stored version 9 of /brand/report.pdf" in result


def test_versioned_already_present_suggests_moving_the_disk_copy(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _ok_response(
            {
                "path": "/brand/report.pdf",
                "size": 512,
                "content_type": "application/pdf",
                "outcome": "already_present",
            }
        ),
    )
    result = file_tools._handle_fetch_file({"path": "/brand/report.pdf", "version": 1})
    assert "move the on-disk copy aside first" in result


def test_store_miss_404_names_the_path_and_recoverability_rule(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    def raise_404(request, timeout):
        raise _http_error(404)

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)
    result = file_tools._handle_fetch_file({"path": "/tmp/draft.md"})
    error = json.loads(result)["error"]
    assert "/tmp/draft.md" in error
    assert "recoverable" in error


def test_non_404_http_error_reports_status(hook_env, monkeypatch: pytest.MonkeyPatch):
    def raise_503(request, timeout):
        raise _http_error(503)

    monkeypatch.setattr(urllib.request, "urlopen", raise_503)
    result = file_tools._handle_fetch_file({"path": "~/report.xlsx"})
    assert "status 503" in json.loads(result)["error"]


def test_transport_failure_is_a_tool_error(hook_env, monkeypatch: pytest.MonkeyPatch):
    def raise_urlerror(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_urlerror)
    result = file_tools._handle_fetch_file({"path": "~/report.xlsx"})
    assert "could not reach durable storage" in json.loads(result)["error"]


def test_invalid_hook_body_is_a_tool_error(hook_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _ok_response({"nope": True})
    )
    result = file_tools._handle_fetch_file({"path": "~/report.xlsx"})
    assert "invalid response" in json.loads(result)["error"]


def test_fetch_file_is_registered_in_the_file_toolset():
    from tools.registry import registry

    tool = registry.get_entry("fetch_file")
    assert tool is not None
    assert tool.toolset == "file"
    assert tool.schema["parameters"]["required"] == ["path"]
