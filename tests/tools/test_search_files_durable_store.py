"""search_files durable-store fallback: env-gated, name-search only.

A file search that finds nothing on disk asks the in-sprite proxy's search hook
(OMNIO_FILE_SEARCH_HOOK) for stored files matching the same pattern. These tests
pin that contract with the hook fully mocked — no live network — and pin the two
behaviours that must NOT happen: no fallback on content search, and no trace of
the store when the hook is unconfigured.
"""

import io
import json
import urllib.error
import urllib.request

import pytest

from tools import file_tools


_HOOK_URL = "http://127.0.0.1:8642/internal/search-files"
_STORED = {
    "path": "/uploads/conv-1/osix-sample.ics",
    "filename": "osix-sample.ics",
    "content_type": "text/calendar",
    "size_bytes": 812,
    "version": 1,
    "stored_at": "2026-08-05T10:11:12Z",
}


def _ok_response(body: dict) -> io.BytesIO:
    raw = io.BytesIO(json.dumps(body).encode("utf-8"))
    raw.__enter__ = lambda *a: raw  # type: ignore[attr-defined]
    raw.__exit__ = lambda *a: False  # type: ignore[attr-defined]
    return raw


@pytest.fixture
def hook_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(file_tools.SEARCH_FILES_HOOK_ENV, _HOOK_URL)
    monkeypatch.setenv("OMNIO_TOOLBOX_BRAND", "brand-uuid-1")
    monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "svc-token")


class _RecordingHook:
    """Captures the requests the fallback makes and replies with `body`."""

    def __init__(self, body: dict) -> None:
        self.body = body
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request, timeout):  # noqa: ANN001
        self.requests.append(request)
        return _ok_response(self.body)


def _search(pattern: str, target: str = "files", **kwargs):
    return json.loads(file_tools.search_tool(pattern=pattern, target=target, **kwargs))


def test_empty_name_search_falls_back_to_the_durable_store(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    hook = _RecordingHook({"files": [_STORED]})
    monkeypatch.setattr(urllib.request, "urlopen", hook)

    result = _search("*.ics", path=str(tmp_path))

    assert result["durable_store"]["files"] == [
        {
            "path": "/uploads/conv-1/osix-sample.ics",
            "size_bytes": 812,
            "stored_at": "2026-08-05T10:11:12Z",
        }
    ]
    assert "fetch_file" in result["durable_store"]["note"]
    assert len(hook.requests) == 1


def test_fallback_request_carries_the_pattern_brand_and_service_token(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    hook = _RecordingHook({"files": [_STORED]})
    monkeypatch.setattr(urllib.request, "urlopen", hook)

    _search("*.ics", path=str(tmp_path), limit=25)

    request = hook.requests[0]
    assert request.full_url == _HOOK_URL
    assert request.get_header("X-omnio-service-token") == "svc-token"
    assert json.loads(request.data.decode("utf-8")) == {
        "pattern": "*.ics",
        "brand": "brand-uuid-1",
        "limit": 25,
    }


def test_name_search_with_disk_hits_never_asks_the_store(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    (tmp_path / "on-disk.ics").write_text("BEGIN:VCALENDAR")
    hook = _RecordingHook({"files": [_STORED]})
    monkeypatch.setattr(urllib.request, "urlopen", hook)

    result = _search("*.ics", path=str(tmp_path))

    assert result["files"]
    assert "durable_store" not in result
    assert hook.requests == []


def test_content_search_never_falls_back_to_the_store(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    hook = _RecordingHook({"files": [_STORED]})
    monkeypatch.setattr(urllib.request, "urlopen", hook)

    result = _search("nothing-matches-this", target="content", path=str(tmp_path))

    assert "durable_store" not in result
    assert hook.requests == []


def test_no_fallback_without_the_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv(file_tools.SEARCH_FILES_HOOK_ENV, raising=False)
    hook = _RecordingHook({"files": [_STORED]})
    monkeypatch.setattr(urllib.request, "urlopen", hook)

    result = _search("*.ics", path=str(tmp_path))

    assert "durable_store" not in result
    assert hook.requests == []


def test_empty_store_result_adds_no_section(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setattr(urllib.request, "urlopen", _RecordingHook({"files": []}))

    result = _search("*.ics", path=str(tmp_path))

    assert "durable_store" not in result


@pytest.mark.parametrize(
    "body",
    [{}, {"files": "nope"}, {"files": [{"no_path": 1}]}],
)
def test_malformed_store_body_adds_no_section(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path, body: dict
):
    monkeypatch.setattr(urllib.request, "urlopen", _RecordingHook(body))

    result = _search("*.ics", path=str(tmp_path))

    assert "durable_store" not in result


def test_store_http_error_leaves_the_search_result_intact(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    def raising(request, timeout):  # noqa: ANN001
        raise urllib.error.HTTPError(_HOOK_URL, 503, "err", hdrs=None, fp=io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "urlopen", raising)

    result = _search("*.ics", path=str(tmp_path))

    assert "durable_store" not in result
    assert "error" not in result


def test_store_timeout_leaves_the_search_result_intact(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    def timing_out(request, timeout):  # noqa: ANN001
        raise TimeoutError("store hung")

    monkeypatch.setattr(urllib.request, "urlopen", timing_out)

    result = _search("*.ics", path=str(tmp_path))

    assert "durable_store" not in result
    assert "error" not in result


def test_description_advertises_the_store_only_when_the_hook_is_set(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(file_tools.SEARCH_FILES_HOOK_ENV, raising=False)
    assert file_tools._build_dynamic_search_files_schema() == {}
    assert file_tools.SEARCH_FILES_SCHEMA["description"] == (
        file_tools.SEARCH_FILES_BASE_DESCRIPTION
    )

    monkeypatch.setenv(file_tools.SEARCH_FILES_HOOK_ENV, _HOOK_URL)
    overrides = file_tools._build_dynamic_search_files_schema()
    assert overrides["description"] == (
        file_tools.SEARCH_FILES_BASE_DESCRIPTION + file_tools.SEARCH_FILES_DURABLE_SUFFIX
    )
    assert "fetch_file" in overrides["description"]


@pytest.fixture
def conversation_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(file_tools, "_conversation_tools_loaded", lambda: True)


def test_description_states_its_scope_where_conversation_tools_are_loaded(
    hook_env, conversation_tools
):
    description = file_tools._build_dynamic_search_files_schema()["description"]

    assert description.endswith(file_tools.SEARCH_FILES_CONVERSATION_SCOPE)


def test_description_omits_other_conversations_without_the_conversation_tools(
    hook_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(file_tools, "_conversation_tools_loaded", lambda: False)

    description = file_tools._build_dynamic_search_files_schema()["description"]

    assert "search_conversations" not in description


def test_empty_search_points_to_other_conversations_when_nothing_is_stored(
    hook_env, conversation_tools, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setattr(urllib.request, "urlopen", _RecordingHook({"files": []}))

    result = _search("*.ics", path=str(tmp_path))

    assert result["_hint"] == file_tools.SEARCH_FILES_OTHER_CONVERSATIONS_HINT


def test_stored_matches_leave_out_the_other_conversations_hint(
    hook_env, conversation_tools, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setattr(urllib.request, "urlopen", _RecordingHook({"files": [_STORED]}))

    result = _search("*.ics", path=str(tmp_path))

    assert "_hint" not in result


def test_empty_search_has_no_hint_without_the_conversation_tools(
    hook_env, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setattr(file_tools, "_conversation_tools_loaded", lambda: False)
    monkeypatch.setattr(urllib.request, "urlopen", _RecordingHook({"files": []}))

    result = _search("*.ics", path=str(tmp_path))

    assert "_hint" not in result


def test_empty_search_has_no_hint_without_the_durable_store(
    conversation_tools, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.delenv(file_tools.SEARCH_FILES_HOOK_ENV, raising=False)

    result = _search("*.ics", path=str(tmp_path))

    assert "_hint" not in result


def test_conversation_tools_are_detected_from_the_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        file_tools.registry, "get_entry", lambda name: object() if name == "search_conversations" else None
    )

    assert file_tools._conversation_tools_loaded() is True
