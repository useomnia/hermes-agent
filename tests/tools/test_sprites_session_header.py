"""Toolbox and durable-file requests name the session that made them.

The Omnio proxy maps ``X-Hermes-Session-Id`` to the session's conversation so
the Toolbox can select that conversation's home. Only the session bound to the
calling context may be sent; the process-wide ``HERMES_SESSION_ID`` mirror
names whichever session ran last and must never be used.
"""

import contextvars
import io
import json
import urllib.request

import pytest

from gateway import session_context
from tools import file_tools

SESSION = "0f1e2d3c-4b5a-4968-8776-655443322110"


def _in_session(session_id, function):
    def run():
        session_context._SESSION_ID.set(session_id)
        return function()

    return contextvars.copy_context().run(run)


def _environment():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    return env


class _Response:
    def __init__(self, body: bytes):
        self._body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, limit=-1):
        return self._body.read(limit)


@pytest.fixture
def toolbox_requests(monkeypatch):
    import tools.environments.sprites as sprites_module

    captured: list[dict[str, str]] = []

    def fake_open(request, timeout):
        captured.append({key.lower(): value for key, value in request.header_items()})
        return _Response(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)
    monkeypatch.setattr(
        sprites_module.SpritesEnvironment, "sync_projected_path", lambda self, path: None
    )
    return captured


@pytest.fixture
def hook_requests(monkeypatch):
    captured: list[dict[str, str]] = []

    def fake_urlopen(request, timeout):
        captured.append({key.lower(): value for key, value in request.header_items()})
        return _Response(json.dumps({"outcome": "already_present", "path": "/home/a"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv(file_tools.FETCH_FILE_HOOK_ENV, "http://127.0.0.1:1/internal/fetch-file")
    monkeypatch.setenv(file_tools.SEARCH_FILES_HOOK_ENV, "http://127.0.0.1:1/internal/search-files")
    monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "svc-token")
    return captured


def test_current_session_id_should_return_the_context_session():
    assert _in_session(SESSION, session_context.current_session_id) == SESSION


def test_current_session_id_should_ignore_the_process_environment(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", SESSION)

    assert contextvars.Context().run(session_context.current_session_id) is None


def test_current_session_id_should_treat_a_cleared_session_as_unbound():
    assert _in_session("", session_context.current_session_id) is None


def test_toolbox_request_should_name_the_calling_session(toolbox_requests):
    env = _environment()

    _in_session(SESSION, lambda: env._request_json("/files", {"operation": "stat"}))

    assert toolbox_requests[0]["x-hermes-session-id"] == SESSION


def test_toolbox_request_should_omit_the_session_outside_a_session(
    toolbox_requests, monkeypatch
):
    monkeypatch.setenv("HERMES_SESSION_ID", SESSION)
    env = _environment()

    contextvars.Context().run(lambda: env._request_json("/files", {"operation": "stat"}))

    assert "x-hermes-session-id" not in toolbox_requests[0]


def test_raw_file_read_should_name_the_calling_session(toolbox_requests):
    env = _environment()

    _in_session(SESSION, lambda: env.read_file_bytes("/home/report.csv", max_bytes=10))

    assert toolbox_requests[0]["x-hermes-session-id"] == SESSION


def test_fetch_file_hook_should_name_the_calling_session(hook_requests):
    _in_session(SESSION, lambda: file_tools._handle_fetch_file({"path": "~/report.csv"}))

    assert hook_requests[0]["x-hermes-session-id"] == SESSION


def test_search_hook_should_name_the_calling_session(hook_requests):
    _in_session(SESSION, lambda: file_tools._durable_store_matches("*.csv", 5))

    assert hook_requests[0]["x-hermes-session-id"] == SESSION


def test_hooks_should_keep_the_service_token(hook_requests):
    _in_session(SESSION, lambda: file_tools._handle_fetch_file({"path": "~/report.csv"}))

    assert hook_requests[0]["x-omnio-service-token"] == "svc-token"


def test_exec_should_name_the_calling_session(toolbox_requests):
    # Exec runs on a worker thread; the session must survive the thread hop.
    env = _environment()
    env.cwd = "/home"

    def run():
        handle = env._run_bash("true", timeout=5)
        handle._done.wait(5)

    _in_session(SESSION, run)

    exec_headers = [headers for headers in toolbox_requests if "x-request-id" in headers]
    assert exec_headers[0]["x-hermes-session-id"] == SESSION
