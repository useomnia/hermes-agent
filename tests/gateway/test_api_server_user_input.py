"""Tests for POST /v1/omnio/user-input — answer delivery for request_user_input.

Covers auth, request validation, and that a valid answer releases the matching
blocked waiter (the agent worker parked inside the plugin's await_user_input).
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.user_input as user_input
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

SESSION = "sess-1"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [
        mw for mw in (cors_middleware, security_headers_middleware) if mw is not None
    ]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/omnio/user-input", adapter._handle_omnio_user_input)
    return app


@pytest.fixture(autouse=True)
def _clean_state():
    user_input._waits.clear()
    yield
    user_input._waits.clear()


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.asyncio
async def test_requires_auth():
    auth_adapter = _make_adapter(api_key="sk-secret")
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": "hi"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_delivers_the_answer_to_a_blocked_waiter(adapter):
    entry = user_input._InputWait("call-1")
    user_input._waits[SESSION] = [entry]

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": "Tuesday at 3pm", "toolCallId": "call-1"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
        assert resp.status == 200
        body = await resp.json()

    assert body["object"] == "omnio.user_input_response"
    assert body["resolved"] is True
    assert entry.answer == "Tuesday at 3pm" and entry.event.is_set()


@pytest.mark.asyncio
async def test_reports_not_resolved_when_no_call_is_waiting(adapter):
    # Stale card (already answered / timed out / turn ended): a valid request,
    # but nothing is parked — the chat treats resolved:false as a stale card.
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": "hi", "toolCallId": "call-x"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
        assert resp.status == 200
        body = await resp.json()
    assert body["resolved"] is False


@pytest.mark.asyncio
async def test_rejects_a_missing_response(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"toolCallId": "call-1"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_a_non_string_response(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": {"not": "a string"}},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_a_missing_session_id(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/omnio/user-input", json={"response": "hi"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_invalid_json(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            data=b"not json",
            headers={
                "X-Hermes-Session-Id": SESSION,
                "Content-Type": "application/json",
            },
        )
    assert resp.status == 400
