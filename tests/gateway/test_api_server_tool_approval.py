"""Mixed-version tests for the Omnio connector approval resolve endpoint."""

import threading
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.mcp_tool as mcp_tool
import tools.tool_approval as tool_approval
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

SESSION = "sess-1"
GATED = "mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT"
SIBLING = "mcp_connectors_GMAIL_SEND_EMAIL"


@pytest.fixture(autouse=True)
def _clean_approval_state():
    tool_approval.register_always_approval_authority(None)
    tool_approval._session_approved.clear()
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    tool_approval.clear_session(SESSION)
    mcp_tool._mcp_tool_read_only_hints.clear()
    mcp_tool._track_mcp_tool_read_only(GATED, False)
    mcp_tool._track_mcp_tool_read_only(SIBLING, False)
    yield
    tool_approval.register_always_approval_authority(None)
    tool_approval._session_approved.clear()
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    tool_approval.clear_session(SESSION)
    mcp_tool._mcp_tool_read_only_hints.clear()


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/omnio/tool-approval",
        adapter._handle_omnio_tool_approval,
    )
    return app


def _park_waiter() -> tuple[threading.Thread, dict[str, str | None]]:
    result: dict[str, str | None] = {}
    tool_approval.register_tool_approval_notify(SESSION, lambda _event: None)
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "choice",
            tool_approval.await_tool_approval(SESSION, GATED, {}, "call-1"),
        )
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if tool_approval._wait_registry.pending_count(SESSION):
            break
        time.sleep(0.01)
    else:
        pytest.fail("the tool-approval waiter should be parked")
    return thread, result


@pytest.mark.asyncio
async def test_old_omnia_request_can_omit_the_optional_tools_field():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    waiter, result = _park_waiter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={"tool": GATED, "scope": "session", "toolCallId": "call-1"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["recorded"] is True
    waiter.join(timeout=3)
    assert result["choice"] == "session"
    assert tool_approval.is_tool_approved(SESSION, GATED) is True
    assert tool_approval.is_tool_approved(SESSION, SIBLING) is False


@pytest.mark.asyncio
async def test_new_omnia_request_can_supply_optional_sibling_tools():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    waiter, result = _park_waiter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={
                "tool": GATED,
                "scope": "session",
                "toolCallId": "call-1",
                "tools": [GATED, SIBLING],
            },
        )
        body = await response.json()

    assert response.status == 200
    assert body["recorded"] is True
    waiter.join(timeout=3)
    assert result["choice"] == "session"
    assert tool_approval.is_tool_approved(SESSION, GATED) is True
    assert tool_approval.is_tool_approved(SESSION, SIBLING) is True


@pytest.mark.asyncio
async def test_omnia_request_accepts_skip_for_a_live_waiter():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    waiter, result = _park_waiter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={"tool": GATED, "scope": "skip", "toolCallId": "call-1"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["scope"] == "skip"
    assert body["recorded"] is True
    waiter.join(timeout=3)
    assert result["choice"] == "skip"
    assert tool_approval.consume_tool_approval_decision(SESSION, "call-1") == "skip"
    assert tool_approval.is_tool_approved(SESSION, GATED) is False
    assert tool_approval.is_always_approved(GATED) is False


@pytest.mark.asyncio
async def test_omnia_request_rejects_an_invalid_scope():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    waiter, _result = _park_waiter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={"tool": GATED, "scope": "forever", "toolCallId": "call-1"},
        )
        body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "invalid_approval_scope"
    tool_approval.clear_session(SESSION)
    waiter.join(timeout=3)
    assert not waiter.is_alive()
