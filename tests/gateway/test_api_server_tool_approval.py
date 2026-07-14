"""Mixed-version tests for the Omnio connector approval resolve endpoint."""

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
    tool_approval._waits.clear()
    tool_approval._decisions.clear()
    mcp_tool._mcp_tool_read_only_hints.clear()
    mcp_tool._track_mcp_tool_read_only(GATED, False)
    mcp_tool._track_mcp_tool_read_only(SIBLING, False)
    yield
    tool_approval.register_always_approval_authority(None)
    tool_approval._session_approved.clear()
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    tool_approval._waits.clear()
    tool_approval._decisions.clear()
    mcp_tool._mcp_tool_read_only_hints.clear()


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/omnio/tool-approval",
        adapter._handle_omnio_tool_approval,
    )
    return app


def _park_waiter() -> None:
    tool_approval._waits[SESSION] = [tool_approval._ApprovalWait(GATED, "call-1")]


@pytest.mark.asyncio
async def test_old_omnia_request_can_omit_the_optional_tools_field():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    _park_waiter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={"tool": GATED, "scope": "session", "toolCallId": "call-1"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["recorded"] is True
    assert tool_approval.is_tool_approved(SESSION, GATED) is True
    assert tool_approval.is_tool_approved(SESSION, SIBLING) is False


@pytest.mark.asyncio
async def test_new_omnia_request_can_supply_optional_sibling_tools():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    _park_waiter()

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
    assert tool_approval.is_tool_approved(SESSION, GATED) is True
    assert tool_approval.is_tool_approved(SESSION, SIBLING) is True
