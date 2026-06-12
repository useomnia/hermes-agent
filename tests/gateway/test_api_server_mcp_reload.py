"""Tests for POST /v1/mcp/reload — mid-session MCP reconnect.

Covers auth and that the endpoint runs the same shutdown+discover the
/reload-mcp slash command uses, reporting which servers were added/removed.
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.mcp_tool as mcp_tool
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/mcp/reload", adapter._handle_mcp_reload)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.mark.asyncio
async def test_reload_requires_auth(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/mcp/reload")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_reload_reconnects_and_reports_added_servers(adapter, monkeypatch):
    # Start with no connected servers; the (mocked) discover adds one + returns
    # its tools, exercising the shutdown -> discover -> diff path.
    monkeypatch.setattr(mcp_tool, "_servers", {})

    def _fake_discover():
        mcp_tool._servers["connectors"] = object()
        return ["GMAIL_CREATE_EMAIL_DRAFT", "GOOGLE_ANALYTICS_RUN_REPORT"]

    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", _fake_discover)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/mcp/reload")
        assert resp.status == 200
        body = await resp.json()

    assert body["object"] == "hermes.mcp.reload"
    assert body["servers"] == ["connectors"]
    assert body["added"] == ["connectors"]
    assert body["removed"] == []
    assert body["tools"] == 2
