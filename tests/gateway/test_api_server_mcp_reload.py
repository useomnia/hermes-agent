"""Tests for POST /v1/mcp/reload — mid-session MCP reconnect.

Covers auth and that the endpoint runs the same shutdown+discover the
/reload-mcp slash command uses, reporting which servers were added/removed.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

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


@pytest.mark.asyncio
async def test_reload_does_not_append_a_session_db_nudge(adapter, monkeypatch):
    # The only caller (Omnia's OpenAI chat path) is client-authoritative for
    # history, so a session-DB nudge would never reach its agent — the awareness
    # is injected client-side instead. The reload must therefore NOT touch the
    # session DB, even when a session id is present.
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: [])
    db = MagicMock()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: db)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/mcp/reload", headers={"X-Hermes-Session-Id": "sess-1"})
        assert resp.status == 200

    db.append_message.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_reloads_are_serialized(adapter, monkeypatch):
    """Two reloads firing at once must not interleave.

    The handler's asyncio.Lock serializes the teardown+rebuild so a second
    reload waits for the first to finish, instead of racing
    shutdown_mcp_servers()/discover_mcp_tools() (the second would otherwise
    hit the "nothing to shut down" fast path and stop the MCP loop while the
    first is still rebuilding, corrupting the server registry).
    """
    state = {"active": 0, "max": 0, "starts": 0}
    guard = threading.Lock()

    def _enter() -> None:
        with guard:
            state["active"] += 1
            state["starts"] += 1
            state["max"] = max(state["max"], state["active"])

    def _exit() -> None:
        with guard:
            state["active"] -= 1

    # shutdown enters the critical section, discover leaves it — so the section
    # spans the whole shutdown->discover a single reload performs. If two reloads
    # overlap, "active" reaches 2 and "max" records it.
    def _fake_shutdown() -> None:
        _enter()
        time.sleep(0.05)

    def _fake_discover():
        time.sleep(0.05)
        _exit()
        return []

    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", _fake_shutdown)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", _fake_discover)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        r1, r2 = await asyncio.gather(
            cli.post("/v1/mcp/reload"),
            cli.post("/v1/mcp/reload"),
        )
        assert r1.status == 200
        assert r2.status == 200

    # Both reloads ran to completion (neither was dropped) ...
    assert state["starts"] == 2
    # ... but never at the same time.
    assert state["max"] == 1
