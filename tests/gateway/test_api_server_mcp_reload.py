"""Tests for POST /v1/mcp/reload — mid-session MCP reconnect.

Covers auth and that the endpoint runs the same shutdown+discover the
/reload-mcp slash command uses, reporting which servers were added/removed.
"""

import asyncio
import json
import threading
import time
from urllib.error import HTTPError
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.mcp_tool as mcp_tool
import tools.tool_approval as tool_approval
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
    mws = [
        mw for mw in (cors_middleware, security_headers_middleware) if mw is not None
    ]
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


@pytest.fixture(autouse=True)
def _clean_tool_approvals():
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    tool_approval.register_always_approval_authority(None)
    yield
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    tool_approval.register_always_approval_authority(None)


def _stub_mcp_reload(monkeypatch) -> None:
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: [])


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
        resp = await cli.post(
            "/v1/mcp/reload", headers={"X-Hermes-Session-Id": "sess-1"}
        )
        assert resp.status == 200

    db.append_message.assert_not_called()


@pytest.mark.asyncio
async def test_reload_refreshes_injected_connector_toolkit_approvals(
    adapter, monkeypatch
):
    _stub_mcp_reload(monkeypatch)
    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "agent-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")
    tool_approval.record_always_approval("mcp_connectors_GMAIL_SEND_EMAIL")
    monkeypatch.setattr(
        adapter,
        "_fetch_omnio_connector_toolkit_approvals",
        AsyncMock(return_value=["mcp_connectors_NOTION_CREATE_NOTION_PAGE"]),
    )
    monkeypatch.setattr(
        adapter,
        "_is_omnio_connector_toolkit_approval_granted",
        MagicMock(return_value=True),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/mcp/reload")
        assert resp.status == 200

    assert tool_approval.is_always_approved("mcp_connectors_GMAIL_SEND_EMAIL") is False
    assert (
        tool_approval.is_always_approved("mcp_connectors_NOTION_CREATE_NOTION_PAGE")
        is True
    )


@pytest.mark.asyncio
async def test_reload_fetch_failure_clears_injected_and_local_always(
    adapter, monkeypatch
):
    _stub_mcp_reload(monkeypatch)
    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "agent-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")
    tool_approval.record_always_approval("mcp_connectors_GMAIL_SEND_EMAIL")
    tool_approval.replace_injected_always_approvals([
        "mcp_connectors_NOTION_UPDATE_PAGE"
    ])
    monkeypatch.setattr(
        adapter,
        "_fetch_omnio_connector_toolkit_approvals",
        AsyncMock(side_effect=RuntimeError("omnia down")),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/mcp/reload")
        assert resp.status == 200

    assert tool_approval.is_always_approved("mcp_connectors_GMAIL_SEND_EMAIL") is False
    assert (
        tool_approval.is_always_approved("mcp_connectors_NOTION_UPDATE_PAGE") is False
    )


class _AuthorityResponse:
    def __init__(self, payload: object, status: int = 200):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode()


def _configure_authority(monkeypatch) -> None:
    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "agent-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")


def _standing_candidate(tool: str) -> None:
    tool_approval.replace_injected_always_approvals([tool])


def test_authority_checks_the_exact_tool_without_a_positive_cache(adapter, monkeypatch):
    tool = "mcp_connectors_GMAIL_SEND_EMAIL"
    _configure_authority(monkeypatch)
    _standing_candidate(tool)
    responses = iter([
        _AuthorityResponse({"tools": [tool]}),
        _AuthorityResponse({"tools": ["mcp_connectors_GMAIL_READ_EMAIL"]}),
    ])
    urlopen = MagicMock(side_effect=lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr("gateway.platforms.api_server.urlopen", urlopen)
    tool_approval.register_always_approval_authority(
        adapter._is_omnio_connector_toolkit_approval_granted
    )

    assert tool_approval.is_always_approved(tool) is True
    assert tool_approval.is_always_approved(tool) is False
    assert urlopen.call_count == 2


def test_old_omnia_404_fails_closed_for_a_warm_candidate(adapter, monkeypatch):
    tool = "mcp_connectors_GMAIL_SEND_EMAIL"
    _configure_authority(monkeypatch)
    _standing_candidate(tool)

    def old_endpoint_404(request, **_kwargs):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("gateway.platforms.api_server.urlopen", old_endpoint_404)
    tool_approval.register_always_approval_authority(
        adapter._is_omnio_connector_toolkit_approval_granted
    )

    assert tool_approval.is_always_approved(tool) is False


@pytest.mark.parametrize(
    "response",
    [
        _AuthorityResponse({"tools": "mcp_connectors_GMAIL_SEND_EMAIL"}),
        _AuthorityResponse({"tools": [123]}),
        _AuthorityResponse({"granted": True}),
        _AuthorityResponse(b"not-json"),
        _AuthorityResponse({"tools": []}, status=503),
    ],
)
def test_malformed_or_non_2xx_authority_fails_closed(adapter, monkeypatch, response):
    tool = "mcp_connectors_GMAIL_SEND_EMAIL"
    _configure_authority(monkeypatch)
    _standing_candidate(tool)
    monkeypatch.setattr(
        "gateway.platforms.api_server.urlopen",
        MagicMock(return_value=response),
    )
    tool_approval.register_always_approval_authority(
        adapter._is_omnio_connector_toolkit_approval_granted
    )

    assert tool_approval.is_always_approved(tool) is False


def test_authority_timeout_fails_closed(adapter, monkeypatch):
    tool = "mcp_connectors_GMAIL_SEND_EMAIL"
    _configure_authority(monkeypatch)
    _standing_candidate(tool)
    monkeypatch.setattr(
        "gateway.platforms.api_server.urlopen",
        MagicMock(side_effect=TimeoutError("authority timed out")),
    )
    tool_approval.register_always_approval_authority(
        adapter._is_omnio_connector_toolkit_approval_granted
    )

    assert tool_approval.is_always_approved(tool) is False


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
