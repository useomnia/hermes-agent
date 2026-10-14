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
import tools.omnio_approval_state as omnio_approval_state
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
    omnio_approval_state.register_conversation_grant_loader(None)
    yield
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    omnio_approval_state.register_conversation_grant_loader(None)


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
        AsyncMock(return_value=(["mcp_connectors_NOTION_CREATE_NOTION_PAGE"], None)),
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


class _OmniaResponse:
    def __init__(self, payload: object):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode()


def _configure_omnia(monkeypatch, *, turn_id: str = "turn-1") -> None:
    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "agent-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")
    monkeypatch.setenv("HERMES_ORIGIN_TURN_ID", turn_id)


def test_startup_registers_the_conversation_grant_loader(adapter, monkeypatch):
    _configure_omnia(monkeypatch)

    adapter._omnia_approval_source(clear_snapshot=True)

    assert (
        omnio_approval_state.conversation_grant_loader()
        == adapter._load_omnio_conversation_tool_approvals
    )


def test_conversation_grants_are_asked_for_by_the_runs_turn(adapter, monkeypatch):
    _configure_omnia(monkeypatch, turn_id="turn-42")
    urlopen = MagicMock(
        return_value=_OmniaResponse(
            {
                "tools": [],
                "toolSlugs": [],
                "conversation": {
                    "tools": ["mcp__connectors__GMAIL_SEND_EMAIL"],
                    "toolSlugs": ["GMAIL_SEND_EMAIL"],
                },
            }
        )
    )
    monkeypatch.setattr("gateway.platforms.api_server.urlopen", urlopen)

    loaded = adapter._load_omnio_conversation_tool_approvals()

    assert loaded == (["mcp__connectors__GMAIL_SEND_EMAIL"], ["GMAIL_SEND_EMAIL"])
    request = urlopen.call_args.args[0]
    assert "brand=brand-1" in request.full_url
    assert "turn=turn-42" in request.full_url
    assert request.get_header("Authorization") == "Bearer agent-token"


def test_saved_chat_grant_skips_the_card_after_a_gateway_restart(adapter, monkeypatch):
    tool = "mcp__connectors__GMAIL_SEND_EMAIL"
    _configure_omnia(monkeypatch)
    monkeypatch.setattr(
        "gateway.platforms.api_server.urlopen",
        MagicMock(
            return_value=_OmniaResponse(
                {
                    "tools": [],
                    "conversation": {"tools": [tool], "toolSlugs": ["GMAIL_SEND_EMAIL"]},
                }
            )
        ),
    )
    adapter._omnia_approval_source(clear_snapshot=True)

    try:
        assert tool_approval.is_tool_approved("restarted-session", tool) is True
    finally:
        tool_approval.clear_session("restarted-session")


def test_conversation_grants_wait_for_a_turn_to_ask_about(adapter, monkeypatch):
    _configure_omnia(monkeypatch, turn_id="")
    urlopen = MagicMock()
    monkeypatch.setattr("gateway.platforms.api_server.urlopen", urlopen)

    assert adapter._load_omnio_conversation_tool_approvals() is None
    urlopen.assert_not_called()


def test_conversation_grants_retry_when_omnia_does_not_know_the_turn(
    adapter, monkeypatch
):
    _configure_omnia(monkeypatch)

    def turn_404(request, **_kwargs):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("gateway.platforms.api_server.urlopen", turn_404)

    assert adapter._load_omnio_conversation_tool_approvals() is None


def test_conversation_grants_are_empty_on_an_omnia_without_chat_grants(
    adapter, monkeypatch
):
    _configure_omnia(monkeypatch)
    monkeypatch.setattr(
        "gateway.platforms.api_server.urlopen",
        MagicMock(return_value=_OmniaResponse({"tools": [], "toolSlugs": []})),
    )

    assert adapter._load_omnio_conversation_tool_approvals() == ([], None)


@pytest.mark.parametrize(
    "configured",
    [
        {"OMNIA_BASE_URL": ""},
        {"OMNIA_API_TOKEN": ""},
        {"OMNIO_BRAND_ID": ""},
        {"OMNIO_TOOL_APPROVAL_DURABLE_DISABLED": "1"},
    ],
)
def test_conversation_grants_are_empty_without_a_usable_omnia(
    adapter, monkeypatch, configured
):
    _configure_omnia(monkeypatch)
    for name, value in configured.items():
        monkeypatch.setenv(name, value)
    urlopen = MagicMock()
    monkeypatch.setattr("gateway.platforms.api_server.urlopen", urlopen)

    assert adapter._load_omnio_conversation_tool_approvals() == ([], None)
    urlopen.assert_not_called()


def test_conversation_grant_load_fails_when_omnia_errors(adapter, monkeypatch):
    _configure_omnia(monkeypatch)

    def unavailable(request, **_kwargs):
        raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

    monkeypatch.setattr("gateway.platforms.api_server.urlopen", unavailable)

    with pytest.raises(HTTPError):
        adapter._load_omnio_conversation_tool_approvals()


@pytest.mark.parametrize(
    "payload",
    [
        {"tools": [], "conversation": {"tools": "mcp__connectors__GMAIL_SEND_EMAIL"}},
        {"tools": [], "conversation": {"tools": [], "toolSlugs": "nope"}},
        {"tools": [], "conversation": []},
        ["not-an-object"],
        b"not-json",
    ],
)
def test_malformed_conversation_grants_fail_the_load(adapter, monkeypatch, payload):
    _configure_omnia(monkeypatch)
    monkeypatch.setattr(
        "gateway.platforms.api_server.urlopen",
        MagicMock(return_value=_OmniaResponse(payload)),
    )

    with pytest.raises(ValueError):
        adapter._load_omnio_conversation_tool_approvals()


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


def _stub_targeted_reload(monkeypatch) -> dict:
    calls = {"full": 0, "targeted": [], "live": []}
    monkeypatch.setattr(mcp_tool, "_servers", {})

    def _full_shutdown():
        calls["full"] += 1

    def _targeted(names, *, live=False):
        calls["targeted"].append(list(names))
        calls["live"].append(live)
        return {name: "refreshed" for name in names}

    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", _full_shutdown)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: [])
    monkeypatch.setattr(mcp_tool, "reload_mcp_servers", _targeted)
    return calls


@pytest.mark.asyncio
async def test_named_reload_reloads_only_the_named_servers(adapter, monkeypatch):
    calls = _stub_targeted_reload(monkeypatch)

    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", json={"servers": ["connectors"]})

    assert resp.status == 200
    assert calls == {"full": 0, "targeted": [["connectors"]], "live": [False]}


@pytest.mark.asyncio
async def test_live_named_reload_asks_for_a_live_reload(adapter, monkeypatch):
    calls = _stub_targeted_reload(monkeypatch)

    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", json={"servers": ["connectors"], "live": True})

    assert resp.status == 200
    assert calls == {"full": 0, "targeted": [["connectors"]], "live": [True]}


@pytest.mark.asyncio
async def test_named_reload_reports_each_servers_outcome(adapter, monkeypatch):
    _stub_targeted_reload(monkeypatch)

    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", json={"servers": ["connectors", "omnia"]})
        body = await resp.json()

    assert body["results"] == {"connectors": "refreshed", "omnia": "refreshed"}


@pytest.mark.asyncio
async def test_reload_without_a_server_list_reconnects_every_server(adapter, monkeypatch):
    calls = _stub_targeted_reload(monkeypatch)

    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", json={})
        body = await resp.json()

    assert (calls["full"], calls["targeted"], "results" in body) == (1, [], False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        '{"servers": []}',
        '{"servers": "connectors"}',
        '{"servers": [""]}',
        '{"servers": [1]}',
        "not json",
        '["connectors"]',
        '{"servers": ["connectors"], "live": "yes"}',
        '{"live": true}',
    ],
)
async def test_malformed_server_list_is_rejected_without_reloading(adapter, monkeypatch, payload):
    calls = _stub_targeted_reload(monkeypatch)

    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.post(
            "/v1/mcp/reload", data=payload, headers={"Content-Type": "application/json"}
        )

    assert (resp.status, calls["full"], calls["targeted"]) == (400, 0, [])


@pytest.mark.asyncio
async def test_capabilities_advertise_live_named_reloads(adapter):
    app = _create_app(adapter)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        body = await resp.json()

    assert body["features"]["mcp_named_reload"] == {"apiVersion": 1, "live": True}
