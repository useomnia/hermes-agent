"""Mixed-version tests for the Omnio connector approval resolve endpoint."""

import asyncio
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
    tool_approval._injected_always_approved_slugs.clear()
    tool_approval.clear_session(SESSION)
    mcp_tool._mcp_tool_read_only_hints.clear()
    mcp_tool._track_mcp_tool_read_only(GATED, False)
    mcp_tool._track_mcp_tool_read_only(SIBLING, False)
    yield
    tool_approval.register_always_approval_authority(None)
    tool_approval._session_approved.clear()
    tool_approval._always_approved.clear()
    tool_approval._injected_always_approved.clear()
    tool_approval._injected_always_approved_slugs.clear()
    tool_approval.clear_session(SESSION)
    mcp_tool._mcp_tool_read_only_hints.clear()


@pytest.mark.asyncio
async def test_connect_overlaps_approval_snapshot_with_readiness(
    monkeypatch: pytest.MonkeyPatch,
):
    """The listener is ready while the control-plane snapshot is in flight.

    The first agent-builder thread still joins that snapshot, so moving it off
    the connect critical path is real overlap rather than a consistency race.
    """
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "key": "sk-test-strong-key-0123456789",
            },
        )
    )
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def fetch(**_kwargs):
        fetch_started.set()
        await release_fetch.wait()
        return [GATED], ["GMAIL_CREATE_EMAIL_DRAFT"]

    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "test-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")
    monkeypatch.setattr(
        adapter,
        "_fetch_omnio_connector_toolkit_approvals",
        fetch,
    )

    try:
        assert await asyncio.wait_for(adapter.connect(), timeout=1) is True
        assert adapter.is_connected is True
        await asyncio.wait_for(fetch_started.wait(), timeout=1)
        assert adapter._omnio_approval_refresh_task is not None
        assert not adapter._omnio_approval_refresh_task.done()

        join = asyncio.create_task(
            asyncio.to_thread(adapter._wait_for_omnio_approval_snapshot)
        )
        await asyncio.sleep(0)
        assert not join.done()

        release_fetch.set()
        await asyncio.wait_for(join, timeout=1)
        assert GATED in tool_approval._injected_always_approved
    finally:
        release_fetch.set()
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancels_approval_refresh_and_releases_joiner(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "key": "sk-test-strong-key-0123456789",
            },
        )
    )
    fetch_started = asyncio.Event()
    never_release = asyncio.Event()

    async def fetch(**_kwargs):
        fetch_started.set()
        await never_release.wait()
        return [GATED], None

    monkeypatch.setenv("OMNIA_BASE_URL", "https://omnia.test")
    monkeypatch.setenv("OMNIA_API_TOKEN", "test-token")
    monkeypatch.setenv("OMNIO_BRAND_ID", "brand-1")
    monkeypatch.setattr(
        adapter,
        "_fetch_omnio_connector_toolkit_approvals",
        fetch,
    )

    assert await adapter.connect() is True
    await asyncio.wait_for(fetch_started.wait(), timeout=1)
    join = asyncio.create_task(
        asyncio.to_thread(adapter._wait_for_omnio_approval_snapshot)
    )
    await asyncio.sleep(0)
    assert not join.done()

    await adapter.disconnect()
    await asyncio.wait_for(join, timeout=1)
    assert adapter._omnio_approval_refresh_task is None
    assert tool_approval._injected_always_approved == set()


def test_every_agent_build_joins_the_startup_approval_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))

    def joined() -> None:
        raise LookupError("snapshot joined before agent imports")

    monkeypatch.setattr(adapter, "_wait_for_omnio_approval_snapshot", joined)

    with pytest.raises(LookupError, match="snapshot joined before agent imports"):
        adapter._create_agent()


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    app.router.add_post(
        "/v1/omnio/tool-approval",
        adapter._handle_omnio_tool_approval,
    )
    return app


def _park_waiter(
    surface_key: str = SESSION,
    *,
    grant_session_key: str | None = None,
) -> tuple[threading.Thread, dict[str, str | None]]:
    result: dict[str, str | None] = {}
    tool_approval.register_tool_approval_notify(
        surface_key,
        lambda _event: None,
        grant_session_key=grant_session_key,
    )
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "choice",
            tool_approval.await_tool_approval(surface_key, GATED, {}, "call-1"),
        )
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if tool_approval._wait_registry.pending_count(surface_key):
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
            json={
                "tool": GATED,
                "scope": "session",
                "toolCallId": "call-1",
            },
        )
        body = await response.json()

    assert response.status == 200
    assert body["recorded"] is True
    waiter.join(timeout=3)
    assert result["choice"] == "session"
    assert tool_approval.is_tool_approved(SESSION, GATED) is True
    assert tool_approval.is_tool_approved(SESSION, SIBLING) is False


@pytest.mark.asyncio
async def test_old_omnia_request_resolves_run_surface_without_surface_id():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    waiter, result = _park_waiter(
        "run-owned-surface",
        grant_session_key=SESSION,
    )

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={
                "tool": GATED,
                "scope": "once",
                "toolCallId": "call-1",
            },
        )
        body = await response.json()

    assert response.status == 200
    assert body["recorded"] is True
    waiter.join(timeout=3)
    assert result["choice"] == "once"


@pytest.mark.asyncio
async def test_old_omnia_request_without_call_or_surface_uses_fifo():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    waiter, result = _park_waiter(
        "run-owned-fifo-surface",
        grant_session_key=SESSION,
    )

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={"tool": GATED, "scope": "deny"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["recorded"] is True
    waiter.join(timeout=3)
    assert result["choice"] == "deny"


@pytest.mark.asyncio
async def test_surface_id_still_requires_call_id():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/omnio/tool-approval",
            headers={"X-Hermes-Session-Id": SESSION},
            json={"tool": GATED, "scope": "once", "surfaceId": "run-exact"},
        )
        body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "approval_missing_tool_call"


@pytest.mark.asyncio
async def test_concurrent_streaming_chat_requests_get_distinct_approval_surfaces():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    calls: list[tuple[str, str]] = []
    both_started = asyncio.Event()

    async def run_agent(**kwargs):
        calls.append(
            (
                kwargs["approval_session_key"],
                kwargs["approval_surface_key"],
            )
        )
        if len(calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=3)
        kwargs["stream_delta_callback"]("done")
        return (
            {"final_response": "done", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "same conversation"}],
        "stream": True,
    }
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(adapter, "_run_agent", run_agent)
            first, second = await asyncio.gather(
                client.post("/v1/chat/completions", json=payload),
                client.post("/v1/chat/completions", json=payload),
            )
            await asyncio.gather(first.text(), second.text())

    assert calls[0][0] == calls[1][0]
    assert calls[0][1] != calls[1][1]
    assert all(surface_key.startswith("chatcmpl-") for _, surface_key in calls)


@pytest.mark.asyncio
async def test_run_agent_keeps_concurrent_streaming_approval_surfaces_isolated(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    barrier = threading.Barrier(2)
    observed_surfaces: dict[str, str] = {}
    first_events: list[dict] = []
    second_events: list[dict] = []

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, **_kwargs):
            surface_key = tool_approval.get_current_tool_approval_surface_key()
            observed_surfaces[user_message] = surface_key
            barrier.wait(timeout=3)
            notify = tool_approval._wait_registry.surface_value(surface_key)
            assert notify is not None
            notify({"request": user_message})
            return {"final_response": "done", "messages": [], "api_calls": 1}

    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **kwargs: FakeAgent(kwargs["session_id"]),
    )

    await asyncio.gather(
        adapter._run_agent(
            user_message="first",
            conversation_history=[],
            session_id=SESSION,
            approval_session_key=SESSION,
            approval_surface_key="chat-stream-first",
            approval_notify=first_events.append,
        ),
        adapter._run_agent(
            user_message="second",
            conversation_history=[],
            session_id=SESSION,
            approval_session_key=SESSION,
            approval_surface_key="chat-stream-second",
            approval_notify=second_events.append,
        ),
    )

    assert observed_surfaces == {
        "first": "chat-stream-first",
        "second": "chat-stream-second",
    }
    assert first_events == [{"request": "first"}]
    assert second_events == [{"request": "second"}]


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
                "surfaceId": SESSION,
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
            json={
                "tool": GATED,
                "scope": "skip",
                "toolCallId": "call-1",
                "surfaceId": SESSION,
            },
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
            json={
                "tool": GATED,
                "scope": "forever",
                "toolCallId": "call-1",
                "surfaceId": SESSION,
            },
        )
        body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "invalid_approval_scope"
    tool_approval.clear_session(SESSION)
    waiter.join(timeout=3)
    assert not waiter.is_alive()
