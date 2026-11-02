"""Expected API source misses stay quiet without hiding failed delivery."""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from gateway.config import GatewayConfig, Platform
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB
from tools import async_delegation


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIO_WAKE_HOOK", raising=False)
    gateway = GatewayRunner.__new__(GatewayRunner)
    gateway.session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    database = gateway.session_store._db
    assert database is not None
    gateway._session_db = AsyncSessionDB(database)
    gateway._session_sources = OrderedDict()
    gateway._completion_delivery_lock = threading.Lock()
    gateway._completion_deliveries_inflight = set()
    gateway._completion_deliveries_delivered = OrderedDict()
    gateway._completion_delivery_retention = 100
    api = APIServerAdapter.__new__(APIServerAdapter)
    api._host, api._port, api._api_key = "127.0.0.1", 1, "test-api-key"
    api._model_name = "hermes"
    gateway.adapters = {Platform.API_SERVER: api}
    yield gateway
    database.close()


@asynccontextmanager
async def serve(path, handler):
    app = web.Application()
    app.router.add_post(path, handler)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    try:
        await site.start()
        listening = site._server
        assert isinstance(listening, asyncio.Server)
        yield listening.sockets[0].getsockname()[1]
    finally:
        await server.cleanup()


def completion(runner, **overrides):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_test",
        "session_key": "run_test_approval",
        "origin_session_id": "parent-session",
        "parent_session_id": "parent-session",
        "origin_turn_id": "origin-turn",
        "subagent_ids": ["child-test"],
        "status": "completed",
        **overrides,
    }
    runner.session_store._db.create_session(evt["parent_session_id"], "api_server")
    async_delegation._persist_dispatch({**evt, "dispatched_at": time.time()})
    async_delegation._persist_completion(evt, {"summary": "child finished"})
    return evt


def messages(caplog, level=logging.WARNING):
    return [record.getMessage() for record in caplog.records
            if record.name in {"gateway.run", "gateway.wake"} and record.levelno >= level]


@pytest.mark.asyncio
@pytest.mark.parametrize("routing", [
    {},
    {"session_key": ""},
    {"platform": "api_server"},
    {"platform": "api_server", "chat_id": "different-chat-id"},
])
async def test_api_hook_accepts_once_without_false_routing_warning(
    runner, monkeypatch, caplog, routing,
):
    evt = completion(runner, **routing)
    observed = []

    async def accept(request):
        observed.append({
            "payload": await request.json(),
            "token": request.headers.get("X-Omnio-Service-Token"),
            "during": async_delegation.get_durable_delegation(evt["delegation_id"]),
        })
        return web.json_response({"status": "steered"}, status=202)

    caplog.set_level(logging.INFO)
    async with serve("/internal/subagent-wake", accept) as port:
        monkeypatch.setenv("OMNIO_WAKE_HOOK", f"http://127.0.0.1:{port}/internal/subagent-wake")
        monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "test-service-token")
        assert await runner._deliver_completion_notification("child finished", evt) is True
        assert await runner._deliver_completion_notification("child finished", evt) is None

    assert len(observed) == 1
    assert observed[0]["payload"] == {
        "origin_turn_id": "origin-turn", "delegation_id": "deleg_test",
        "subagent_ids": ["child-test"], "session_id": "parent-session", "text": "child finished",
    }
    assert observed[0]["token"] == "test-service-token"
    assert observed[0]["during"]["delivery_state"] == "pending"
    row = async_delegation.get_durable_delegation(evt["delegation_id"])
    assert row is not None
    assert (row["delivery_state"], row["delivery_attempts"]) == ("delivered", 1)
    assert messages(caplog) == []
    assert any("wake hook delivered" in text for text in messages(caplog, logging.INFO))
    assert not any("via self-post" in text for text in messages(caplog, logging.INFO))


@pytest.mark.asyncio
async def test_legacy_raw_session_self_post_is_quiet_and_keeps_session_header(
    runner, caplog,
):
    evt = completion(
        runner, session_key="raw-parent", parent_session_id="raw-parent",
        origin_session_id="", origin_turn_id="",
    )
    observed = []

    async def accept(request):
        observed.append((
            request.headers.get("X-Hermes-Session-Id"),
            request.headers.get("Authorization"), await request.json(),
        ))
        return web.json_response({"choices": []})

    caplog.set_level(logging.INFO)
    async with serve("/v1/chat/completions", accept) as port:
        runner.adapters[Platform.API_SERVER]._port = port
        assert await runner._deliver_completion_notification("child finished", evt) is True

    assert observed == [("raw-parent", "Bearer test-api-key", {
        "model": "hermes", "messages": [{"role": "user", "content": "child finished"}],
        "stream": False,
    })]
    row = async_delegation.get_durable_delegation(evt["delegation_id"])
    assert row is not None
    assert (row["delivery_state"], row["delivery_attempts"]) == ("delivered", 1)
    assert messages(caplog) == []
    assert any("wake self-post delivered" in text for text in messages(caplog, logging.INFO))


@pytest.mark.asyncio
@pytest.mark.parametrize("evt", [
    {},
    {"session_key": "   ", "origin_session_id": "   "},
    {"session_key": "agent:broken"},
    {"session_key": "agent:broken", "origin_session_id": "parent-session"},
    {"session_key": "agent:main:telegram:dm:"},
    {"session_key": "opaque-key", "chat_id": "chat"},
    {"session_key": "opaque-key", "chat_type": "group"},
    {"session_key": "opaque-key", "thread_id": "thread"},
    {"session_key": "opaque-key", "platform": "telegram"},
])
async def test_malformed_or_partial_messaging_route_still_warns(runner, caplog, evt):
    runner.adapters.clear()
    caplog.set_level(logging.WARNING)
    assert await runner._inject_watch_notification("child finished", evt) is None
    assert any("Synthetic event source unresolvable" in text for text in messages(caplog))


@pytest.mark.asyncio
async def test_invalid_platform_warning_remains_visible(runner, caplog):
    runner.adapters.clear()
    caplog.set_level(logging.WARNING)
    assert await runner._inject_watch_notification("child finished", {
        "session_key": "opaque-key", "origin_session_id": "parent-session",
        "platform": "unregistered_completion_test", "chat_type": "dm", "chat_id": "chat",
    }) is None
    assert any("invalid platform metadata" in text for text in messages(caplog))


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_state", ["missing", "push"])
async def test_recognised_api_route_without_adapter_keeps_final_failure_warning(
    runner, caplog, adapter_state,
):
    evt = completion(runner)
    if adapter_state == "missing":
        runner.adapters.clear()
    else:
        runner.adapters[Platform.API_SERVER].supports_push_delivery = True
    caplog.set_level(logging.WARNING)
    assert await runner._deliver_completion_notification("child finished", evt) is None
    row = async_delegation.get_durable_delegation(evt["delegation_id"])
    assert row is not None
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 1)
    assert not any("Synthetic event source unresolvable" in text for text in messages(caplog))
    assert any("Dropping watch notification for raw session parent-session" in text
               for text in messages(caplog))


@pytest.mark.asyncio
@pytest.mark.parametrize("source_location", ["store", "cache"])
async def test_saved_messaging_source_wins_before_raw_api_recognition(
    runner, caplog, source_location,
):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", thread_id="42")
    runner.session_store._ensure_loaded()
    if source_location == "store":
        runner.session_store._entries["run_test_approval"] = SimpleNamespace(origin=source)
    else:
        runner._session_sources["run_test_approval"] = source
    push = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = push
    caplog.set_level(logging.WARNING)
    assert await runner._inject_watch_notification("child finished", {
        "type": "async_delegation", "session_key": "run_test_approval",
        "origin_session_id": "parent-session",
    }) is True
    push.handle_message.assert_awaited_once()
    assert push.handle_message.await_args.args[0].source is source
    assert messages(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status,requests_expected,accepted,delivery_state,diagnostic", [
    (503, 4, False, "pending", "wake failed for session parent-session"),
    (404, 1, True, "delivered", "wake_dropped"),
])
async def test_real_hook_failure_stays_visible_with_existing_durable_disposition(
    runner, monkeypatch, caplog, status, requests_expected, accepted, delivery_state, diagnostic,
):
    from gateway import wake

    monkeypatch.setattr(wake, "_RETRY_DELAYS_SECONDS", (0, 0, 0))
    evt = completion(runner)
    observed = []

    async def refuse(request):
        observed.append(await request.json())
        return web.json_response({"error": "test refusal"}, status=status)

    caplog.set_level(logging.WARNING)
    async with serve("/internal/subagent-wake", refuse) as port:
        monkeypatch.setenv("OMNIO_WAKE_HOOK", f"http://127.0.0.1:{port}/internal/subagent-wake")
        monkeypatch.setenv("OMNIO_INTERNAL_TOKEN", "test-service-token")
        assert await runner._deliver_completion_notification("child finished", evt) is accepted

    assert len(observed) == requests_expected
    assert all(payload["session_id"] == "parent-session" for payload in observed)
    row = async_delegation.get_durable_delegation(evt["delegation_id"])
    assert row is not None
    assert (row["delivery_state"], row["delivery_attempts"]) == (delivery_state, 1)
    assert any(diagnostic in text for text in messages(caplog))
    assert any(str(status) in text for text in messages(caplog))
