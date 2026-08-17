"""Fork-owned contract tests for Omnio's resumable ``/v1/runs`` Turn log.

These tests intentionally live outside the upstream-owned runs test module.
They cover the versioned Omnio contract: immutable Responses-native frames,
independent sequence-number cursors, bounded retention, safe tool projection,
and server-authoritative session continuity.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient, TestServer

import tools.tool_approval as tool_approval
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.turn_event_log import (
    CursorExpiredError,
    DELTA_COALESCE_BYTES,
    DELTA_COALESCE_SECONDS,
    OMNIO_EXTENSION_EVENT_TYPES,
    TERMINAL_FRAME_RESERVE_BYTES,
    TurnEventEmitter,
    TurnEventLogStore,
    UnknownRunError,
)
from hermes_constants import MAX_TODO_ITEMS
from hermes_state import SessionDB


_AUTH_HEADERS = {"Authorization": "Bearer omnio-test-key"}


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _make_adapter(*, api_key: str = "") -> APIServerAdapter:
    # Supplying an explicit empty key keeps ambient API_SERVER_KEY out of these
    # process-local HTTP contract tests.
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs", adapter._handle_recoverable_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_profile_app(adapter: APIServerAdapter) -> web.Application:
    @web.middleware
    async def profile_context(request: web.Request, handler):
        token = _api_request_profile.set(request.match_info["profile"])
        try:
            return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[profile_context])
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
    app.router.add_get("/p/{profile}/v1/runs", adapter._handle_recoverable_runs)
    app.router.add_get(
        "/p/{profile}/v1/runs/{run_id}/events", adapter._handle_run_events
    )
    return app


def _make_multiplex_app(adapter: APIServerAdapter) -> web.Application:
    class _Runner:
        config = GatewayConfig(multiplex_profiles=True)

    adapter.gateway_runner = _Runner()
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    routes = (
        ("POST", "/v1/runs", adapter._handle_runs),
        ("GET", "/v1/runs", adapter._handle_recoverable_runs),
        ("GET", "/v1/runs/{run_id}/events", adapter._handle_run_events),
    )
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


def _agent(
    run: Callable[..., Dict[str, Any]],
    *,
    interrupt: Optional[Callable[[Optional[str]], None]] = None,
) -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.side_effect = run
    if interrupt is not None:
        agent.interrupt.side_effect = interrupt
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


def _sse_events(body: str) -> List[Dict[str, Any]]:
    lines = body.splitlines()
    assert not any(line.startswith("event:") for line in lines)
    assert not any(line.startswith("id:") for line in lines)
    return [
        json.loads(line.removeprefix("data: "))
        for line in lines
        if line.startswith("data: ")
    ]


def _log_events(log: Any) -> List[Dict[str, Any]]:
    """Decode a run's retained frames without going through the HTTP layer."""
    return [
        json.loads(stored.frame.removeprefix(b"data: ").strip())
        for stored in log.events
    ]


async def _read_one_sse_event(response: ClientResponse) -> Dict[str, Any]:
    frame = await asyncio.wait_for(response.content.readuntil(b"\n\n"), 2.0)
    data_line = next(
        line for line in frame.decode("utf-8").splitlines() if line.startswith("data: ")
    )
    return json.loads(data_line.removeprefix("data: "))


async def _wait_for_terminal(
    adapter: APIServerAdapter,
    run_id: str,
    *,
    timeout: float = 3.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        log = adapter._turn_event_logs.get_log(run_id)
        if log is not None and log.terminal:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not become terminal: {run_id}")


class _JsonRequest:
    def __init__(
        self,
        body: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.body = body
        self.headers = headers or {}

    async def json(self) -> Dict[str, Any]:
        return self.body


async def _run_without_http_server(
    adapter: APIServerAdapter, body: Dict[str, Any]
) -> tuple[web.Response, List[Dict[str, Any]]]:
    response = await adapter._handle_runs(_JsonRequest(body))  # type: ignore[arg-type]
    payload = json.loads(response.text)
    run_id = payload["run_id"]
    await _wait_for_terminal(adapter, run_id)
    log = adapter._turn_event_logs.get_log(run_id)
    assert log is not None
    events = [
        json.loads(stored.frame.removeprefix(b"data: ").strip())
        for stored in log.events
    ]
    return response, events


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 3.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert event.is_set()


def _install_log_store(
    adapter: APIServerAdapter,
    *,
    clock: Callable[[], float],
    budget: int = 32 * 1024 * 1024,
    retention: float = 300.0,
) -> TurnEventLogStore:
    store = TurnEventLogStore(
        clock=clock,
        run_log_ring_budget_bytes=budget,
        terminal_retention_seconds=retention,
    )
    adapter._turn_event_logs = store
    return store


@pytest.mark.asyncio
async def test_capabilities_stamp_turn_event_log_without_changing_legacy_boolean() -> (
    None
):
    adapter = _make_adapter()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/v1/capabilities")
        payload = await response.json()

    assert response.status == 200
    assert payload["turn_event_log_api_version"] == 2
    assert payload["features"]["run_events_sse"] is True


def test_omnio_extension_event_types_are_explicit_and_namespaced() -> None:
    expected = {
        "response.omnio.interaction",
        "response.omnio.interaction_completed",
        "response.omnio.client_event",
        "response.omnio.gen_ui",
        "response.omnio.task_list",
        "response.omnio.warmup",
        "response.omnio.subagent_start",
        "response.omnio.subagent_complete",
        "response.omnio.interrupted_history",
        "response.omnio.approval_request",
        "response.omnio.approval_responded",
        "response.omnio.steer_missed",
        "response.omnio.log_pressure",
    }
    assert OMNIO_EXTENSION_EVENT_TYPES == expected

    store = TurnEventLogStore()
    store.create_run("run_extensions", "session-extensions")
    emitter = TurnEventEmitter(store, "run_extensions", "session-extensions")
    emitter.response_started()
    for event_type in sorted(expected):
        fields: Dict[str, Any] = {"surface": event_type}
        if event_type == "response.omnio.interaction_completed":
            fields = {"answered": True, "choice": "continue"}
        emitter.omnio_event(event_type, **fields)
    emitter.response_completed()

    log = store.get_log("run_extensions")
    assert log is not None
    events = [
        json.loads(stored.frame.removeprefix(b"data: ").strip())
        for stored in log.events
    ]
    extensions = [
        event for event in events if event["type"].startswith("response.omnio.")
    ]
    assert {event["type"] for event in extensions} == expected
    assert [event["sequence_number"] for event in events] == list(
        range(1, len(events) + 1)
    )
    completed = next(
        event
        for event in extensions
        if event["type"] == "response.omnio.interaction_completed"
    )
    assert completed["answered"] is True
    assert completed["choice"] == "continue"
    with pytest.raises(ValueError, match="unknown Omnio response event"):
        emitter.omnio_event("response.omnio.unregistered")


def test_task_list_emitter_bounds_and_filters_todo_items() -> None:
    store = TurnEventLogStore()
    store.create_run("run_tasks", "session-tasks")
    emitter = TurnEventEmitter(store, "run_tasks", "session-tasks")
    emitter.response_started()
    emitter.task_list([
        {
            "id": str(index),
            "content": f"task {index}",
            "status": "pending",
            "ignored": "not projected",
        }
        for index in range(MAX_TODO_ITEMS + 1)
    ] + [None, {"id": 123, "ignored": "empty after filtering"}])
    emitter.response_completed()

    log = store.get_log("run_tasks")
    assert log is not None
    event = next(
        json.loads(stored.frame.removeprefix(b"data: ").strip())
        for stored in log.events
        if b'response.omnio.task_list' in stored.frame
    )
    assert len(event["todos"]) == MAX_TODO_ITEMS
    assert event["todos"][0] == {
        "id": "0",
        "content": "task 0",
        "status": "pending",
    }
    assert event["todos"][-1]["id"] == str(MAX_TODO_ITEMS - 1)


@pytest.mark.asyncio
async def test_two_concurrent_subscribers_receive_the_same_complete_stream() -> None:
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_fanout", "session-fanout")
    emitter = TurnEventEmitter(store, "run_fanout", "session-fanout")
    emitter.response_started()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        first = await client.get("/v1/runs/run_fanout/events")
        second = await client.get("/v1/runs/run_fanout/events")

        emitter.output_text_start("message-1")
        emitter.output_text_delta("message-1", "hello")
        emitter.output_text_done("message-1")
        emitter.response_completed()

        first_body, second_body = await asyncio.gather(first.text(), second.text())

    first_events = _sse_events(first_body)
    second_events = _sse_events(second_body)
    assert first_events == second_events
    assert [event["sequence_number"] for event in first_events] == list(range(1, 8))
    assert [event["type"] for event in first_events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert first_events[0]["response"] == {
        "id": "resp_fanout",
        "status": "in_progress",
        "created_at": first_events[0]["response"]["created_at"],
    }
    assert first_events[-1]["response"] == {
        "id": "resp_fanout",
        "status": "completed",
        "created_at": first_events[0]["response"]["created_at"],
    }
    assert first_events[2]["output_index"] == 0
    assert first_events[2]["item"] == {
        "id": "message-1",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
        "started_at": first_events[2]["item"]["started_at"],
    }
    assert isinstance(first_events[2]["item"]["started_at"], float)
    assert first_events[5]["item"]["content"] == [
        {"type": "output_text", "text": "hello"}
    ]
    # The closed item reports the real interval: started when it opened,
    # completed no earlier than that.
    done_item = first_events[5]["item"]
    assert done_item["started_at"] == first_events[2]["item"]["started_at"]
    assert done_item["completed_at"] >= done_item["started_at"]
    assert all("seq" not in event for event in first_events)
    assert all(
        "runId" not in event and "threadId" not in event for event in first_events
    )


def test_first_delta_flushes_immediately_and_later_small_deltas_coalesce() -> None:
    store = TurnEventLogStore()
    store.create_run("run_delta_coalesce", "session-delta-coalesce")
    emitter = TurnEventEmitter(store, "run_delta_coalesce", "session-delta-coalesce")
    emitter.response_started()
    emitter.output_text_start("message-1")
    log = store.get_log("run_delta_coalesce")
    assert log is not None

    emitter.output_text_delta("message-1", "he")
    # The item's first delta protects time-to-first-token: it mints its own
    # frame rather than waiting on a coalescing window.
    assert _log_events(log)[-1] == {
        "type": "response.output_text.delta",
        "item_id": "message-1",
        "output_index": 0,
        "content_index": 0,
        "delta": "he",
        "sequence_number": 4,
    }

    emitter.output_text_delta("message-1", "llo")
    emitter.output_text_delta("message-1", " world")
    # Neither the byte nor the age bound has been crossed, so these stay
    # buffered instead of minting one frame per provider delta.
    assert _log_events(log)[-1]["delta"] == "he"

    emitter.output_text_done("message-1")
    events = _log_events(log)
    delta_events = [e for e in events if e["type"] == "response.output_text.delta"]
    assert [e["delta"] for e in delta_events] == ["he", "llo world"]
    # output_text_done's guaranteed flush-before-done lands the coalesced
    # tail ahead of the done/item-done pair, and no content_part events are
    # minted at all — nobody reads them.
    assert [e["type"] for e in events[-3:]] == [
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
    ]
    assert events[-2]["text"] == "hello world"


def test_delta_buffer_flushes_once_it_reaches_the_byte_threshold() -> None:
    store = TurnEventLogStore()
    store.create_run("run_delta_size", "session-delta-size")
    emitter = TurnEventEmitter(store, "run_delta_size", "session-delta-size")
    emitter.response_started()
    emitter.output_text_start("message-1")
    log = store.get_log("run_delta_size")
    assert log is not None

    emitter.output_text_delta("message-1", "a")  # first delta flushes alone
    chunk = "b" * (DELTA_COALESCE_BYTES - 1)
    emitter.output_text_delta("message-1", chunk)
    assert _log_events(log)[-1]["delta"] == "a"  # still under the byte cap

    emitter.output_text_delta("message-1", "c")  # tips the buffer to 512 bytes
    events = _log_events(log)
    assert events[-1]["type"] == "response.output_text.delta"
    assert events[-1]["delta"] == chunk + "c"


def test_delta_buffer_flushes_once_it_ages_past_the_time_threshold() -> None:
    clock = _Clock()
    store = TurnEventLogStore(clock=clock)
    store.create_run("run_delta_age", "session-delta-age")
    emitter = TurnEventEmitter(store, "run_delta_age", "session-delta-age")
    emitter.response_started()
    emitter.output_text_start("message-1")
    log = store.get_log("run_delta_age")
    assert log is not None

    emitter.output_text_delta("message-1", "a")  # first delta flushes alone
    emitter.output_text_delta("message-1", "b")
    assert _log_events(log)[-1]["delta"] == "a"  # neither bound crossed yet

    clock.now += DELTA_COALESCE_SECONDS + 0.001
    emitter.output_text_delta("message-1", "c")  # now past the age deadline
    events = _log_events(log)
    assert events[-1]["type"] == "response.output_text.delta"
    assert events[-1]["delta"] == "bc"


def test_pending_text_delta_flushes_before_any_other_event_type() -> None:
    store = TurnEventLogStore()
    store.create_run("run_delta_preempt", "session-delta-preempt")
    emitter = TurnEventEmitter(store, "run_delta_preempt", "session-delta-preempt")
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "a")  # first delta flushes alone
    emitter.output_text_delta("message-1", "b")  # buffered, below both bounds

    # A function-call boundary must never jump ahead of "b" — this is the
    # emitter's own invariant, independent of caller-side close-item
    # ordering.
    emitter.function_call_start("call-1", "search")

    log = store.get_log("run_delta_preempt")
    assert log is not None
    events = _log_events(log)
    assert [e["type"] for e in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.added",
    ]
    assert [e["delta"] for e in events[3:5]] == ["a", "b"]


def test_pending_delta_flushes_before_a_failed_terminal_event() -> None:
    store = TurnEventLogStore()
    store.create_run("run_delta_failure", "session-delta-failure")
    emitter = TurnEventEmitter(store, "run_delta_failure", "session-delta-failure")
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "a")  # first delta flushes alone
    emitter.output_text_delta("message-1", "unflushed tail")  # left buffered

    emitter.response_failed("boom", code="provider_error")

    log = store.get_log("run_delta_failure")
    assert log is not None
    events = _log_events(log)
    assert [e["type"] for e in events][-2:] == [
        "response.output_text.delta",
        "response.failed",
    ]
    assert events[-2]["delta"] == "unflushed tail"
    assert log.terminal


def test_pending_delta_flushes_before_an_interrupted_terminal_event() -> None:
    store = TurnEventLogStore()
    store.create_run("run_delta_interrupt", "session-delta-interrupt")
    emitter = TurnEventEmitter(store, "run_delta_interrupt", "session-delta-interrupt")
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "a")  # first delta flushes alone
    emitter.output_text_delta("message-1", "cut off mid-word")  # left buffered

    emitter.response_incomplete()

    log = store.get_log("run_delta_interrupt")
    assert log is not None
    events = _log_events(log)
    assert [e["type"] for e in events][-2:] == [
        "response.output_text.delta",
        "response.incomplete",
    ]
    assert events[-2]["delta"] == "cut off mid-word"
    assert log.terminal


def test_sequence_numbers_stay_contiguous_across_multiple_coalesced_flushes() -> None:
    store = TurnEventLogStore()
    store.create_run("run_delta_contiguous", "session-delta-contiguous")
    emitter = TurnEventEmitter(
        store, "run_delta_contiguous", "session-delta-contiguous"
    )
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "first")  # flush #1: item's first delta
    chunk = "x" * DELTA_COALESCE_BYTES
    emitter.output_text_delta("message-1", chunk)  # flush #2: crosses the byte cap
    emitter.output_text_delta("message-1", "tail")  # left buffered
    emitter.output_text_done("message-1")  # flush #3: flush-before-done

    log = store.get_log("run_delta_contiguous")
    assert log is not None
    events = _log_events(log)
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))
    delta_events = [e for e in events if e["type"] == "response.output_text.delta"]
    assert [e["delta"] for e in delta_events] == ["first", chunk, "tail"]


def test_reasoning_deltas_also_coalesce_and_flush_before_done() -> None:
    store = TurnEventLogStore()
    store.create_run("run_reasoning_coalesce", "session-reasoning-coalesce")
    emitter = TurnEventEmitter(
        store, "run_reasoning_coalesce", "session-reasoning-coalesce"
    )
    emitter.response_started()
    emitter.reasoning_start("reasoning-1")
    emitter.reasoning_text_delta("reasoning-1", "thinking")  # first delta, flushes alone
    emitter.reasoning_text_delta("reasoning-1", " more")  # left buffered

    log = store.get_log("run_reasoning_coalesce")
    assert log is not None
    assert _log_events(log)[-1]["delta"] == "thinking"

    emitter.reasoning_text_done("reasoning-1")
    events = _log_events(log)
    delta_events = [
        e for e in events if e["type"] == "response.reasoning_text.delta"
    ]
    assert [e["delta"] for e in delta_events] == ["thinking", " more"]
    assert [e["type"] for e in events[-3:]] == [
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",
    ]


@pytest.mark.asyncio
async def test_cursor_replays_exact_sequence_numbers_then_follows_live() -> None:
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_cursor", "session-cursor")
    emitter = TurnEventEmitter(store, "run_cursor", "session-cursor")
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "retained")

    async with TestClient(TestServer(_make_app(adapter))) as client:
        # "retained" is the item's first delta, so it flushed immediately as
        # sequence 4; subscribing after=3 replays it.
        response = await client.get("/v1/runs/run_cursor/events?after=3")

        # "live" is a second delta on the same item: it stays buffered until
        # output_text_done's guaranteed flush-before-done emits it.
        emitter.output_text_delta("message-1", "live")
        emitter.output_text_done("message-1")
        emitter.response_completed()
        body = await response.text()

    events = _sse_events(body)
    assert [event["sequence_number"] for event in events] == list(range(4, 9))
    assert [event.get("delta") for event in events[:2]] == ["retained", "live"]


@pytest.mark.asyncio
async def test_events_response_stamps_replay_boundary_recorded_at_connect() -> None:
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_replay_header", "session-replay-header")
    emitter = TurnEventEmitter(store, "run_replay_header", "session-replay-header")
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "recorded")
    recorded_high_water = store.get_log(  # type: ignore[union-attr]
        "run_replay_header"
    ).sequence_number_high_water

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/v1/runs/run_replay_header/events?after=0")
        connect_boundary = response.headers["X-Omnio-Replay-Through"]

        emitter.output_text_delta("message-1", "live")
        emitter.output_text_done("message-1")
        emitter.response_completed()
        body = await response.text()

    assert connect_boundary == str(recorded_high_water)
    events = _sse_events(body)
    assert events[-1]["sequence_number"] > recorded_high_water


@pytest.mark.asyncio
async def test_terminal_replay_stamps_boundary_covering_every_frame() -> None:
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_replay_terminal", "session-replay-terminal")
    emitter = TurnEventEmitter(store, "run_replay_terminal", "session-replay-terminal")
    emitter.response_started()
    emitter.output_text_start("message-1")
    emitter.output_text_delta("message-1", "already finished")
    emitter.output_text_done("message-1")
    emitter.response_completed()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/v1/runs/run_replay_terminal/events?after=0")
        boundary = int(response.headers["X-Omnio-Replay-Through"])
        body = await response.text()

    events = _sse_events(body)
    assert events
    assert all(event["sequence_number"] <= boundary for event in events)


@pytest.mark.asyncio
async def test_tombstone_caught_up_response_stamps_replay_boundary() -> None:
    clock = _Clock()
    adapter = _make_adapter()
    store = _install_log_store(adapter, clock=clock)
    store.create_run("run_replay_tombstone", "session-replay-tombstone")
    emitter = TurnEventEmitter(
        store, "run_replay_tombstone", "session-replay-tombstone"
    )
    emitter.response_started()
    emitter.response_completed()
    high_water = store.get_log(  # type: ignore[union-attr]
        "run_replay_tombstone"
    ).sequence_number_high_water
    clock.now += 300.0

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(
            f"/v1/runs/run_replay_tombstone/events?after={high_water}"
        )
        body = await response.text()

    assert response.status == 200
    assert response.headers["X-Omnio-Replay-Through"] == str(high_water)
    assert _sse_events(body) == []


@pytest.mark.asyncio
async def test_live_run_rejects_cursor_ahead_of_high_water() -> None:
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_future_cursor", "session-future-cursor")
    emitter = TurnEventEmitter(store, "run_future_cursor", "session-future-cursor")
    emitter.response_started()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/v1/runs/run_future_cursor/events?after=3")
        payload = await response.json()

    assert response.status == 400
    assert payload == {"error": "invalid_cursor"}


@pytest.mark.asyncio
async def test_terminal_run_accepts_cursor_ahead_of_high_water_as_caught_up() -> None:
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_terminal_cursor", "session-terminal-cursor")
    emitter = TurnEventEmitter(store, "run_terminal_cursor", "session-terminal-cursor")
    emitter.response_started()
    emitter.response_completed()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/v1/runs/run_terminal_cursor/events?after=99")
        body = await response.text()

    assert response.status == 200
    assert _sse_events(body) == []


@pytest.mark.asyncio
async def test_disconnect_does_not_destroy_log_and_reconnect_resumes_by_cursor() -> (
    None
):
    adapter = _make_adapter()
    store = adapter._turn_event_logs
    store.create_run("run_reconnect", "session-reconnect")
    emitter = TurnEventEmitter(store, "run_reconnect", "session-reconnect")
    emitter.response_started()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        first = await client.get("/v1/runs/run_reconnect/events")
        observed = await _read_one_sse_event(first)
        assert observed["sequence_number"] == 1
        first.close()
        await asyncio.sleep(0)

        emitter.output_text_start("message-1")
        emitter.output_text_delta("message-1", "survived disconnect")
        emitter.output_text_done("message-1")
        emitter.response_completed()

        resumed = await client.get(
            f"/v1/runs/run_reconnect/events?after={observed['sequence_number']}"
        )
        resumed_body = await resumed.text()

    events = _sse_events(resumed_body)
    assert [event["sequence_number"] for event in events] == list(range(2, 8))
    assert (
        next(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        == "survived disconnect"
    )
    assert store.get_log("run_reconnect") is not None


@pytest.mark.asyncio
async def test_cursor_expiry_is_410_only_when_tombstone_proves_history_loss() -> None:
    clock = _Clock()
    adapter = _make_adapter()
    store = _install_log_store(adapter, clock=clock)
    store.create_run("run_expired", "session-expired")
    emitter = TurnEventEmitter(store, "run_expired", "session-expired")
    emitter.response_started()
    emitter.response_completed()
    high_water = store.get_log(  # type: ignore[union-attr]
        "run_expired"
    ).sequence_number_high_water
    clock.now += 300.0

    async with TestClient(TestServer(_make_app(adapter))) as client:
        expired = await client.get("/v1/runs/run_expired/events?after=0")
        expired_payload = await expired.json()
        fully_observed = await client.get(
            f"/v1/runs/run_expired/events?after={high_water}"
        )
        fully_observed_body = await fully_observed.text()
        unknown = await client.get("/v1/runs/run_never_seen/events?after=0")

    assert expired.status == 410
    assert expired_payload == {"error": "cursor_expired"}
    assert fully_observed.status == 200
    assert _sse_events(fully_observed_body) == []
    assert unknown.status == 404


@pytest.mark.asyncio
async def test_recoverable_runs_are_isolated_by_owning_profile() -> None:
    adapter = _make_adapter()

    async with TestClient(TestServer(_make_profile_app(adapter))) as client:
        with patch.object(
            adapter,
            "_create_agent",
            return_value=_agent(
                lambda **_kwargs: {
                    "final_response": "done",
                    "messages": [],
                }
            ),
        ):
            started = await client.post("/p/foo/v1/runs", json={"input": "hello"})
            run_id = (await started.json())["run_id"]
            await _wait_for_terminal(adapter, run_id)

        owner_response = await client.get("/p/foo/v1/runs?recoverable=1")
        foreign_response = await client.get("/p/bar/v1/runs?recoverable=1")
        owner_payload = await owner_response.json()
        foreign_payload = await foreign_response.json()

    assert [item["runId"] for item in owner_payload["data"]] == [run_id]
    assert foreign_payload["data"] == []
    owned_log = adapter._turn_event_logs.get_log(run_id)
    assert owned_log is not None
    assert owned_log.owner_profile == "foo"


@pytest.mark.asyncio
async def test_run_event_cursor_lookup_is_isolated_by_owning_profile() -> None:
    adapter = _make_adapter()

    async with TestClient(TestServer(_make_profile_app(adapter))) as client:
        with patch.object(
            adapter,
            "_create_agent",
            return_value=_agent(
                lambda **_kwargs: {
                    "final_response": "done",
                    "messages": [],
                }
            ),
        ):
            started = await client.post("/p/foo/v1/runs", json={"input": "hello"})
            run_id = (await started.json())["run_id"]
            await _wait_for_terminal(adapter, run_id)

        foreign_response = await client.get(f"/p/bar/v1/runs/{run_id}/events")
        owner_response = await client.get(f"/p/foo/v1/runs/{run_id}/events")
        owner_body = await owner_response.text()

    assert foreign_response.status == 404
    assert owner_response.status == 200
    assert _sse_events(owner_body)[-1]["type"] == "response.completed"


@pytest.mark.parametrize(
    ("create_prefix", "read_prefix"),
    (("", "/p/default"), ("/p/default", "")),
)
@pytest.mark.asyncio
async def test_native_and_mirrored_default_profile_routes_share_turn_logs(
    create_prefix: str,
    read_prefix: str,
) -> None:
    adapter = _make_adapter()

    async with TestClient(TestServer(_make_multiplex_app(adapter))) as client:
        with patch.object(
            adapter,
            "_create_agent",
            return_value=_agent(
                lambda **_kwargs: {
                    "final_response": "done",
                    "messages": [],
                }
            ),
        ):
            started = await client.post(
                f"{create_prefix}/v1/runs",
                json={"input": "hello"},
            )
            run_id = (await started.json())["run_id"]
            await _wait_for_terminal(adapter, run_id)

        events_response = await client.get(f"{read_prefix}/v1/runs/{run_id}/events")
        events_body = await events_response.text()
        recoverable_response = await client.get(f"{read_prefix}/v1/runs?recoverable=1")
        recoverable_payload = await recoverable_response.json()

    assert events_response.status == 200
    events = _sse_events(events_body)
    assert events[-1]["type"] == "response.completed"
    assert events[0]["response"]["id"] == f"resp_{run_id.removeprefix('run_')}"
    assert [item["runId"] for item in recoverable_payload["data"]] == [run_id]
    owned_log = adapter._turn_event_logs.get_log(run_id)
    assert owned_log is not None
    assert owned_log.owner_profile == "default"


@pytest.mark.asyncio
async def test_ring_evicts_oldest_frames_instead_of_failing_the_run() -> None:
    """Regression for the incident: hitting the byte budget must never kill a run.

    A budget this small (4 KiB) is blown by the single oversized delta below,
    which is itself larger than the whole budget — the pathological case
    ``_make_room`` has to admit rather than refuse. The run still completes
    normally; the ring just evicts the delta frame to make room for what
    follows, advancing its floor.
    """
    adapter = _make_adapter()
    _install_log_store(adapter, clock=_Clock(), budget=4_096)
    built_agent: Optional[MagicMock] = None

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal built_agent

        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["stream_delta_callback"]("x" * 8_192)
            return {"final_response": "x" * 8_192, "messages": []}

        built_agent = _agent(run)
        return built_agent

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "overflow"})
            run_id = (await started.json())["run_id"]
            # Attach only once the run is fully terminal: a still-live run
            # keeps evicting concurrently with this GET, and a consumer that
            # falls behind that gets legitimately closed mid-stream (see
            # test_eviction_past_attached_cursor_closes_stream_for_reattach)
            # — that race isn't what this test is about.
            await _wait_for_terminal(adapter, run_id)
            events_response = await client.get(f"/v1/runs/{run_id}/events")
            body = await events_response.text()

    events = _sse_events(body)
    log = adapter._turn_event_logs.get_log(run_id)
    assert started.status == 202
    assert log is not None
    assert log.terminal
    assert log.failure_reason is None
    assert log.floor > 0, "the oversized delta frame must have been evicted"
    # The client attached after the evicted delta frame fell off the ring, so
    # its replay resumes at the floor — it never sees response.failed/failed.
    assert events[-1]["type"] == "response.completed"
    assert adapter._run_statuses[run_id]["status"] == "completed"
    assert built_agent is not None
    built_agent.interrupt.assert_not_called()
    assert events_response.headers["X-Omnio-Replay-From"] == str(log.floor)


@pytest.mark.asyncio
async def test_none_final_response_does_not_mask_structured_run_failure() -> None:
    adapter = _make_adapter()
    failed_result = {
        "final_response": None,
        "failed": True,
        "error": "original agent failure",
        "messages": [],
    }

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(
            adapter,
            "_create_agent",
            return_value=_agent(lambda **_kwargs: failed_result),
        ):
            started = await client.post("/v1/runs", json={"input": "fail"})
            run_id = (await started.json())["run_id"]
            response = await client.get(f"/v1/runs/{run_id}/events")
            events = _sse_events(await response.text())

    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "run_failed"
    assert adapter._run_statuses[run_id]["error"] == "original agent failure"


@pytest.mark.asyncio
async def test_successful_run_emits_file_annotations_before_terminal_with_contiguous_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_response = "The final report is at `/brand/final report.pdf`."
    annotation = {
        "type": "file_path",
        "path": "/brand/final report.pdf",
        "filename": "final report.pdf",
        "content_type": "application/pdf",
        "size_label": "2 KB",
        "size_bytes": 1536,
    }
    finalize = AsyncMock(return_value=[annotation])
    monkeypatch.setenv(
        api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
        "http://127.0.0.1:8642/internal/turn-finalize",
    )
    monkeypatch.setattr(
        api_server_module, "_request_turn_finalize_annotations", finalize
    )
    adapter = _make_adapter()

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_agent(
            lambda **_kwargs: {
                "final_response": final_response,
                "messages": [],
            }
        ),
    ):
        started, events = await _run_without_http_server(
            adapter,
            {
                "input": "make report",
                "turn_id": "turn-1",
                "session_id": "session-1",
            },
        )

    assert started.status == 202
    annotation_event = next(
        event
        for event in events
        if event["type"] == "response.output_text.annotation.added"
    )
    message_event = next(
        event
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "message"
    )
    assert events[-2] == annotation_event
    assert events[-1]["type"] == "response.completed"
    assert [event["sequence_number"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert annotation_event == {
        "type": "response.output_text.annotation.added",
        "item_id": message_event["item"]["id"],
        "output_index": message_event["output_index"],
        "content_index": 0,
        "annotation_index": 0,
        "annotation": annotation,
        "sequence_number": annotation_event["sequence_number"],
    }
    finalize.assert_awaited_once_with(
        "http://127.0.0.1:8642/internal/turn-finalize",
        {
            "run_id": json.loads(started.text)["run_id"],
            "turn_id": "turn-1",
            "session_id": "session-1",
            "final_text": final_response,
            "message_item_id": message_event["item"]["id"],
            "output_index": message_event["output_index"],
        },
    )


def _file_annotation(path: str) -> Dict[str, Any]:
    return {
        "type": "file_path",
        "path": path,
        "filename": path.rsplit("/", 1)[-1],
        "content_type": "application/pdf",
        "size_label": "2 KB",
        "size_bytes": 1536,
    }


@pytest.mark.asyncio
async def test_finalize_hook_annotates_every_emitted_message_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file handed over mid-turn belongs to the block that named it.

    The reply names the draft, then runs another tool and closes on a
    different file. Scanning only the last block would leave the draft
    unannotated, so its chat link would render as plain text forever.
    """
    handover_text = "Here's the draft: `/brand/draft.pdf`."
    final_text = "Download `/brand/final.pdf`."

    async def finalize(_hook_url: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = (
            "/brand/draft.pdf"
            if "draft" in payload["final_text"]
            else "/brand/final.pdf"
        )
        return [_file_annotation(path)]

    finalize_mock = AsyncMock(side_effect=finalize)
    monkeypatch.setenv(
        api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
        "http://127.0.0.1:8642/internal/turn-finalize",
    )
    monkeypatch.setattr(
        api_server_module, "_request_turn_finalize_annotations", finalize_mock
    )
    adapter = _make_adapter()

    def create_agent(**kwargs: Any) -> MagicMock:
        stream = kwargs["stream_delta_callback"]
        tool_start = kwargs["tool_start_callback"]
        tool_complete = kwargs["tool_complete_callback"]

        def run(**_kwargs: Any) -> Dict[str, Any]:
            stream(handover_text)
            tool_start("call-finalize", "terminal", {"command": "make report"})
            tool_complete(
                "call-finalize",
                "terminal",
                {"command": "make report"},
                "done",
            )
            stream("\n\n" + final_text)
            return {
                "final_response": final_text,
                "messages": [],
            }

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=create_agent):
        started, events = await _run_without_http_server(
            adapter,
            {"input": "make report", "turn_id": "turn-1"},
        )

    message_events = [
        event
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "message"
    ]
    assert [event["item"]["content"][0]["text"] for event in message_events] == [
        handover_text,
        final_text,
    ]
    assert message_events[0]["item"]["id"] != message_events[1]["item"]["id"]

    run_id = json.loads(started.text)["run_id"]
    payloads = {
        call.args[1]["message_item_id"]: call.args[1]
        for call in finalize_mock.await_args_list
    }
    assert len(payloads) == 2
    for text, message_event in zip(
        [handover_text, final_text], message_events, strict=True
    ):
        payload = payloads[message_event["item"]["id"]]
        assert payload["final_text"] == text
        assert payload["output_index"] == message_event["output_index"]
        assert payload["run_id"] == run_id
        assert payload["turn_id"] == "turn-1"

    annotations = [
        event
        for event in events
        if event["type"] == "response.output_text.annotation.added"
    ]
    assert [
        (event["item_id"], event["annotation"]["path"]) for event in annotations
    ] == [
        (message_events[0]["item"]["id"], "/brand/draft.pdf"),
        (message_events[1]["item"]["id"], "/brand/final.pdf"),
    ]
    assert events[-1]["type"] == "response.completed"
    assert [event["sequence_number"] for event in events] == list(
        range(1, len(events) + 1)
    )


@pytest.mark.asyncio
async def test_message_block_is_annotated_while_the_turn_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block is annotated when it closes, not when the turn ends.

    `request_user_input` blocks its turn for as long as the user takes to
    answer, so a hand-over that waited for the turn to finish would leave the
    file unopenable for exactly as long as the question stands. The fake agent
    here refuses to move on until the block's hook has run, so an end-of-turn
    annotation deadlocks this test instead of passing it.
    """
    handover_text = "Here's the draft: `/brand/draft.pdf`."
    hook_called = threading.Event()

    async def finalize(_hook_url: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "draft" not in payload["final_text"]:
            return []
        hook_called.set()
        return [_file_annotation("/brand/draft.pdf")]

    monkeypatch.setenv(
        api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
        "http://127.0.0.1:8642/internal/turn-finalize",
    )
    monkeypatch.setattr(
        api_server_module, "_request_turn_finalize_annotations", finalize
    )
    adapter = _make_adapter()

    def create_agent(**kwargs: Any) -> MagicMock:
        stream = kwargs["stream_delta_callback"]
        tool_start = kwargs["tool_start_callback"]
        tool_complete = kwargs["tool_complete_callback"]

        def run(**_kwargs: Any) -> Dict[str, Any]:
            stream(handover_text)
            tool_start("call-ask", "terminal", {"command": "ask"})
            assert hook_called.wait(10), "the block was not annotated mid-turn"
            tool_complete("call-ask", "terminal", {"command": "ask"}, "answered")
            stream("\n\nAll set.")
            return {"final_response": "All set.", "messages": []}

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=create_agent):
        _started, events = await _run_without_http_server(
            adapter,
            {"input": "draft it", "turn_id": "turn-1"},
        )

    handover_item = next(
        event
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "message"
        and event["item"]["content"][0]["text"] == handover_text
    )
    annotations = [
        event
        for event in events
        if event["type"] == "response.output_text.annotation.added"
    ]
    assert [event["item_id"] for event in annotations] == [
        handover_item["item"]["id"]
    ]
    assert annotations[0]["annotation"]["path"] == "/brand/draft.pdf"
    assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_stop_accepted_during_finalize_hook_cancels_without_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = {
        "type": "file_path",
        "path": "/brand/report.pdf",
        "filename": "report.pdf",
        "content_type": "application/pdf",
        "size_label": "2 KB",
        "size_bytes": 1536,
    }
    monkeypatch.setenv(
        api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
        "http://127.0.0.1:8642/internal/turn-finalize",
    )
    adapter = _make_adapter()

    async def finalize(_hook_url: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        adapter._stopping_run_ids.add(payload["run_id"])
        await asyncio.sleep(0)
        return [annotation]

    monkeypatch.setattr(
        api_server_module, "_request_turn_finalize_annotations", finalize
    )
    with patch.object(
        adapter,
        "_create_agent",
        return_value=_agent(
            lambda **_kwargs: {
                "final_response": "Download /brand/report.pdf",
                "messages": [],
            }
        ),
    ):
        started, events = await _run_without_http_server(
            adapter,
            {"input": "make report", "turn_id": "turn-1"},
        )

    run_id = json.loads(started.text)["run_id"]
    assert events[-1]["type"] == "response.incomplete"
    assert adapter._run_statuses[run_id]["status"] == "cancelled"
    assert not any(
        event["type"] == "response.output_text.annotation.added" for event in events
    )


@pytest.mark.asyncio
async def test_successful_run_without_finalize_hook_does_not_call_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV, raising=False)
    finalize = AsyncMock()
    monkeypatch.setattr(
        api_server_module, "_request_turn_finalize_annotations", finalize
    )
    adapter = _make_adapter()

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_agent(
            lambda **_kwargs: {
                "final_response": "Report: /brand/report.pdf",
                "messages": [],
            }
        ),
    ):
        started, events = await _run_without_http_server(
            adapter, {"input": "make report"}
        )

    assert started.status == 202
    assert not any(
        event["type"] == "response.output_text.annotation.added" for event in events
    )
    finalize.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_run_without_message_item_emits_no_file_annotations() -> None:
    adapter = _make_adapter()

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_agent(
            lambda **_kwargs: {
                "final_response": "",
                "messages": [],
            }
        ),
    ):
        started, events = await _run_without_http_server(adapter, {"input": "do work"})

    assert started.status == 202
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]


@pytest.mark.parametrize(
    ("result_fields", "terminal_type"),
    [
        ({"failed": True, "error": "agent failed"}, "response.failed"),
        ({"interrupted": True}, "response.incomplete"),
    ],
)
@pytest.mark.asyncio
async def test_failed_and_cancelled_runs_do_not_emit_file_annotations(
    monkeypatch: pytest.MonkeyPatch,
    result_fields: Dict[str, Any],
    terminal_type: str,
) -> None:
    monkeypatch.setenv(
        api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
        "http://127.0.0.1:8642/internal/turn-finalize",
    )
    finalize = AsyncMock(return_value=[])
    monkeypatch.setattr(
        api_server_module, "_request_turn_finalize_annotations", finalize
    )
    adapter = _make_adapter()
    result = {
        "final_response": "Report: /brand/report.txt",
        "messages": [],
        **result_fields,
    }

    with patch.object(
        adapter,
        "_create_agent",
        return_value=_agent(lambda **_kwargs: result),
    ):
        started, events = await _run_without_http_server(
            adapter, {"input": "make report"}
        )

    assert started.status == 202
    assert events[-1]["type"] == terminal_type
    assert not any(
        event["type"] == "response.output_text.annotation.added" for event in events
    )
    finalize.assert_not_awaited()


@pytest.mark.parametrize("hook_status", [500, "timeout"])
@pytest.mark.asyncio
async def test_finalize_hook_failure_still_emits_terminal_promptly(
    monkeypatch: pytest.MonkeyPatch,
    hook_status: int | str,
) -> None:
    async def finalize(_request: web.Request) -> web.Response:
        if hook_status == "timeout":
            await asyncio.sleep(1.0)
            return web.json_response({"annotations": []})
        return web.json_response({"annotations": []}, status=hook_status)

    hook_app = web.Application()
    hook_app.router.add_post("/internal/turn-finalize", finalize)
    monkeypatch.setenv(api_server_module._OMNIO_TURN_FINALIZE_TIMEOUT_ENV, "0.02")
    adapter = _make_adapter()

    async with TestServer(hook_app) as server:
        monkeypatch.setenv(
            api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
            str(server.make_url("/internal/turn-finalize")),
        )
        started_at = asyncio.get_running_loop().time()
        with patch.object(
            adapter,
            "_create_agent",
            return_value=_agent(
                lambda **_kwargs: {
                    "final_response": "Report: /brand/report.pdf",
                    "messages": [],
                }
            ),
        ):
            started, events = await _run_without_http_server(
                adapter,
                {"input": "make report", "turn_id": "turn-1"},
            )
        elapsed = asyncio.get_running_loop().time() - started_at

    assert started.status == 202
    assert elapsed < 0.5
    assert events[-1]["type"] == "response.completed"
    assert not any(
        event["type"] == "response.output_text.annotation.added" for event in events
    )


@pytest.mark.asyncio
async def test_annotation_batch_over_budget_evicts_instead_of_failing_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch (the annotation append) too big for remaining headroom evicts
    older frames to make room rather than rejecting the whole batch — the
    ring never partially-fails a run the way the old cap-exceeded path did.
    """
    final_response = "Reports: /brand/first.txt and /brand/second.txt"
    annotations = [
        {
            "type": "file_path",
            "path": f"/brand/{name}.txt",
            "filename": f"{name}.txt",
            "content_type": "text/plain",
            "size_label": "11 B",
            "size_bytes": 11,
        }
        for name in ("first", "second")
    ]
    monkeypatch.setenv(
        api_server_module._OMNIO_TURN_FINALIZE_HOOK_ENV,
        "http://127.0.0.1:8642/internal/turn-finalize",
    )
    monkeypatch.setattr(
        api_server_module,
        "_request_turn_finalize_annotations",
        AsyncMock(return_value=annotations),
    )

    clock = _Clock(now=1_000_000_000.0)
    dry_store = TurnEventLogStore(clock=clock)
    dry_store.create_run("run_" + "a" * 32, "run_" + "a" * 32)
    dry_emitter = TurnEventEmitter(
        dry_store,
        "run_" + "a" * 32,
        "run_" + "a" * 32,
    )
    dry_emitter.response_started()
    dry_emitter.output_text_start("msg_" + "b" * 32)
    dry_emitter.output_text_delta("msg_" + "b" * 32, final_response)
    dry_emitter.output_text_done("msg_" + "b" * 32)
    dry_log = dry_store.get_log("run_" + "a" * 32)
    assert dry_log is not None
    base_wire_bytes = dry_log.wire_bytes
    dry_emitter.output_text_annotations_added("msg_" + "b" * 32, annotations)
    first_annotation_bytes = dry_log.events[-2].frame

    adapter = _make_adapter()
    budget = base_wire_bytes + len(first_annotation_bytes) + TERMINAL_FRAME_RESERVE_BYTES
    _install_log_store(adapter, clock=clock, budget=budget)
    with patch.object(
        adapter,
        "_create_agent",
        return_value=_agent(
            lambda **_kwargs: {
                "final_response": final_response,
                "messages": [],
            }
        ),
    ):
        started, events = await _run_without_http_server(
            adapter, {"input": "make reports"}
        )

    run_id = json.loads(started.text)["run_id"]
    log = adapter._turn_event_logs.get_log(run_id)
    assert log is not None
    assert events[-1]["type"] == "response.completed"
    assert adapter._run_statuses[run_id]["status"] == "completed"
    assert log.failure_reason is None
    assert log.floor > 0, "older frames must have been evicted to fit the batch"
    annotation_events = [
        event
        for event in events
        if event["type"] == "response.output_text.annotation.added"
    ]
    assert len(annotation_events) == len(annotations)


def _fill_events(store: TurnEventLogStore, run_id: str, count: int, size: int) -> None:
    """Append ``count`` ordinary frames of roughly ``size`` bytes each."""
    for index in range(count):
        store.append_payload(run_id, {"type": "response.omnio.warmup", "pad": "x" * size})


def test_eviction_preserves_sequence_numbering_and_high_water() -> None:
    """Eviction must never renumber or rewind sequence numbers/high_water."""
    store = TurnEventLogStore(run_log_ring_budget_bytes=2_048)
    run_id = "run_" + "a" * 32
    store.create_run(run_id, run_id)

    _fill_events(store, run_id, 20, 100)

    log = store.get_log(run_id)
    assert log is not None
    assert log.floor > 0, "a budget this small must have evicted something"
    assert log.high_water == 20, "high_water counts every frame ever minted"
    assert log.events, "the most recent frames must still be retained"
    # Sequence numbers are contiguous and monotonic from floor+1 onward, and
    # never repeat or rewind despite the front of the list having been
    # popped repeatedly.
    sequence_numbers = [event.sequence_number for event in log.events]
    assert sequence_numbers == list(range(log.floor + 1, log.high_water + 1))


def test_frames_after_below_floor_serves_from_floor() -> None:
    store = TurnEventLogStore(run_log_ring_budget_bytes=2_048)
    run_id = "run_" + "b" * 32
    store.create_run(run_id, run_id)
    _fill_events(store, run_id, 20, 100)

    log = store.get_log(run_id)
    assert log is not None
    assert log.floor > 0

    # A cursor at 0 (or anywhere below the floor) is clamped to the floor —
    # it never raises, and the returned floor matches the log's.
    result = log.frames_after(0)
    assert result.floor == log.floor
    assert [event.sequence_number for event in result.frames] == list(
        range(log.floor + 1, log.high_water + 1)
    )

    # A cursor already at or above the floor is unaffected by it.
    caught_up = log.frames_after(log.high_water)
    assert caught_up.frames == []


@pytest.mark.asyncio
async def test_replay_from_header_only_stamped_when_cursor_truncated() -> None:
    clock = _Clock()
    adapter = _make_adapter()
    store = _install_log_store(adapter, clock=clock, budget=2_048)
    run_id = "run_" + "c" * 32
    store.create_run(run_id, run_id)
    _fill_events(store, run_id, 20, 100)
    store.mark_terminal(run_id, "completed")
    log = store.get_log(run_id)
    assert log is not None
    assert log.floor > 0

    async with TestClient(TestServer(_make_app(adapter))) as client:
        truncated = await client.get(f"/v1/runs/{run_id}/events?after=0")
        await truncated.text()
        caught_up = await client.get(
            f"/v1/runs/{run_id}/events?after={log.floor}"
        )
        await caught_up.text()

    assert truncated.status == 200
    assert truncated.headers["X-Omnio-Replay-From"] == str(log.floor)
    assert caught_up.status == 200
    assert "X-Omnio-Replay-From" not in caught_up.headers


@pytest.mark.asyncio
async def test_attached_consumer_above_floor_streams_normally_through_unrelated_eviction() -> (
    None
):
    """Eviction of frames an attached consumer never asked for (older than
    its cursor) must not disturb its stream. Only a jump *past* its cursor
    should ever close it — see the sibling test below.
    """
    adapter = _make_adapter()
    store = _install_log_store(adapter, clock=_Clock(), budget=4_096)
    run_id = "run_" + "g" * 32
    store.create_run(run_id, run_id)

    def pad(size: int) -> Dict[str, Any]:
        return {"type": "response.omnio.warmup", "pad": "x" * size}

    for _ in range(10):
        store.append_payload(run_id, pad(50))
    log = store.get_log(run_id)
    assert log is not None
    cursor = log.high_water  # attach right at the current tip

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(f"/v1/runs/{run_id}/events?after={cursor}")
        assert response.status == 200
        assert "X-Omnio-Replay-From" not in response.headers

        # A couple more frames — small enough that any eviction they cause
        # only ever removes frames at or before the attached cursor, never
        # past it.
        store.append_payload(run_id, pad(50))
        store.append_payload(run_id, pad(50))
        assert log.floor <= cursor, "eviction must not have reached the attached cursor"

        first = await _read_one_sse_event(response)
        second = await _read_one_sse_event(response)
        assert [first["sequence_number"], second["sequence_number"]] == [
            cursor + 1,
            cursor + 2,
        ]

        store.mark_terminal(run_id, "completed")
        # Draining to the natural end confirms the handler reached its
        # normal terminal close path, not the mid-stream jump-guard close.
        tail = await asyncio.wait_for(response.content.read(), 2.0)

    assert b"data: " not in tail
    assert b"stream closed" in tail


@pytest.mark.asyncio
async def test_eviction_past_attached_cursor_closes_stream_for_reattach() -> None:
    """If the ring evicts past a slow, already-attached consumer's cursor,
    its stream must close cleanly instead of silently resuming from the new
    floor — an undeclared sequence jump neither the proxy's projector nor
    the browser reducer will accept. The declared-jump contract only exists
    at attach time (X-Omnio-Replay-From), so the client must reattach to get
    a correctly-stamped floor rather than be handed one mid-stream.
    """
    adapter = _make_adapter()
    store = _install_log_store(adapter, clock=_Clock(), budget=4_096)
    run_id = "run_" + "h" * 32
    store.create_run(run_id, run_id)

    def pad(size: int) -> Dict[str, Any]:
        return {"type": "response.omnio.warmup", "pad": "x" * size}

    for _ in range(5):
        store.append_payload(run_id, pad(50))
    log = store.get_log(run_id)
    assert log is not None
    cursor = log.high_water

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(f"/v1/runs/{run_id}/events?after={cursor}")
        assert response.status == 200
        assert "X-Omnio-Replay-From" not in response.headers

        # Blow well past the attached cursor while this consumer never
        # reads — the ring doesn't know or care that anyone is attached.
        for _ in range(80):
            store.append_payload(run_id, pad(50))
        assert log.floor > cursor, "the ring must have evicted past the attached cursor"

        # The stream must close cleanly — no data frames, just the sentinel
        # close comment — rather than silently jump this consumer forward.
        tail = await asyncio.wait_for(response.content.read(), 2.0)
        assert b"data: " not in tail
        assert b"stream closed" in tail

        reattached_count = log.high_water - log.floor
        reattached = await client.get(f"/v1/runs/{run_id}/events?after={cursor}")
        assert reattached.headers["X-Omnio-Replay-From"] == str(log.floor)
        reattached_events = [
            await _read_one_sse_event(reattached) for _ in range(reattached_count)
        ]
        reattached.close()

    assert [event["sequence_number"] for event in reattached_events] == list(
        range(log.floor + 1, log.high_water + 1)
    )


def test_terminal_frame_always_appendable_under_a_full_ring() -> None:
    """The terminal frame must never be dropped, even when a single ordinary
    frame already occupies the entire ring budget."""
    store = TurnEventLogStore(run_log_ring_budget_bytes=512)
    run_id = "run_" + "d" * 32
    store.create_run(run_id, run_id)

    # One frame bigger than the whole budget — the pathological case
    # `_make_room` has to admit rather than refuse.
    huge = store.append_payload(
        run_id, {"type": "response.omnio.warmup", "pad": "x" * 4_096}
    )
    assert huge is not None

    terminal = store.append_payload(
        run_id, {"type": "response.completed"}, force_terminal=True
    )
    assert terminal is not None
    log = store.get_log(run_id)
    assert log is not None
    # The oversized frame had to be evicted to make room; the terminal frame
    # is nonetheless present and is the log's last retained frame.
    assert log.events[-1].sequence_number == terminal.sequence_number


def test_log_pressure_emitted_once_per_crossing_with_rearm() -> None:
    """Fires once past the trigger ratio, stays silent while still above it,
    and only fires again after occupancy has fallen back under the rearm
    ratio and climbed past the trigger a second time.

    Occupancy is driven directly rather than through realistic ring/eviction
    dynamics — this test is purely about the hysteresis state machine. The
    budget is comfortably larger than TERMINAL_FRAME_RESERVE_BYTES so the
    tiny frames appended below never themselves trigger eviction and muddy
    the manually-set occupancy.
    """
    budget = 10 * TERMINAL_FRAME_RESERVE_BYTES
    store = TurnEventLogStore(run_log_ring_budget_bytes=budget)
    run_id = "run_" + "e" * 32
    store.create_run(run_id, run_id)
    emitter = TurnEventEmitter(store, run_id, run_id)
    log = store.get_log(run_id)
    assert log is not None

    def pressure_events() -> List[Dict[str, Any]]:
        return [
            json.loads(stored.frame.removeprefix(b"data: ").strip())
            for stored in log.events
            if json.loads(stored.frame.removeprefix(b"data: ").strip())["type"]
            == "response.omnio.log_pressure"
        ]

    above_trigger = int(budget * 0.75)
    below_rearm = int(budget * 0.2)

    log.wire_bytes = above_trigger
    emitter.omnio_event("response.omnio.warmup")
    first_crossing = pressure_events()
    assert len(first_crossing) == 1
    assert first_crossing[0]["occupancy_bytes"] > 0
    assert first_crossing[0]["budget_bytes"] == budget
    assert log.log_pressure_armed is False

    # Still above the trigger — must not fire again without re-arming.
    emitter.omnio_event("response.omnio.warmup")
    assert len(pressure_events()) == 1

    # Drop below the rearm ratio: re-arms, but this append alone doesn't
    # cross the trigger, so it still doesn't fire.
    log.wire_bytes = below_rearm
    emitter.omnio_event("response.omnio.warmup")
    assert len(pressure_events()) == 1
    assert log.log_pressure_armed is True

    # Climb back past the trigger: fires a second, independent hint.
    log.wire_bytes = above_trigger
    emitter.omnio_event("response.omnio.warmup")
    assert len(pressure_events()) == 2


@pytest.mark.asyncio
async def test_terminal_logs_evict_after_completion_grace_and_live_logs_do_not_age() -> (
    None
):
    clock = _Clock()
    adapter = _make_adapter()
    store = _install_log_store(adapter, clock=clock)

    store.create_run("run_live", "session-live")
    live = TurnEventEmitter(store, "run_live", "session-live")
    live.response_started()

    clock.now += 1.0
    store.create_run("run_terminal", "session-terminal")
    terminal = TurnEventEmitter(store, "run_terminal", "session-terminal")
    terminal.response_started()
    terminal.response_failed("failed", code="test_failure")

    clock.now += 299.0
    async with TestClient(TestServer(_make_app(adapter))) as client:
        in_grace = await client.get("/v1/runs?recoverable=1")
        in_grace_payload = await in_grace.json()

        clock.now += 1.0
        after_grace = await client.get("/v1/runs?recoverable=1")
        after_grace_payload = await after_grace.json()

    in_grace_by_id = {item["runId"]: item for item in in_grace_payload["data"]}
    assert set(in_grace_by_id) == {"run_live", "run_terminal"}
    assert in_grace_by_id["run_live"] == {
        "runId": "run_live",
        "status": "running",
        "sessionId": "session-live",
        "sequence_number": 2,
        "createdAt": 1_000.0,
        "completedAt": None,
        "failureReason": None,
    }
    assert in_grace_by_id["run_terminal"]["status"] == "failed"
    assert in_grace_by_id["run_terminal"]["sequence_number"] == 3
    assert in_grace_by_id["run_terminal"]["failureReason"] == "test_failure"
    assert [item["runId"] for item in after_grace_payload["data"]] == ["run_live"]
    assert store.get_log("run_live") is not None
    assert store.get_log("run_terminal") is None


def test_tombstones_keep_only_the_most_recent_completed_runs() -> None:
    clock = _Clock()
    store = TurnEventLogStore(
        clock=clock,
        terminal_retention_seconds=0,
        tombstone_limit=2,
    )
    for index in range(3):
        run_id = f"run_tombstone_{index}"
        store.create_run(run_id, f"session-{index}")
        emitter = TurnEventEmitter(store, run_id, f"session-{index}")
        emitter.response_started()
        emitter.response_completed()
        clock.now += 1.0

    store.sweep()

    with pytest.raises(UnknownRunError):
        store.lookup_for_cursor("run_tombstone_0", 0)
    with pytest.raises(CursorExpiredError):
        store.lookup_for_cursor("run_tombstone_1", 0)
    with pytest.raises(CursorExpiredError):
        store.lookup_for_cursor("run_tombstone_2", 0)


@pytest.mark.asyncio
async def test_runs_rotate_output_items_at_each_contiguous_block_boundary() -> None:
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["reasoning_callback"]("reasoning one")
            callbacks["stream_delta_callback"]("message one")
            callbacks["tool_start_callback"]("call-one", "terminal", {})
            callbacks["tool_complete_callback"]("call-one", "terminal", {}, "ok")
            callbacks["stream_delta_callback"]("\n\nmessage two")
            callbacks["stream_delta_callback"]("\n\ncontinued")
            callbacks["tool_start_callback"]("call-two", "terminal", {})
            callbacks["tool_complete_callback"]("call-two", "terminal", {}, "ok")
            callbacks["reasoning_callback"]("reasoning two")
            callbacks["stream_delta_callback"]("\n\nmessage three")
            return {"final_response": "message three", "messages": []}

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=build_agent):
        _, events = await _run_without_http_server(adapter, {"input": "do work"})

    boundaries = [
        event
        for event in events
        if event["type"] in {
            "response.output_item.added",
            "response.output_item.done",
        }
    ]
    assert [
        (
            event["type"],
            event["output_index"],
            event["item"]["type"],
        )
        for event in boundaries
    ] == [
        ("response.output_item.added", 0, "reasoning"),
        ("response.output_item.done", 0, "reasoning"),
        ("response.output_item.added", 1, "message"),
        ("response.output_item.done", 1, "message"),
        ("response.output_item.added", 2, "function_call"),
        ("response.output_item.done", 2, "function_call"),
        ("response.output_item.added", 3, "message"),
        ("response.output_item.done", 3, "message"),
        ("response.output_item.added", 4, "function_call"),
        ("response.output_item.done", 4, "function_call"),
        ("response.output_item.added", 5, "reasoning"),
        ("response.output_item.done", 5, "reasoning"),
        ("response.output_item.added", 6, "message"),
        ("response.output_item.done", 6, "message"),
    ]
    added = boundaries[::2]
    done = boundaries[1::2]
    assert [event["item"]["id"] for event in added] == [
        event["item"]["id"] for event in done
    ]
    assert len({event["item"]["id"] for event in added}) == 7
    assert [
        event["item"]["content"][0]["text"]
        for event in done
        if event["item"]["type"] == "message"
    ] == ["message one", "message two\n\ncontinued", "message three"]
    assert [
        event["item"]["content"][0]["text"]
        for event in done
        if event["item"]["type"] == "reasoning"
    ] == ["reasoning one", "reasoning two"]
    assert [event["sequence_number"] for event in events] == list(
        range(1, len(events) + 1)
    )


@pytest.mark.asyncio
async def test_run_result_tool_calls_never_started_live_are_not_replayed_at_run_end() -> (
    None
):
    """A tool_call the model saw only as an error result (rejected before
    dispatch — scope/plugin/guardrail block, invalid tool name) must not be
    fabricated into a run-end function_call item: the executor and
    conversation loop emit those calls live, and the gateway's run-end sweep
    only closes calls that actually started live.
    """
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            # No tool_start_callback/tool_complete_callback fired — this
            # simulates a call blocked before dispatch, whose only trace is
            # in the final message history the mock agent returns.
            return {
                "final_response": "done",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-rejected",
                                "function": {
                                    "name": "blocked_tool",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-rejected",
                        "content": '{"error": "blocked"}',
                    },
                ],
            }

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=build_agent):
        _, events = await _run_without_http_server(adapter, {"input": "do work"})

    assert not any(
        event["type"] == "response.output_item.added"
        and event["item"]["type"] == "function_call"
        for event in events
    )


@pytest.mark.asyncio
async def test_run_result_tool_call_started_live_but_never_completed_is_closed_at_run_end() -> (
    None
):
    """A call that started live (tool_start_callback fired) but never got a
    tool_complete_callback — e.g. abandoned on interrupt — must still get
    its response.output_item.done at run end so the stream doesn't hang
    with an unclosed function_call item.
    """
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"]("call-abandoned", "terminal", {})
            return {"final_response": "done", "messages": []}

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=build_agent):
        _, events = await _run_without_http_server(adapter, {"input": "do work"})

    boundaries = [
        (event["type"], event["item"]["type"])
        for event in events
        if event["type"] in {
            "response.output_item.added",
            "response.output_item.done",
        }
    ]
    assert ("response.output_item.added", "function_call") in boundaries
    assert ("response.output_item.done", "function_call") in boundaries


@pytest.mark.asyncio
async def test_previewed_content_before_todo_is_not_synthesized_again() -> None:
    adapter = _make_adapter()
    answer = "The report is ready."

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["stream_delta_callback"](answer)
            callbacks["tool_start_callback"](
                "call-todo-housekeeping",
                "todo",
                {
                    "merge": True,
                    "todos": [{"id": "report", "status": "completed"}],
                },
            )
            callbacks["tool_complete_callback"](
                "call-todo-housekeeping",
                "todo",
                {"merge": True},
                json.dumps({
                    "todos": [
                        {
                            "id": "report",
                            "content": "Prepare report",
                            "status": "completed",
                        }
                    ]
                }),
            )
            return {
                "final_response": answer,
                "response_previewed": True,
                "messages": [],
            }

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=build_agent):
        _, events = await _run_without_http_server(
            adapter,
            {"input": "prepare the report"},
        )

    message_items = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "message"
    ]
    assert len(message_items) == 1
    assert message_items[0]["content"] == [
        {"type": "output_text", "text": answer}
    ]


@pytest.mark.asyncio
async def test_todo_completion_emits_full_sanitized_task_list_before_call_done() -> None:
    adapter = _make_adapter()
    secret = "sk-proj-" + "a" * 48

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"](
                "call-todo",
                "todo",
                {
                    "merge": True,
                    "todos": [
                        {"id": "second", "status": "completed"},
                    ],
                },
            )
            callbacks["tool_complete_callback"](
                "call-todo",
                "todo",
                {"merge": True},
                json.dumps({
                    "todos": [
                        {
                            "id": "first",
                            "content": "kept from the canonical result",
                            "status": "pending",
                            "private": "RESULT_EXTRA_MUST_NOT_LEAK",
                        },
                        {
                            "id": "second",
                            "content": f"uses {secret}",
                            "status": "completed",
                        },
                    ],
                    "summary": {"total": 2},
                }),
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    with patch.object(adapter, "_create_agent", side_effect=build_agent):
        _, events = await _run_without_http_server(adapter, {"input": "update plan"})

    task_event = next(
        event for event in events if event["type"] == "response.omnio.task_list"
    )
    assert task_event["todos"][0] == {
        "id": "first",
        "content": "kept from the canonical result",
        "status": "pending",
    }
    assert task_event["todos"][1]["id"] == "second"
    assert task_event["todos"][1]["status"] == "completed"
    assert secret not in task_event["todos"][1]["content"]
    call_done_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "response.output_item.done"
        and event["item"].get("call_id") == "call-todo"
    )
    assert events.index(task_event) < call_done_index
    assert [event["sequence_number"] for event in events] == list(
        range(1, len(events) + 1)
    )
    serialized = json.dumps(events)
    assert "RESULT_EXTRA_MUST_NOT_LEAK" not in serialized
    assert not any(
        event.get("item", {}).get("type") == "function_call_output"
        for event in events
    )


@pytest.mark.asyncio
async def test_gated_tool_progress_emits_correlated_interaction_extensions() -> None:
    adapter = _make_adapter()
    secret = "sk-proj-" + "b" * 48
    registered: Dict[str, Any] = {}

    def register_notify(
        surface_key: str,
        callback: Callable[[dict], None],
        *,
        grant_session_key: Optional[str] = None,
    ) -> object:
        registered["surface_key"] = surface_key
        registered["grant_session_key"] = grant_session_key
        registered["callback"] = callback
        return "notify-token"

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"](
                "call-gated", "mcp__crm__write", {"record": "hidden"}
            )
            registered["callback"]({
                "tool": "mcp__crm__write",
                "toolCallId": "call-gated",
                "status": "running",
                "interaction": {
                    "kind": "approval",
                    "question": f"Approve credential {secret}?",
                    "options": ["once", "deny"],
                    "approval": {"detail": f"nested {secret}"},
                },
            })
            callbacks["tool_progress_callback"](
                "tool.progress",
                "mcp__crm__write",
                None,
                None,
                interaction={"kind": "approval"},
            )
            callbacks["tool_progress_callback"](
                "tool.progress",
                "mcp__crm__write",
                None,
                None,
                toolCallId="missing-interaction",
            )
            callbacks["tool_complete_callback"](
                "call-gated",
                "mcp__crm__write",
                {},
                json.dumps({"status": "ok"}),
            )

            callbacks["tool_start_callback"](
                "call-timeout", "mcp__crm__write", {"record": "hidden"}
            )
            registered["callback"]({
                "tool": "mcp__crm__write",
                "toolCallId": "call-timeout",
                "status": "running",
                "interaction": {
                    "kind": "approval",
                    "question": "Approve another write?",
                    "options": ["once", "deny"],
                },
            })
            callbacks["tool_complete_callback"](
                "call-timeout",
                "mcp__crm__write",
                {},
                json.dumps({"status": "approval_no_response"}),
            )

            callbacks["tool_start_callback"](
                "call-input",
                "request_user_input",
                {"prompt": "Choose one"},
            )
            callbacks["tool_progress_callback"](
                "tool.progress",
                "request_user_input",
                None,
                None,
                toolCallId="call-input",
                interaction={"kind": "approval", "question": "duplicate"},
            )
            callbacks["tool_complete_callback"](
                "call-input",
                "request_user_input",
                {},
                json.dumps({"status": "answered", "response": "yes"}),
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    unregister_notify = MagicMock()
    with (
        patch.object(adapter, "_create_agent", side_effect=build_agent),
        patch(
            "tools.tool_approval.register_tool_approval_notify",
            side_effect=register_notify,
        ),
        patch(
            "tools.tool_approval.unregister_tool_approval_notify",
            unregister_notify,
        ),
        patch("tools.tool_approval.is_gated_tool", return_value=True),
        patch(
            "tools.tool_approval.consume_tool_approval_decision",
            side_effect=["once", None],
        ),
        patch(
            "tools.tool_approval.consume_tool_approval_completion_reason",
            return_value="expired",
        ),
    ):
        started, events = await _run_without_http_server(
            adapter, {"input": "write record"}
        )

    interactions = [
        event
        for event in events
        if event["type"] == "response.omnio.interaction"
    ]
    assert len(interactions) == 3
    gated = next(
        event for event in interactions if event.get("tool_call_id") == "call-gated"
    )
    assert gated["tool_call_id"] == "call-gated"
    assert gated["interaction"]["kind"] == "approval"
    assert secret not in gated["interaction"]["question"]
    assert secret not in gated["interaction"]["approval"]["detail"]
    argument_derived = next(
        event for event in interactions if "tool_call_id" not in event
    )
    assert argument_derived["interaction"] == {"prompt": "Choose one"}

    completed = [
        event
        for event in events
        if event["type"] == "response.omnio.interaction_completed"
    ]
    assert len(completed) == 2
    assert [
        {
            key: event[key]
            for key in ("tool_call_id", "timed_out", "choice")
            if key in event
        }
        for event in completed
    ] == [
        {
            "tool_call_id": "call-gated",
            "timed_out": False,
            "choice": "once",
        },
        {
            "tool_call_id": "call-timeout",
            "timed_out": True,
        },
    ]
    gated_index = events.index(gated)
    completed_index = events.index(completed[0])
    call_done_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "response.output_item.done"
        and event["item"].get("call_id") == "call-gated"
    )
    assert gated_index < completed_index < call_done_index
    assert registered["surface_key"] == json.loads(started.text)["run_id"]
    assert registered["grant_session_key"] == json.loads(started.text)["run_id"]
    unregister_notify.assert_called_once_with(
        registered["surface_key"], "notify-token"
    )


@pytest.mark.asyncio
async def test_runs_tool_approval_resolves_in_conversation_namespace_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    session_id = "conversation-tool-approval"
    tool_name = "mcp_connectors_TEST_WRITE"
    guard_results: List[Optional[str]] = []
    run_ids: List[str] = []
    tool_approval.clear_session(session_id)
    monkeypatch.setattr(tool_approval, "_approval_timeout", lambda: 2)

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            call_id = f"call-{len(guard_results) + 1}"
            callbacks["tool_start_callback"](call_id, tool_name, {})
            guard_result = tool_approval.maybe_require_tool_approval(
                tool_name,
                call_id,
                {},
            )
            guard_results.append(guard_result)
            callbacks["tool_complete_callback"](
                call_id,
                tool_name,
                {},
                json.dumps({"status": "ok"}),
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    try:
        with (
            patch.object(adapter, "_create_agent", side_effect=build_agent),
            patch.object(tool_approval, "is_gated_tool", return_value=True),
            patch.object(
                tool_approval,
                "mcp_tool_has_read_only_hint",
                return_value=True,
            ),
        ):
            started = await adapter._handle_runs(  # type: ignore[arg-type]
                _JsonRequest({
                    "input": "draft email",
                    "session_id": session_id,
                })
            )
            first_run_id = json.loads(started.text)["run_id"]
            run_ids.append(first_run_id)

            deadline = asyncio.get_running_loop().time() + 2
            while (
                not tool_approval._wait_registry.pending_count(first_run_id)
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.01)
            assert tool_approval._wait_registry.pending_count(first_run_id) == 1
            assert tool_approval._wait_registry.pending_count(session_id) == 0
            assert adapter._run_approval_sessions[first_run_id] == first_run_id

            approval = await adapter._handle_omnio_tool_approval(  # type: ignore[arg-type]
                _JsonRequest(
                    {
                        "tool": tool_name,
                        "scope": "session",
                        "toolCallId": "call-1",
                        "surfaceId": first_run_id,
                    },
                    headers={"X-Hermes-Session-Id": session_id},
                )
            )
            assert approval.status == 200
            assert json.loads(approval.text)["recorded"] is True
            await _wait_for_terminal(adapter, first_run_id)

            assert guard_results == [None]
            assert tool_approval.is_tool_approved(session_id, tool_name) is True
            assert tool_approval.is_tool_approved(first_run_id, tool_name) is False

            second_started, second_events = await _run_without_http_server(
                adapter,
                {
                    "input": "draft another email",
                    "session_id": session_id,
                },
            )
            second_run_id = json.loads(second_started.text)["run_id"]
            run_ids.append(second_run_id)
            assert guard_results == [None, None]
            assert not any(
                event.get("tool_call_id") == "call-2"
                and event["type"] == "response.omnio.interaction"
                for event in second_events
            )
    finally:
        tool_approval.clear_session(session_id)
        for run_id in run_ids:
            tool_approval.clear_session(run_id)


@pytest.mark.asyncio
async def test_concurrent_runs_legacy_resolution_is_ambiguous_without_surface_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    session_id = "conversation-concurrent-approvals"
    tool_name = "mcp_connectors_TEST_WRITE"
    tool_call_id = "call-shared-across-runs"
    guard_results: List[Optional[str]] = []
    run_ids: List[str] = []
    tool_approval.clear_session(session_id)
    monkeypatch.setattr(tool_approval, "_approval_timeout", lambda: 3)

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"](tool_call_id, tool_name, {})
            guard_result = tool_approval.maybe_require_tool_approval(
                tool_name,
                tool_call_id,
                {},
            )
            guard_results.append(guard_result)
            callbacks["tool_complete_callback"](
                tool_call_id,
                tool_name,
                {},
                guard_result or json.dumps({"status": "ok"}),
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    def run_events(run_id: str) -> List[Dict[str, Any]]:
        log = adapter._turn_event_logs.get_log(run_id)
        assert log is not None
        return [
            json.loads(stored.frame.removeprefix(b"data: ").strip())
            for stored in log.events
        ]

    try:
        with (
            patch.object(adapter, "_create_agent", side_effect=build_agent),
            patch.object(tool_approval, "is_gated_tool", return_value=True),
            patch.object(
                tool_approval,
                "mcp_tool_has_read_only_hint",
                return_value=True,
            ),
        ):
            first_started = await adapter._handle_runs(  # type: ignore[arg-type]
                _JsonRequest({"input": "first write", "session_id": session_id})
            )
            second_started = await adapter._handle_runs(  # type: ignore[arg-type]
                _JsonRequest({"input": "second write", "session_id": session_id})
            )
            first_run_id = json.loads(first_started.text)["run_id"]
            second_run_id = json.loads(second_started.text)["run_id"]
            run_ids.extend((first_run_id, second_run_id))

            deadline = asyncio.get_running_loop().time() + 3
            while asyncio.get_running_loop().time() < deadline:
                if (
                    tool_approval._wait_registry.pending_count(first_run_id) == 1
                    and tool_approval._wait_registry.pending_count(second_run_id) == 1
                ):
                    break
                await asyncio.sleep(0.01)
            assert tool_approval._wait_registry.pending_count(first_run_id) == 1
            assert tool_approval._wait_registry.pending_count(second_run_id) == 1
            assert tool_approval._wait_registry.pending_count(session_id) == 0

            first_interaction = next(
                event
                for event in run_events(first_run_id)
                if event["type"] == "response.omnio.interaction"
            )
            second_interaction = next(
                event
                for event in run_events(second_run_id)
                if event["type"] == "response.omnio.interaction"
            )
            first_call_id = first_interaction["tool_call_id"]
            second_call_id = second_interaction["tool_call_id"]
            first_surface_id = first_interaction["interaction"]["approval"][
                "surface_id"
            ]
            second_surface_id = second_interaction["interaction"]["approval"][
                "surface_id"
            ]
            assert first_call_id == second_call_id == tool_call_id
            assert first_surface_id == first_run_id
            assert second_surface_id == second_run_id

            # A legacy payload cannot identify which run emitted the card when
            # call ids collide. Preserve its contract by releasing one matching
            # surface in the conversation, without crossing that namespace.
            ambiguous_approval = await adapter._handle_omnio_tool_approval(  # type: ignore[arg-type]
                _JsonRequest(
                    {
                        "tool": tool_name,
                        "scope": "once",
                        "toolCallId": first_call_id,
                    },
                    headers={"X-Hermes-Session-Id": session_id},
                )
            )
            assert ambiguous_approval.status == 200
            assert json.loads(ambiguous_approval.text)["recorded"] is True

            deadline = asyncio.get_running_loop().time() + 3
            while asyncio.get_running_loop().time() < deadline:
                pending_run_ids = [
                    candidate_run_id
                    for candidate_run_id in (first_run_id, second_run_id)
                    if tool_approval._wait_registry.pending_count(candidate_run_id)
                ]
                terminal_run_ids = [
                    candidate_run_id
                    for candidate_run_id in (first_run_id, second_run_id)
                    if (
                        adapter._turn_event_logs.get_log(candidate_run_id)
                        and adapter._turn_event_logs.get_log(candidate_run_id).terminal
                    )
                ]
                if len(pending_run_ids) == len(terminal_run_ids) == 1:
                    break
                await asyncio.sleep(0.01)
            assert len(pending_run_ids) == len(terminal_run_ids) == 1

            remaining_run_id = pending_run_ids[0]
            remaining_surface_id = (
                first_surface_id
                if remaining_run_id == first_run_id
                else second_surface_id
            )
            remaining_call_id = (
                first_call_id
                if remaining_run_id == first_run_id
                else second_call_id
            )
            exact_approval = await adapter._handle_omnio_tool_approval(  # type: ignore[arg-type]
                _JsonRequest(
                    {
                        "tool": tool_name,
                        "scope": "deny",
                        "toolCallId": remaining_call_id,
                        "surfaceId": remaining_surface_id,
                    },
                    headers={"X-Hermes-Session-Id": session_id},
                )
            )
            assert json.loads(exact_approval.text)["recorded"] is True
            await _wait_for_terminal(adapter, remaining_run_id)

            assert sum(result is None for result in guard_results) == 1
            denied_results = [
                json.loads(result)
                for result in guard_results
                if isinstance(result, str)
            ]
            assert [result["status"] for result in denied_results] == [
                "approval_denied"
            ]
    finally:
        tool_approval.clear_session(session_id)
        for run_id in run_ids:
            tool_approval.clear_session(run_id)


@pytest.mark.asyncio
async def test_multiplex_profiles_isolate_tool_approval_session_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    session_id = "shared-profile-session-id"
    tool_name = "mcp_connectors_TEST_WRITE"
    guard_results: List[Optional[str]] = []
    run_ids: List[str] = []
    call_counter = 0

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal call_counter
        call_counter += 1
        call_id = f"call-profile-{call_counter}"

        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"](call_id, tool_name, {})
            guard_result = tool_approval.maybe_require_tool_approval(
                tool_name,
                call_id,
                {},
            )
            guard_results.append(guard_result)
            callbacks["tool_complete_callback"](
                call_id,
                tool_name,
                {},
                guard_result or json.dumps({"status": "ok"}),
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async def start_run(profile: str, prompt: str) -> str:
        token = _api_request_profile.set(profile)
        try:
            response = await adapter._handle_runs(  # type: ignore[arg-type]
                _JsonRequest({"input": prompt, "session_id": session_id})
            )
        finally:
            _api_request_profile.reset(token)
        run_id = json.loads(response.text)["run_id"]
        run_ids.append(run_id)
        return run_id

    async def resolve(
        profile: str,
        run_id: str,
        call_id: str,
        scope: str,
    ) -> web.Response:
        token = _api_request_profile.set(profile)
        try:
            return await adapter._handle_omnio_tool_approval(  # type: ignore[arg-type]
                _JsonRequest(
                    {
                        "tool": tool_name,
                        "scope": scope,
                        "toolCallId": call_id,
                        "surfaceId": run_id,
                    },
                    headers={"X-Hermes-Session-Id": session_id},
                )
            )
        finally:
            _api_request_profile.reset(token)

    coder_grant_key = adapter._scoped_tool_approval_session_key(
        session_id,
        "coder",
    )
    writer_grant_key = adapter._scoped_tool_approval_session_key(
        session_id,
        "writer",
    )
    tool_approval.clear_session(coder_grant_key)
    tool_approval.clear_session(writer_grant_key)
    monkeypatch.setattr(tool_approval, "_approval_timeout", lambda: 3)

    try:
        with (
            patch.object(adapter, "_create_agent", side_effect=build_agent),
            patch.object(tool_approval, "is_gated_tool", return_value=True),
            patch.object(
                tool_approval,
                "mcp_tool_has_read_only_hint",
                return_value=True,
            ),
        ):
            coder_run_id = await start_run("coder", "first profile write")
            deadline = asyncio.get_running_loop().time() + 3
            while (
                not tool_approval._wait_registry.pending_count(coder_run_id)
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.01)
            coder_log = adapter._turn_event_logs.get_log(coder_run_id)
            assert coder_log is not None
            coder_events = [
                json.loads(stored.frame.removeprefix(b"data: ").strip())
                for stored in coder_log.events
            ]
            coder_interaction = next(
                event
                for event in coder_events
                if event["type"] == "response.omnio.interaction"
            )
            wrong_profile_approval = await resolve(
                "writer",
                coder_run_id,
                coder_interaction["tool_call_id"],
                "session",
            )
            assert json.loads(wrong_profile_approval.text)["recorded"] is False
            assert tool_approval._wait_registry.pending_count(coder_run_id) == 1

            coder_approval = await resolve(
                "coder",
                coder_run_id,
                coder_interaction["tool_call_id"],
                "session",
            )
            assert json.loads(coder_approval.text)["recorded"] is True
            await _wait_for_terminal(adapter, coder_run_id)
            assert tool_approval.is_tool_approved(coder_grant_key, tool_name) is True
            assert tool_approval.is_tool_approved(writer_grant_key, tool_name) is False

            writer_run_id = await start_run("writer", "second profile write")
            deadline = asyncio.get_running_loop().time() + 3
            while (
                not tool_approval._wait_registry.pending_count(writer_run_id)
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.01)
            assert tool_approval._wait_registry.pending_count(writer_run_id) == 1
            writer_log = adapter._turn_event_logs.get_log(writer_run_id)
            assert writer_log is not None and not writer_log.terminal
            writer_events = [
                json.loads(stored.frame.removeprefix(b"data: ").strip())
                for stored in writer_log.events
            ]
            writer_interaction = next(
                event
                for event in writer_events
                if event["type"] == "response.omnio.interaction"
            )
            writer_approval = await resolve(
                "writer",
                writer_run_id,
                writer_interaction["tool_call_id"],
                "deny",
            )
            assert json.loads(writer_approval.text)["recorded"] is True
            await _wait_for_terminal(adapter, writer_run_id)

            assert guard_results[0] is None
            assert json.loads(guard_results[1] or "{}")["status"] == (
                "approval_denied"
            )
    finally:
        tool_approval.clear_session(coder_grant_key)
        tool_approval.clear_session(writer_grant_key)
        for run_id in run_ids:
            tool_approval.clear_session(run_id)


@pytest.mark.asyncio
async def test_tool_approval_timeout_interrupts_the_turn_before_another_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    session_id = "conversation-approval-timeout"
    tool_name = "mcp_connectors_TEST_WRITE"
    continued_after_timeout = False
    interrupted = threading.Event()
    built_agent: Optional[MagicMock] = None
    tool_approval.clear_session(session_id)
    monkeypatch.setattr(tool_approval, "_approval_timeout", lambda: 0)

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal built_agent, continued_after_timeout

        def run(**_kwargs: Any) -> Dict[str, Any]:
            nonlocal continued_after_timeout
            callbacks["tool_start_callback"]("call-timeout", tool_name, {})
            guard_result = tool_approval.maybe_require_tool_approval(
                tool_name,
                "call-timeout",
                {},
            )
            callbacks["tool_complete_callback"](
                "call-timeout",
                tool_name,
                {},
                guard_result,
            )
            continued_after_timeout = not interrupted.is_set()
            return {
                "final_response": "",
                "messages": [],
                "interrupted": interrupted.is_set(),
            }

        built_agent = _agent(run, interrupt=lambda _message=None: interrupted.set())
        return built_agent

    try:
        with (
            patch.object(adapter, "_create_agent", side_effect=build_agent),
            patch.object(tool_approval, "is_gated_tool", return_value=True),
            patch.object(
                tool_approval,
                "mcp_tool_has_read_only_hint",
                return_value=True,
            ),
        ):
            _, events = await _run_without_http_server(
                adapter,
                {"input": "write record", "session_id": session_id},
            )

        assert built_agent is not None
        built_agent.interrupt.assert_called_once_with(
            "awaiting user approval (tool approval timed out)"
        )
        assert continued_after_timeout is False
        completed = next(
            event
            for event in events
            if event["type"] == "response.omnio.interaction_completed"
            and event.get("tool_call_id") == "call-timeout"
        )
        assert completed["timed_out"] is True
        assert events[-1]["type"] == "response.incomplete"
    finally:
        tool_approval.clear_session(session_id)


@pytest.mark.asyncio
async def test_user_input_timeout_interrupts_the_run_and_stamps_timed_out() -> None:
    adapter = _make_adapter()
    session_id = "conversation-user-input-timeout"
    continued_after_timeout = False
    interrupted = threading.Event()
    built_agent: Optional[MagicMock] = None

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal built_agent, continued_after_timeout

        def run(**_kwargs: Any) -> Dict[str, Any]:
            nonlocal continued_after_timeout
            callbacks["tool_start_callback"]("call-ask", "request_user_input", {})
            callbacks["tool_progress_callback"](
                "tool.progress",
                tool="request_user_input",
                toolCallId="call-ask",
                status="running",
                interaction={"kind": "choice", "question": "Which one?", "options": []},
            )
            callbacks["tool_complete_callback"](
                "call-ask",
                "request_user_input",
                {},
                json.dumps({"status": "no_response"}),
            )
            continued_after_timeout = not interrupted.is_set()
            return {
                "final_response": "",
                "messages": [],
                "interrupted": interrupted.is_set(),
            }

        built_agent = _agent(run, interrupt=lambda _message=None: interrupted.set())
        return built_agent

    with (
        patch.object(adapter, "_create_agent", side_effect=build_agent),
        patch(
            "tools.user_input.consume_user_input_completion_reason",
            return_value="expired",
        ),
    ):
        _, events = await _run_without_http_server(
            adapter,
            {"input": "ask the user", "session_id": session_id},
        )

    assert built_agent is not None
    built_agent.interrupt.assert_called_once_with(
        "awaiting user interaction (request_user_input)"
    )
    assert continued_after_timeout is False
    completed = next(
        event
        for event in events
        if event["type"] == "response.omnio.interaction_completed"
        and event.get("tool_call_id") == "call-ask"
    )
    assert completed["timed_out"] is True
    assert events[-1]["type"] == "response.incomplete"


@pytest.mark.asyncio
async def test_answered_user_input_completes_the_card_without_interrupting() -> None:
    adapter = _make_adapter()
    session_id = "conversation-user-input-answered"
    built_agent: Optional[MagicMock] = None

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal built_agent

        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"]("call-ask", "request_user_input", {})
            callbacks["tool_progress_callback"](
                "tool.progress",
                tool="request_user_input",
                toolCallId="call-ask",
                status="running",
                interaction={"kind": "choice", "question": "Which one?", "options": []},
            )
            callbacks["tool_complete_callback"](
                "call-ask",
                "request_user_input",
                {},
                json.dumps({"status": "answered", "response": "Brand A"}),
            )
            return {"final_response": "picked", "messages": []}

        built_agent = _agent(run)
        return built_agent

    with patch.object(adapter, "_create_agent", side_effect=build_agent):
        _, events = await _run_without_http_server(
            adapter,
            {"input": "ask the user", "session_id": session_id},
        )

    assert built_agent is not None
    built_agent.interrupt.assert_not_called()
    # The answered card is recorded by the answer's own delivery path — the run
    # emits no completion of its own.
    assert not any(
        event["type"] == "response.omnio.interaction_completed" for event in events
    )
    assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_cancelled_gated_tool_closes_interaction_without_timeout() -> None:
    adapter = _make_adapter()
    session_id = "conversation-cancelled-approval"
    registered: Dict[str, Any] = {}

    def register_notify(
        surface_key: str,
        callback: Callable[[dict], None],
        *,
        grant_session_key: Optional[str] = None,
    ) -> object:
        registered["surface_key"] = surface_key
        registered["grant_session_key"] = grant_session_key
        registered["callback"] = callback
        return "notify-token"

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"](
                "call-cancelled", "mcp__crm__write", {}
            )
            registered["callback"]({
                "tool": "mcp__crm__write",
                "toolCallId": "call-cancelled",
                "status": "running",
                "interaction": {
                    "kind": "approval",
                    "question": "Approve write?",
                    "options": ["once", "deny"],
                },
            })
            callbacks["tool_complete_callback"](
                "call-cancelled",
                "mcp__crm__write",
                {},
                json.dumps({"status": "approval_no_response"}),
            )
            return {"final_response": "cancelled", "messages": []}

        return _agent(run)

    with (
        patch.object(adapter, "_create_agent", side_effect=build_agent),
        patch(
            "tools.tool_approval.register_tool_approval_notify",
            side_effect=register_notify,
        ),
        patch("tools.tool_approval.unregister_tool_approval_notify"),
        patch("tools.tool_approval.is_gated_tool", return_value=True),
        patch(
            "tools.tool_approval.consume_tool_approval_decision",
            return_value=None,
        ),
        patch(
            "tools.tool_approval.consume_tool_approval_completion_reason",
            return_value="cancelled",
        ),
    ):
        _, events = await _run_without_http_server(
            adapter,
            {"input": "write record", "session_id": session_id},
        )

    interaction = next(
        event
        for event in events
        if event["type"] == "response.omnio.interaction"
        and event.get("tool_call_id") == "call-cancelled"
    )
    completed = next(
        event
        for event in events
        if event["type"] == "response.omnio.interaction_completed"
        and event.get("tool_call_id") == "call-cancelled"
    )
    assert "choice" not in completed
    assert "timed_out" not in completed
    call_done = next(
        event
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"].get("call_id") == "call-cancelled"
    )
    assert interaction["sequence_number"] < completed["sequence_number"]
    assert completed["sequence_number"] < call_done["sequence_number"]


@pytest.mark.asyncio
async def test_cancel_before_approval_token_publish_unregisters_late_surface() -> None:
    adapter = _make_adapter()
    session_id = "conversation-cancel-before-token-publish"
    register_started = threading.Event()
    allow_register_return = threading.Event()
    late_unregister_finished = threading.Event()
    agent_ran = threading.Event()
    registered_surface: Dict[str, str] = {}
    real_register = tool_approval.register_tool_approval_notify
    real_unregister = tool_approval.unregister_tool_approval_notify

    def delayed_register(
        surface_key: str,
        callback: Callable[[dict], None],
        *,
        grant_session_key: Optional[str] = None,
    ) -> object:
        token = real_register(
            surface_key,
            callback,
            grant_session_key=grant_session_key,
        )
        registered_surface["key"] = surface_key
        register_started.set()
        allow_register_return.wait(timeout=5)
        return token

    def observed_unregister(surface_key: str, token: object) -> None:
        real_unregister(surface_key, token)
        late_unregister_finished.set()

    def run_agent(**_kwargs: Any) -> Dict[str, Any]:
        agent_ran.set()
        return {"final_response": "must not run", "messages": []}

    try:
        with (
            patch.object(adapter, "_create_agent", return_value=_agent(run_agent)),
            patch(
                "tools.tool_approval.register_tool_approval_notify",
                side_effect=delayed_register,
            ),
            patch(
                "tools.tool_approval.unregister_tool_approval_notify",
                side_effect=observed_unregister,
            ),
        ):
            response = await adapter._handle_runs(  # type: ignore[arg-type]
                _JsonRequest({"input": "write", "session_id": session_id})
            )
            run_id = json.loads(response.text)["run_id"]
            await _wait_for_thread_event(register_started)
            assert registered_surface["key"] == run_id
            assert tool_approval._wait_registry.has_surface(run_id)

            task = adapter._active_run_tasks[run_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            allow_register_return.set()
            await _wait_for_thread_event(late_unregister_finished)

        assert not tool_approval._wait_registry.has_surface(run_id)
        assert tool_approval._wait_registry.pending_count(run_id) == 0
        assert not agent_ran.is_set()
    finally:
        allow_register_return.set()
        tool_approval.clear_session(session_id)
        surface_key = registered_surface.get("key")
        if surface_key:
            tool_approval.clear_session(surface_key)


@pytest.mark.asyncio
async def test_runs_projection_allows_only_allowlisted_tool_args_and_no_results() -> (
    None
):
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["reasoning_callback"]("thinking")
            callbacks["tool_start_callback"](
                "call-allowed",
                "request_user_input",
                {"prompt": "visible allowlisted args"},
            )
            callbacks["tool_complete_callback"](
                "call-allowed",
                "request_user_input",
                {"prompt": "visible allowlisted args"},
                {"answer": "RESULT_MUST_NOT_LEAK"},
            )
            callbacks["tool_start_callback"](
                "call-client-event",
                "emit_client_event",
                {"name": "toast", "payload": {"message": "visible"}},
            )
            callbacks["tool_complete_callback"](
                "call-client-event",
                "emit_client_event",
                {"name": "toast", "payload": {"message": "visible"}},
                "CLIENT_EVENT_RESULT_MUST_NOT_LEAK",
            )
            callbacks["tool_start_callback"](
                "call-gen-ui",
                "render_component",
                {"component": "chart", "props": {"title": "visible"}},
            )
            callbacks["tool_complete_callback"](
                "call-gen-ui",
                "render_component",
                {"component": "chart", "props": {"title": "visible"}},
                "GEN_UI_RESULT_MUST_NOT_LEAK",
            )
            callbacks["tool_start_callback"](
                "call-hidden",
                "terminal",
                {"command": "NON_ALLOWLISTED_ARGS_MUST_NOT_LEAK"},
            )
            callbacks["tool_complete_callback"](
                "call-hidden",
                "terminal",
                {"command": "NON_ALLOWLISTED_ARGS_MUST_NOT_LEAK"},
                "SECOND_RESULT_MUST_NOT_LEAK",
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use tools"})
            run_id = (await started.json())["run_id"]
            response = await client.get(f"/v1/runs/{run_id}/events")
            body = await response.text()

    events = _sse_events(body)
    args_deltas = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    ]
    args_done = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.done"
    ]
    starts = {
        event["item"]["call_id"]: event
        for event in events
        if event["type"] == "response.output_item.added"
        and event["item"]["type"] == "function_call"
    }
    ends = {
        event["item"]["call_id"]: event
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    }

    assert starts["call-allowed"]["item"]["name"] == "request_user_input"
    assert starts["call-client-event"]["item"]["name"] == "emit_client_event"
    assert starts["call-gen-ui"]["item"]["name"] == "render_component"
    assert starts["call-hidden"]["item"]["name"] == "terminal"
    assert set(ends) == {
        "call-allowed",
        "call-client-event",
        "call-gen-ui",
        "call-hidden",
    }
    assert len(args_deltas) == 3
    assert len(args_done) == 3
    arguments_by_item = {
        event["item_id"]: json.loads(event["delta"]) for event in args_deltas
    }
    assert arguments_by_item == {
        starts["call-allowed"]["item"]["id"]: {"prompt": "visible allowlisted args"},
        starts["call-client-event"]["item"]["id"]: {
            "name": "toast",
            "payload": {"message": "visible"},
        },
        starts["call-gen-ui"]["item"]["id"]: {
            "component": "chart",
            "props": {"title": "visible"},
        },
    }
    assert {
        event["item_id"]: json.loads(event["arguments"]) for event in args_done
    } == arguments_by_item
    assert (
        json.loads(ends["call-allowed"]["item"]["arguments"])
        == arguments_by_item[starts["call-allowed"]["item"]["id"]]
    )
    assert "arguments" not in starts["call-allowed"]["item"]
    assert "arguments" not in starts["call-hidden"]["item"]
    assert "arguments" not in ends["call-hidden"]["item"]
    extension_events = {
        event["type"]: event
        for event in events
        if event["type"]
        in {
            "response.omnio.interaction",
            "response.omnio.client_event",
            "response.omnio.gen_ui",
        }
    }
    assert extension_events["response.omnio.interaction"]["interaction"] == {
        "prompt": "visible allowlisted args"
    }
    assert extension_events["response.omnio.client_event"]["client_event"] == {
        "name": "toast",
        "payload": {"message": "visible"},
    }
    assert extension_events["response.omnio.gen_ui"]["gen_ui"] == {
        "component": "chart",
        "props": {"title": "visible"},
    }
    assert "NON_ALLOWLISTED_ARGS_MUST_NOT_LEAK" not in body
    assert "RESULT_MUST_NOT_LEAK" not in body
    assert "SECOND_RESULT_MUST_NOT_LEAK" not in body
    assert not any(
        event.get("item", {}).get("type") == "function_call_output" for event in events
    )
    assert [
        event["type"]
        for event in events
        if event["type"].startswith("response.reasoning_text")
    ] == [
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
    ]
    reasoning_boundaries = [
        event
        for event in events
        if event["type"]
        in {
            "response.output_item.added",
            "response.output_item.done",
        }
        and event["item"]["type"] == "reasoning"
    ]
    assert [event["type"] for event in reasoning_boundaries] == [
        "response.output_item.added",
        "response.output_item.done",
    ]


@pytest.mark.asyncio
async def test_runs_emit_function_call_boundary_while_arguments_are_generating() -> None:
    """A streamed tool name opens the native item before execution starts."""
    adapter = _make_adapter()
    generating = threading.Event()
    continue_execution = threading.Event()
    executed = threading.Event()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_gen_event_callback"]("terminal", "call-early")
            generating.set()
            assert continue_execution.wait(2.0)
            executed.set()
            callbacks["tool_start_callback"](
                "call-early", "terminal", {"command": "hidden"}
            )
            callbacks["tool_complete_callback"](
                "call-early", "terminal", {"command": "hidden"}, "hidden result"
            )
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use a tool"})
            run_id = (await started.json())["run_id"]
            await _wait_for_thread_event(generating)
            response = await client.get(f"/v1/runs/{run_id}/events")

            seen: List[Dict[str, Any]] = []
            while True:
                event = await _read_one_sse_event(response)
                seen.append(event)
                if (
                    event["type"] == "response.output_item.added"
                    and event["item"]["type"] == "function_call"
                ):
                    break

            boundary = seen[-1]
            assert boundary["item"] == {
                "id": "fc_call-early",
                "type": "function_call",
                "status": "in_progress",
                "name": "terminal",
                "call_id": "call-early",
                "arguments": "",
                "started_at": boundary["item"]["started_at"],
            }
            assert not executed.is_set()
            assert not any(
                event["type"].startswith("response.function_call_arguments")
                for event in seen
            )

            continue_execution.set()
            remaining = _sse_events(await response.text())

    events = seen + remaining
    call_events = [
        event
        for event in events
        if event.get("item_id") == "fc_call-early"
        or (
            event.get("item", {}).get("id") == "fc_call-early"
            if isinstance(event.get("item"), dict)
            else False
        )
    ]
    assert [event["type"] for event in call_events] == [
        "response.output_item.added",
        "response.output_item.done",
    ]
    assert call_events[-1]["item"]["arguments"] == ""
    assert "hidden" not in json.dumps(events)


@pytest.mark.asyncio
async def test_runs_streamed_duplicate_or_blank_call_ids_fall_back_to_execution_ids() -> None:
    """Ambiguous generation hints never duplicate or orphan Responses items."""
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_gen_event_callback"]("first_tool", "call-shared")
            callbacks["tool_gen_event_callback"]("second_tool", "call-shared")
            callbacks["tool_gen_event_callback"]("blank_id_tool", "")
            for call_id, name in (
                ("call-shared", "first_tool"),
                ("call-shared_d2", "second_tool"),
                ("call-from-execution", "blank_id_tool"),
            ):
                callbacks["tool_start_callback"](call_id, name, {})
                callbacks["tool_complete_callback"](call_id, name, {}, "ok")
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use tools"})
            run_id = (await started.json())["run_id"]
            response = await client.get(f"/v1/runs/{run_id}/events")
            events = _sse_events(await response.text())

    added = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.added"
        and event["item"]["type"] == "function_call"
    ]
    completed = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    ]
    assert [item["call_id"] for item in added] == [
        "call-shared",
        "call-shared_d2",
        "call-from-execution",
    ]
    assert [item["call_id"] for item in completed] == [
        "call-shared",
        "call-shared_d2",
        "call-from-execution",
    ]
    assert [event["type"] for event in events][-1] == "response.completed"


@pytest.mark.asyncio
async def test_runs_failure_closes_an_early_function_call_before_the_terminal_event() -> None:
    """An interrupted generation never leaves an in-progress call item open."""
    adapter = _make_adapter()
    generating = threading.Event()
    fail_now = threading.Event()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_gen_event_callback"]("terminal", "call-abandoned")
            generating.set()
            assert fail_now.wait(2.0)
            raise RuntimeError("stream ended during tool arguments")

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use a tool"})
            run_id = (await started.json())["run_id"]
            await _wait_for_thread_event(generating)
            response = await client.get(f"/v1/runs/{run_id}/events")
            while True:
                event = await _read_one_sse_event(response)
                if (
                    event["type"] == "response.output_item.added"
                    and event["item"].get("id") == "fc_call-abandoned"
                ):
                    break
            fail_now.set()
            events = _sse_events(await response.text())

    assert [event["type"] for event in events] == [
        "response.output_item.done",
        "response.failed",
    ]
    assert events[0]["item"] == {
        "id": "fc_call-abandoned",
        "type": "function_call",
        "status": "incomplete",
        "name": "terminal",
        "call_id": "call-abandoned",
        "arguments": "",
        "started_at": events[0]["item"]["started_at"],
        "completed_at": events[0]["item"]["completed_at"],
    }


@pytest.mark.asyncio
async def test_runs_retry_marks_first_early_call_incomplete_before_second_executes() -> None:
    """A dropped stream attempt cannot complete a fictional function call."""
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            # First stream attempt identifies a tool, then drops before its
            # arguments reached execution. The provider retry advertises and
            # executes a distinct call ID.
            callbacks["tool_gen_event_callback"]("terminal", "call-first")
            callbacks["tool_gen_event_aborted_callback"]("call-first")
            callbacks["tool_gen_event_callback"]("read_file", "call-second")
            callbacks["tool_start_callback"]("call-second", "read_file", {})
            callbacks["tool_complete_callback"]("call-second", "read_file", {}, "ok")
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use a tool"})
            run_id = (await started.json())["run_id"]
            response = await client.get(f"/v1/runs/{run_id}/events")
            events = _sse_events(await response.text())

    function_items = [
        event["item"]
        for event in events
        if event["type"] in {"response.output_item.added", "response.output_item.done"}
        and event.get("item", {}).get("type") == "function_call"
    ]
    assert [
        (item["call_id"], item["status"])
        for item in function_items
    ] == [
        ("call-first", "in_progress"),
        ("call-first", "incomplete"),
        ("call-second", "in_progress"),
        ("call-second", "completed"),
    ]
    assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_runs_retry_reuses_early_call_id_for_its_real_execution() -> None:
    """A deterministic retry ID keeps one function-call lifecycle."""
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_gen_event_callback"]("read_file", "call-reused")
            callbacks["tool_gen_event_aborted_callback"]("call-reused")
            callbacks["tool_gen_event_callback"]("read_file", "call-reused")
            callbacks["tool_start_callback"]("call-reused", "read_file", {})
            callbacks["tool_complete_callback"]("call-reused", "read_file", {}, "ok")
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use a tool"})
            run_id = (await started.json())["run_id"]
            response = await client.get(f"/v1/runs/{run_id}/events")
            events = _sse_events(await response.text())

    function_items = [
        event["item"]
        for event in events
        if event["type"] in {"response.output_item.added", "response.output_item.done"}
        and event.get("item", {}).get("type") == "function_call"
    ]
    assert [
        (item["call_id"], item["status"])
        for item in function_items
    ] == [
        ("call-reused", "in_progress"),
        ("call-reused", "completed"),
    ]


@pytest.mark.asyncio
async def test_runs_retry_reopens_call_id_with_a_unique_item_lifecycle() -> None:
    """A→B→A retries terminate each item even when A's ID is reused."""
    adapter = _make_adapter()

    def build_agent(**callbacks: Any) -> MagicMock:
        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_gen_event_callback"]("first", "call-a")
            callbacks["tool_gen_event_aborted_callback"]("call-a")
            callbacks["tool_gen_event_callback"]("second", "call-b")
            callbacks["tool_gen_event_aborted_callback"]("call-b")
            callbacks["tool_gen_event_callback"]("third", "call-a")
            callbacks["tool_start_callback"]("call-a", "third", {})
            callbacks["tool_complete_callback"]("call-a", "third", {}, "ok")
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "use tools"})
            run_id = (await started.json())["run_id"]
            response = await client.get(f"/v1/runs/{run_id}/events")
            events = _sse_events(await response.text())

    items = [
        event["item"]
        for event in events
        if event["type"] in {"response.output_item.added", "response.output_item.done"}
        and event.get("item", {}).get("type") == "function_call"
    ]
    added = items[::2]
    done = items[1::2]
    assert [item["id"] for item in added] == ["fc_call-a", "fc_call-b", "fc_call-a_2"]
    assert [item["status"] for item in done] == ["incomplete", "incomplete", "completed"]
    assert [item["id"] for item in done] == [item["id"] for item in added]
    assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_runs_session_continuity_uses_authoritative_history_on_second_turn(
    tmp_path,
) -> None:
    adapter = _make_adapter(api_key="omnio-test-key")
    session_db = SessionDB(tmp_path / "state.db")
    session_db.create_session("conversation-1", "api_server")
    adapter._session_db = session_db
    server_history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    loaded_histories: List[List[Dict[str, str]]] = []

    def build_agent(**_callbacks: Any) -> MagicMock:
        def run(**kwargs: Any) -> Dict[str, Any]:
            loaded_histories.append(kwargs["conversation_history"])
            if kwargs["user_message"] == "first question":
                session_db.append_message("conversation-1", "user", "first question")
                session_db.append_message("conversation-1", "assistant", "first answer")
            return {"final_response": "done", "messages": []}

        return _agent(run)

    try:
        async with TestClient(TestServer(_make_app(adapter))) as client:
            with patch.object(adapter, "_create_agent", side_effect=build_agent):
                first = await client.post(
                    "/v1/runs",
                    json={"input": "first question", "session_id": "conversation-1"},
                    headers=_AUTH_HEADERS,
                )
                first_run_id = (await first.json())["run_id"]
                await _wait_for_terminal(adapter, first_run_id)

                second = await client.post(
                    "/v1/runs",
                    json={"input": "follow up", "session_id": "conversation-1"},
                    headers=_AUTH_HEADERS,
                )
                second_run_id = (await second.json())["run_id"]
                await _wait_for_terminal(adapter, second_run_id)

        assert first.status == 202
        assert second.status == 202
        normalized_histories = [
            [
                {"role": message["role"], "content": message["content"]}
                for message in history
            ]
            for history in loaded_histories
        ]
        assert normalized_histories == [[], server_history]
    finally:
        session_db.close()


@pytest.mark.asyncio
async def test_session_id_ignores_stale_body_history_but_no_session_honours_it() -> (
    None
):
    adapter = _make_adapter(api_key="omnio-test-key")
    stale_history = [{"role": "assistant", "content": "stale client replay"}]
    server_history = [{"role": "assistant", "content": "server truth"}]
    loaded_histories: List[List[Dict[str, str]]] = []

    def build_agent(**_callbacks: Any) -> MagicMock:
        def run(**kwargs: Any) -> Dict[str, Any]:
            loaded_histories.append(kwargs["conversation_history"])
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with (
            patch.object(adapter, "_create_agent", side_effect=build_agent),
            patch.object(
                adapter,
                "_conversation_history_for_session",
                new=AsyncMock(return_value=server_history),
            ),
        ):
            with_session = await client.post(
                "/v1/runs",
                json={
                    "input": "follow up",
                    "session_id": "conversation-1",
                    "conversation_history": stale_history,
                },
                headers=_AUTH_HEADERS,
            )
            with_session_id = (await with_session.json())["run_id"]
            await _wait_for_terminal(adapter, with_session_id)

            without_session = await client.post(
                "/v1/runs",
                json={
                    "input": "standalone",
                    "conversation_history": stale_history,
                },
                headers=_AUTH_HEADERS,
            )
            without_session_id = (await without_session.json())["run_id"]
            await _wait_for_terminal(adapter, without_session_id)

    assert loaded_histories == [server_history, stale_history]


@pytest.mark.asyncio
async def test_session_id_without_gateway_key_uses_legacy_empty_history() -> None:
    adapter = _make_adapter()
    loaded_histories: List[List[Dict[str, str]]] = []

    def build_agent(**_callbacks: Any) -> MagicMock:
        def run(**kwargs: Any) -> Dict[str, Any]:
            loaded_histories.append(kwargs["conversation_history"])
            return {"final_response": "done", "messages": []}

        return _agent(run)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with (
            patch.object(adapter, "_create_agent", side_effect=build_agent),
            patch.object(
                adapter,
                "_conversation_history_for_session",
                new=AsyncMock(
                    return_value=[{"role": "assistant", "content": "server"}]
                ),
            ) as hydrate_history,
        ):
            response = await client.post(
                "/v1/runs",
                json={"input": "follow up", "session_id": "conversation-1"},
            )
            payload = await response.json()
            await _wait_for_terminal(adapter, payload["run_id"])

    assert response.status == 202
    assert loaded_histories == [[]]
    hydrate_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_wind_down_is_logged_and_visible_to_attached_subscriber() -> None:
    adapter = _make_adapter()
    running = threading.Event()
    interrupted = threading.Event()
    built_agent: Optional[MagicMock] = None

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal built_agent

        def interrupt(_message: Optional[str] = None) -> None:
            interrupted.set()

        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["tool_start_callback"](
                "call-stopped", "terminal", {"command": "long task"}
            )
            running.set()
            interrupted.wait(timeout=3.0)
            callbacks["tool_complete_callback"](
                "call-stopped",
                "terminal",
                {"command": "long task"},
                "CANCELLED_TOOL_RESULT_MUST_NOT_LEAK",
            )
            return {
                "final_response": "",
                "interrupted": True,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Operation interrupted by user.",
                    }
                ],
            }

        built_agent = _agent(run, interrupt=interrupt)
        return built_agent

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "long task"})
            run_id = (await started.json())["run_id"]
            await _wait_for_thread_event(running)

            attached = await client.get(f"/v1/runs/{run_id}/events")
            stopped = await client.post(f"/v1/runs/{run_id}/stop")
            body = await attached.text()

    events = _sse_events(body)
    assert stopped.status == 200
    assert [event["type"] for event in events][-3:] == [
        "response.output_item.done",
        "response.omnio.interrupted_history",
        "response.incomplete",
    ]
    assert events[-2]["message"] == "Operation interrupted by user."
    assert events[-1]["response"]["status"] == "incomplete"
    assert events[-1]["response"]["incomplete_details"] == {"reason": "cancelled"}
    assert "CANCELLED_TOOL_RESULT_MUST_NOT_LEAK" not in body
    assert adapter._run_statuses[run_id]["status"] == "cancelled"
    assert built_agent is not None
    built_agent.interrupt.assert_called_once_with("Stop requested via API")


def test_turn_finalize_timeout_defaults_and_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(api_server_module._OMNIO_TURN_FINALIZE_TIMEOUT_ENV, raising=False)
    assert (
        api_server_module._turn_finalize_timeout_seconds()
        == api_server_module._OMNIO_TURN_FINALIZE_TIMEOUT_DEFAULT_SECONDS
    )

    monkeypatch.setenv(api_server_module._OMNIO_TURN_FINALIZE_TIMEOUT_ENV, "12.5")
    assert api_server_module._turn_finalize_timeout_seconds() == 12.5

    for invalid in ("", "not-a-number", "0", "-3"):
        monkeypatch.setenv(api_server_module._OMNIO_TURN_FINALIZE_TIMEOUT_ENV, invalid)
        assert (
            api_server_module._turn_finalize_timeout_seconds()
            == api_server_module._OMNIO_TURN_FINALIZE_TIMEOUT_DEFAULT_SECONDS
        )


@pytest.mark.asyncio
async def test_runs_register_the_user_input_surface_so_questions_park_and_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a registered interactive surface the blocking wait returns
    # "no_surface" instantly and every question degrades to no_response — the
    # run worker must register the run's session key for the run's lifetime.
    # The answer arrives keyed by conversation session and must translate to
    # the run-scoped wait.
    monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "5")
    adapter = _make_adapter()
    captured: Dict[str, Any] = {}

    def run(**_kwargs: Any) -> Dict[str, Any]:
        import tools.user_input as user_input
        from tools.approval import get_current_session_key

        session_key = get_current_session_key()
        captured["surface"] = user_input._wait_registry.has_surface(session_key)
        captured["answer"] = user_input.await_user_input(session_key, "call-q")
        return {"final_response": "done", "messages": []}

    app = _make_app(adapter)
    app.router.add_post("/v1/omnio/user-input", adapter._handle_omnio_user_input)
    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_create_agent", return_value=_agent(run)):
            started = await client.post(
                "/v1/runs", json={"input": "ask me", "session_id": "conv-session-1"}
            )
            run_id = (await started.json())["run_id"]

            import tools.user_input as user_input

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 3.0
            while loop.time() < deadline:
                if user_input._wait_registry.pending_count(run_id):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("the question never parked on the run surface")

            answered = await client.post(
                "/v1/omnio/user-input",
                json={"response": "Option B", "toolCallId": "call-q"},
                headers={"X-Hermes-Session-Id": "conv-session-1"},
            )
            assert (await answered.json())["resolved"] is True
            await _wait_for_terminal(adapter, run_id)

    assert captured["surface"] is True
    assert captured["answer"] == "Option B"


@pytest.mark.asyncio
async def test_runs_thread_delegation_sync_only_into_session_context() -> None:
    """The Omnio proxy's ``delegation_sync_only`` on ``POST /v1/runs`` must
    reach the running agent via ``HERMES_DELEGATION_SYNC_ONLY`` — bound by
    ``_bind_api_server_session`` exactly like ``turn_id`` -> ``HERMES_ORIGIN_TURN_ID``
    — so ``tools.async_delegation._current_delegation_sync_only()`` (read by
    ``delegate_task``) sees it while the run is live."""
    from gateway.session_context import get_session_env

    adapter = _make_adapter()
    captured: Dict[str, Any] = {}

    def run(**_kwargs: Any) -> Dict[str, Any]:
        captured["delegation_sync_only"] = get_session_env(
            "HERMES_DELEGATION_SYNC_ONLY"
        )
        return {"final_response": "done", "messages": []}

    with patch.object(adapter, "_create_agent", return_value=_agent(run)):
        started, _events = await _run_without_http_server(
            adapter,
            {
                "input": "run headless",
                "session_id": "session-sync-only",
                "delegation_sync_only": True,
            },
        )

    assert started.status == 202
    assert captured["delegation_sync_only"] == "1"


@pytest.mark.asyncio
async def test_runs_default_delegation_sync_only_false_when_omitted() -> None:
    """Callers that never pass ``delegation_sync_only`` (every non-Omnio and
    most Omnio deployments) must not force delegate_task's synchronous
    fallback."""
    from gateway.session_context import get_session_env

    adapter = _make_adapter()
    captured: Dict[str, Any] = {}

    def run(**_kwargs: Any) -> Dict[str, Any]:
        captured["delegation_sync_only"] = get_session_env(
            "HERMES_DELEGATION_SYNC_ONLY"
        )
        return {"final_response": "done", "messages": []}

    with patch.object(adapter, "_create_agent", return_value=_agent(run)):
        started, _events = await _run_without_http_server(
            adapter,
            {"input": "run normally", "session_id": "session-normal"},
        )

    assert started.status == 202
    assert captured["delegation_sync_only"] == ""
