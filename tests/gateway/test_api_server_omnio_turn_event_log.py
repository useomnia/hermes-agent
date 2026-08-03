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

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.turn_event_log import (
    CursorExpiredError,
    OMNIO_EXTENSION_EVENT_TYPES,
    TurnEventEmitter,
    TurnEventLogStore,
    UnknownRunError,
)
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
    cap: int = 8 * 1024 * 1024,
    retention: float = 300.0,
) -> TurnEventLogStore:
    store = TurnEventLogStore(
        clock=clock,
        run_log_cap_bytes=cap,
        terminal_retention_seconds=retention,
        on_cap_exceeded=adapter._handle_run_log_cap_exceeded,
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
        "response.omnio.warmup",
        "response.omnio.subagent_start",
        "response.omnio.subagent_complete",
        "response.omnio.interrupted_history",
        "response.omnio.approval_request",
        "response.omnio.approval_responded",
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
    assert [event["sequence_number"] for event in first_events] == list(range(1, 10))
    assert [event["type"] for event in first_events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
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
    }
    assert first_events[7]["item"]["content"] == [
        {"type": "output_text", "text": "hello"}
    ]
    assert all("seq" not in event for event in first_events)
    assert all(
        "runId" not in event and "threadId" not in event for event in first_events
    )


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
        response = await client.get("/v1/runs/run_cursor/events?after=4")

        emitter.output_text_delta("message-1", "live")
        emitter.output_text_done("message-1")
        emitter.response_completed()
        body = await response.text()

    events = _sse_events(body)
    assert [event["sequence_number"] for event in events] == list(range(5, 11))
    assert [event.get("delta") for event in events[:2]] == ["retained", "live"]


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
    assert [event["sequence_number"] for event in events] == list(range(2, 10))
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
async def test_log_cap_interrupts_run_and_emits_machine_readable_run_error() -> None:
    adapter = _make_adapter()
    assert adapter._turn_event_logs.run_log_cap_bytes == 8 * 1024 * 1024
    _install_log_store(adapter, clock=_Clock(), cap=4_096)
    built_agent: Optional[MagicMock] = None

    def build_agent(**callbacks: Any) -> MagicMock:
        nonlocal built_agent

        def run(**_kwargs: Any) -> Dict[str, Any]:
            callbacks["stream_delta_callback"]("x" * 8_192)
            return {"final_response": "", "messages": []}

        built_agent = _agent(run)
        return built_agent

    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=build_agent):
            started = await client.post("/v1/runs", json={"input": "overflow"})
            run_id = (await started.json())["run_id"]
            events_response = await client.get(f"/v1/runs/{run_id}/events")
            body = await events_response.text()

    events = _sse_events(body)
    log = adapter._turn_event_logs.get_log(run_id)
    assert started.status == 202
    assert log is not None
    assert log.terminal
    assert log.failure_reason == "log_cap_exceeded"
    assert log.wire_bytes <= 4_096
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "log_cap_exceeded"
    assert adapter._run_statuses[run_id]["status"] == "failed"
    assert built_agent is not None
    built_agent.interrupt.assert_called_once_with("Turn event log cap exceeded")


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
