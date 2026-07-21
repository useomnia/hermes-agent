"""Tests for semantic client-event forwarding in api_server.py.

Verifies that event-emitter tools have their call arguments forwarded on the
``running`` progress event under a ``clientEvent`` key, while non-emitter tools,
non-dict args, and the ``completed`` event remain unaffected.
"""

import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _extract_tool_progress_events(body: str) -> list[dict]:
    """Parse all ``hermes.tool.progress`` SSE events from the response body."""
    events = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "event: hermes.tool.progress":
            continue
        for follow in lines[i + 1 : i + 4]:
            if follow.startswith("data: "):
                try:
                    payload = json.loads(follow[len("data: ") :])
                except json.JSONDecodeError:
                    break
                events.append(payload)
                break
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClientEventForwarding:
    """Verify semantic client events on tool-progress events."""

    @pytest.mark.asyncio
    async def test_client_event_running_event_snapshots_args(self):
        """A client event retains the args present when the tool starts."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        payload = {"id": "abc"}
        tool_args = {"version": 1, "name": "some_event", "payload": payload}
        expected_event = {"version": 1, "name": "some_event", "payload": {"id": "abc"}}

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_1", "emit_client_event", tool_args)
                    tool_args["name"] = "mutated_event"
                    payload["id"] = "mutated"
                if tc_cb:
                    tc_cb("call_1", "emit_client_event", tool_args, '{"status":"emitted"}')
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok")
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "go"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = _extract_tool_progress_events(body)
        running = [e for e in events if e.get("status") == "running" and e.get("tool") == "emit_client_event"]
        assert len(running) == 1, f"expected 1 running event, got {events}"
        assert running[0]["clientEvent"] == expected_event
        assert "client" not in running[0]

    @pytest.mark.asyncio
    async def test_non_emitter_tool_has_no_client_event(self):
        """A tool that is not an event emitter gets no ``clientEvent`` key."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_1", "terminal", {"command": "ls"})
                if tc_cb:
                    tc_cb("call_1", "terminal", {"command": "ls"}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("done")
                return (
                    {"final_response": "done", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "go"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = _extract_tool_progress_events(body)
        for event in events:
            assert "clientEvent" not in event, f"non-emitter tool got clientEvent: {event}"

    @pytest.mark.asyncio
    async def test_client_event_emitter_with_non_dict_args_has_no_client_event(self):
        """An event emitter with non-dict args gets no ``clientEvent`` key."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # Pass a string instead of dict for args
                if ts_cb:
                    ts_cb("call_1", "emit_client_event", "not-a-dict")
                if tc_cb:
                    tc_cb("call_1", "emit_client_event", "not-a-dict", '{"status":"emitted"}')
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok")
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "go"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = _extract_tool_progress_events(body)
        running = [e for e in events if e.get("status") == "running"]
        for event in running:
            assert "clientEvent" not in event, f"non-dict args got clientEvent: {event}"

    @pytest.mark.asyncio
    async def test_request_user_input_interaction_unchanged(self):
        """Regression: request_user_input still gets ``interaction`` key,
        not ``clientEvent``, and the two do not interfere."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        rui_args = {"kind": "text", "prompt": "What is your name?"}

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_rui_1", "request_user_input", rui_args)
                if tc_cb:
                    tc_cb(
                        "call_rui_1",
                        "request_user_input",
                        rui_args,
                        json.dumps({"status": "answered", "response": "Alice"}),
                    )
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok")
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "go"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = _extract_tool_progress_events(body)
        running = [e for e in events if e.get("status") == "running" and e.get("tool") == "request_user_input"]
        assert len(running) == 1
        assert running[0]["interaction"] == rui_args
        assert "clientEvent" not in running[0]

    @pytest.mark.asyncio
    async def test_completed_event_has_no_client_event(self):
        """An event emitter's ``completed`` event carries no ``clientEvent`` key."""
        adapter = _make_adapter()
        app = _create_app(adapter)

        tool_args = {"version": 1, "name": "some_event", "payload": {}}

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_1", "emit_client_event", tool_args)
                if tc_cb:
                    tc_cb("call_1", "emit_client_event", tool_args, '{"status":"emitted"}')
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok")
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "go"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = _extract_tool_progress_events(body)
        completed = [e for e in events if e.get("status") == "completed" and e.get("tool") == "emit_client_event"]
        assert len(completed) == 1, f"expected 1 completed event, got {events}"
        assert "clientEvent" not in completed[0], f"completed event has clientEvent: {completed[0]}"
