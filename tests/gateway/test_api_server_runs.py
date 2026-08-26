"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(
    adapter: APIServerAdapter,
    *,
    include_profile_routes: bool = False,
) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    if include_profile_routes:
        for method, path, handler in adapter._http_route_table():
            app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


async def _wait_for_thread_event(
    event: threading.Event,
    timeout: float = 3.0,
) -> bool:
    """Wait for executor-thread progress without blocking the asyncio loop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return event.is_set()


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_turn_id_setup_failure_closes_reserved_run(self, tmp_path):
        """A reservation must not remain queued when Turn-log setup fails."""
        adapter = _make_adapter()
        store = RunIdempotencyStore(tmp_path / "state.db")
        adapter._run_idempotency = store
        app = _create_runs_app(adapter)
        body = {"input": "hello", "turn_id": "turn-setup-failure"}

        with patch.object(
            adapter._turn_event_logs,
            "create_run",
            side_effect=RuntimeError("simulated log setup failure"),
        ):
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post("/v1/runs", json=body)
                first_data = await first.json()

        record = store.get("turn-setup-failure")
        assert first.status == 503
        assert first_data["error"]["code"] == "run_initialization_failed"
        assert record is not None
        assert record.status == "failed"
        assert record.failure_reason == "run_initialization_failed"

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                retry = await cli.post("/v1/runs", json=body)
                retry_data = await retry.json()

        assert retry.status == 202
        assert retry_data["run_id"] == first_data.get("run_id", record.run_id)
        assert retry_data["status"] == "failed"
        assert retry_data["idempotent"] is True
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_id_retry_returns_original_run_without_second_agent(self, tmp_path):
        adapter = _make_adapter()
        adapter._run_idempotency = RunIdempotencyStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)
        body = {"input": "hello", "turn_id": "turn-retry"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                first = await cli.post("/v1/runs", json=body)
                first_data = await first.json()
                await asyncio.sleep(0.05)
                second = await cli.post("/v1/runs", json=body)
                second_data = await second.json()

        assert first.status == 202
        assert second.status == 202
        assert second_data["run_id"] == first_data["run_id"]
        assert second_data["idempotent"] is True
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_id_changed_request_returns_409(self, tmp_path):
        adapter = _make_adapter()
        adapter._run_idempotency = RunIdempotencyStore(tmp_path / "state.db")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                assert (await cli.post(
                    "/v1/runs", json={"input": "hello", "turn_id": "turn-mismatch"}
                )).status == 202
                response = await cli.post(
                    "/v1/runs", json={"input": "changed", "turn_id": "turn-mismatch"}
                )
                data = await response.json()

        assert response.status == 409
        assert data["error"]["code"] == "turn_id_conflict"
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_id_retry_after_adapter_restart_keeps_run_identity(self, tmp_path):
        path = tmp_path / "state.db"
        first_adapter = _make_adapter()
        first_adapter._run_idempotency = RunIdempotencyStore(path)
        first_app = _create_runs_app(first_adapter)
        body = {"input": "hello", "turn_id": "turn-restart"}
        async with TestClient(TestServer(first_app)) as cli:
            with patch.object(first_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                first = await cli.post("/v1/runs", json=body)
                first_data = await first.json()

        second_adapter = _make_adapter()
        second_adapter._run_idempotency = RunIdempotencyStore(path)
        second_app = _create_runs_app(second_adapter)
        async with TestClient(TestServer(second_app)) as cli:
            with patch.object(second_adapter, "_create_agent") as mock_create:
                second = await cli.post("/v1/runs", json=body)
                second_data = await second.json()

        assert second.status == 202
        assert second_data["run_id"] == first_data["run_id"]
        assert second_data["idempotent"] is True
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_id_isolated_between_multiplex_profiles(self, tmp_path, monkeypatch):
        """The shared listener must use each profile's own state.db relation."""
        from gateway.config import GatewayConfig

        profile_homes = {
            name: tmp_path / name for name in ("foo", "bar")
        }
        for home in profile_homes.values():
            home.mkdir()
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex=True: list(profile_homes.items()),
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: profile_homes[name],
        )

        adapter = _make_adapter()
        adapter.gateway_runner = MagicMock(config=GatewayConfig(multiplex_profiles=True))
        app = web.Application(
            middlewares=[adapter._make_profile_prefix_middleware()]
        )
        app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
        body = {"input": "hello", "turn_id": "same-turn"}

        with patch.object(adapter, "_create_agent") as mock_create:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            mock_create.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                foo = await cli.post("/p/foo/v1/runs", json=body)
                bar = await cli.post("/p/bar/v1/runs", json=body)
                foo_data = await foo.json()
                bar_data = await bar.json()
                await asyncio.sleep(0.05)

        assert foo.status == bar.status == 202
        assert foo_data["run_id"] != bar_data["run_id"]
        assert mock_create.call_count == 2
        assert (profile_homes["foo"] / "state.db").exists()
        assert (profile_homes["bar"] / "state.db").exists()

    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_non_object_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=[{"input": "hello"}])
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_start_empty_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": ""})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_history_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "conversation_history": {"role": "user"}},
            )
        assert resp.status == 400
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_with_valid_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202

    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_status_completed_run_includes_output_and_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                assert status["output"] == "done"
                assert status["usage"]["total_tokens"] == 6
                assert status["last_event"] == "run.completed"

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body

    @pytest.mark.asyncio
    async def test_pending_steer_event_precedes_terminal_event(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "Use the other approach",
                }
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_resp.text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        event_types = [event["type"] for event in events]
        missed_index = event_types.index("response.omnio.steer_missed")
        terminal_index = event_types.index("response.completed")
        assert events[missed_index]["text"] == "Use the other approach"
        assert missed_index < terminal_index



    @pytest.mark.asyncio
    async def test_approval_response_without_pending_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 409
                approval_data = await approval_resp.json()
                assert approval_data["error"]["code"] in {
                    "approval_not_active",
                    "approval_not_pending",
                }

    @pytest.mark.asyncio
    async def test_approval_string_false_does_not_resolve_all(self, adapter):
        """Quoted false must not fan out approval resolution across the queue."""
        app = _create_runs_app(adapter)
        run_id = "run_bool_parse"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_approval_sessions[run_id] = "session-123"

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "all": "false"},
                )

        assert approval_resp.status == 200
        mock_resolve.assert_called_once_with(
            "session-123",
            "once",
            resolve_all=False,
        )

    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                assert await _wait_for_thread_event(victim_ready)
                assert await _wait_for_thread_event(attacker_ready)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


    @pytest.mark.asyncio
    async def test_events_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:
    def test_sweep_keeps_transport_with_active_subscriber(self, adapter):
        run_id = "run_subscribed"
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_streams_created[run_id] = 0
        adapter._run_stream_subscribers.add(run_id)

        adapter._sweep_orphaned_runs_once(time.time())

        assert adapter._run_streams[run_id] is queue
        assert run_id in adapter._run_streams_created

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert await _wait_for_thread_event(agent_ready)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

    @pytest.mark.asyncio
    async def test_expired_transport_stops_buffering_new_deltas(self, adapter):
        """An unconsumed expired queue must not grow for the rest of a live run."""
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert await _wait_for_thread_event(agent_ready)
                expired_queue = adapter._run_streams[run_id]
                stream_delta = mock_create.call_args.kwargs["stream_delta_callback"]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                adapter._sweep_orphaned_runs_once(time.time())
                before = expired_queue.qsize()
                stream_delta("must-not-buffer")
                mock_agent.interrupt("finish test")
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert expired_queue.qsize() == before

    @pytest.mark.asyncio
    async def test_expired_orphan_run_state_is_reaped(self, adapter):
        run_id = "run_expired_orphan"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0
        adapter._run_approval_sessions[run_id] = run_id

        pending = approval_mod._ApprovalEntry({
            "command": "bash -c orphaned",
            "description": "orphaned approval",
            "pattern_keys": ["shell-c"],
        })
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]

        with patch(
            "gateway.platforms.api_server.asyncio.sleep",
            side_effect=[None, asyncio.CancelledError()],
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert run_id not in adapter._run_streams
        assert run_id not in adapter._run_streams_created
        assert run_id not in adapter._run_approval_sessions
        assert pending.event.is_set()
        with approval_mod._lock:
            assert run_id not in approval_mod._gateway_queues


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_before_agent_creation_prevents_run_start(self, adapter):
        """A stop accepted while queued must prevent agent construction."""
        app = _create_runs_app(adapter)
        original_create_task = asyncio.create_task
        task_started = asyncio.Event()
        allow_task = asyncio.Event()

        def _delayed_create_task(coro):
            async def _delayed():
                task_started.set()
                await allow_task.wait()
                return await coro

            return original_create_task(_delayed())

        with patch("gateway.platforms.api_server.asyncio.create_task", side_effect=_delayed_create_task), \
             patch.object(adapter, "_create_agent") as mock_create:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                await task_started.wait()

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                allow_task.set()

                for _ in range(20):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                mock_create.assert_not_called()
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert await _wait_for_thread_event(started)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                assert await _wait_for_thread_event(agent_ready)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.5)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/stop")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_stop_already_completed_run_returns_404(self, adapter):
        """Stopping a run that already finished should return 404 (refs cleaned up)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                # Start and wait for completion
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)

                # Run should be done, refs cleaned up
                assert run_id not in adapter._active_run_agents

                # Stop should return 404
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_interrupt_exception_does_not_crash(self, adapter):
        """If agent.interrupt() raises, stop should still succeed."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()

                # Override the interrupt side_effect to raise. Still trip
                # ``interrupted`` so the slow_run thread unblocks at teardown
                # — without this the agent thread blocks the full 10s
                # timeout and the test teardown waits the same amount.
                def _raising_interrupt(message=None):
                    interrupted.set()
                    raise RuntimeError("interrupt failed")

                mock_agent.interrupt = MagicMock(side_effect=_raising_interrupt)
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                assert await _wait_for_thread_event(agent_ready)
                await asyncio.sleep(0.1)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                assert await _wait_for_thread_event(agent_ready)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestSteerRun:
    @staticmethod
    def _make_active_agent(adapter, run_id="run_steer"):
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent._executing_tools = False
        task = MagicMock()
        task.done.return_value = False
        adapter._run_statuses[run_id] = {
            "object": "hermes.run",
            "run_id": run_id,
            "status": "running",
        }
        adapter._active_run_agents[run_id] = agent
        adapter._active_run_tasks[run_id] = task
        adapter._run_lifecycles[run_id] = {
            "accepting": True,
            "agent": agent,
            "pending": [],
            "lock": asyncio.Lock(),
        }
        return agent

    def test_route_table_registers_steer_next_to_stop(self, adapter):
        routes = {
            (method, path): handler.__name__
            for method, path, handler in adapter._http_route_table()
        }
        assert routes[("POST", "/v1/runs/{run_id}/stop")] == "_handle_stop_run"
        assert routes[("POST", "/v1/runs/{run_id}/steer")] == "_handle_steer_run"

    @pytest.mark.asyncio
    async def test_redirects_capable_active_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent._supports_active_turn_redirect = True
        agent.redirect.return_value = True
        agent.steer.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "  change direction  "},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data == {"status": "redirected"}
        agent.redirect.assert_called_once_with("change direction")
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_soft_steer_skips_redirect_when_redirect_is_supported(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent._supports_active_turn_redirect = True
        agent.redirect.return_value = True
        agent.steer.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "do not interrupt", "mode": "steer"},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data == {"status": "queued"}
        agent.redirect.assert_not_called()
        agent.steer.assert_called_once_with("do not interrupt")

    @pytest.mark.asyncio
    async def test_tool_boundary_redirect_reports_queued(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent._supports_active_turn_redirect = True
        agent._executing_tools = True
        agent.redirect.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "apply after the tool"},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data == {"status": "queued"}
        agent.redirect.assert_called_once_with("apply after the tool")
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_codex_soft_steer_uses_native_turn_steer(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent.api_mode = "codex_app_server"
        agent._supports_active_turn_redirect = False
        agent.redirect.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "native soft correction", "mode": "steer"},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data == {"status": "queued"}
        agent.redirect.assert_called_once_with("native soft correction")
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_steer_when_redirect_is_unavailable(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent._supports_active_turn_redirect = False
        agent.steer.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "keep going"},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data["status"] == "queued"
        agent.redirect.assert_not_called()
        agent.steer.assert_called_once_with("keep going")

    @pytest.mark.asyncio
    async def test_falls_back_to_steer_when_redirect_raises(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent._supports_active_turn_redirect = True
        agent.redirect.side_effect = RuntimeError("redirect unavailable")
        agent.steer.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "use the fallback"},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data["status"] == "queued"
        agent.steer.assert_called_once_with("use the fallback")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": "   "}])
    async def test_rejects_missing_or_blank_text(self, adapter, body):
        app = _create_runs_app(adapter)
        self._make_active_agent(adapter)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post("/v1/runs/run_steer/steer", json=body)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_returns_400_when_both_steer_paths_are_rejected(self, adapter):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)
        agent._supports_active_turn_redirect = True
        agent.redirect.return_value = False
        agent.steer.return_value = False

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "not accepted"},
            )
            response_data = await response.json()

        assert response.status == 400
        assert response_data["error"]["code"] == "steer_failed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["soft", "", None, {"kind": "steer"}])
    async def test_rejects_invalid_mode(self, adapter, mode):
        app = _create_runs_app(adapter)
        agent = self._make_active_agent(adapter)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_steer/steer",
                json={"text": "keep going", "mode": mode},
            )
            response_data = await response.json()

        assert response.status == 400
        assert response_data["error"]["code"] == "steer_invalid_mode"
        agent.redirect.assert_not_called()
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_latches_steers_until_agent_is_published_in_order(self, adapter):
        app = _create_runs_app(adapter)
        create_started = threading.Event()
        allow_create = threading.Event()
        run_started = threading.Event()
        allow_run_finish = threading.Event()

        agent = MagicMock()
        agent._supports_active_turn_redirect = True
        agent.redirect.return_value = True
        agent.steer.return_value = True
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        def _create_agent(**_kwargs):
            create_started.set()
            assert allow_create.wait(5)
            return agent

        def _run_conversation(*_args, **_kwargs):
            run_started.set()
            assert allow_run_finish.wait(5)
            return {"final_response": "done"}

        agent.run_conversation.side_effect = _run_conversation

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                start_response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_response.json())["run_id"]
                assert await _wait_for_thread_event(create_started)

                first = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"text": "first correction", "mode": "steer"},
                )
                second = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"text": "second correction"},
                )

                assert first.status == 200
                assert await first.json() == {"status": "queued"}
                assert second.status == 200
                assert await second.json() == {"status": "queued"}
                agent.steer.assert_not_called()

                allow_create.set()
                assert await _wait_for_thread_event(run_started)
                agent.steer.assert_called_once_with(
                    "first correction\nsecond correction"
                )
                agent.redirect.assert_not_called()

                allow_run_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_finalize_race_surfaces_accepted_steer_before_terminal(self, adapter):
        app = _create_runs_app(adapter)
        run_started = threading.Event()
        allow_run_finish = threading.Event()
        run_returned = threading.Event()
        steer_entered = threading.Event()
        allow_steer_return = threading.Event()
        pending: list[str] = []

        agent = MagicMock()
        agent._supports_active_turn_redirect = False
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        def _run_conversation(*_args, **_kwargs):
            run_started.set()
            assert allow_run_finish.wait(5)
            run_returned.set()
            return {"final_response": "done"}

        def _steer(text):
            steer_entered.set()
            assert allow_steer_return.wait(5)
            pending.append(text)
            return True

        def _drain_pending_steer():
            if not pending:
                return None
            text = "\n".join(pending)
            pending.clear()
            return text

        agent.run_conversation.side_effect = _run_conversation
        agent.steer.side_effect = _steer
        agent._drain_pending_steer.side_effect = _drain_pending_steer

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=agent):
                start_response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_response.json())["run_id"]
                assert await _wait_for_thread_event(run_started)

                steer_request = asyncio.create_task(
                    cli.post(
                        f"/v1/runs/{run_id}/steer",
                        json={"text": "retain this exact text"},
                    )
                )
                assert await _wait_for_thread_event(steer_entered)

                allow_run_finish.set()
                assert await _wait_for_thread_event(run_returned)
                await asyncio.sleep(0)
                allow_steer_return.set()

                steer_response = await steer_request
                assert steer_response.status == 200
                assert await steer_response.json() == {"status": "queued"}

                events_response = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_response.text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        event_types = [event["type"] for event in events]
        missed_index = event_types.index("response.omnio.steer_missed")
        terminal_index = event_types.index("response.completed")
        assert events[missed_index]["text"] == "retain this exact text"
        assert missed_index < terminal_index

    @pytest.mark.asyncio
    async def test_unknown_run_uses_stop_not_found_error_shape(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            stop_response = await cli.post("/v1/runs/run_unknown/stop")
            steer_response = await cli.post(
                "/v1/runs/run_unknown/steer",
                json={"text": "hello"},
            )
            stop_data = await stop_response.json()
            steer_data = await steer_response.json()

        assert stop_response.status == 404
        assert steer_response.status == 404
        assert steer_data == stop_data

    @pytest.mark.asyncio
    async def test_prefixed_route_reaches_steer_handler(self, adapter):
        app = _create_runs_app(adapter, include_profile_routes=True)
        agent = self._make_active_agent(adapter, run_id="run_prefixed")
        agent._supports_active_turn_redirect = False
        agent.steer.return_value = True

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/p/coder/v1/runs/run_prefixed/steer",
                json={"text": "profile guidance"},
            )
            response_data = await response.json()

        assert response.status == 200
        assert response_data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_capabilities_advertise_steer(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.get("/v1/capabilities")
            payload = await response.json()

        assert response.status == 200
        assert payload["features"]["run_steer"] is True
        assert payload["endpoints"]["run_steer"] == {
            "method": "POST",
            "path": "/v1/runs/{run_id}/steer",
        }


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
