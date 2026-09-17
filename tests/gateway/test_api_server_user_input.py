"""Tests for POST /v1/omnio/user-input — answer delivery for request_user_input.

Covers auth, request validation, and that a valid answer releases the matching
blocked waiter (the agent worker parked inside the plugin's await_user_input).
"""

import threading
import time
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.user_input as user_input
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

SESSION = "sess-1"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [
        mw for mw in (cors_middleware, security_headers_middleware) if mw is not None
    ]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/omnio/user-input", adapter._handle_omnio_user_input)
    app.router.add_post("/v1/runs/{run_id}/user-input", adapter._handle_run_user_input)
    return app


@pytest.fixture(autouse=True)
def _clean_state():
    user_input.clear_session(SESSION)
    yield
    user_input.clear_session(SESSION)


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.asyncio
async def test_requires_auth():
    auth_adapter = _make_adapter(api_key="sk-secret")
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": "hi"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_delivers_the_answer_to_a_blocked_waiter(adapter):
    user_input.register_user_input_session(SESSION)
    result: dict[str, str | None] = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "answer", user_input.await_user_input(SESSION, "call-1")
        )
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if user_input._wait_registry.pending_count(SESSION):
            break
        time.sleep(0.01)
    else:
        pytest.fail("the user-input waiter should be parked")

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": "Tuesday at 3pm", "toolCallId": "call-1"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
        assert resp.status == 200
        body = await resp.json()

    assert body["object"] == "omnio.user_input_response"
    assert body["resolved"] is True
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result["answer"] == "Tuesday at 3pm"


@pytest.mark.asyncio
async def test_resolves_a_wait_parked_under_the_active_run_id(adapter):
    # Turn-path runs park the wait under the run id (the run's session
    # context), while the answer arrives keyed by conversation session — the
    # endpoint must translate across the two namespaces.
    run_id = "run_user_input_ns"
    user_input.clear_session(run_id)
    user_input.register_user_input_session(run_id)
    adapter._active_run_agents[run_id] = object()
    adapter._run_statuses[run_id] = {"session_id": SESSION}
    result: dict[str, str | None] = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "answer", user_input.await_user_input(run_id, "call-1")
        )
    )
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if user_input._wait_registry.pending_count(run_id):
                break
            time.sleep(0.01)
        else:
            pytest.fail("the user-input waiter should be parked")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/omnio/user-input",
                json={"response": "Brand A", "toolCallId": "call-1"},
                headers={"X-Hermes-Session-Id": SESSION},
            )
            assert resp.status == 200
            body = await resp.json()

        assert body["resolved"] is True
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert result["answer"] == "Brand A"
    finally:
        user_input.clear_session(run_id)
        adapter._active_run_agents.pop(run_id, None)
        adapter._run_statuses.pop(run_id, None)


@pytest.mark.asyncio
async def test_reports_not_resolved_when_no_call_is_waiting(adapter):
    # Stale card (already answered / timed out / turn ended): a valid request,
    # but nothing is parked — the chat treats resolved:false as a stale card.
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": "hi", "toolCallId": "call-x"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
        assert resp.status == 200
        body = await resp.json()
    assert body["resolved"] is False


@pytest.mark.asyncio
async def test_rejects_a_missing_response(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"toolCallId": "call-1"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_a_non_string_response(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            json={"response": {"not": "a string"}},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_a_missing_session_id(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/omnio/user-input", json={"response": "hi"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_rejects_invalid_json(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/user-input",
            data=b"not json",
            headers={
                "X-Hermes-Session-Id": SESSION,
                "Content-Type": "application/json",
            },
        )
    assert resp.status == 400


@pytest.fixture
def exact_run(adapter):
    run_id = "run_exact_question"
    user_input.register_user_input_session(run_id)
    agent = Mock()
    adapter._active_run_agents[run_id] = agent
    adapter._run_statuses[run_id] = {"session_id": SESSION, "status": "running"}
    adapter._turn_event_logs.create_run(
        run_id, session_id=SESSION, owner_profile=adapter._effective_request_profile(),
    )
    yield run_id, agent
    user_input.clear_session(run_id)


def _park_exact(run_id, call_id="call-1"):
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault(
        "answer", user_input.await_user_input(run_id, call_id),
    ))
    thread.start()
    deadline = time.monotonic() + 3
    while not user_input._wait_registry.pending_count(run_id):
        if time.monotonic() > deadline:
            pytest.fail("question did not park")
        time.sleep(0.01)
    return thread, result


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["answer", "skip", "supersede"])
async def test_exact_resolution_commits_before_releasing_the_question(adapter, exact_run, action):
    run_id, agent = exact_run
    waiter, result = _park_exact(run_id)
    agent.interrupt.side_effect = lambda _message: result.update(interrupted=True)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            f"/v1/runs/{run_id}/user-input",
            json={"toolCallId": "call-1", "response": "chosen", "action": action},
            headers={"X-Hermes-Session-Id": SESSION},
        )
    waiter.join(timeout=3)
    assert response.status == 200
    assert result["answer"] == "chosen"
    assert bool(result.get("interrupted")) == (action == "supersede")
    assert user_input._wait_registry.pending_count(run_id) == 0


@pytest.mark.asyncio
async def test_exact_resolution_replays_without_answering_a_later_question(adapter, exact_run):
    run_id, _agent = exact_run
    first, _ = _park_exact(run_id)
    body = {"toolCallId": "call-1", "response": "first"}
    async with TestClient(TestServer(_create_app(adapter))) as client:
        url = f"/v1/runs/{run_id}/user-input"
        headers = {"X-Hermes-Session-Id": SESSION}
        assert (await client.post(url, json=body, headers=headers)).status == 200
        first.join(timeout=3)
        second, result = _park_exact(run_id, "call-2")
        replay = await client.post(url, json=body, headers=headers)
        assert (await replay.json())["replayed"] is True
        assert user_input._wait_registry.pending_count(run_id) == 1
        conflict = await client.post(url, json={**body, "response": "different"}, headers=headers)
        assert conflict.status == 409
        user_input.clear_session(run_id)
        second.join(timeout=3)
        assert result["answer"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("session,call_id,status", [("other", "call-1", 404), (SESSION, "stale", 200)])
async def test_exact_resolution_rejects_stale_identity(adapter, exact_run, session, call_id, status):
    run_id, agent = exact_run
    waiter, _ = _park_exact(run_id)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            f"/v1/runs/{run_id}/user-input",
            json={"toolCallId": call_id, "response": "wrong", "action": "supersede"},
            headers={"X-Hermes-Session-Id": session},
        )
        assert response.status == status
        assert user_input._wait_registry.pending_count(run_id) == 1
    agent.interrupt.assert_not_called()
    user_input.clear_session(run_id)
    waiter.join(timeout=3)


@pytest.mark.asyncio
async def test_supersession_failure_keeps_the_question_waiting(adapter, exact_run):
    run_id, agent = exact_run
    waiter, _ = _park_exact(run_id)
    agent.interrupt.side_effect = RuntimeError("cannot interrupt")
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            f"/v1/runs/{run_id}/user-input",
            json={"toolCallId": "call-1", "response": "skip", "action": "supersede"},
            headers={"X-Hermes-Session-Id": SESSION},
        )
        assert response.status == 500
        assert user_input._wait_registry.pending_count(run_id) == 1
    user_input.clear_session(run_id)
    waiter.join(timeout=3)


@pytest.mark.asyncio
async def test_exact_resolution_projects_only_the_answer_from_shared_state(adapter, exact_run):
    import json

    run_id, _ = exact_run
    waiter, result = _park_exact(run_id)
    envelope = json.dumps({"_omnio_interaction_answer": 1, "response": "Continue", "ag_ui_state": {"selection": "A"}})
    async with TestClient(TestServer(_create_app(adapter))) as client:
        url = f"/v1/runs/{run_id}/user-input"
        headers = {"X-Hermes-Session-Id": SESSION}
        response = await client.post(url, json={"toolCallId": "call-1", "response": envelope}, headers=headers)
        assert response.status == 200
        waiter.join(timeout=3)
        assert result["answer"] == envelope
        conflict = await client.post(url, json={"toolCallId": "call-1", "response": "different"}, headers=headers)
        assert (await conflict.json())["choice"] == "Continue"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, [], {"toolCallId": "call-1", "response": 3}, {"toolCallId": "call-1", "response": "x", "action": []}])
async def test_exact_resolution_rejects_invalid_commands(adapter, exact_run, body):
    run_id, agent = exact_run
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(f"/v1/runs/{run_id}/user-input", json=body, headers={"X-Hermes-Session-Id": SESSION})
        assert response.status == 400
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_exact_resolution_requires_authentication():
    adapter = _make_adapter(api_key="secret")
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/v1/runs/unknown/user-input", json={"toolCallId": "call-1", "response": "x"})
        assert response.status == 401
