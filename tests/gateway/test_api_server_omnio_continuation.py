"""Behavior contract for no-user continuations on ``/v1/runs``."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.tool_approval as tool_approval
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    TURN_CONTINUATION_API_VERSION,
    APIServerAdapter,
    _parse_continuation,
)
from gateway.run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB

AUTH = {"Authorization": "Bearer test-key"}
SESSION = "conversation-session"


def _managed(turn_id: str = "turn-continue") -> dict:
    return {
        "turn_id": turn_id,
        "session_id": SESSION,
        "omnio_managed": {
            "version": 1,
            "submission_id": "12345678-1234-4234-8234-123456789abc",
            "execution_fingerprint": "a" * 64,
        },
    }


def _continue(close=None, **extra) -> dict:
    return {"input": None, "continuation": {"close": close}, **_managed(), **extra}


def _call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session(SESSION, source="api_server")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def adapter(db, tmp_path):
    value = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    value._session_db = db
    value._run_idempotency = RunIdempotencyStore(tmp_path / "state.db")
    return value


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


def _sse_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "continued"}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


async def _wait_for_run(agent: MagicMock) -> dict:
    for _ in range(200):
        if agent.run_conversation.called:
            return agent.run_conversation.call_args.kwargs
        await asyncio.sleep(0.01)
    raise AssertionError("the continuation never reached the agent")


class TestParseContinuation:
    def test_no_close(self):
        assert _parse_continuation({}) is None
        assert _parse_continuation({"close": None}) is None

    def test_answer_matches_the_plugin_result_shape(self):
        close = _parse_continuation(
            {"close": {"kind": "answer", "tool_call_id": "q1", "response": "Blue", "ag_ui_state": {"a": 1}}}
        )
        assert close["kind"] == "answer"
        assert json.loads(close["content"]) == {
            "status": "answered",
            "response": "Blue",
            "ag_ui_state": {"a": 1},
        }

    def test_late_deny_carries_the_live_denial(self):
        close = _parse_continuation(
            {"close": {"kind": "approval", "tool_call_id": "w1", "scope": "deny"}}
        )
        assert json.loads(close["content"])["status"] == "approval_denied"

    def test_allow_carries_no_result(self):
        close = _parse_continuation(
            {"close": {"kind": "approval", "tool_call_id": "w1", "scope": "once"}}
        )
        assert close["content"] is None

    @pytest.mark.parametrize(
        "value",
        [
            [],
            {"close": {"kind": "interrupted", "tool_call_id": "x"}},
            {"close": {"kind": "answer", "tool_call_id": "q1"}},
            {"close": {"kind": "answer", "tool_call_id": "", "response": "x"}},
            {"close": {"kind": "approval", "tool_call_id": "w1", "scope": "forever"}},
            {"close": {"kind": "approval", "tool_call_id": "w1", "scope": "once", "tool": {"name": "x"}}},
            {"close": {"kind": "resume"}},
            {"close": None, "extra": 1},
        ],
    )
    def test_rejects_malformed(self, value):
        with pytest.raises(ValueError):
            _parse_continuation(value)


@pytest.mark.asyncio
async def test_capabilities_advertise_continuation(adapter):
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.get("/v1/capabilities", headers=AUTH)
        body = await response.json()
    assert body["turn_continuation_api_version"] == TURN_CONTINUATION_API_VERSION


@pytest.mark.asyncio
async def test_continuation_requires_a_turn_and_session(adapter):
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            headers=AUTH,
            json={"input": None, "continuation": {}, "session_id": SESSION},
        )
        body = await response.json()
    assert response.status == 400
    assert body["error"]["code"] == "invalid_continuation"


@pytest.mark.asyncio
async def test_continuation_adds_no_input(adapter):
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/runs", headers=AUTH, json={**_continue(), "input": "hello"}
        )
    assert response.status == 400


@pytest.mark.asyncio
async def test_answer_closes_the_question_then_runs_without_a_user_message(adapter, db):
    db.append_message(SESSION, "user", "pick a colour")
    db.append_message(SESSION, "assistant", "", tool_calls=[_call("q1", "request_user_input")])
    agent = _agent()
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs",
                headers=AUTH,
                json=_continue({"kind": "answer", "tool_call_id": "q1", "response": "Blue"}),
            )
            kwargs = await _wait_for_run(agent)

    assert response.status == 202
    assert kwargs["continuation"] is True
    assert kwargs["user_message"] == ""
    history = kwargs["conversation_history"]
    assert [m["role"] for m in history] == ["user", "assistant", "tool"]
    assert json.loads(history[-1]["content"]) == {"status": "answered", "response": "Blue"}


@pytest.mark.asyncio
async def test_failed_turn_continues_from_its_saved_work(adapter, db):
    db.append_message(SESSION, "user", "research this")
    db.append_message(
        SESSION,
        "assistant",
        "",
        tool_calls=[_call("c1", "web_search"), _call("c2", "terminal")],
    )
    db.append_message(SESSION, "tool", "results", tool_call_id="c1", tool_name="web_search")
    agent = _agent()
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs", headers=AUTH, json=_continue({"kind": "interrupted"})
            )
            kwargs = await _wait_for_run(agent)

    assert response.status == 202
    history = kwargs["conversation_history"]
    assert [(m["role"], m.get("tool_call_id")) for m in history[-2:]] == [
        ("tool", "c1"),
        ("tool", "c2"),
    ]
    assert json.loads(history[-1]["content"])["status"] == "interrupted"


@pytest.mark.asyncio
async def test_replay_after_the_tail_moved_on_returns_the_same_run(adapter, db):
    db.append_message(SESSION, "user", "hello")
    agent = _agent()
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent) as create_agent:
            first = await client.post("/v1/runs", headers=AUTH, json=_continue())
            first_body = await first.json()
            await _wait_for_run(agent)
            # The continuation answered; a retry must not try to close this tail.
            db.append_message(SESSION, "assistant", "all done")
            second = await client.post("/v1/runs", headers=AUTH, json=_continue())
            second_body = await second.json()

    assert first.status == second.status == 202
    assert second_body == {
        "run_id": first_body["run_id"],
        "status": second_body["status"],
        "idempotent": True,
    }
    create_agent.assert_called_once()


@pytest.mark.asyncio
async def test_final_answer_is_not_resumable_and_reserves_nothing(adapter, db, tmp_path):
    db.append_message(SESSION, "user", "hello")
    db.append_message(SESSION, "assistant", "all done")
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent") as create_agent:
            response = await client.post("/v1/runs", headers=AUTH, json=_continue())
            body = await response.json()

    assert response.status == 409
    assert body["error"]["code"] == "continuation_not_resumable"
    create_agent.assert_not_called()
    assert adapter._run_idempotency.get("turn-continue") is None


@pytest.mark.asyncio
async def test_a_different_answer_conflicts(adapter, db):
    db.append_message(SESSION, "user", "pick")
    db.append_message(SESSION, "assistant", "", tool_calls=[_call("q1", "request_user_input")])
    db.append_message(SESSION, "tool", '{"status": "answered", "response": "Red"}', tool_call_id="q1", tool_name="request_user_input")
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            headers=AUTH,
            json=_continue({"kind": "answer", "tool_call_id": "q1", "response": "Blue"}),
        )
        body = await response.json()
    assert response.status == 409
    assert body["error"]["code"] == "continuation_conflict"


@pytest.mark.asyncio
async def test_late_nested_approval_grants_the_exact_call_and_tells_the_model(adapter, db):
    db.append_message(SESSION, "user", "write it")
    db.append_message(SESSION, "assistant", "", tool_calls=[_call("x1", "execute_code")])
    db.append_message(SESSION, "tool", "approval not granted", tool_call_id="x1", tool_name="execute_code")
    agent = _agent()
    tool = {"name": "mcp_connectors_write", "arguments": {"id": 7}}
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent) as create_agent:
            response = await client.post(
                "/v1/runs",
                headers=AUTH,
                json=_continue(
                    {"kind": "approval", "tool_call_id": "nested_x1_0", "scope": "once", "tool": tool}
                ),
            )
            await _wait_for_run(agent)

    assert response.status == 202
    prompt = create_agent.call_args.kwargs["ephemeral_system_prompt"]
    assert "approved `mcp_connectors_write`" in prompt
    grant_key = adapter._scoped_tool_approval_session_key(SESSION, adapter._effective_request_profile())
    assert tool_approval.consume_once_approval(grant_key, "nested_new_0", tool["name"], tool["arguments"])
    assert not tool_approval.consume_once_approval(grant_key, "nested_new_1", tool["name"], tool["arguments"])


def test_expired_approval_leaves_its_call_open():
    agent = MagicMock()
    agent._omnio_skip_persist_tool_call_ids = None
    APIServerAdapter._interrupt_for_expired_tool_approval(
        agent, {"toolCallId": "w1", "interaction": {"timed_out": True}}
    )
    assert agent._omnio_skip_persist_tool_call_ids == {"w1"}
    agent.interrupt.assert_called_once()


def test_answered_approval_is_not_left_open():
    agent = MagicMock()
    agent._omnio_skip_persist_tool_call_ids = None
    APIServerAdapter._interrupt_for_expired_tool_approval(
        agent, {"toolCallId": "w1", "interaction": {"timed_out": False}}
    )
    assert agent._omnio_skip_persist_tool_call_ids is None
    agent.interrupt.assert_not_called()


def test_open_question_reads_as_pending_only_in_the_request_copy():
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "assistant", "tool_calls": [_call("answered", "request_user_input"), _call("pending", "request_user_input")]},
        {"role": "tool", "tool_call_id": "answered", "content": '{"status":"answered"}'},
    ]
    saved = json.loads(json.dumps(messages))
    outgoing = sanitize_api_messages([dict(message) for message in messages])

    assert messages == saved
    pending = next(m for m in outgoing if m.get("tool_call_id") == "pending")
    assert json.loads(pending["content"])["status"] == "pending"


def test_missing_ordinary_result_keeps_the_generic_stub():
    from agent.agent_runtime_helpers import sanitize_api_messages

    outgoing = sanitize_api_messages(
        [{"role": "assistant", "tool_calls": [_call("ordinary", "read_file")]}]
    )
    stub = next(m for m in outgoing if m.get("tool_call_id") == "ordinary")
    assert stub["content"] == "[Result unavailable — see context summary above]"


def test_timed_out_interaction_rows_are_never_saved(db):
    from agent.message_sanitization import close_interrupted_tool_sequence
    from agent.tool_executor import _mark_omnio_timeout_tool_result
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = SESSION
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._session_persist_lock = None
    agent._omnio_skip_persist_tool_call_ids = {"q1"}
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_call("q1", "request_user_input")]},
        {"role": "tool", "tool_call_id": "q1", "tool_name": "request_user_input", "content": '{"status":"no_response"}'},
    ]
    _mark_omnio_timeout_tool_result(agent, messages[-1], "q1")
    assert close_interrupted_tool_sequence(messages)

    AIAgent._flush_messages_to_session_db(agent, messages, [])

    rows = db.get_messages(SESSION)
    assert [row["role"] for row in rows] == ["assistant"]


@pytest.mark.asyncio
async def test_the_closing_step_is_the_runs_first_event(adapter, db):
    db.append_message(SESSION, "user", "pick")
    db.append_message(
        SESSION,
        "assistant",
        "",
        tool_calls=[_call("q1", "request_user_input"), _call("c2", "terminal")],
    )
    agent = _agent()
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs",
                headers=AUTH,
                json=_continue({"kind": "answer", "tool_call_id": "q1", "response": "Blue"}),
            )
            run_id = (await response.json())["run_id"]
            await _wait_for_run(agent)
            events = await client.get(f"/v1/runs/{run_id}/events?after=0", headers=AUTH)
            body = await events.text()

    omnio = [e for e in _sse_events(body) if e.get("type", "").startswith("response.omnio.")]
    assert omnio[0]["type"] == "response.omnio.continuation"
    assert omnio[0]["closed"] == [
        {"tool_call_id": "q1", "kind": "answer"},
        {"tool_call_id": "c2", "kind": "interrupted"},
    ]
    assert omnio[1]["type"] == "response.omnio.interaction_completed"
    assert (omnio[1]["tool_call_id"], omnio[1]["choice"]) == ("q1", "Blue")


@pytest.mark.asyncio
async def test_unmanaged_continuation_is_keyed_by_turn_id(adapter, db):
    """The Omnio proxy identifies runs by turn_id alone."""
    db.append_message(SESSION, "user", "hello")
    agent = _agent()
    body = {"input": None, "continuation": {}, "turn_id": "turn-plain", "session_id": SESSION}
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent) as create_agent:
            first = await client.post("/v1/runs", headers=AUTH, json=body)
            first_body = await first.json()
            await _wait_for_run(agent)
            db.append_message(SESSION, "assistant", "all done")
            second = await client.post("/v1/runs", headers=AUTH, json=body)
            second_body = await second.json()
            other_session = await client.post(
                "/v1/runs", headers=AUTH, json={**body, "session_id": "other"}
            )

    assert first.status == second.status == 202
    assert second_body["run_id"] == first_body["run_id"]
    assert second_body["idempotent"] is True
    assert other_session.status == 409
    create_agent.assert_called_once()
