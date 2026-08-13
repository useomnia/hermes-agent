"""Behavior contracts for durable Omnio HITL resume."""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import tools.tool_approval as tool_approval
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from run_agent import AIAgent


AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def adapter(db):
    value = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    value._session_db = db
    return value


def _tool_call(call_id: str = "call-1", name: str = "request_user_input") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"question":"Which?"}'},
    }


def _durable_grant(scope: str, name: str) -> dict:
    return {
        "scope": scope,
        "tool_name": name,
        "arguments": '{"question":"Which?"}',
    }


async def _seed_dangling(db: SessionDB, session_id: str, call: dict) -> None:
    db.create_session(session_id, "api_server")
    db.append_message(session_id, "assistant", "", tool_calls=[call])


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/continuations/prepare", adapter._handle_prepare_continuation
    )
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_post(
        "/api/sessions/{session_id}/interactions/{tool_call_id}/resolve",
        adapter._handle_resolve_interaction,
    )
    return app


@pytest.mark.asyncio
async def test_prepared_resume_keeps_interaction_unresolved_until_admission_and_replays(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    await _seed_dangling(db, "prepared-resume", _tool_call("call-prepared"))
    continuation_id = "interaction-resume:call-prepared"
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "continued",
        "messages": [],
        "interrupted": False,
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    async with TestClient(TestServer(_app(adapter))) as client:
        prepared = await client.post(
            "/v1/continuations/prepare",
            headers=AUTH,
            json={
                "session_id": "prepared-resume",
                "turn_id": continuation_id,
                "tool_call_id": "call-prepared",
            },
        )
        prepared_payload = await prepared.json()
        replayed = await client.post(
            "/v1/continuations/prepare",
            headers=AUTH,
            json={
                "session_id": "prepared-resume",
                "turn_id": continuation_id,
                "tool_call_id": "call-prepared",
            },
        )
        assert prepared_payload["run_id"] == (await replayed.json())["run_id"]
        assert [row["role"] for row in db.get_messages("prepared-resume")] == [
            "assistant"
        ]
        premature = await client.post(
            "/v1/runs",
            headers=AUTH,
            json={
                "session_id": "prepared-resume",
                "input": None,
                "turn_id": continuation_id,
                "start_prepared": prepared_payload["run_id"],
            },
        )
        resolved = await client.post(
            "/api/sessions/prepared-resume/interactions/call-prepared/resolve",
            headers=AUTH,
            json={
                "kind": "input",
                "response": "accepted",
                "resolutionId": continuation_id,
            },
        )
        repeated = await client.post(
            "/api/sessions/prepared-resume/interactions/call-prepared/resolve",
            headers=AUTH,
            json={
                "kind": "input",
                "response": "accepted",
                "resolutionId": continuation_id,
            },
        )
        prepared_after_resolve = await client.post(
            "/v1/continuations/prepare",
            headers=AUTH,
            json={
                "session_id": "prepared-resume",
                "turn_id": continuation_id,
                "tool_call_id": "call-prepared",
            },
        )
        assert prepared_after_resolve.status == 202, await prepared_after_resolve.text()
        assert (await prepared_after_resolve.json())["run_id"] == prepared_payload["run_id"]
        with patch.object(adapter, "_create_agent", return_value=agent):
            started = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={
                    "session_id": "prepared-resume",
                    "input": None,
                    "turn_id": continuation_id,
                    "start_prepared": prepared_payload["run_id"],
                },
            )
            started_payload = await started.json()

    assert prepared.status == replayed.status == 202
    assert premature.status == 409
    assert resolved.status == repeated.status == 200
    assert started.status == 202
    assert started_payload["run_id"] == prepared_payload["run_id"]
    rows = db.get_messages("prepared-resume")
    assert [row["role"] for row in rows] == ["assistant", "tool"]
    assert (
        rows[0]["display_metadata"]["_omnio_continuation_claim"]["phase"]
        == "started"
    )


@pytest.mark.asyncio
async def test_legacy_resume_takes_over_prepared_claim_after_legacy_resolve(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    """An old Omnia retry can finish a claim left by a newer Hermes prepare."""
    await _seed_dangling(db, "legacy-takeover", _tool_call("call-legacy"))
    continuation_id = "interaction-resume:call-legacy"
    legacy_turn_id = "legacy-omnia-turn-id"
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "continued",
        "messages": [],
        "interrupted": False,
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    async with TestClient(TestServer(_app(adapter))) as client:
        prepared = await client.post(
            "/v1/continuations/prepare",
            headers=AUTH,
            json={
                "session_id": "legacy-takeover",
                "turn_id": continuation_id,
                "tool_call_id": "call-legacy",
            },
        )
        prepared_payload = await prepared.json()
        resolved = await client.post(
            "/api/sessions/legacy-takeover/interactions/call-legacy/resolve",
            headers=AUTH,
            json={"kind": "input", "response": "accepted"},
        )
        with patch.object(adapter, "_create_agent", return_value=agent):
            started = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={
                    "session_id": "legacy-takeover",
                    "input": None,
                    "turn_id": legacy_turn_id,
                },
            )
            started_payload = await started.json()

    assert prepared.status == 202
    assert resolved.status == 200
    assert started.status == 202
    assert started_payload["run_id"] != prepared_payload["run_id"]
    rows = db.get_messages("legacy-takeover")
    assert [row["role"] for row in rows] == ["assistant", "tool"]
    assert rows[0]["display_metadata"]["_omnio_continuation_claim"] == {
        "continuation_id": legacy_turn_id,
        "run_id": started_payload["run_id"],
    }


def test_resolution_id_cannot_resolve_a_different_tool_call(db: SessionDB) -> None:
    db.create_session("bound-resolution", "api_server")
    db.append_message(
        "bound-resolution",
        "assistant",
        "",
        tool_calls=[_tool_call("call-a"), _tool_call("call-b")],
    )
    status, _ = db.claim_pending_continuation(
        "bound-resolution",
        "interaction-resume:call-a",
        "run-bound",
        phase="prepared",
        tool_call_id="call-a",
    )
    assert status == "claimed"
    resolved, _ = db.resolve_pending_interaction(
        "bound-resolution",
        "call-b",
        expected_tool_name="request_user_input",
        tool_result_content='{"status":"answered","response":"wrong"}',
        resolution_id="interaction-resume:call-a",
    )
    assert resolved == "not_resumable"
    assert [row["role"] for row in db.get_messages("bound-resolution")] == [
        "assistant"
    ]


@pytest.mark.asyncio
async def test_resolve_input_is_durable_once_and_requires_the_active_dangling_tail(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    await _seed_dangling(db, "input-session", _tool_call())
    await _seed_dangling(db, "moved-on", _tool_call("call-moved"))
    db.append_message("moved-on", "user", "new message")

    async with TestClient(TestServer(_app(adapter))) as client:
        resolved = await client.post(
            "/api/sessions/input-session/interactions/call-1/resolve",
            headers=AUTH,
            json={"kind": "input", "response": "Brand A"},
        )
        repeated = await client.post(
            "/api/sessions/input-session/interactions/call-1/resolve",
            headers=AUTH,
            json={"kind": "input", "response": "Brand A"},
        )
        unknown = await client.post(
            "/api/sessions/input-session/interactions/not-a-call/resolve",
            headers=AUTH,
            json={"kind": "input", "response": "Brand A"},
        )
        moved_on = await client.post(
            "/api/sessions/moved-on/interactions/call-moved/resolve",
            headers=AUTH,
            json={"kind": "input", "response": "Brand A"},
        )
        resolved_payload = await resolved.json()
        statuses = (repeated.status, unknown.status, moved_on.status)

    assert resolved.status == 200
    assert resolved_payload == {"resolved": True}
    assert statuses == (409, 404, 409)
    rows = db.get_messages("input-session")
    assert [row["role"] for row in rows] == ["assistant", "tool"]
    assert rows[-1]["content"] == (
        '{"status": "answered", "response": "Brand A"}'
    )


@pytest.mark.asyncio
async def test_resolve_input_persists_genui_state_in_live_plugin_shape(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    await _seed_dangling(db, "genui-input", _tool_call("call-genui"))
    shared_state = {
        "competitors": {
            "competitors": [{"name": "Razer", "domain": "razer.com"}]
        }
    }

    async with TestClient(TestServer(_app(adapter))) as client:
        resolved = await client.post(
            "/api/sessions/genui-input/interactions/call-genui/resolve",
            headers=AUTH,
            json={
                "kind": "input",
                "response": "Continue",
                "agUiState": shared_state,
            },
        )

    assert resolved.status == 200
    expected = json.dumps(
        {
            "status": "answered",
            "response": "Continue",
            "ag_ui_state": shared_state,
        },
        ensure_ascii=False,
    )
    rows = db.get_messages("genui-input")
    assert rows[-1]["content"] == expected
    provider_tool = db.get_messages_as_conversation("genui-input")[-1]
    assert provider_tool["role"] == "tool"
    assert provider_tool["tool_call_id"] == "call-genui"
    assert provider_tool["tool_name"] == "request_user_input"
    assert provider_tool["content"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_state", [None, [], "state", 1, True])
async def test_resolve_input_rejects_non_object_genui_state(
    adapter: APIServerAdapter,
    db: SessionDB,
    invalid_state,
) -> None:
    session_id = f"invalid-genui-{type(invalid_state).__name__}"
    await _seed_dangling(db, session_id, _tool_call("call-invalid-state"))

    async with TestClient(TestServer(_app(adapter))) as client:
        resolved = await client.post(
            f"/api/sessions/{session_id}/interactions/call-invalid-state/resolve",
            headers=AUTH,
            json={
                "kind": "input",
                "response": "Continue",
                "agUiState": invalid_state,
            },
        )

    assert resolved.status == 400
    assert [row["role"] for row in db.get_messages(session_id)] == ["assistant"]


@pytest.mark.asyncio
async def test_concurrent_input_resolves_append_exactly_one_tool_row(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    await _seed_dangling(db, "concurrent-input", _tool_call())

    async with TestClient(TestServer(_app(adapter))) as client:
        first, second = await asyncio.gather(
            client.post(
                "/api/sessions/concurrent-input/interactions/call-1/resolve",
                headers=AUTH,
                json={"kind": "input", "response": "Brand A"},
            ),
            client.post(
                "/api/sessions/concurrent-input/interactions/call-1/resolve",
                headers=AUTH,
                json={"kind": "input", "response": "Brand A"},
            ),
        )

    assert sorted((first.status, second.status)) == [200, 409]
    rows = db.get_messages("concurrent-input")
    assert [row["role"] for row in rows] == ["assistant", "tool"]
    assert sum(row["tool_call_id"] == "call-1" for row in rows) == 1


@pytest.mark.asyncio
async def test_resolve_input_after_sibling_result_builds_continuation_without_user(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    input_call = _tool_call("call-input")
    sibling_call = _tool_call("call-read", "read_file")
    db.create_session("mixed-siblings", "api_server")
    db.append_message(
        "mixed-siblings",
        "assistant",
        "",
        tool_calls=[input_call, sibling_call],
    )
    db.append_message(
        "mixed-siblings",
        "tool",
        '{"content":"sibling result"}',
        tool_name="read_file",
        tool_call_id="call-read",
    )

    captured = {}
    agent = MagicMock()
    agent.run_conversation.side_effect = lambda **kwargs: captured.update(kwargs) or {
        "final_response": "continued",
        "messages": [],
        "interrupted": False,
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    async with TestClient(TestServer(_app(adapter))) as client:
        resolved = await client.post(
            "/api/sessions/mixed-siblings/interactions/call-input/resolve",
            headers=AUTH,
            json={"kind": "input", "response": "Brand A"},
        )
        with patch.object(adapter, "_create_agent", return_value=agent):
            continued = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={"session_id": "mixed-siblings", "input": None},
            )
            for _ in range(100):
                if captured:
                    break
                await asyncio.sleep(0.01)

    assert resolved.status == 200
    assert continued.status == 202
    rows = db.get_messages("mixed-siblings")
    assert [row["role"] for row in rows] == ["assistant", "tool", "tool"]
    assert [row["tool_call_id"] for row in rows[1:]] == ["call-read", "call-input"]
    assert captured["continuation"] is True
    assert [message["role"] for message in captured["conversation_history"]] == [
        "assistant", "tool", "tool",
    ]
    assert all(
        message["role"] != "user" for message in captured["conversation_history"]
    )


@pytest.mark.asyncio
async def test_resolve_approval_denial_and_one_shot_grant(
    adapter: APIServerAdapter, db: SessionDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_name = "mcp_connectors_TEST_WRITE"
    await _seed_dangling(db, "deny-session", _tool_call("call-deny", tool_name))
    await _seed_dangling(db, "once-session", _tool_call("call-once", tool_name))
    monkeypatch.setattr(tool_approval, "is_gated_tool", lambda _name: True)

    async with TestClient(TestServer(_app(adapter))) as client:
        denied = await client.post(
            "/api/sessions/deny-session/interactions/call-deny/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "deny"}},
        )
        granted = await client.post(
            "/api/sessions/once-session/interactions/call-once/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "once"}},
        )

    assert denied.status == granted.status == 200
    assert json.loads(db.get_messages("deny-session")[-1]["content"])["status"] == "approval_denied"
    assert [row["role"] for row in db.get_messages("once-session")] == ["assistant"]

    token = tool_approval.set_current_tool_approval_session_key("once-session")
    try:
        wrong_call = tool_approval.maybe_require_tool_approval(
            tool_name, "different-call", {"question": "Which?"}
        )
        assert tool_approval.maybe_require_tool_approval(tool_name, "call-once", {"question": "Which?"}) is None
        second = tool_approval.maybe_require_tool_approval(tool_name, "call-once", {"question": "Which?"})
    finally:
        tool_approval.reset_current_tool_approval_session_key(token)
        tool_approval.clear_session("once-session")
    assert json.loads(wrong_call)["status"] == "approval_error"
    assert json.loads(second)["status"] == "approval_error"


@pytest.mark.asyncio
async def test_resolve_approval_after_sibling_tool_result_claims_exact_call(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    tool_name = "mcp_connectors_TEST_WRITE"
    db.create_session("approval-siblings", "api_server")
    db.append_message(
        "approval-siblings",
        "assistant",
        "",
        tool_calls=[
            _tool_call("call-approval", tool_name),
            _tool_call("call-read", "read_file"),
        ],
    )
    db.append_message(
        "approval-siblings",
        "tool",
        '{"content":"read result"}',
        tool_call_id="call-read",
        tool_name="read_file",
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/approval-siblings/interactions/call-approval/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "once"}},
        )

    assert response.status == 200
    rows = db.get_messages("approval-siblings")
    assert [row["role"] for row in rows] == ["assistant", "tool"]
    assert rows[0]["display_metadata"] == {
        "_omnio_resolved_approvals": {
            "call-approval": _durable_grant("once", tool_name)
        }
    }
    assert tool_approval.consume_once_approval(
        "approval-siblings",
        "call-approval",
        tool_name,
        {"question": "Which?"},
    )
    tool_approval.clear_session("approval-siblings")


@pytest.mark.asyncio
async def test_accepted_approval_resolution_claim_is_sequentially_idempotent(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    tool_name = "mcp_connectors_TEST_WRITE"
    call = _tool_call("call-claimed", tool_name)
    db.create_session("claimed-once", "api_server")
    db.append_message(
        "claimed-once",
        "assistant",
        "",
        tool_calls=[call],
        display_kind="existing-kind",
        display_metadata={"existing": "preserved"},
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        first = await client.post(
            "/api/sessions/claimed-once/interactions/call-claimed/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "once"}},
        )
        repeated = await client.post(
            "/api/sessions/claimed-once/interactions/call-claimed/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "once"}},
        )
        conflicting_deny = await client.post(
            "/api/sessions/claimed-once/interactions/call-claimed/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "deny"}},
        )

    assert (first.status, repeated.status, conflicting_deny.status) == (200, 409, 409)
    rows = db.get_messages("claimed-once")
    assert [row["role"] for row in rows] == ["assistant"]
    assert rows[0]["display_kind"] == "existing-kind"
    assert rows[0]["display_metadata"] == {
        "existing": "preserved",
        "_omnio_resolved_approvals": {
            "call-claimed": _durable_grant("once", tool_name)
        },
    }
    tool_approval.clear_session("claimed-once")


@pytest.mark.asyncio
async def test_concurrent_accepted_approval_resolves_claim_exactly_once(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    tool_name = "mcp_connectors_TEST_WRITE"
    await _seed_dangling(
        db,
        "concurrent-approval",
        _tool_call("call-concurrent-approval", tool_name),
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        first, second = await asyncio.gather(
            client.post(
                "/api/sessions/concurrent-approval/interactions/"
                "call-concurrent-approval/resolve",
                headers=AUTH,
                json={"kind": "approval", "decision": {"scope": "once"}},
            ),
            client.post(
                "/api/sessions/concurrent-approval/interactions/"
                "call-concurrent-approval/resolve",
                headers=AUTH,
                json={"kind": "approval", "decision": {"scope": "once"}},
            ),
        )

    assert sorted((first.status, second.status)) == [200, 409]
    rows = db.get_messages("concurrent-approval")
    assert [row["role"] for row in rows] == ["assistant"]
    assert rows[0]["display_metadata"] == {
        "_omnio_resolved_approvals": {
            "call-concurrent-approval": _durable_grant("once", tool_name)
        }
    }
    assert tool_approval.consume_once_approval(
        "concurrent-approval",
        "call-concurrent-approval",
        tool_name,
        {"question": "Which?"},
    )
    assert not tool_approval.consume_once_approval(
        "concurrent-approval",
        "call-concurrent-approval",
        tool_name,
        {"question": "Which?"},
    )
    tool_approval.clear_session("concurrent-approval")


@pytest.mark.asyncio
async def test_resolve_approval_maps_session_and_always_scopes_to_existing_grants(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    tool_name = "mcp_connectors_TEST_WRITE"
    await _seed_dangling(db, "session-grant", _tool_call("call-session", tool_name))
    await _seed_dangling(db, "always-grant", _tool_call("call-always", tool_name))

    async with TestClient(TestServer(_app(adapter))) as client:
        session = await client.post(
            "/api/sessions/session-grant/interactions/call-session/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "session"}},
        )
        always = await client.post(
            "/api/sessions/always-grant/interactions/call-always/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "always"}},
        )

    assert session.status == always.status == 200
    assert tool_approval.is_tool_approved("session-grant", tool_name)
    assert tool_name in tool_approval._always_approved
    tool_approval.clear_session("session-grant")
    tool_approval._always_approved.discard(tool_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["once", "session", "always"])
async def test_credit_approval_scopes_resume_as_exact_once_grants(
    adapter: APIServerAdapter,
    db: SessionDB,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    tool_name = "mcp_paid_GENERATE"
    session_id = f"credit-{scope}"
    call_id = f"call-credit-{scope}"
    await _seed_dangling(db, session_id, _tool_call(call_id, tool_name))
    monkeypatch.setattr(
        tool_approval,
        "mcp_tool_credits_meta",
        lambda name: {"strategy": "fixed", "credits": 1}
        if name == tool_name else None,
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            f"/api/sessions/{session_id}/interactions/{call_id}/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": scope}},
        )

    assert response.status == 200
    grant_key = adapter._scoped_tool_approval_session_key(session_id, None)
    assert not tool_approval.is_tool_approved(grant_key, tool_name)
    assert tool_name not in tool_approval._always_approved
    assert tool_approval.consume_once_approval(
        grant_key,
        call_id,
        tool_name,
        {"question": "Which?"},
    )
    assert not tool_approval.consume_once_approval(
        grant_key,
        call_id,
        tool_name,
        {"question": "Which?"},
    )
    assistant = db.get_messages(session_id)[0]
    assert assistant["display_metadata"]["_omnio_resolved_approvals"] == {
        call_id: _durable_grant(scope, tool_name)
    }
    tool_approval.clear_session(grant_key)


@pytest.mark.asyncio
async def test_runs_continuation_accepts_only_tool_or_dangling_tool_call_tails(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    db.create_session("answered", "api_server")
    db.append_message("answered", "assistant", "", tool_calls=[_tool_call("call-answered")])
    db.append_message("answered", "tool", '{"status":"answered","response":"A"}', tool_call_id="call-answered")
    db.create_session("invalid", "api_server")
    db.append_message("invalid", "assistant", "done")
    await _seed_dangling(db, "dangling", _tool_call("call-dangling"))

    captured = {}
    agent = MagicMock()
    agent.run_conversation.side_effect = lambda **kwargs: captured.update(kwargs) or {
        "final_response": "continued", "messages": [], "interrupted": False,
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            accepted = await client.post("/v1/runs", headers=AUTH, json={"session_id": "answered", "input": None})
            dangling = await client.post("/v1/runs", headers=AUTH, json={"session_id": "dangling", "input": None})
            invalid = await client.post("/v1/runs", headers=AUTH, json={"session_id": "invalid", "input": None})
            missing = await client.post("/v1/runs", headers=AUTH, json={"input": None})
            invalid_payload = await invalid.json()

    assert accepted.status == 202
    assert dangling.status == 202
    assert invalid.status == 400
    assert invalid_payload["error"]["code"] == "invalid_continuation"
    assert missing.status == 400


@pytest.mark.asyncio
async def test_concurrent_duplicate_continuations_enqueue_only_one_run(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    db.create_session("continuation-race", "api_server")
    db.append_message(
        "continuation-race",
        "assistant",
        "",
        tool_calls=[_tool_call("answered-race")],
    )
    db.append_message(
        "continuation-race",
        "tool",
        '{"status":"answered","response":"A"}',
        tool_call_id="answered-race",
    )
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "continued",
        "messages": [],
        "interrupted": False,
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            first, second = await asyncio.gather(*[
                client.post(
                    "/v1/runs",
                    headers=AUTH,
                    json={
                        "session_id": "continuation-race",
                        "input": None,
                        "turn_id": "client-continuation-1",
                    },
                )
                for _ in range(2)
            ])
            payloads = [await first.json(), await second.json()]

    assert sorted((first.status, second.status)) in ([202, 202], [202, 409])
    accepted_run_ids = {
        payload["run_id"]
        for response, payload in zip((first, second), payloads)
        if response.status == 202
    }
    assert len(accepted_run_ids) == 1
    for _ in range(50):
        if agent.run_conversation.call_count:
            break
        await asyncio.sleep(0.01)
    assert agent.run_conversation.call_count == 1
    tail = db.get_messages("continuation-race")[-1]
    assert tail["display_metadata"]["_omnio_continuation_claim"] == {
        "continuation_id": "client-continuation-1",
        "run_id": accepted_run_ids.pop(),
    }


@pytest.mark.asyncio
async def test_same_continuation_id_reclaims_after_adapter_restart(
    adapter: APIServerAdapter, db: SessionDB,
) -> None:
    db.create_session("restart-claim", "api_server")
    db.append_message(
        "restart-claim",
        "assistant",
        "",
        tool_calls=[_tool_call("restart-answer")],
    )
    db.append_message(
        "restart-claim",
        "tool",
        '{"status":"answered","response":"A"}',
        tool_call_id="restart-answer",
    )

    def _agent() -> MagicMock:
        value = MagicMock()
        value.run_conversation.return_value = {
            "final_response": "continued",
            "messages": [],
            "interrupted": False,
        }
        value.session_prompt_tokens = value.session_completion_tokens = 0
        value.session_total_tokens = 0
        return value

    first_agent = _agent()
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=first_agent):
            first = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={
                    "session_id": "restart-claim",
                    "input": None,
                    "turn_id": "stable-retry-id",
                },
            )
            first_payload = await first.json()

    restarted = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    restarted._session_db = db
    second_agent = _agent()
    async with TestClient(TestServer(_app(restarted))) as client:
        with patch.object(restarted, "_create_agent", return_value=second_agent):
            second = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={
                    "session_id": "restart-claim",
                    "input": None,
                    "turn_id": "stable-retry-id",
                },
            )
            second_payload = await second.json()

    assert first.status == second.status == 202
    assert first_payload["run_id"] != second_payload["run_id"]
    tail = db.get_messages("restart-claim")[-1]
    assert tail["display_metadata"]["_omnio_continuation_claim"] == {
        "continuation_id": "stable-retry-id",
        "run_id": second_payload["run_id"],
    }


@pytest.mark.asyncio
async def test_durable_once_approval_survives_fresh_adapter_and_executes(
    adapter: APIServerAdapter,
    db: SessionDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_name = "mcp_connectors_TEST_WRITE"
    call_id = "restart-approval"
    await _seed_dangling(
        db, "approval-restart", _tool_call(call_id, tool_name)
    )
    monkeypatch.setattr(
        tool_approval, "is_gated_tool", lambda name: name == tool_name
    )
    approval_wait = MagicMock(return_value=None)
    monkeypatch.setattr(tool_approval, "await_tool_approval", approval_wait)

    async with TestClient(TestServer(_app(adapter))) as client:
        resolved = await client.post(
            f"/api/sessions/approval-restart/interactions/{call_id}/resolve",
            headers=AUTH,
            json={"kind": "approval", "decision": {"scope": "once"}},
        )
    assert resolved.status == 200
    tool_approval.clear_session("approval-restart")

    restarted = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    restarted._session_db = db
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def _run_after_restart(**kwargs):
        assistant = kwargs["conversation_history"][0]
        call = assistant["tool_calls"][0]
        durable = assistant["display_metadata"][
            "_omnio_resolved_approvals"
        ][call_id]
        assert tool_approval.rehydrate_resolved_approval(
            tool_approval.get_current_tool_approval_session_key(),
            call_id,
            tool_name,
            call["function"]["arguments"],
            durable,
        )
        assert tool_approval.maybe_require_tool_approval(
            tool_name, call_id, {"question": "Which?"}
        ) is None
        db.append_message(
            "approval-restart",
            "tool",
            '{"ok":true}',
            tool_call_id=call_id,
            tool_name=tool_name,
        )
        return {
            "final_response": "continued",
            "messages": [],
            "interrupted": False,
        }

    agent.run_conversation.side_effect = _run_after_restart
    async with TestClient(TestServer(_app(restarted))) as client:
        with patch.object(restarted, "_create_agent", return_value=agent):
            continued = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={
                    "session_id": "approval-restart",
                    "input": None,
                    "turn_id": "approval-restart-turn",
                },
            )

    assert continued.status == 202
    for _ in range(50):
        if len(db.get_messages("approval-restart")) == 2:
            break
        await asyncio.sleep(0.01)
    rows = db.get_messages("approval-restart")
    assert [row["role"] for row in rows] == ["assistant", "tool"]
    assert rows[-1]["content"] == '{"ok":true}'
    assert approval_wait.call_count == 0
    tool_approval.clear_session("approval-restart")


@pytest.mark.asyncio
async def test_approval_replay_cannot_escalate_stored_scope(
    adapter: APIServerAdapter,
    db: SessionDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "scope-replay"
    tool_name = "mcp_connectors_TEST_WRITE"
    await _seed_dangling(db, "scope-replay", _tool_call(call_id, tool_name))
    monkeypatch.setattr(tool_approval, "is_credit_gated_tool", lambda _name: False)
    once = MagicMock()
    session = MagicMock()
    always = MagicMock()
    monkeypatch.setattr(tool_approval, "record_once_approval", once)
    monkeypatch.setattr(tool_approval, "record_session_approval", session)
    monkeypatch.setattr(tool_approval, "record_always_approval", always)

    async with TestClient(TestServer(_app(adapter))) as client:
        prepared = await client.post(
            "/v1/continuations/prepare",
            headers=AUTH,
            json={
                "session_id": "scope-replay",
                "turn_id": f"interaction-resume:{call_id}",
                "tool_call_id": call_id,
            },
        )
        assert prepared.status == 202
        first = await client.post(
            f"/api/sessions/scope-replay/interactions/{call_id}/resolve",
            headers=AUTH,
            json={
                "kind": "approval",
                "resolutionId": f"interaction-resume:{call_id}",
                "decision": {"scope": "once"},
            },
        )
        replay = await client.post(
            f"/api/sessions/scope-replay/interactions/{call_id}/resolve",
            headers=AUTH,
            json={
                "kind": "approval",
                "resolutionId": f"interaction-resume:{call_id}",
                "decision": {"scope": "always"},
            },
        )

    assert first.status == replay.status == 200
    assert always.call_count == 0
    assert once.call_count == 2
    assert session.call_count == 0


@pytest.mark.asyncio
async def test_resolution_id_cannot_replay_input_as_approval(
    adapter: APIServerAdapter,
    db: SessionDB,
) -> None:
    call_id = "kind-bound"
    await _seed_dangling(db, "kind-bound", _tool_call(call_id, "request_user_input"))
    async with TestClient(TestServer(_app(adapter))) as client:
        prepared = await client.post(
            "/v1/continuations/prepare",
            headers=AUTH,
            json={
                "session_id": "kind-bound",
                "turn_id": f"interaction-resume:{call_id}",
                "tool_call_id": call_id,
            },
        )
        assert prepared.status == 202
        first = await client.post(
            f"/api/sessions/kind-bound/interactions/{call_id}/resolve",
            headers=AUTH,
            json={
                "kind": "input",
                "resolutionId": f"interaction-resume:{call_id}",
                "response": "ok",
            },
        )
        replay = await client.post(
            f"/api/sessions/kind-bound/interactions/{call_id}/resolve",
            headers=AUTH,
            json={
                "kind": "approval",
                "resolutionId": f"interaction-resume:{call_id}",
                "decision": {"scope": "always"},
            },
        )
    assert first.status == 200
    assert replay.status == 404


@pytest.mark.asyncio
async def test_codex_app_server_route_rejects_before_continuation_claim(
    db: SessionDB,
) -> None:
    db.create_session("codex-continuation", "api_server")
    db.append_message(
        "codex-continuation",
        "assistant",
        "",
        tool_calls=[_tool_call("codex-answer")],
    )
    db.append_message(
        "codex-continuation",
        "tool",
        '{"status":"answered","response":"A"}',
        tool_call_id="codex-answer",
    )
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True,
        extra={
            "key": "test-key",
            "model_routes": {
                "codex": {"model": "gpt-5", "provider": "codex"}
            },
        },
    ))
    adapter._session_db = db

    with patch(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        return_value={"api_mode": "codex_app_server"},
    ) as resolve_runtime:
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={
                    "session_id": "codex-continuation",
                    "input": None,
                    "model": "codex",
                    "turn_id": "must-not-claim",
                },
            )
            payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == "invalid_continuation"
    resolve_runtime.assert_called_once_with("codex", target_model="gpt-5")
    assert db.get_messages("codex-continuation")[-1].get(
        "display_metadata"
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "result"),
    [
        (
            "request_user_input",
            '{"status":"no_response"}',
        ),
        (
            "mcp_connectors_TEST_WRITE",
            '{"status":"approval_no_response"}',
        ),
    ],
)
async def test_runs_timeout_callback_marks_only_expired_hitl_sentinels_for_skip_persist(
    adapter: APIServerAdapter,
    db: SessionDB,
    tool_name: str,
    result: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback consumes the reason from the exact registered run key."""
    await _seed_dangling(db, "timeout-session", _tool_call("expired-call", tool_name))
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent._omnio_skip_persist_tool_call_ids = None
    observed = {}

    def _build_agent(**kwargs):
        agent.tool_complete_callback = kwargs["tool_complete_callback"]
        return agent

    def _run_conversation(**_kwargs):
        if tool_name == "request_user_input":
            from tools.approval import get_current_session_key
            from tools.user_input import await_user_input

            wait_key = get_current_session_key()
            wait_result = await_user_input(wait_key, "expired-call")
        else:
            from tools.tool_approval import (
                await_tool_approval,
                get_current_tool_approval_surface_key,
            )

            wait_key = get_current_tool_approval_surface_key()
            wait_result = await_tool_approval(
                wait_key,
                tool_name,
                {"tool": tool_name, "toolCallId": "expired-call"},
                "expired-call",
            )
        observed.update(wait_key=wait_key, wait_result=wait_result)
        agent.tool_complete_callback("expired-call", tool_name, {}, result)
        return {"final_response": "", "messages": [], "interrupted": True}

    agent.run_conversation.side_effect = _run_conversation
    monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "0")
    monkeypatch.setenv("OMNIO_TOOL_APPROVAL_TIMEOUT", "0")
    monkeypatch.setattr(
        tool_approval,
        "is_gated_tool",
        lambda name: name == "mcp_connectors_TEST_WRITE",
    )
    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=_build_agent):
            started = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={"session_id": "timeout-session", "input": None},
            )
            run_id = (await started.json())["run_id"]
            assert started.status == 202
            for _ in range(50):
                if agent.interrupt.called:
                    break
                await asyncio.sleep(0.01)
    assert agent.interrupt.called
    assert agent._omnio_skip_persist_tool_call_ids == {"expired-call"}
    assert observed == {"wait_key": run_id, "wait_result": None}
    rows = db.get_messages("timeout-session")
    assert [row["role"] for row in rows] == ["assistant"]
    assert rows[0]["tool_calls"] == [_tool_call("expired-call", tool_name)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "result"),
    [
        ("request_user_input", '{"status":"no_response"}'),
        (
            "mcp_connectors_TEST_WRITE",
            '{"status":"approval_no_response"}',
        ),
    ],
)
async def test_disconnected_hitl_waits_are_also_marked_skip_persist(
    adapter: APIServerAdapter,
    db: SessionDB,
    tool_name: str,
    result: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_dangling(
        db, "cancelled-session", _tool_call("cancelled-call", tool_name)
    )
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent._omnio_skip_persist_tool_call_ids = None

    def _build_agent(**kwargs):
        agent.tool_complete_callback = kwargs["tool_complete_callback"]
        return agent

    def _run_conversation(**_kwargs):
        if tool_name == "request_user_input":
            from tools.approval import get_current_session_key
            from tools.user_input import await_user_input, clear_session

            wait_key = get_current_session_key()
            release = threading.Timer(0.05, clear_session, args=(wait_key,))
            release.start()
            wait_result = await_user_input(wait_key, "cancelled-call")
        else:
            from tools.tool_approval import (
                await_tool_approval,
                get_current_tool_approval_surface_key,
            )

            wait_key = get_current_tool_approval_surface_key()
            release = threading.Timer(
                0.05, tool_approval._wait_registry.clear, args=(wait_key,)
            )
            release.start()
            wait_result = await_tool_approval(
                wait_key,
                tool_name,
                {"tool": tool_name, "toolCallId": "cancelled-call"},
                "cancelled-call",
            )
        release.join()
        assert wait_result is None
        agent.tool_complete_callback(
            "cancelled-call", tool_name, {}, result
        )
        return {"final_response": "", "messages": [], "interrupted": True}

    agent.run_conversation.side_effect = _run_conversation
    monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "5")
    monkeypatch.setenv("OMNIO_TOOL_APPROVAL_TIMEOUT", "5")
    monkeypatch.setattr(
        tool_approval,
        "is_gated_tool",
        lambda name: name == "mcp_connectors_TEST_WRITE",
    )
    if tool_name != "request_user_input":
        # The run's own unregister cleanup can remove the registry reason as
        # the executor completes. Pin the callback seam to the cancellation
        # verdict while the user-input parameter exercises the real registry.
        monkeypatch.setattr(
            tool_approval,
            "consume_tool_approval_completion_reason",
            lambda _surface, _call: "cancelled",
        )

    async with TestClient(TestServer(_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=_build_agent):
            started = await client.post(
                "/v1/runs",
                headers=AUTH,
                json={"session_id": "cancelled-session", "input": None},
            )
            assert started.status == 202
            for _ in range(100):
                if agent._omnio_skip_persist_tool_call_ids:
                    break
                await asyncio.sleep(0.01)
    assert agent._omnio_skip_persist_tool_call_ids == {"cancelled-call"}


def test_pending_hitl_result_is_request_only_and_preserves_real_siblings() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [_tool_call("answered"), _tool_call("pending")]},
        {"role": "tool", "tool_call_id": "answered", "content": '{"status":"answered"}'},
    ]
    from agent.agent_runtime_helpers import sanitize_api_messages

    outgoing = sanitize_api_messages([dict(message) for message in messages])
    assert messages == [
        {"role": "assistant", "tool_calls": [_tool_call("answered"), _tool_call("pending")]},
        {"role": "tool", "tool_call_id": "answered", "content": '{"status":"answered"}'},
    ]
    pending = next(message for message in outgoing if message.get("tool_call_id") == "pending")
    assert json.loads(pending["content"]) == {
        "status": "pending",
        "note": "The user has not answered this request yet; it is still open. Do not assume an answer.",
    }
    assert next(message for message in outgoing if message.get("tool_call_id") == "answered")["content"] == '{"status":"answered"}'


def test_missing_ordinary_tool_result_keeps_generic_unavailable_stub() -> None:
    from agent.agent_runtime_helpers import sanitize_api_messages

    outgoing = sanitize_api_messages([
        {
            "role": "assistant",
            "tool_calls": [_tool_call("ordinary", "read_file")],
        }
    ])

    stub = next(
        message for message in outgoing
        if message.get("tool_call_id") == "ordinary"
    )
    assert stub["content"] == "[Result unavailable — see context summary above]"


def test_expired_interaction_rows_are_not_persisted(db: SessionDB) -> None:
    """The durable tail is the original assistant tool-call, and nothing else."""
    db.create_session("expired", "api_server")
    agent = object.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "expired"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._session_persist_lock = None
    agent._omnio_skip_persist_tool_call_ids = {"call-1"}
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call()]},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "tool_name": "request_user_input",
            "content": '{"status":"no_response"}',
        },
    ]
    from agent.message_sanitization import close_interrupted_tool_sequence
    from agent.tool_executor import _mark_omnio_timeout_tool_result

    _mark_omnio_timeout_tool_result(agent, messages[-1], "call-1")
    assert messages[-1]["_omnio_skip_persist"] is True
    assert agent._omnio_skip_persist_tool_call_ids == set()
    assert close_interrupted_tool_sequence(messages)
    assert messages[-1]["_omnio_skip_persist"] is True

    AIAgent._flush_messages_to_session_db(agent, messages, [])

    rows = db.get_messages("expired")
    assert [row["role"] for row in rows] == ["assistant"]
    assert rows[0]["tool_calls"] == [_tool_call()]
