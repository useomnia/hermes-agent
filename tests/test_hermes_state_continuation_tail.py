"""SessionDB.close_continuation_tail: making a saved tail resumable without a user message."""

import json

import pytest

from hermes_state import SessionDB

INTERRUPTED = json.dumps({"status": "interrupted"})


def _is_interaction(name: str) -> bool:
    return name == "request_user_input"


def _call(call_id: str, name: str, arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("s1", source="api_server")
    yield session_db
    session_db.close()


def close(db, step, **kwargs):
    return db.close_continuation_tail(
        "s1",
        step,
        interrupted_content=INTERRUPTED,
        is_interaction_tool=_is_interaction,
        **kwargs,
    )


def tool_rows(db):
    return [
        (m["tool_call_id"], m["content"])
        for m in db.get_messages_as_conversation("s1")
        if m["role"] == "tool"
    ]


def assistant_calls(db, *calls):
    db.append_message("s1", "user", "do the thing")
    db.append_message("s1", "assistant", "", tool_calls=list(calls))


class TestTailShapes:
    def test_empty_session_is_not_resumable(self, db):
        assert close(db, None) == ("not_resumable", {"reason": "empty_session"})

    def test_user_tail_resumes_with_nothing_to_close(self, db):
        db.append_message("s1", "user", "hello")
        assert close(db, None) == ("ok", {"tail": "user", "closed": []})
        assert close(db, {"kind": "interrupted"})[0] == "ok"

    def test_user_tail_has_no_call_to_answer(self, db):
        db.append_message("s1", "user", "hello")
        status, info = close(
            db, {"kind": "answer", "tool_call_id": "c1", "content": "{}"}
        )
        assert (status, info["reason"]) == ("not_found", "no_pending_call")

    def test_final_assistant_text_is_not_resumable(self, db):
        db.append_message("s1", "user", "hello")
        db.append_message("s1", "assistant", "all done")
        status, info = close(db, {"kind": "interrupted"})
        assert (status, info["reason"]) == ("not_resumable", "final_assistant_message")

    def test_resolved_block_resumes_with_nothing_to_close(self, db):
        assistant_calls(db, _call("c1", "terminal"))
        db.append_message("s1", "tool", "ok", tool_call_id="c1", tool_name="terminal")
        assert close(db, None) == ("ok", {"tail": "tool_calls", "closed": []})

    def test_unresolved_block_needs_a_closing_step(self, db):
        assistant_calls(db, _call("c1", "terminal"))
        status, info = close(db, None)
        assert (status, info["reason"]) == ("not_resumable", "unresolved_calls")
        assert tool_rows(db) == []

    def test_foreign_tool_results_are_not_resumable(self, db):
        assistant_calls(db, _call("c1", "terminal"))
        db.append_message("s1", "tool", "stray", tool_call_id="other", tool_name="terminal")
        status, info = close(db, {"kind": "interrupted"})
        assert (status, info["reason"]) == ("not_resumable", "foreign_tool_results")


class TestInterrupted:
    def test_every_unresolved_call_gets_the_interrupted_result(self, db):
        assistant_calls(db, _call("c1", "terminal"), _call("c2", "web_search"))
        db.append_message("s1", "tool", "done", tool_call_id="c1", tool_name="terminal")
        assert close(db, {"kind": "interrupted"}) == (
            "ok",
            {"tail": "tool_calls", "closed": ["c2"]},
        )
        assert tool_rows(db) == [("c1", "done"), ("c2", INTERRUPTED)]

    def test_replay_closes_nothing_more(self, db):
        assistant_calls(db, _call("c1", "terminal"))
        close(db, {"kind": "interrupted"})
        assert close(db, {"kind": "interrupted"}) == (
            "ok",
            {"tail": "tool_calls", "closed": []},
        )
        assert len(tool_rows(db)) == 1


class TestAnswer:
    def test_answer_closes_the_question_and_interrupts_siblings(self, db):
        assistant_calls(db, _call("q1", "request_user_input"), _call("c2", "terminal"))
        answer = {"kind": "answer", "tool_call_id": "q1", "content": '{"status": "answered"}'}
        assert close(db, answer) == ("ok", {"tail": "tool_calls", "closed": ["q1", "c2"]})
        assert tool_rows(db) == [("q1", '{"status": "answered"}'), ("c2", INTERRUPTED)]

    def test_same_answer_replays(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        answer = {"kind": "answer", "tool_call_id": "q1", "content": '{"a": 1}'}
        close(db, answer)
        assert close(db, answer) == ("ok", {"replayed": True})
        assert len(tool_rows(db)) == 1

    def test_different_answer_conflicts(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        close(db, {"kind": "answer", "tool_call_id": "q1", "content": '{"a": 1}'})
        assert close(db, {"kind": "answer", "tool_call_id": "q1", "content": '{"a": 2}'}) == (
            "conflict",
            {"replayed": True},
        )

    def test_same_answer_replays_over_a_rebuilt_bare_result(self, db):
        # A session rebuilt from saved history records a late answer as its bare text.
        assistant_calls(db, _call("q1", "request_user_input"))
        db.append_message("s1", "tool", "Yes", tool_call_id="q1", tool_name="request_user_input")
        answer = json.dumps({"status": "answered", "response": "Yes"})

        assert close(db, {"kind": "answer", "tool_call_id": "q1", "content": answer}) == (
            "ok",
            {"replayed": True},
        )
        assert tool_rows(db) == [("q1", "Yes")]

    def test_same_answer_with_shared_state_replays(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        close(db, {"kind": "answer", "tool_call_id": "q1", "content": json.dumps(
            {"status": "answered", "response": "Yes"}
        )})
        answer = json.dumps({"status": "answered", "response": "Yes", "ag_ui_state": {"k": 1}})

        assert close(db, {"kind": "answer", "tool_call_id": "q1", "content": answer})[0] == "ok"

    def test_different_answer_conflicts_with_a_rebuilt_bare_result(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        db.append_message("s1", "tool", "No", tool_call_id="q1", tool_name="request_user_input")
        answer = json.dumps({"status": "answered", "response": "Yes"})

        assert close(db, {"kind": "answer", "tool_call_id": "q1", "content": answer})[0] == (
            "conflict"
        )

    def test_answer_does_not_replay_over_a_skip(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        close(db, {"kind": "answer", "tool_call_id": "q1", "content": json.dumps(
            {"status": "skipped"}
        )})
        answer = json.dumps({"status": "answered", "response": "Yes"})

        assert close(db, {"kind": "answer", "tool_call_id": "q1", "content": answer})[0] == (
            "conflict"
        )

    def test_interrupted_question_cannot_be_answered(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        close(db, {"kind": "interrupted"})
        status, _ = close(db, {"kind": "answer", "tool_call_id": "q1", "content": "{}"})
        assert status == "conflict"

    def test_answer_must_target_a_question(self, db):
        assistant_calls(db, _call("c1", "terminal"))
        status, info = close(db, {"kind": "answer", "tool_call_id": "c1", "content": "{}"})
        assert (status, info["reason"]) == ("not_found", "wrong_call_kind")

    def test_unknown_call(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        status, info = close(db, {"kind": "answer", "tool_call_id": "nope", "content": "{}"})
        assert (status, info["reason"]) == ("not_found", "unknown_call")


class TestApproval:
    def _grants(self, db):
        rows = db.get_messages_as_conversation("s1")
        assistant = next(m for m in rows if m["role"] == "assistant")
        return (assistant.get("display_metadata") or {}).get("_omnio_resolved_approvals")

    def test_allow_records_a_grant_without_a_result(self, db):
        assistant_calls(db, _call("w1", "mcp_write", '{"x": 1}'))
        approval = {"kind": "approval", "tool_call_id": "w1", "scope": "once", "content": None}
        assert close(db, approval) == ("ok", {"tail": "tool_calls", "closed": ["w1"]})
        assert tool_rows(db) == []
        assert self._grants(db) == {
            "w1": {"scope": "once", "tool_name": "mcp_write", "arguments": '{"x": 1}'}
        }

    def test_same_allow_replays_even_after_the_call_ran(self, db):
        assistant_calls(db, _call("w1", "mcp_write"))
        approval = {"kind": "approval", "tool_call_id": "w1", "scope": "once", "content": None}
        close(db, approval)
        db.append_message("s1", "tool", "written", tool_call_id="w1", tool_name="mcp_write")
        assert close(db, approval) == ("ok", {"replayed": True})

    def test_different_scope_conflicts(self, db):
        assistant_calls(db, _call("w1", "mcp_write"))
        close(db, {"kind": "approval", "tool_call_id": "w1", "scope": "once", "content": None})
        status, _ = close(
            db, {"kind": "approval", "tool_call_id": "w1", "scope": "always", "content": None}
        )
        assert status == "conflict"

    def test_deny_records_the_denial_result(self, db):
        assistant_calls(db, _call("w1", "mcp_write"))
        deny = {"kind": "approval", "tool_call_id": "w1", "scope": "deny", "content": '{"status": "approval_denied"}'}
        assert close(db, deny)[0] == "ok"
        assert tool_rows(db) == [("w1", '{"status": "approval_denied"}')]
        assert close(db, deny) == ("ok", {"replayed": True})

    def test_allow_after_deny_conflicts(self, db):
        assistant_calls(db, _call("w1", "mcp_write"))
        close(db, {"kind": "approval", "tool_call_id": "w1", "scope": "deny", "content": "{}"})
        status, _ = close(
            db, {"kind": "approval", "tool_call_id": "w1", "scope": "once", "content": None}
        )
        assert status == "conflict"

    def test_approval_cannot_target_a_question(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        status, info = close(
            db, {"kind": "approval", "tool_call_id": "q1", "scope": "once", "content": None}
        )
        assert (status, info["reason"]) == ("not_found", "wrong_call_kind")


class TestContextNotes:
    @pytest.mark.parametrize("tail", ["user", "answered", "interrupted", "denied", "resolved"])
    def test_notes_follow_the_closed_tail(self, db, tail):
        from agent.context_notes import context_note_message

        step = None
        if tail == "user":
            db.append_message("s1", "user", "hello")
        else:
            assistant_calls(db, _call("c1", "request_user_input" if tail == "answered" else "mcp_write"))
            if tail == "answered":
                step = {"kind": "answer", "tool_call_id": "c1", "content": "yes"}
            elif tail == "denied":
                step = {"kind": "approval", "tool_call_id": "c1", "scope": "deny", "content": "denied"}
            elif tail == "interrupted":
                step = {"kind": "interrupted"}
            else:
                db.append_message("s1", "tool", "done", tool_call_id="c1")

        status, _ = close(db, step, notes=["Project: alpha", "File edited"], continuation_id="turn-1")
        assert status == "ok"
        rows = db.get_messages_as_conversation("s1")
        assert [{"role": row["role"], "content": row["content"]} for row in rows[-2:]] == [
            context_note_message("Project: alpha"), context_note_message("File edited")
        ]
        assert rows[-3]["role"] == ("user" if tail == "user" else "tool")

    def test_notes_replay_once_before_run_admission(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        step = {"kind": "answer", "tool_call_id": "q1", "content": "yes"}
        for _ in range(2):
            assert close(db, step, notes=["Project: alpha"], continuation_id="turn-1")[0] == "ok"
        assert len(db.get_messages_as_conversation("s1")) == 4

    def test_changed_notes_conflict_without_appending(self, db):
        db.append_message("s1", "user", "hello")
        close(db, None, notes=["alpha"], continuation_id="turn-1")
        assert close(db, None, notes=["beta"], continuation_id="turn-1")[0] == "conflict"
        assert len(db.get_messages_as_conversation("s1")) == 2

    def test_approval_defers_notes_without_overwriting_its_grant(self, db):
        assistant_calls(db, _call("w1", "mcp_write"))
        step = {"kind": "approval", "tool_call_id": "w1", "scope": "once", "content": None}
        assert close(db, step, notes=["Project: alpha"], continuation_id="turn-1")[0] == "ok"
        rows = db.get_messages_as_conversation("s1")
        assert [row["role"] for row in rows] == ["user", "assistant"]
        metadata = rows[-1]["display_metadata"]
        assert metadata["_omnio_resolved_approvals"]["w1"]["scope"] == "once"
        assert metadata["_omnio_continuation_notes"]["turn-1"]["deferred"] is True

    def test_rejected_close_does_not_save_notes(self, db):
        assistant_calls(db, _call("q1", "request_user_input"))
        assert close(db, {"kind": "answer", "tool_call_id": "unknown", "content": "yes"},
                     notes=["alpha"], continuation_id="turn-1")[0] == "not_found"
        assert len(db.get_messages_as_conversation("s1")) == 2
