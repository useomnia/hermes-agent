"""Behavior contracts for incremental tool-call persistence (#49045).

A destructive or process-terminating tool that runs during tool execution
must not lose the just-executed assistant(tool_calls) block or the tool
results that were produced before it fired.  These tests pin the contract:

    1. run_conversation flushes the assistant tool-call turn to the session
       DB BEFORE handing control to _execute_tool_calls (so a tool that
       restarts/kills the process never orphans the tool-call block).
    2. The SEQUENTIAL tool path flushes each tool result to the session DB
       immediately after appending it — BEFORE the next tool dispatches.
    3. The CONCURRENT tool path flushes each tool result in append order.

These exercise the REAL production dispatch surfaces:

    * sequential -> ``run_agent.handle_function_call`` (tool_executor ~1256/1298)
    * concurrent -> ``agent._invoke_tool`` (tool_executor ~539)

Mocking the genuine dispatch surface keeps the tests deterministic (no real
``web_search`` / network) AND mutation-survivable: the ordering assertions
read snapshots captured at flush time, so removing any production flush call
makes the corresponding assertion fail.
"""

import copy
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from agent.tool_dispatch_helpers import make_tool_result_message
from hermes_state import SessionDB
from run_agent import AIAgent
import tools.tool_approval as tool_approval
from tools.tool_approval import ToolApprovalDenial


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


# ---------------------------------------------------------------------------
# Contract 1: run_conversation persists the assistant tool-call block BEFORE
# tool execution begins.
# ---------------------------------------------------------------------------
def test_run_conversation_flushes_assistant_tool_call_before_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]

    # Record a deep snapshot of the message list at every flush so the
    # assertion does not depend on later mutations.
    flush_snapshots: list[list] = []

    def _record_flush(messages, conversation_history=None):
        flush_snapshots.append(copy.deepcopy(messages))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    # Capture observations at execute time into module-level lists rather than
    # asserting inside _execute_tool_calls — run_conversation's outer loop
    # swallows exceptions, so an in-callback assertion would never surface.
    executed = {"count": 0}
    snapshot_at_execute: list = []

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        executed["count"] += 1
        # Record the DB state observed at the moment tool execution begins.
        snapshot_at_execute.append(
            copy.deepcopy(flush_snapshots[-1]) if flush_snapshots else None
        )
        # Simulate the tool producing a result (as the real path would).
        messages.append(make_tool_result_message("web_search", "search result", "c1"))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("search something")

    assert executed["count"] == 1, "_execute_tool_calls was never reached"
    # The assistant tool-call block MUST have been flushed before execution.
    last = snapshot_at_execute[0]
    assert last is not None, "no flush occurred before tool execution"
    assert last[-1]["role"] == "assistant"
    assert last[-1]["tool_calls"][0]["id"] == "c1"
    assert result["final_response"] == "done"


# ---------------------------------------------------------------------------
# Contract 2: the SEQUENTIAL path flushes each tool result immediately, BEFORE
# the next tool dispatches.  Dispatch goes through run_agent.handle_function_call
# (the real production surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_sequential_flushes_each_tool_result_before_next_dispatch():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    # Ordered event log interleaving real dispatches and DB flushes.
    events: list = []

    def _fake_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # The result for call N must have been flushed before call N+1 fires.
        events.append(("dispatch", kwargs.get("tool_call_id")))
        return f"result-{kwargs.get('tool_call_id')}"

    def _record_flush(flush_messages, conversation_history=None):
        # Snapshot the tail tool result that triggered this flush.
        tail = flush_messages[-1]
        events.append(("flush", tail.get("role"), tail.get("tool_call_id")))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch("run_agent.handle_function_call", side_effect=_fake_dispatch) as disp,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # The mock proves we exercised the REAL sequential dispatch surface.
    assert disp.call_count == 2, "sequential path did not dispatch via handle_function_call"

    # Both tool results landed, in order.
    assert [m["role"] for m in messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Ordering contract: each tool result is flushed AFTER its own dispatch
    # and BEFORE the next dispatch. Expected interleaving:
    #   dispatch c1 -> flush c1 -> dispatch c2 -> flush c2
    assert events == [
        ("dispatch", "c1"),
        ("flush", "tool", "c1"),
        ("dispatch", "c2"),
        ("flush", "tool", "c2"),
    ]


# ---------------------------------------------------------------------------
# Contract 3: the CONCURRENT path flushes each collected tool result in append
# order.  Dispatch goes through agent._invoke_tool (the real concurrent
# surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_concurrent_flushes_each_tool_result_in_order():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    invoked_ids: list = []

    def _fake_invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        invoked_ids.append(tool_call_id)
        return f"result-{tool_call_id}"

    # Each flush must observe exactly one more tool result than the previous
    # flush, in append order — i.e. the tail tool_call_id sequence is c1, c2.
    flushed_tool_ids: list = []
    flush_lengths: list = []

    def _record_flush(flush_messages, conversation_history=None):
        flushed_tool_ids.append(flush_messages[-1]["tool_call_id"])
        flush_lengths.append(len([m for m in flush_messages if m.get("role") == "tool"]))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch.object(agent, "_invoke_tool", side_effect=_fake_invoke) as inv,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    # Proves the real concurrent dispatch surface was exercised.
    assert inv.call_count == 2, "concurrent path did not dispatch via _invoke_tool"
    assert sorted(invoked_ids) == ["c1", "c2"]

    # Results appended in deterministic order.
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Each tool result was flushed exactly once, in append order, with the
    # running tool count growing by one each time (1 then 2).  Removing either
    # production flush call breaks one of these assertions.
    assert flushed_tool_ids == ["c1", "c2"]
    assert flush_lengths == [1, 2]


def test_execute_tool_calls_sequential_marks_blocked_result_as_no_effect():
    agent = _make_agent()
    tool_call = _mock_tool_call(name="mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT", call_id="c-blocked")
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    with (
        patch(
            "hermes_cli.plugins.resolve_pre_tool_block",
            return_value="connector approval is required",
        ),
        patch("run_agent.handle_function_call") as dispatch,
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    dispatch.assert_not_called()
    assert messages[0]["effect_disposition"] == "none"


def test_execute_tool_calls_sequential_marks_approval_denial_as_no_effect():
    agent = _make_agent()
    tool_call = _mock_tool_call(name="mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT", call_id="c-denied-seq")
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    with (
        patch(
            "tools.tool_approval.maybe_require_tool_approval",
            return_value=ToolApprovalDenial(
                '{"status":"approval_denied","error":"It was NOT performed."}'
            ),
        ),
        patch("model_tools.registry.dispatch") as dispatch,
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    dispatch.assert_not_called()
    assert messages[0]["effect_disposition"] == "none"


def test_execute_tool_calls_concurrent_marks_approval_denial_as_no_effect():
    agent = _make_agent()
    tool_call = _mock_tool_call(name="mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT", call_id="c-denied")
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    with (
        patch(
            "tools.tool_approval.maybe_require_tool_approval",
            return_value=ToolApprovalDenial(
                '{"status":"approval_denied","error":"It was NOT performed."}'
            ),
        ),
        patch("model_tools.registry.dispatch") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    dispatch.assert_not_called()
    assert messages[0]["effect_disposition"] == "none"


def test_execute_tool_calls_concurrent_does_not_trust_provider_approval_like_content():
    agent = _make_agent()
    tool_call = _mock_tool_call(name="mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT", call_id="c-provider")
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    with patch.object(
        agent,
        "_invoke_tool",
        return_value='{"status":"approval_denied","error":"It was NOT performed."}',
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    assert "effect_disposition" not in messages[0]


def test_continuation_reuses_answered_tool_tail_without_creating_a_user_turn():
    """A late HITL answer resumes from the durable tool result, not a new user row."""
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Thanks — I'll use Brand A."
    )
    db = SessionDB(Path(tempfile.mkdtemp(prefix="hermes-resume-db-")) / "state.db")
    prior_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "input-1",
                "type": "function",
                "function": {
                    "name": "request_user_input",
                    "arguments": '{"question":"Which brand?"}',
                },
            }
        ],
    }
    answered_tool = {
        "role": "tool",
        "tool_call_id": "input-1",
        "name": "request_user_input",
        "content": '{"status":"answered","response":"Brand A"}',
    }
    db.create_session("resume-session", "api_server")
    db.append_message("resume-session", "assistant", "", tool_calls=prior_assistant["tool_calls"])
    db.append_message(
        "resume-session", "tool", answered_tool["content"],
        tool_name="request_user_input", tool_call_id="input-1",
    )
    agent.session_id = "resume-session"
    agent._session_db = db
    agent._session_db_created = True
    agent._last_flushed_db_idx = 2
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = "resume-session"

    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            None,
            conversation_history=db.get_messages_as_conversation("resume-session"),
            continuation=True,
        )

    provider_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert [message["role"] for message in provider_messages[-2:]] == ["assistant", "tool"]
    assert provider_messages[-1]["content"] == answered_tool["content"]
    assert all(message.get("role") != "user" for message in result["messages"])
    assert result["final_response"] == "Thanks — I'll use Brand A."
    assert [message["role"] for message in db.get_messages("resume-session")] == [
        "assistant", "tool", "assistant",
    ]
    db.close()


def test_dangling_approval_continuation_consumes_once_grant_and_persists_real_result(
    monkeypatch,
):
    """A granted dangling call runs once without re-emitting an approval card."""
    agent = _make_agent()
    tool_name = "mcp_connectors_TEST_WRITE"
    function_args = {"target": "Brand A"}
    bridge_args = {"name": tool_name, "arguments": function_args}
    call = {
        "id": "approval-1",
        "type": "function",
        "function": {
            "name": "tool_call",
            "arguments": '{"name":"mcp_connectors_TEST_WRITE","arguments":{"target":"Brand A"}}',
        },
    }
    sibling_call = {
        "id": "read-1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
    }
    db = SessionDB(Path(tempfile.mkdtemp(prefix="hermes-approval-db-")) / "state.db")
    db.create_session("approval-session", "api_server")
    db.append_message(
        "approval-session",
        "assistant",
        "",
        tool_calls=[call, sibling_call],
        display_metadata={
            "_omnio_resolved_approvals": {
                "approval-1": {
                    "scope": "once",
                    "tool_name": "tool_call",
                    "arguments": json.dumps(bridge_args, separators=(",", ":")),
                }
            }
        },
    )
    db.append_message(
        "approval-session",
        "tool",
        '{"content":"existing sibling"}',
        tool_name="read_file",
        tool_call_id="read-1",
    )
    agent.session_id = "approval-session"
    agent._session_db = db
    agent._session_db_created = True
    agent._last_flushed_db_idx = 2
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = "approval-session"
    agent.client.chat.completions.create.return_value = _mock_response("completed")
    monkeypatch.setattr(tool_approval, "is_gated_tool", lambda name: name == tool_name)
    monkeypatch.setattr(
        "tools.tool_search.resolve_underlying_call",
        lambda args: (tool_name, args["arguments"], None),
    )
    approval_wait = MagicMock()
    monkeypatch.setattr(tool_approval, "await_tool_approval", approval_wait)
    # Simulate a fresh gateway process: only SessionDB survives.
    tool_approval.clear_session("approval-session")

    def _execute(assistant_message, messages, effective_task_id):
        assert len(assistant_message.tool_calls) == 1
        pending = assistant_message.tool_calls[0]
        assert pending.id == "approval-1"
        assert tool_approval.maybe_require_tool_approval(
            tool_name,
            pending.id,
            function_args,
        ) is None
        messages.append(make_tool_result_message(tool_name, '{"ok":true}', pending.id))
        agent._flush_messages_to_session_db(messages, [])

    token = tool_approval.set_current_tool_approval_session_key("approval-session")
    try:
        with (
            patch.object(agent, "_execute_tool_calls", side_effect=_execute),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation(
                None,
                conversation_history=db.get_messages_as_conversation("approval-session"),
                continuation=True,
            )
    finally:
        tool_approval.reset_current_tool_approval_session_key(token)
        tool_approval.clear_session("approval-session")

    assert approval_wait.call_count == 0
    rows = db.get_messages("approval-session")
    assert [row["role"] for row in rows] == [
        "assistant", "tool", "tool", "assistant",
    ]
    assert rows[1]["tool_call_id"] == "read-1"
    assert rows[2]["content"] == '{"ok":true}'
    db.close()


def test_denied_dangling_approval_continuation_never_redispatches_or_adds_user():
    """A durable late Deny resumes from its result without asking again."""
    agent = _make_agent()
    tool_name = "mcp_connectors_TEST_WRITE"
    call = {
        "id": "approval-denied",
        "type": "function",
        "function": {"name": tool_name, "arguments": '{"target":"Brand A"}'},
    }
    denial = tool_approval._denial_result("deny")
    db = SessionDB(Path(tempfile.mkdtemp(prefix="hermes-denial-db-")) / "state.db")
    db.create_session("approval-denied-session", "api_server")
    db.append_message(
        "approval-denied-session",
        "assistant",
        "",
        tool_calls=[call],
        display_metadata={
            "_omnio_resolved_approvals": {
                "approval-denied": {
                    "scope": "deny",
                    "tool_name": tool_name,
                    "arguments": '{"target":"Brand A"}',
                }
            }
        },
    )
    db.append_message(
        "approval-denied-session",
        "tool",
        denial,
        tool_name=tool_name,
        tool_call_id="approval-denied",
        effect_disposition="none",
    )
    agent.session_id = "approval-denied-session"
    agent._session_db = db
    agent._session_db_created = True
    agent._last_flushed_db_idx = 2
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = "approval-denied-session"
    agent.client.chat.completions.create.return_value = _mock_response("denied")

    with (
        patch.object(agent, "_execute_tool_calls") as execute,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            None,
            conversation_history=db.get_messages_as_conversation(
                "approval-denied-session"
            ),
            continuation=True,
        )

    execute.assert_not_called()
    assert all(message.get("role") != "user" for message in result["messages"])
    assert result["final_response"] == "denied"
    assert [
        message["role"] for message in db.get_messages("approval-denied-session")
    ] == [
        "assistant",
        "tool",
        "assistant",
    ]
    db.close()


def test_normal_turn_normalizes_dangling_tail_only_in_provider_copy():
    """A later user turn sees a provider-valid pending result without DB repair."""
    agent = _make_agent()
    db = SessionDB(Path(tempfile.mkdtemp(prefix="hermes-pending-db-")) / "state.db")
    dangling_call = {
        "id": "pending-1",
        "type": "function",
        "function": {
            "name": "request_user_input",
            "arguments": '{"question":"Which brand?"}',
        },
    }
    db.create_session("pending-session", "api_server")
    db.append_message("pending-session", "assistant", "", tool_calls=[dangling_call])
    before = db.get_messages("pending-session")
    agent.client.chat.completions.create.return_value = _mock_response("new answer")

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(
            "Please continue with something else",
            conversation_history=db.get_messages_as_conversation("pending-session"),
        )

    provider_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    pending = next(
        message
        for message in provider_messages
        if message.get("tool_call_id") == "pending-1"
    )
    assert pending["name"] == "request_user_input"
    assert pending["content"] == (
        '{"status": "pending", "note": "The user has not answered this request yet; '
        'it is still open. Do not assume an answer."}'
    )
    assert db.get_messages("pending-session") == before
    db.close()
