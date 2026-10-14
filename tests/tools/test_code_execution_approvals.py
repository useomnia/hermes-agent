"""Approval behavior through generated scripts and both real RPC protocols.

The registered MCP handler is the external boundary: its recorded effects prove
whether dispatch happened. Scripts, sockets/files, gate, and dispatcher are real.
"""

import json
import threading

import pytest

import tools.code_execution_tool as execution
import tools.mcp_tool as mcp
import tools.tool_approval as approvals
from model_tools import handle_function_call
from tools.approval import reset_current_session_key, set_current_session_key
from tools.environments.local import LocalEnvironment
from tools.interrupt import ToolExecutionScope, bind_execution_scope, set_interrupt
from tools.registry import registry

WRITE = "mcp__connectors__APPROVAL_TEST_WRITE"
READ = "mcp__connectors__APPROVAL_TEST_READ"
SESSION = "nested-approval-test"


@pytest.fixture
def setup(monkeypatch, tmp_path):
    token = set_current_session_key(SESSION)
    approvals.clear_session(SESSION)
    monkeypatch.setattr(approvals, "_DISABLED_FROZEN", False)
    monkeypatch.setattr(approvals, "is_always_approved", lambda _name: False)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("OMNIO_TOOL_APPROVAL_TIMEOUT", "10")
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 10, "mode": "strict"})
    monkeypatch.setattr("tools.approval.check_execute_code_guard", lambda *_a, **_kw: {"approved": True})
    env = LocalEnvironment(cwd=str(tmp_path))
    monkeypatch.setattr(execution, "_get_or_create_env", lambda _task: (env, "ssh"))
    effects = []
    for tool in (WRITE, READ):
        registry.register(
            name=tool, toolset="mcp-connectors", description="Approval test boundary",
            schema={"name": tool, "description": "Approval test boundary", "parameters": {
                "type": "object", "properties": {"value": {"type": "integer"}},
            }},
            handler=lambda args, **_kw: effects.append(args.get("value")) or json.dumps({"ok": True}),
        )
        mcp._track_mcp_tool_read_only(tool, tool == READ)
    yield effects
    for tool in (WRITE, READ):
        registry.deregister(tool)
        mcp._mcp_tool_read_only_hints.pop(tool, None)
    approvals.clear_session(SESSION)
    reset_current_session_key(token)


def run_script(transport, code):
    run = execution.execute_code if transport == "socket" else execution._execute_remote
    return json.loads(run(code, "nested-approval-task", [WRITE, READ]))


def record_events(events, choice=None, before_answer=None):
    def notify(event):
        events.append(event)
        if event.get("status") != "running":
            return
        if before_answer is not None:
            before_answer()
        if choice is not None:
            assert approvals.resolve_tool_approval(
                SESSION, WRITE, choice, event["toolCallId"], surface_key=SESSION,
            )
    approvals.register_tool_approval_notify(SESSION, notify)


@pytest.mark.parametrize("transport", ["socket", "files"])
def test_publishes_distinct_answerable_approvals_for_repeated_calls(setup, transport):
    events = []
    record_events(events, "once")
    result = run_script(transport, f"from hermes_tools import {WRITE}\nfor i in range(2):\n print({WRITE}(value=i))")
    assert result["status"] == "success", result
    assert setup == [0, 1]
    opened = [e for e in events if e["status"] == "running"]
    closed = [e for e in events if e["status"] == "completed"]
    assert len(opened) == len(closed) == 2
    ids = [e["toolCallId"] for e in opened]
    assert all(ids) and len(set(ids)) == 2
    assert [e["toolCallId"] for e in closed] == ids
    assert all(e["interaction"]["answered"] == "once" for e in closed)
    assert approvals._wait_registry.pending_count(SESSION) == 0
    assert not any(key[0] == SESSION for key in approvals._decisions)


@pytest.mark.parametrize("transport", ["socket", "files"])
@pytest.mark.parametrize("choice", ["deny", "skip"])
def test_denial_never_dispatches_the_nested_write(setup, transport, choice):
    events = []
    record_events(events, choice)
    result = run_script(transport, f"from hermes_tools import {WRITE}\nprint({WRITE}(value=1))")
    assert setup == []
    assert f"approval_{'denied' if choice == 'deny' else 'skipped'}" in result["output"]
    assert events[-1]["interaction"]["answered"] == choice


@pytest.mark.parametrize("transport", ["socket", "files"])
def test_read_only_nested_calls_need_no_approval(setup, transport):
    events = []
    record_events(events)
    result = run_script(transport, f"from hermes_tools import {READ}\nprint({READ}(value=3))")
    assert result["status"] == "success", result
    assert setup == [3]
    assert events == []


@pytest.mark.parametrize("transport", ["socket", "files"])
def test_approval_expiry_closes_the_nested_card(setup, transport, monkeypatch):
    monkeypatch.setenv("OMNIO_TOOL_APPROVAL_TIMEOUT", "0")
    events = []
    record_events(events)
    result = run_script(transport, f"from hermes_tools import {WRITE}\nprint({WRITE}())")
    assert setup == []
    assert "approval_no_response" in result["output"]
    assert events[-1]["completed"] is True
    assert events[-1]["interaction"]["timed_out"] is True


@pytest.mark.parametrize("transport", ["socket", "files"])
def test_script_timeout_releases_waiter_and_rejects_late_answer(setup, transport, monkeypatch):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 2, "mode": "strict"})
    events = []
    record_events(events)
    result = run_script(transport, f"from hermes_tools import {WRITE}\nprint({WRITE}())")
    assert events and events[0]["status"] == "running", result
    assert events[-1]["completed"] is True
    assert events[-1]["interaction"]["timed_out"] is False
    assert approvals._wait_registry.pending_count(SESSION) == 0
    assert approvals.resolve_tool_approval(SESSION, WRITE, "once", events[0]["toolCallId"], surface_key=SESSION) is False
    assert setup == []


def test_cancellation_after_approval_cannot_admit_the_write(setup):
    scope = ToolExecutionScope(threading.Event())
    events = []

    def notify(event):
        events.append(event)
        if event.get("status") == "running":
            assert approvals.resolve_tool_approval(SESSION, WRITE, "once", event["toolCallId"])
            scope.cancel()

    approvals.register_tool_approval_notify(SESSION, notify)
    with bind_execution_scope(scope):
        result = json.loads(handle_function_call(WRITE, {}, tool_call_id="nested-race"))
    assert result["status"] == "interrupted"
    assert setup == []
    assert events[-1]["completed"] is True


def test_completion_uses_the_surface_that_published_the_card(setup, monkeypatch):
    old_events, replacement_events = [], []
    approvals.register_tool_approval_notify(SESSION, old_events.append)
    wait = approvals._wait_registry.wait

    def replace_before_parking(*args, **kwargs):
        record_events(replacement_events, "once")
        return wait(*args, **kwargs)

    monkeypatch.setattr(approvals._wait_registry, "wait", replace_before_parking)
    result = json.loads(handle_function_call(WRITE, {"value": 4}, tool_call_id="replacement"))
    assert result == {"ok": True}
    assert setup == [4]
    assert old_events == []
    assert [event["status"] for event in replacement_events] == ["running", "completed"]


@pytest.mark.parametrize("transport", ["socket", "files"])
def test_parent_interrupt_releases_the_nested_worker(setup, transport):
    owner = threading.get_ident()
    events = []
    record_events(events, before_answer=lambda: set_interrupt(True, thread_id=owner))
    try:
        run_script(transport, f"from hermes_tools import {WRITE}\nprint({WRITE}())")
    finally:
        set_interrupt(False, thread_id=owner)
    assert setup == []
    assert approvals._wait_registry.pending_count(SESSION) == 0
    assert events[-1]["completed"] is True
    assert events[-1]["interaction"]["timed_out"] is False


def test_cancelled_wait_rejects_answer_before_the_worker_wakes(setup):
    scope = ToolExecutionScope(threading.Event())
    events = []

    def notify(event):
        events.append(event)
        if event.get("status") == "running":
            scope.cancel()
            assert approvals.resolve_tool_approval(SESSION, WRITE, "once", event["toolCallId"]) is False

    approvals.register_tool_approval_notify(SESSION, notify)
    with bind_execution_scope(scope):
        result = json.loads(handle_function_call(WRITE, {}, tool_call_id="nested-cancelled"))
    assert result["status"] == "approval_no_response"
    assert setup == []


def test_missing_nested_identity_fails_without_parking(setup):
    events = []
    record_events(events)
    with bind_execution_scope(ToolExecutionScope(threading.Event())):
        result = json.loads(handle_function_call(WRITE, {}))
    assert result["status"] == "approval_error"
    assert setup == events == []


def test_publication_failure_does_not_execute_the_write(setup):
    def fail(_event):
        raise RuntimeError("surface closed")
    approvals.register_tool_approval_notify(SESSION, fail)
    result = json.loads(handle_function_call(WRITE, {}, tool_call_id="nested-unpublished"))
    assert result["status"] == "approval_error"
    assert approvals._wait_registry.pending_count(SESSION) == 0
    assert setup == []


def test_late_session_grant_only_applies_to_future_calls(setup):
    scope = ToolExecutionScope(threading.Event())
    events = []

    def notify(event):
        events.append(event)
        if event.get("status") == "running":
            scope.cancel()
            assert approvals.resolve_tool_approval(
                SESSION, WRITE, "session", event["toolCallId"], surface_key=SESSION,
            ) is False

    approvals.register_tool_approval_notify(SESSION, notify)
    with bind_execution_scope(scope):
        rejected = json.loads(handle_function_call(WRITE, {"value": 1}, tool_call_id="abandoned"))
    assert rejected["status"] == "approval_no_response"
    assert setup == []
    assert json.loads(handle_function_call(WRITE, {"value": 2}, tool_call_id="future"))["ok"] is True
    assert setup == [2]
    assert len(events) == 2
    assert "answered" not in events[-1]["interaction"]


def test_cancelling_one_execution_does_not_release_its_sibling(setup):
    from tools.thread_context import propagate_context_to_thread

    scopes = [ToolExecutionScope(threading.Event()), ToolExecutionScope(threading.Event())]
    parked = [threading.Event(), threading.Event()]
    results = {}

    def notify(event):
        if event.get("status") == "running":
            parked[int(event["toolCallId"])].set()

    def run(index):
        with bind_execution_scope(scopes[index]):
            results[index] = json.loads(handle_function_call(WRITE, {"value": index}, tool_call_id=str(index)))

    approvals.register_tool_approval_notify(SESSION, notify)
    threads = [threading.Thread(target=propagate_context_to_thread(run), args=(i,)) for i in range(2)]
    try:
        for thread in threads:
            thread.start()
        assert all(event.wait(3) for event in parked)
        scopes[0].cancel()
        assert approvals.resolve_tool_approval(SESSION, WRITE, "once", "0", surface_key=SESSION) is False
        assert approvals.resolve_tool_approval(SESSION, WRITE, "once", "1", surface_key=SESSION) is True
        for thread in threads:
            thread.join(3)
        assert all(not thread.is_alive() for thread in threads)
        assert results[0]["status"] == "approval_no_response"
        assert results[1]["ok"] is True
        assert setup == [1]
    finally:
        for scope in scopes:
            scope.cancel()
        for thread in threads:
            thread.join(3)


def test_remote_result_redelivery_does_not_repeat_the_approved_call(setup, monkeypatch):
    ship = execution._ship_file_to_remote
    failed = False

    def fail_first_result(env, path, content):
        nonlocal failed
        if "/res_" in path and not failed:
            failed = True
            raise OSError("response transport unavailable")
        return ship(env, path, content)

    monkeypatch.setattr(execution, "_ship_file_to_remote", fail_first_result)
    events = []
    record_events(events, "once")
    result = run_script("files", f"from hermes_tools import {WRITE}\nprint({WRITE}(value=4))")
    assert result["status"] == "success", result
    assert setup == [4]
    assert len(events) == 2
    assert events[0]["toolCallId"] == events[1]["toolCallId"]
    assert result["inner_tool_calls"] == {WRITE: 1}
