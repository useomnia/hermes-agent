"""Deadline, Stop and checkpoint recovery through real script processes."""

import json
import threading

import pytest

import tools.code_execution_tool as execution
from tools.environments.local import LocalEnvironment
from tools.interrupt import ToolExecutionScope, bind_execution_scope, set_interrupt


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 1, "mode": "strict"})
    monkeypatch.setattr("tools.approval.check_execute_code_guard", lambda *_a, **_kw: {"approved": True})
    env = LocalEnvironment(cwd=str(tmp_path))
    monkeypatch.setattr(execution, "_get_or_create_env", lambda _task: (env, "ssh"))
    calls = []

    def read(name, args, **_kwargs):
        calls.append(args["query"])
        return json.dumps({"value": args["query"]})

    monkeypatch.setattr("model_tools.handle_function_call", read)
    yield calls
    set_interrupt(False)
    env.cleanup()


def run(transport, code):
    runner = execution.execute_code if transport == "local" else execution._execute_remote
    return json.loads(runner(code, "runtime-test", ["web_search"]))


@pytest.mark.parametrize("reason", ["timeout", "interrupted", "cancelled"])
def test_ended_execution_rejects_nested_admission_with_true_cause(runtime, reason):
    scope = ToolExecutionScope(threading.Event())
    if reason == "timeout":
        scope.start(0)
    elif reason == "interrupted":
        set_interrupt(True)
    else:
        scope.cancel()
    log, counter = [], [0]
    dispatcher = execution._NestedToolDispatcher("test", log, counter, 10, {"web_search"}, scope)
    result = json.loads(dispatcher.dispatch("web_search", {"query": "must not start"}))
    assert result["termination_reason"] == reason
    assert result["status"] == ("timeout" if reason == "timeout" else "interrupted")
    assert runtime == []
    assert log == [] and counter == [0]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_timeout_keeps_printed_and_saved_progress_without_replay(runtime, tmp_path, transport):
    checkpoint = tmp_path / "checkpoint.json"
    first = run(transport, f"""
from hermes_tools import web_search
import json, time
from pathlib import Path
result = web_search(query="first")
Path({str(checkpoint)!r}).write_text(json.dumps(result))
print("completed first")
time.sleep(999)
web_search(query="must not start")
""")
    assert first["status"] == "timeout", first
    assert first["termination_reason"] == "timeout"
    assert "completed first" in first["output"]
    assert "user sent" not in first["output"]
    assert json.loads(checkpoint.read_text()) == {"value": "first"}
    assert runtime == ["first"]
    second = run(transport, f"""
from hermes_tools import web_search
import json
from pathlib import Path
assert json.loads(Path({str(checkpoint)!r}).read_text())["value"] == "first"
print(web_search(query="remaining")["value"])
""")
    assert second["status"] == "success", second
    assert "remaining" in second["output"]
    assert runtime == ["first", "remaining"]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_enclosing_deadline_is_not_an_interrupt(runtime, monkeypatch, transport):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 30, "mode": "strict"})
    parent = ToolExecutionScope(threading.Event())
    parent.start(1)
    with bind_execution_scope(parent):
        result = run(transport, 'import time; print("before deadline"); time.sleep(999)')
    assert result["status"] == "timeout", result
    assert result["termination_reason"] == "timeout"
    assert "before deadline" in result["output"]
    assert "user sent" not in result["output"]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_stop_retains_output_and_fences_later_calls(runtime, monkeypatch, tmp_path, transport):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 30, "mode": "strict"})
    ready = tmp_path / "ready"
    owner = threading.get_ident()
    finished = threading.Event()

    def stop_when_ready():
        # The marker is written by the real child after its first result.
        for _ in range(500):
            if ready.exists():
                set_interrupt(True, thread_id=owner)
                return
            if finished.wait(0.01):
                return

    watcher = threading.Thread(target=stop_when_ready)
    watcher.start()
    try:
        result = run(transport, f"""
from hermes_tools import web_search
from pathlib import Path
import time
print(web_search(query="first")["value"])
Path({str(ready)!r}).touch()
time.sleep(999)
web_search(query="must not start")
""")
    finally:
        finished.set()
        watcher.join(timeout=6)
        set_interrupt(False)
    assert not watcher.is_alive()
    assert result["status"] == "interrupted", result
    assert result["termination_reason"] == "interrupted"
    assert "first" in result["output"]
    assert "user sent" not in result["output"]
    assert runtime == ["first"]
