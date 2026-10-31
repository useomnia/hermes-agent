"""Deadline, Stop and checkpoint recovery through real script processes."""

import json
import os
import socket
import threading
import time

import psutil
import pytest

import tools.code_execution_tool as execution
from tools.environments.local import LocalEnvironment
from tools.interrupt import ToolExecutionScope, bind_execution_scope, current_execution_scope, set_interrupt


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 1, "max_timeout": 1200, "mode": "strict"})
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


def run(transport, code, **kwargs):
    runner = execution.execute_code if transport == "local" else execution._execute_remote
    return json.loads(runner(code, "runtime-test", ["web_search"], **kwargs))


@pytest.mark.parametrize("transport", ["local", "files"])
@pytest.mark.parametrize("requested", [0, -1, 1.5, True, "600", float("inf"), float("nan"), 1201, 10**1000])
def test_invalid_allowance_starts_no_script(runtime, tmp_path, transport, requested):
    marker = tmp_path / "must-not-exist"
    result = run(transport, f"from pathlib import Path; Path({str(marker)!r}).touch()", timeout=requested)
    assert "timeout" in result["error"] and "1200" in result["error"], result
    assert not marker.exists()
    assert runtime == []


@pytest.mark.parametrize("transport", ["local", "files"])
def test_requested_allowance_can_extend_the_default(runtime, transport):
    result = run(transport, 'import time; print("started"); time.sleep(1.2); print("completed")', timeout=5)
    assert result["status"] == "success", result
    assert "completed" in result["output"]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_requested_allowance_can_shorten_the_default(runtime, monkeypatch, transport):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 30, "mode": "strict"})
    result = run(transport, 'import time; print("before requested deadline"); time.sleep(999)', timeout=1)
    assert result["status"] == "timeout", result
    assert result["termination_reason"] == "timeout"
    assert "before requested deadline" in result["output"]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_maximum_allowance_is_accepted(runtime, transport):
    result = run(transport, 'print("accepted maximum")', timeout=1200)
    assert result["status"] == "success", result
    assert "accepted maximum" in result["output"]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_legacy_host_ceiling_rejects_unadvertised_extensions(runtime, monkeypatch, tmp_path, transport):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 900})
    marker = tmp_path / "must-not-run"
    result = run(transport, f"from pathlib import Path; Path({str(marker)!r}).touch()", timeout=901)
    assert "900" in result["error"], result
    assert not marker.exists()


@pytest.mark.parametrize("requested, config, expected", [
    (None, {}, 300), (None, {"timeout": 450}, 450), (None, {"timeout": 5000}, 1200),
    (None, {"timeout": 0}, 300), (None, {"timeout": "bad"}, 300),
    (None, {"timeout": True}, 300), (None, {"timeout": float("nan")}, 300),
    (600, {"timeout": 1, "max_timeout": 1200}, 600), (1200, {"timeout": 1, "max_timeout": 1200}, 1200),
    (600.0, {"timeout": 1, "max_timeout": 1200}, 600),
])
def test_registered_tool_forwards_the_requested_allowance(runtime, monkeypatch, requested, config, expected):
    from tools.registry import registry

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setattr(execution, "_load_config", lambda: config)
    seen = []

    def execute(command, **kwargs):
        if "python3 -u" in command:
            seen.append(kwargs["timeout"])
        return {"output": "OK", "returncode": 0}

    env, _ = execution._get_or_create_env("runtime-test")
    monkeypatch.setattr(env, "execute", execute)
    args = {"code": "print('ok')"}
    if requested is not None:
        args["timeout"] = requested
    result = registry.dispatch("execute_code", args, enabled_tools=[])
    assert json.loads(result)["status"] == "success", result
    assert seen == [expected]


@pytest.mark.parametrize("transport", ["local", "files"])
def test_printed_progress_does_not_renew_the_deadline(runtime, transport):
    result = run(transport, 'import time\nwhile True:\n print("progress")\n time.sleep(0.02)', timeout=1)
    assert result["status"] == "timeout", result
    assert "progress" in result["output"]
    assert result["duration_seconds"] < 10


@pytest.mark.skipif(os.name == "nt", reason="Requires Unix domain sockets")
@pytest.mark.parametrize("transport", ["uds", "stream", "file"])
def test_rpc_wait_uses_the_script_allowance(monkeypatch, tmp_path, transport):
    # Keep the peer connected without returning a result; the generated client
    # must use its allowance rather than a separate fixed five-minute wait.
    endpoint = f"/tmp/code-timeout-{os.getpid()}-{transport}.sock"
    done = threading.Event()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(endpoint)
    server.listen(1)
    server.settimeout(3)
    monkeypatch.setenv("HERMES_RPC_SOCKET", endpoint)
    monkeypatch.setenv("HERMES_RPC_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_RPC_TOKEN", "test")

    def hold():
        if transport != "file":
            with server.accept()[0] as client:
                done.wait(5)

    peer = threading.Thread(target=hold)
    peer.start()
    namespace = {}
    started = time.monotonic()
    try:
        exec(execution.generate_hermes_tools_module(["web_search"], transport=transport, timeout=0.05), namespace)
        with pytest.raises((TimeoutError, RuntimeError)):
            namespace["web_search"](query="blocked")
        assert time.monotonic() - started < 2
    finally:
        done.set()
        peer.join(timeout=4)
        if namespace.get("_sock") is not None:
            namespace["_sock"].close()
        server.close()
        os.unlink(endpoint)
    assert not peer.is_alive()


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


@pytest.mark.parametrize("exit_code", [0, 1, 124, 130])
@pytest.mark.parametrize("reason", ["timeout", "interrupted"])
def test_backend_exit_does_not_replace_recorded_cause(runtime, monkeypatch, exit_code, reason):
    env, _ = execution._get_or_create_env("runtime-test")
    execute = env.execute

    def finish_with_cause(command, **kwargs):
        if "python3 -u" in command:
            scope = current_execution_scope()
            assert scope is not None
            if reason == "timeout":
                scope.start(0)
            else:
                set_interrupt(True)
            assert scope.termination_reason() == reason
            return {"output": "retained progress", "returncode": exit_code}
        return execute(command, **kwargs)

    monkeypatch.setattr(env, "execute", finish_with_cause)
    result = run("files", "print('not executed by the backend boundary fixture')")
    assert result["status"] == reason, result
    assert result["termination_reason"] == reason
    assert "retained progress" in result["output"]
    assert result["tool_calls_made"] == 0


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


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX signal handling")
def test_local_deadline_kills_a_script_that_ignores_sigterm(runtime, tmp_path):
    pid_file = tmp_path / "script.pid"
    process = None
    try:
        result = run("local", f"""
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(pid_file)!r}).write_text(str(os.getpid()))
print("saved progress")
time.sleep(999)
""")
        try:
            process = psutil.Process(int(pid_file.read_text()))
        except psutil.NoSuchProcess:
            pass
        else:
            assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
        assert result["status"] == "timeout", result
        assert result["termination_reason"] == "timeout"
        assert "saved progress" in result["output"]
    finally:
        if process is not None and process.is_running():
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("transport", ["local", "files"])
def test_enclosing_deadline_is_not_an_interrupt(runtime, monkeypatch, transport):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 30, "max_timeout": 1200, "mode": "strict"})
    parent = ToolExecutionScope(threading.Event())
    parent.start(1)
    with bind_execution_scope(parent):
        result = run(transport, 'import time; print("before deadline"); time.sleep(999)', timeout=1200)
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
