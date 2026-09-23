"""Real RPC dispatch must not redirect or close another thread's output.

Socket coverage builds on NousResearch/hermes-agent PR #73012. Events force
two overlapping handlers to exit in entry order, reproducing the stream race.
The remote transport exercises actual files and shell commands in a temp dir.
"""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from tools.code_execution_tool import _rpc_poll_loop, _rpc_server_loop


class _OneShotListener:
    def __init__(self, connection):
        self.connection = connection

    def settimeout(self, _timeout):
        pass

    def accept(self):
        if self.connection is None:
            raise socket.timeout()
        connection, self.connection = self.connection, None
        return connection, ("peer", 0)


class _FileEnvironment:
    def __init__(self, stop):
        self.stop = stop

    def execute(self, command, cwd="/", timeout=10):
        result = subprocess.run(
            command, shell=True, cwd=cwd, timeout=timeout,
            capture_output=True, text=True,
        )
        if command.startswith("rm -f "):
            self.stop.set()
        return {"output": result.stdout, "returncode": result.returncode}

    def write_file_content(self, path, content):
        Path(path).write_text(content, encoding="utf-8")
        return True


def _dispatch(transport, directory, task_id, done):
    request = {"tool": "terminal", "args": {"command": "echo hello"}, "token": "tok", "seq": 1}
    counter, call_log = [0], []
    stop = threading.Event()
    options = dict(
        task_id=task_id, tool_call_log=call_log, tool_call_counter=counter,
        max_tool_calls=10, allowed_tools=frozenset({"terminal"}),
        stop_event=stop, rpc_token="tok",
    )
    try:
        if transport == "socket":
            server, client = socket.socketpair()
            with server, client:
                client.settimeout(10)
                client.sendall((json.dumps(request) + "\n").encode())
                client.shutdown(socket.SHUT_WR)
                _rpc_server_loop(_OneShotListener(server), **options)
                reply = json.loads(client.recv(65536))
        else:
            directory.mkdir()
            (directory / "req_000001").write_text(json.dumps(request))
            _rpc_poll_loop(_FileEnvironment(stop), str(directory), **options)
            reply = json.loads((directory / "res_000001").read_text())
        assert counter == [1]
        assert len(call_log) == 1
        return reply
    finally:
        done.set()


@pytest.mark.parametrize("transport", ["socket", "remote"])
@pytest.mark.parametrize("handler_fails", [False, True])
def test_overlapping_dispatch_preserves_other_threads_output(
    transport, handler_fails, tmp_path, monkeypatch,
):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    done = [threading.Event(), threading.Event()]
    stdout, stderr = io.StringIO(), io.StringIO()

    def handler(_name, _args, *, task_id):
        index = int(task_id)
        print("handler chatter")
        print("handler chatter", file=sys.stderr)
        entered[index].set()
        assert release[index].wait(10), "test did not release handler"
        if handler_fails:
            raise RuntimeError("handler failed")
        return json.dumps({"output": "handled"})

    monkeypatch.setattr("model_tools.handle_function_call", handler)
    with monkeypatch.context() as capture:
        capture.setattr(sys, "stdout", stdout)
        capture.setattr(sys, "stderr", stderr)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []
            try:
                for index in range(2):
                    futures.append(pool.submit(
                        _dispatch, transport, tmp_path / str(index), str(index), done[index],
                    ))
                    assert entered[index].wait(10), "dispatch did not reach handler"

                print("unrelated output during dispatch")
                print("unrelated output during dispatch", file=sys.stderr)
                # A exits while B still holds the stream that A used to own.
                release[0].set()
                assert done[0].wait(10)
                release[1].set()
                replies = [future.result(timeout=10) for future in futures]
            finally:
                for event in release:
                    event.set()

        print("output after dispatch")
        print("output after dispatch", file=sys.stderr)

    for reply in replies:
        if handler_fails:
            assert "handler failed" in reply["error"]
        else:
            assert reply["output"] == "handled"
    for stream in (stdout, stderr):
        assert "unrelated output during dispatch" in stream.getvalue()
        assert "output after dispatch" in stream.getvalue()
        assert "handler chatter" not in stream.getvalue()
