"""Exercise real response files and shell commands under the Toolbox limit."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.code_execution_tool import _rpc_poll_loop, _ship_file_to_remote


class BoundedShell:
    def __init__(self, *, publish_failures=0, unlink_failures=0, fail_chunks=False):
        self.publish_failures = publish_failures
        self.unlink_failures = unlink_failures
        self.fail_chunks = fail_chunks
        self.largest_command = 0

    def execute(self, command, **kwargs):
        self.largest_command = max(self.largest_command, len(command))
        if len(command) > 100_000:
            raise ValueError("Toolbox command limit exceeded")
        if command.startswith("mv ") and self.publish_failures:
            self.publish_failures -= 1
            raise OSError("publication unavailable")
        if command.startswith("rm -f ") and self.unlink_failures:
            self.unlink_failures -= 1
            raise OSError("unlink unavailable")
        if command.startswith("printf ") and self.fail_chunks:
            return {"returncode": 1, "output": "write failed"}
        completed = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True,
            timeout=kwargs.get("timeout", 10),
        )
        return {"returncode": completed.returncode, "output": completed.stdout}


class BoundedPoll:
    """Bound the test even when the original implementation replays forever."""
    def __init__(self, request):
        self.request = request
        self.polls = 0

    def is_set(self):
        return self.polls >= 8 or not self.request.exists()

    def wait(self, seconds):
        self.polls += 1


def run_request(tmp_path, monkeypatch, env, result):
    request = tmp_path / "req_000001"
    request.write_text(json.dumps({"tool": "example", "args": {}, "seq": 1, "token": "test"}))
    calls = []

    def dispatch(*args, **kwargs):
        calls.append(args)
        return json.dumps(result, ensure_ascii=False)

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    errors = []
    _rpc_poll_loop(env, str(tmp_path), "test", [], [0], 150,
                   frozenset({"example"}), BoundedPoll(request), "test", errors)
    response = tmp_path / "res_000001"
    return calls, json.loads(response.read_text()) if response.exists() else None, errors


def test_large_response_arrives_completely_after_one_dispatch(tmp_path, monkeypatch):
    env = BoundedShell()
    expected = {"result": "ñ" * 85_000 + "FINAL RECORD"}
    calls, response, errors = run_request(tmp_path, monkeypatch, env, expected)
    assert response == expected
    assert len(calls) == 1
    assert env.largest_command < 100_000
    assert errors == []


@pytest.mark.parametrize("fault", ["publish_failures", "unlink_failures"])
def test_delivery_retry_never_reexecutes_the_tool(tmp_path, monkeypatch, fault):
    calls, response, errors = run_request(
        tmp_path, monkeypatch, BoundedShell(**{fault: 1}), {"result": "complete"}
    )
    assert len(calls) == 1
    assert response == {"result": "complete"}
    assert not errors


def test_exhausted_delivery_returns_explicit_transport_failure(tmp_path, monkeypatch):
    calls, response, errors = run_request(
        tmp_path, monkeypatch, BoundedShell(publish_failures=3), {"result": "complete"}
    )
    assert len(calls) == 1
    assert response.get("_rpc_transport_error")
    assert errors == [response["_rpc_transport_error"]]


def test_nonzero_chunk_write_cannot_publish_a_partial_file(tmp_path):
    with pytest.raises(RuntimeError, match="Remote file operation failed"):
        _ship_file_to_remote(BoundedShell(fail_chunks=True), str(tmp_path / "response"), "data")
    assert not (tmp_path / "response").exists()


def test_empty_file_is_shipped_without_a_missing_base64_file(tmp_path):
    target = tmp_path / "empty"
    _ship_file_to_remote(BoundedShell(), str(target), "")
    assert target.read_bytes() == b""


def test_first_class_file_transport_is_used(tmp_path):
    class FileEnvironment:
        def write_file_content(self, path, content):
            Path(path).write_text(content)
            return True

    target = tmp_path / "response"
    _ship_file_to_remote(FileEnvironment(), str(target), "a" * 150_000)
    assert target.read_text() == "a" * 150_000


def test_permanent_delivery_failure_stops_polling_without_replay(tmp_path, monkeypatch):
    calls, response, errors = run_request(
        tmp_path, monkeypatch, BoundedShell(publish_failures=100), {"result": "complete"}
    )
    assert len(calls) == 1
    assert response is None
    assert len(errors) == 1
