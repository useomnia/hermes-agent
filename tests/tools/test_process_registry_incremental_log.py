"""Run the non-local poller against a real shell and byte-oriented log files."""

import os
import shlex
import shutil
import subprocess
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry
from tools.process_registry_env_log import log_delta_command
from tests.tools.test_process_registry import _make_session

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="requires a POSIX shell")


def _read(path, offset):
    result = subprocess.run(["sh", "-c", log_delta_command(shlex.quote(str(path)), offset)],
                            capture_output=True, check=True, timeout=5)
    assert not result.stderr
    header, _, data = result.stdout.partition(b"\n")
    size, used = map(int, header.split())
    # Decode independently, exactly as a remote execute() implementation does.
    return size, used, data.decode("utf-8")


@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_every_utf8_write_boundary_is_lossless(tmp_path, shell):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} unavailable")
    path = tmp_path / "log with spaces and 'quotes'"
    source = "start é日本語 🙂 end\n".encode()
    offset = 0
    received = ""
    for end in range(len(source) + 1):
        path.write_bytes(source[:end])
        result = subprocess.run([shell, "-c", log_delta_command(shlex.quote(str(path)), offset)],
                                capture_output=True, check=True, timeout=5)
        header, _, data = result.stdout.partition(b"\n")
        size, used = map(int, header.split())
        assert used == offset
        received += data.decode("utf-8")
        assert size - used == len(data)
        assert source.startswith(received.encode())
        offset = size
    assert received.encode() == source
    assert _read(path, offset) == (len(source), len(source), "")


def test_rotation_and_missing_log_reset_the_offset(tmp_path):
    path = tmp_path / "job.log"
    assert _read(path, 200) == (0, 0, "")
    path.write_text("previous long output", encoding="utf-8")
    size, _, text = _read(path, 0)
    assert text == "previous long output"
    path.write_text("new é", encoding="utf-8")
    assert _read(path, size) == (6, 0, "new é")


class _ShellEnv:
    """Real shell I/O, with deterministic writes between poll iterations."""

    def __init__(self, tmp_path, actions):
        self.log = tmp_path / "job log 'quoted'.log"
        self.pid = tmp_path / "job.pid"
        self.exit = tmp_path / "job.exit"
        self.pid.write_text(str(os.getpid()), encoding="utf-8")
        self.actions = iter(actions)
        self.reads = []
        self.polls = 0
        self.bad_log_read = False

    def execute(self, command, **kwargs):
        if command == f"cat {shlex.quote(str(self.exit))} 2>/dev/null":
            self.polls += 1
            assert self.polls < 12, "poller did not make progress"
            next(self.actions)(self)
        is_log_read = command.startswith("O=")
        if is_log_read and self.bad_log_read:
            self.bad_log_read = False
            return {"output": "backend returned an unusable header"}
        result = subprocess.run(["sh", "-c", command], capture_output=True, timeout=kwargs["timeout"])
        if is_log_read:
            self.reads.append((command, result.stdout))
        return {"output": result.stdout.decode("utf-8"), "exit_code": result.returncode}


def _run_poll(env, session, registry):
    registry._running[session.id] = session
    with patch("tools.process_registry.time.sleep"), patch.object(registry, "_write_checkpoint"):
        registry._env_poller_loop(session, env, str(env.log), str(env.pid), str(env.exit))


def test_poller_sends_only_new_bytes_and_delivers_one_completion(tmp_path):
    def first(env):
        env.log.write_text("hé", encoding="utf-8")

    def append(env):
        with env.log.open("ab") as output:
            output.write("llo 日本語\n".encode())
        env.exit.write_text("7\n", encoding="utf-8")

    env = _ShellEnv(tmp_path, [first, lambda env: None, append])
    registry = ProcessRegistry()
    session = _make_session()
    session.notify_on_complete = True
    received = []
    registry.on_output = lambda _session, text: received.append(text)
    _run_poll(env, session, registry)
    assert received == ["hé", "llo 日本語\n"]
    assert session.output_buffer == "héllo 日本語\n"
    assert env.reads[1][0].startswith("O=3;")
    assert env.reads[1][1] == b"3 3\n"
    assert session.exit_code == 7 and session.completion_reason == "exited"
    event = registry.completion_queue.get_nowait()
    assert event["output"] == session.output_buffer and event["exit_code"] == 7
    registry._move_to_finished(session)
    assert registry.completion_queue.empty()


def test_poller_keeps_offset_through_bad_reads_and_backend_failure(tmp_path):
    def first(env):
        env.log.write_text("before ", encoding="utf-8")

    def transient(env):
        raise ConnectionError("temporary disconnect")

    def malformed(env):
        env.bad_log_read = True
        env.log.write_text("before after", encoding="utf-8")

    def finish(env):
        env.exit.write_text("0", encoding="utf-8")

    env = _ShellEnv(tmp_path, [first, transient, malformed, finish])
    registry = ProcessRegistry()
    session = _make_session()
    received = []
    registry.on_output = lambda _session, text: received.append(text)
    _run_poll(env, session, registry)
    assert received == ["before ", "after"]
    assert env.reads[-1][0].startswith("O=7;")
    assert session.output_buffer == "before after" and session.exit_code == 0


def test_poller_truncation_resets_buffer_and_still_bounds_output(tmp_path):
    def first(env):
        env.log.write_text("old output is much longer", encoding="utf-8")

    def rotate(env):
        env.log.write_text("fresh é", encoding="utf-8")
        env.exit.write_text("0", encoding="utf-8")

    env = _ShellEnv(tmp_path, [first, rotate])
    registry = ProcessRegistry()
    session = _make_session()
    session.max_output_chars = 5
    received = []
    registry.on_output = lambda _session, text: received.append(text)
    _run_poll(env, session, registry)
    assert received == ["old output is much longer", "fresh é"]
    assert session.output_buffer == "esh é"


def test_durable_exit_survives_a_failed_final_log_read(tmp_path):
    def finish(env):
        env.exit.write_text("9", encoding="utf-8")

    env = _ShellEnv(tmp_path, [finish])
    execute = env.execute

    def fail_log(command, **kwargs):
        if command.startswith("O="):
            raise ConnectionError("log read failed after exit marker")
        return execute(command, **kwargs)

    registry = ProcessRegistry()
    session = _make_session(output="already read")
    with patch.object(env, "execute", side_effect=fail_log):
        _run_poll(env, session, registry)
    assert session.exit_code == 9
    assert session.output_buffer == "already read" and session.completion_reason == "exited"
