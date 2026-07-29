"""Tests for BaseEnvironment unified execution model.

Tests _wrap_command(), _extract_cwd_from_output(), _embed_stdin_heredoc(),
init_session() failure handling, and the CWD marker contract.
"""

from unittest.mock import MagicMock

from tools.environments.base import BaseEnvironment


class _TestableEnv(BaseEnvironment):
    """Concrete subclass for testing base class methods."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestWrapCommand:
    def test_basic_shape(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" in wrapped
        assert "cd -- /tmp" in wrapped or "cd -- '/tmp'" in wrapped
        assert "eval 'echo hello'" in wrapped
        assert "__hermes_ec=$?" in wrapped
        assert "export -p >" in wrapped
        assert "pwd -P >" in wrapped
        assert env._cwd_marker in wrapped
        assert "exit $__hermes_ec" in wrapped

    def test_no_snapshot_skips_source(self):
        env = _TestableEnv()
        env._snapshot_ready = False
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" not in wrapped

    def test_single_quote_escaping(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo 'hello world'", "/tmp")

        assert "eval 'echo '\\''hello world'\\'''" in wrapped

    def test_tilde_not_quoted(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~")

        assert "cd -- ~" in wrapped
        assert "cd -- '~'" not in wrapped

    def test_tilde_subpath_with_spaces_uses_home_and_quotes_suffix(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/my repo")

        assert "cd -- $HOME/'my repo'" in wrapped
        assert "cd -- ~/my repo" not in wrapped

    def test_tilde_slash_maps_to_home(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/")

        assert "cd -- $HOME" in wrapped
        assert "cd -- ~/" not in wrapped

    def test_hyphen_prefixed_workdir_is_passed_after_double_dash(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("pwd", "-demo")

        assert "builtin cd -- -demo || exit 126" in wrapped

    def test_cd_failure_exit_126(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "/nonexistent")

        assert "exit 126" in wrapped


class TestExtractCwdFromOutput:
    def test_happy_path(self):
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"hello\n{marker}/home/user{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert env.cwd == "/home/user"
        assert marker not in result["output"]

    def test_missing_marker(self):
        env = _TestableEnv()
        result = {"output": "hello world\n"}
        env._extract_cwd_from_output(result)

        assert env.cwd == "/tmp"  # unchanged

    def test_marker_in_command_output(self):
        """If the marker appears in command output AND as the real marker,
        rfind grabs the last (real) one."""
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"user typed {marker} in their output\nreal output\n{marker}/correct/path{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert env.cwd == "/correct/path"

    def test_output_cleaned(self):
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"hello\n{marker}/tmp{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert "hello" in result["output"]
        assert marker not in result["output"]


class TestEmbedStdinHeredoc:
    def test_heredoc_format(self):
        result = BaseEnvironment._embed_stdin_heredoc("cat", "hello world")

        assert result.startswith("cat << '")
        assert "hello world" in result
        assert "HERMES_STDIN_" in result

    def test_unique_delimiter_each_call(self):
        r1 = BaseEnvironment._embed_stdin_heredoc("cat", "data")
        r2 = BaseEnvironment._embed_stdin_heredoc("cat", "data")

        # Extract delimiters
        d1 = r1.split("'")[1]
        d2 = r2.split("'")[1]
        assert d1 != d2  # UUID-based, should be unique


class TestInitSessionFailure:
    def test_snapshot_ready_false_on_failure(self):
        env = _TestableEnv()

        def failing_run_bash(*args, **kwargs):
            raise RuntimeError("bash not found")

        env._run_bash = failing_run_bash
        env.init_session()

        assert env._snapshot_ready is False

    def test_login_flag_when_snapshot_not_ready(self):
        """When _snapshot_ready=False, execute() should pass login=True to _run_bash."""
        env = _TestableEnv()
        env._snapshot_ready = False

        calls = []
        def mock_run_bash(cmd, *, login=False, timeout=120, stdin_data=None):
            calls.append({"login": login})
            # Return a mock process handle
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            mock.stdout = iter([])
            return mock

        env._run_bash = mock_run_bash
        env.execute("echo test")

        assert len(calls) == 1
        assert calls[0]["login"] is True


class TestCwdMarker:
    def test_marker_contains_session_id(self):
        env = _TestableEnv()
        assert env._session_id in env._cwd_marker

    def test_unique_per_instance(self):
        env1 = _TestableEnv()
        env2 = _TestableEnv()
        assert env1._cwd_marker != env2._cwd_marker


class _MissingTempDirEnv(BaseEnvironment):
    """Env whose temp dir does not exist yet (mirrors the Sprites toolbox,
    where a sandbox restart wipes /tmp and the session dir with it)."""

    def __init__(self, temp_dir, cwd="/tmp", timeout=10):
        self._temp_dir = str(temp_dir)
        super().__init__(cwd=cwd, timeout=timeout)

    def get_temp_dir(self):
        return self._temp_dir

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use subprocess directly")

    def cleanup(self):
        pass


class TestSessionTempDirCreation:
    """The session temp dir must be (re)created before snapshot/cwd writes,
    and a failed write must never leak bash redirect errors into output."""

    def test_wrapped_command_recreates_temp_dir(self, tmp_path):
        env = _MissingTempDirEnv(tmp_path / "gone")
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", "/tmp")

        mkdir_line = wrapped.splitlines()[0]
        assert mkdir_line.startswith("mkdir -p ")
        assert str(tmp_path / "gone") in mkdir_line

    def test_missing_temp_dir_no_stderr_leak_and_state_written(self, tmp_path):
        import subprocess

        env = _MissingTempDirEnv(tmp_path / "wiped" / "session")
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", str(tmp_path))

        result = subprocess.run(
            ["bash", "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "hello" in result.stdout
        assert "No such file or directory" not in result.stdout
        assert "No such file or directory" not in result.stderr
        # The dir was recreated and both session files were written.
        assert (tmp_path / "wiped" / "session").is_dir()
        assert env._snapshot_path.startswith(str(tmp_path / "wiped" / "session"))
        import os
        assert os.path.exists(env._snapshot_path)
        assert os.path.exists(env._cwd_file)

    def test_unwritable_temp_dir_stays_silent(self, tmp_path):
        import subprocess

        # A *file* where the temp dir should be: mkdir -p fails, and the
        # snapshot/cwd redirects fail — none of it may reach the output.
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        env = _MissingTempDirEnv(blocker)
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", str(tmp_path))

        result = subprocess.run(
            ["bash", "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "hello" in result.stdout
        assert "No such file or directory" not in result.stdout
        assert "No such file or directory" not in result.stderr
        assert "Not a directory" not in result.stdout
        assert "Not a directory" not in result.stderr
