"""Recover clipped execution output without repeating the original command."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools import code_execution_tool as execution


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("TERMINAL_ENV", "local")


def test_local_execution_retains_middle_and_does_not_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 10})
    receipt = tmp_path / "executions.txt"
    code = (
        f"with open({str(receipt)!r}, 'a') as f: f.write('ran\\n')\n"
        "print('head\\n' * 12000)\n"
        "print('MIDDLE_RECORD_617')\n"
        "print('tail\\n' * 12000)\n"
    )
    result = json.loads(execution.execute_code(code, enabled_tools=[]))
    assert result["status"] == "success"
    assert "MIDDLE_RECORD_617" not in result["output"]
    saved = Path(result["stdout_spill_path"]).read_text()
    assert "MIDDLE_RECORD_617" in saved
    assert saved.startswith("head\n") and saved.endswith("tail\n\n")
    assert result["stdout_spill_truncated"] is False
    assert receipt.read_text() == "ran\n"


def test_short_output_does_not_create_artifact():
    output, metadata = execution._truncate_stdout_text("small\n")
    assert output == "small\n"
    assert "stdout_spill_path" not in metadata


def test_sprites_stdout_and_tool_results_share_large_file_transport(monkeypatch):
    import io
    import re
    import threading
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit

    from tools import file_tools
    from tools.environments import sprites
    from tools.environments.file_sync import FileSyncManager, iter_sprites_cache_files
    from tools.tool_result_storage import maybe_persist_tool_result

    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    env = sprites.SpritesEnvironment.__new__(sprites.SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example/internal/toolbox"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 30
    env.file_request = lambda _payload: pytest.fail("large result used JSON upload")
    env._cache_sync_lock = threading.Lock()
    env._cache_sync_manager = FileSyncManager(
        get_files_fn=iter_sprites_cache_files,
        upload_fn=env._upload_cache_file,
        delete_fn=lambda _paths: pytest.fail("publication deleted an artifact"),
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda: SimpleNamespace(env=env))
    uploaded = {}

    def upload(request, timeout):
        assert request.get_method() == "PUT"
        query = parse_qs(urlsplit(request.full_url).query)
        assert query["maxBytes"] == [str(len(request.data))]
        uploaded[query["path"][0]] = request.data
        return io.BytesIO(json.dumps({"bytesWritten": len(request.data)}).encode())

    monkeypatch.setattr(sprites._URL_OPENER, "open", upload)
    content = "record é\n" * 280_000 + "FINAL RECORD\n"
    preview, metadata = execution._truncate_stdout_text(content)
    result = maybe_persist_tool_result(content, "connector", "tc_shared", env=env)
    result_path = re.search(r"^Full output saved to: (.+)$", result, re.MULTILINE)

    assert metadata["stdout_truncated"] is True
    assert metadata["stdout_spill_truncated"] is False
    assert result_path is not None
    assert len(preview) < len(content)
    assert len(result) < len(content)
    assert len(uploaded) == 2
    assert uploaded[metadata["stdout_spill_path"]] == content.encode("utf-8")
    assert uploaded[result_path.group(1)] == content.encode("utf-8")


def test_single_line_json_recovery_reads_middle_without_repeating_source(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 10})
    receipt = tmp_path / "executions.txt"
    code = (
        "import json\n"
        f"with open({str(receipt)!r}, 'a') as f: f.write('ran\\n')\n"
        "print(json.dumps({'head': 'h' * 80000, "
        "'finding': {'reference': 'MIDDLE_617', 'count': 37}, 'tail': 't' * 80000}))\n"
    )
    result = json.loads(execution.execute_code(code, enabled_tools=[]))
    assert result["status"] == "success"
    assert "MIDDLE_617" not in result["output"]
    assert result["stdout_spill_truncated"] is False
    assert "JSON" in result["warning"]
    assert "same data" in result["warning"]

    recovery = json.loads(execution.execute_code(
        "import json\n"
        f"with open({result['stdout_spill_path']!r}) as f: data = json.load(f)\n"
        "print(json.dumps(data['finding']))\n",
        enabled_tools=[],
    ))
    assert recovery["status"] == "success"
    assert json.loads(recovery["output"]) == {"reference": "MIDDLE_617", "count": 37}
    assert receipt.read_text() == "ran\n"


def test_full_output_is_redacted_before_it_is_saved():
    secret = "ghp_" + "a" * 36
    _, metadata = execution._truncate_stdout_text("start\n" * 12000 + secret + "\nend\n" * 12000)
    saved = Path(metadata["stdout_spill_path"]).read_text()
    assert secret not in saved
    assert "start" in saved and "end" in saved


def test_spill_limit_is_bytes_and_reported_as_partial(monkeypatch):
    monkeypatch.setattr(execution, "MAX_SPILLED_STDOUT_BYTES", 60_000)
    _, metadata = execution._truncate_stdout_text("ééé\n" * 15000)
    data = Path(metadata["stdout_spill_path"]).read_bytes()
    data.decode("utf-8")
    assert len(data) <= 60_000
    assert metadata["stdout_spill_truncated"] is True
    assert "partial" in metadata["warning"].lower()
    assert "FULL output" not in metadata["warning"]
    assert "missing records" in metadata["warning"]
    assert "JSON" in metadata["warning"]


def test_storage_failure_preserves_result_without_unreadable_path(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(execution, "_spill_full_stdout", unavailable)
    output, metadata = execution._truncate_stdout_text("start\n" * 15000 + "END")
    assert output.endswith("END")
    assert metadata["stdout_truncated"] is True
    assert "stdout_spill_path" not in metadata
    assert "unavailable" in metadata["warning"]


def test_symlink_cache_is_not_followed(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / "cache").symlink_to(outside, target_is_directory=True)
    _, metadata = execution._truncate_stdout_text("data\n" * 15000)
    assert "stdout_spill_path" not in metadata
    assert not list(outside.iterdir())


@pytest.mark.parametrize("backend", ["ssh", "modal", "docker"])
@pytest.mark.skipif(os.name == "nt", reason="The remote-filesystem fixture runs a POSIX shell")
def test_remote_execution_retains_recovery_on_execution_filesystem(tmp_path, monkeypatch, backend):
    remote = tmp_path / "remote"
    remote.mkdir()

    class FileEnvironment:
        def get_temp_dir(self):
            return str(remote)

        def write_file_content(self, path, content):
            target = Path(path)
            assert target.is_relative_to(remote)
            target.write_text(content)
            return True

        def execute(self, command, cwd=None, timeout=30):
            result = subprocess.run(
                command, shell=True, executable="/bin/bash", cwd=cwd or remote,
                env={"PATH": os.environ["PATH"], "HOME": str(remote)},
                capture_output=True, text=True, timeout=timeout,
            )
            return {"output": result.stdout + result.stderr, "returncode": result.returncode}

    monkeypatch.setenv("TERMINAL_ENV", backend)
    monkeypatch.setattr(execution, "_load_config", lambda: {"timeout": 10})
    monkeypatch.setattr(execution, "_get_or_create_env", lambda task: (FileEnvironment(), backend))
    receipt = remote / "executions.txt"
    code = (
        f"with open({str(receipt)!r}, 'a') as f: f.write('ran\\n')\n"
        "print('head\\n' * 12000)\n"
        "print('REMOTE_MIDDLE_RECORD')\n"
        "print('tail\\n' * 12000)\n"
    )
    result = json.loads(execution._execute_remote(code, "recovery-test", []))
    assert result["status"] == "success", result
    assert result["exit_code"] == 0
    assert "REMOTE_MIDDLE_RECORD" not in result["output"]
    saved = Path(result["stdout_spill_path"])
    assert saved.is_relative_to(remote), result
    assert "REMOTE_MIDDLE_RECORD" in saved.read_text()
    assert receipt.read_text() == "ran\n"


def test_remote_publication_failure_does_not_return_a_host_path(monkeypatch):
    class UnavailableEnvironment:
        def execute(self, *args, **kwargs):
            raise OSError("remote unavailable")

    output, metadata = execution._truncate_stdout_text(
        "start\n" * 15000 + "END", env=UnavailableEnvironment(),
    )
    assert output.endswith("END")
    assert metadata["stdout_truncated"] is True
    assert "stdout_spill_path" not in metadata
    assert "unavailable" in metadata["warning"]
