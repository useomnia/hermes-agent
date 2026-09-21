"""Recover clipped execution output without repeating the original command."""
import json
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
