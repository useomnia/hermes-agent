"""Runtime status is published in-process at once and persisted in the background."""

import concurrent.futures
import json

import pytest

from gateway import status
from gateway.run import _require_offline_marker_invalidated


def test_published_state_is_readable_before_it_reaches_disk(monkeypatch, tmp_path):
    path = tmp_path / "gateway_state.json"
    monkeypatch.setattr(status, "_get_runtime_status_path", lambda: path)
    monkeypatch.setattr(status, "_runtime_status_record", None)
    monkeypatch.setattr(status, "_runtime_status_writer", status._RuntimeStatusWriter())

    status.write_runtime_status(gateway_state="starting")
    status.write_runtime_status(gateway_state="running", active_agents=2)

    assert status.read_runtime_status()["gateway_state"] == "running"
    assert status.flush_runtime_status(timeout=5) is True
    on_disk = json.loads(path.read_text())
    assert on_disk["gateway_state"] == "running"
    assert on_disk["active_agents"] == 2


def test_another_profiles_status_is_still_read_from_disk(monkeypatch, tmp_path):
    own = tmp_path / "own.json"
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"gateway_state": "stopped"}))
    monkeypatch.setattr(status, "_get_runtime_status_path", lambda: own)
    monkeypatch.setattr(status, "_runtime_status_record", None)
    monkeypatch.setattr(status, "_runtime_status_writer", status._RuntimeStatusWriter())

    status.write_runtime_status(gateway_state="running")

    assert status.read_runtime_status(other)["gateway_state"] == "stopped"
    assert status.flush_runtime_status(timeout=5) is True


@pytest.mark.asyncio
async def test_start_refuses_work_when_the_quiescence_marker_cannot_be_invalidated():
    failed: concurrent.futures.Future = concurrent.futures.Future()
    failed.set_result(False)
    with pytest.raises(RuntimeError, match="refusing to admit gateway work"):
        await _require_offline_marker_invalidated(failed)

    done: concurrent.futures.Future = concurrent.futures.Future()
    done.set_result(True)
    await _require_offline_marker_invalidated(done)
    await _require_offline_marker_invalidated(None)
