"""Importing tools must leave the store untouched; consumers own recovery."""

import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import threading
from unittest.mock import Mock

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation
from tools.process_registry import ProcessRegistry


def _seed_completion(database_path, delegation_id="retained"):
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "owner",
        "status": "completed",
    }
    with sqlite3.connect(database_path) as connection:
        async_delegation._initialize_schema(connection)
        connection.execute(
            "INSERT INTO async_delegations "
            "(delegation_id, origin_session, state, dispatched_at, completed_at, "
            "updated_at, event_json) VALUES (?, 'owner', 'completed', 1, 1, 1, ?)",
            (delegation_id, json.dumps(event)),
        )


def test_model_tools_import_should_not_create_state_database(tmp_path):
    environment = {
        **os.environ,
        "HERMES_HOME": str(tmp_path),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import model_tools"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert not (tmp_path / "state.db").exists()


def test_registry_construction_should_not_create_state_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ProcessRegistry()

    assert not (tmp_path / "state.db").exists()


def test_model_tools_import_should_not_migrate_existing_ledger(tmp_path):
    database_path = tmp_path / "state.db"
    _seed_completion(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "ALTER TABLE async_delegations DROP COLUMN origin_session_id"
        )
        before = connection.execute("PRAGMA table_info(async_delegations)").fetchall()
    environment = {
        **os.environ,
        "HERMES_HOME": str(tmp_path),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }

    result = subprocess.run(
        [sys.executable, "-c", "import model_tools"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("PRAGMA table_info(async_delegations)").fetchall()
            == before
        )


def test_replay_should_not_create_missing_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert async_delegation.restore_undelivered_completions(queue.Queue()) == 0
    assert not (tmp_path / "state.db").exists()


def test_first_consumer_should_restore_persisted_completion_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_completion(tmp_path / "state.db")
    registry = ProcessRegistry()
    assert registry.completion_queue.empty()

    first = registry.drain_notifications("owner")
    second = registry.drain_notifications("owner")

    assert [event["delegation_id"] for event, _ in first] == ["retained"]
    assert second == []


def test_first_consumer_should_restore_launch_profile_under_secondary_binding(
    tmp_path, monkeypatch
):
    launch = tmp_path / "launch"
    secondary = tmp_path / "secondary"
    launch.mkdir()
    secondary.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    _seed_completion(launch / "state.db", "launch-completion")
    _seed_completion(secondary / "state.db", "secondary-completion")
    registry = ProcessRegistry()
    token = set_hermes_home_override(secondary)
    try:
        events = registry.drain_notifications("owner")
        assert async_delegation._db_path() == secondary / "state.db"
    finally:
        reset_hermes_home_override(token)

    assert [event["delegation_id"] for event, _ in events] == ["launch-completion"]


def test_failed_recovery_should_leave_consumer_usable(monkeypatch, caplog):
    restore = Mock(side_effect=sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(async_delegation, "restore_undelivered_completions", restore)
    registry = ProcessRegistry()

    assert registry.drain_notifications("owner") == []
    assert registry.drain_notifications("owner") == []
    assert restore.call_count == 1
    assert "Could not restore async delegation completions" in caplog.text


def test_simultaneous_consumers_should_enqueue_one_recovered_completion(monkeypatch):
    callers = threading.Barrier(3)
    started = threading.Event()
    release = threading.Event()

    def restore(events):
        started.set()
        assert release.wait(5)
        events.put({"delegation_id": "one"})
        return 1

    monkeypatch.setattr(async_delegation, "restore_undelivered_completions", restore)
    registry = ProcessRegistry()

    def consume():
        callers.wait()
        registry.restore_completions()

    threads = [threading.Thread(target=consume) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        callers.wait()
        assert started.wait(5)
    finally:
        release.set()
        for thread in threads:
            thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert list(registry.completion_queue.queue) == [{"delegation_id": "one"}]
