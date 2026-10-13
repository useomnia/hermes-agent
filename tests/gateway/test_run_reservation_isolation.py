"""Real SQLite boundaries for durable Turn admission (OMNI-275)."""

import sqlite3

import pytest

from gateway.run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB
from hermes_state_errors import is_sqlite_lock_error


def test_session_writer_does_not_block_turn_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session = SessionDB(tmp_path / "state.db")
    store = RunIdempotencyStore()
    blocker = sqlite3.connect(tmp_path / "state.db")
    try:
        blocker.execute("BEGIN IMMEDIATE")
        record, created = store.reserve(
            turn_id="turn-locked-session",
            run_id="run-once",
            request_fingerprint="immutable-input",
            session_id="session",
            owner_profile=None,
        )
        assert created
        assert record.run_id == "run-once"
    finally:
        blocker.rollback()
        blocker.close()
        session.close()
        store.close()


def test_settled_session_opens_while_another_connection_is_writing(tmp_path):
    path = tmp_path / "state.db"
    session = SessionDB(path)
    session.close()
    # A second open settles the fork's separately versioned FTS layout stamp.
    session = SessionDB(path)
    blocker = sqlite3.connect(path)
    reopened = None
    try:
        blocker.execute("BEGIN IMMEDIATE")
        reopened = SessionDB(path)
        assert reopened.get_meta("fts_storage_version") is not None
    finally:
        blocker.rollback()
        blocker.close()
        if reopened is not None:
            reopened.close()
        session.close()


@pytest.mark.parametrize("code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, 517])
def test_fts_constructor_busy_is_classified_by_result_code(code):
    error = sqlite3.OperationalError("vtable constructor failed: messages_fts")
    error.sqlite_errorcode = code
    assert is_sqlite_lock_error(error)


def test_known_non_lock_result_code_is_not_retried_as_busy():
    error = sqlite3.OperationalError("busy looking but damaged")
    error.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
    assert not is_sqlite_lock_error(error)


def test_wrapped_lock_message_still_classifies_as_retryable():
    assert is_sqlite_lock_error("database is locked")
    assert not is_sqlite_lock_error("disk I/O error")


def test_unavailable_reservation_file_never_falls_back_to_memory(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        RunIdempotencyStore(tmp_path)


def test_in_place_upgrade_refuses_to_forget_legacy_reservations(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    legacy = RunIdempotencyStore(tmp_path / "state.db")
    try:
        legacy.reserve(
            turn_id="accepted",
            run_id="original",
            request_fingerprint="same",
            session_id="s",
            owner_profile=None,
        )
        with pytest.raises(RuntimeError, match="fresh reprovisioned runtime"):
            RunIdempotencyStore()
        assert legacy.get("accepted").run_id == "original"
        assert not (tmp_path / "runs_idempotency.db").exists()
    finally:
        legacy.close()


def test_terminal_reservation_does_not_expire_after_upstream_retention_window(tmp_path):
    store = RunIdempotencyStore(tmp_path / "runs_idempotency.db")
    try:
        store.reserve(
            turn_id="turn",
            run_id="original",
            request_fingerprint="same",
            session_id="s",
            owner_profile=None,
        )
        store.update_status(turn_id="turn", status="completed", updated_at=1)
        record, created = store.reserve(
            turn_id="turn",
            run_id="duplicate",
            request_fingerprint="same",
            session_id="s",
            owner_profile=None,
        )
        assert not created
        assert (record.run_id, record.status) == ("original", "completed")
    finally:
        store.close()
