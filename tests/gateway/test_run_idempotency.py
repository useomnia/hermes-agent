"""Durable /v1/runs turn-id admission invariants."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from gateway.run_idempotency import (
    ManagedRunIdentity,
    RunIdempotencyMismatch,
    RunIdempotencyStore,
    request_fingerprint,
)


MANAGED = ManagedRunIdentity(
    submission_id="12345678-1234-4234-8234-123456789abc",
    execution_fingerprint="a" * 64,
)


def test_reservation_survives_store_reopen_and_rejects_changed_request(tmp_path) -> None:
    path = tmp_path / "state.db"
    fingerprint = request_fingerprint({"input": "hello", "session_id": "s"})
    first, is_new = RunIdempotencyStore(path).reserve(
        turn_id="turn-1",
        run_id="run-original",
        request_fingerprint=fingerprint,
        session_id="s",
        owner_profile="default",
    )
    assert is_new is True
    assert first.run_id == "run-original"
    with sqlite3.connect(path) as conn:
        stored = conn.execute(
            "SELECT request_fingerprint FROM api_run_idempotency WHERE turn_id = ?",
            ("turn-1",),
        ).fetchone()[0]
    assert stored == fingerprint
    assert "hello" not in stored

    reopened = RunIdempotencyStore(path)
    same, is_new = reopened.reserve(
        turn_id="turn-1",
        run_id="run-different",
        request_fingerprint=fingerprint,
        session_id="s",
        owner_profile="default",
    )
    assert is_new is False
    assert same.run_id == "run-original"

    with pytest.raises(RunIdempotencyMismatch):
        reopened.reserve(
            turn_id="turn-1",
            run_id="run-different",
            request_fingerprint=request_fingerprint({"input": "changed"}),
            session_id="s",
            owner_profile="default",
        )


def test_concurrent_reservations_create_one_run(tmp_path) -> None:
    path = tmp_path / "state.db"
    fingerprint = request_fingerprint({"input": "hello"})

    def reserve(index: int):
        return RunIdempotencyStore(path).reserve(
            turn_id="turn-race",
            run_id=f"run-{index}",
            request_fingerprint=fingerprint,
            session_id="session",
            owner_profile=None,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(8)))

    assert sum(is_new for _, is_new in results) == 1
    assert len({record.run_id for record, _ in results}) == 1


def test_old_schema_is_upgraded_without_changing_legacy_rows(tmp_path) -> None:
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE api_run_idempotency (
                turn_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL UNIQUE,
                request_fingerprint TEXT NOT NULL,
                session_id TEXT NOT NULL,
                owner_profile TEXT,
                status TEXT NOT NULL,
                failure_reason TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO api_run_idempotency VALUES (
                'turn-old', 'run-old', 'fingerprint', 'session', NULL,
                'completed', NULL, 1.0, 2.0
            )"""
        )

    record = RunIdempotencyStore(path).get("turn-old")

    assert record is not None
    assert record.run_id == "run-old"
    assert record.managed_submission_id is None
    assert record.managed_execution_fingerprint is None
    assert record.cancel_requested is False


def test_managed_and_legacy_reservations_cannot_adopt_each_other(tmp_path) -> None:
    store = RunIdempotencyStore(tmp_path / "state.db")
    fingerprint = request_fingerprint({"input": "hello"})
    store.reserve(
        turn_id="turn-managed",
        run_id="run-managed",
        request_fingerprint=fingerprint,
        session_id="session",
        owner_profile="profile",
        managed_identity=MANAGED,
    )
    store.reserve(
        turn_id="turn-legacy",
        run_id="run-legacy",
        request_fingerprint=fingerprint,
        session_id="session",
        owner_profile="profile",
    )

    with pytest.raises(RunIdempotencyMismatch):
        store.reserve(
            turn_id="turn-managed",
            run_id="ignored",
            request_fingerprint=fingerprint,
            session_id="session",
            owner_profile="profile",
        )
    with pytest.raises(RunIdempotencyMismatch):
        store.reserve(
            turn_id="turn-legacy",
            run_id="ignored",
            request_fingerprint=fingerprint,
            session_id="session",
            owner_profile="profile",
            managed_identity=MANAGED,
        )


@pytest.mark.parametrize(
    ("session_id", "owner_profile", "identity"),
    [
        ("other", "profile", MANAGED),
        ("session", "other", MANAGED),
        (
            "session",
            "profile",
            ManagedRunIdentity(
                submission_id="87654321-4321-4321-8321-cba987654321",
                execution_fingerprint="a" * 64,
            ),
        ),
        (
            "session",
            "profile",
            ManagedRunIdentity(
                submission_id=MANAGED.submission_id,
                execution_fingerprint="b" * 64,
            ),
        ),
    ],
)
def test_managed_reconcile_requires_exact_identity(
    tmp_path, session_id, owner_profile, identity
) -> None:
    store = RunIdempotencyStore(tmp_path / "state.db")
    store.reserve(
        turn_id="turn",
        run_id="run",
        request_fingerprint="launch",
        session_id="session",
        owner_profile="profile",
        managed_identity=MANAGED,
    )

    with pytest.raises(RunIdempotencyMismatch):
        store.reconcile_managed(
            turn_id="turn",
            session_id=session_id,
            owner_profile=owner_profile,
            identity=identity,
        )


def test_reconcile_absent_does_not_create_a_row(tmp_path) -> None:
    path = tmp_path / "state.db"
    store = RunIdempotencyStore(path)

    assert store.reconcile_managed(
        turn_id="absent",
        session_id="session",
        owner_profile=None,
        identity=MANAGED,
    ) is None
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM api_run_idempotency").fetchone()[0] == 0


def test_cancel_before_launch_creates_stable_restart_safe_tombstone(tmp_path) -> None:
    path = tmp_path / "state.db"
    cancelled, is_new = RunIdempotencyStore(path).cancel_managed(
        turn_id="turn",
        run_id="run-tombstone",
        session_id="session",
        owner_profile="profile",
        identity=MANAGED,
    )
    assert is_new is True
    assert cancelled.is_cancel_tombstone
    assert cancelled.request_fingerprint != MANAGED.execution_fingerprint

    reopened = RunIdempotencyStore(path)
    replay, is_new = reopened.reserve(
        turn_id="turn",
        run_id="ignored",
        request_fingerprint=request_fingerprint({"launch": "arrived later"}),
        session_id="session",
        owner_profile="profile",
        managed_identity=MANAGED,
    )
    assert is_new is False
    assert replay.run_id == "run-tombstone"
    assert replay.is_cancel_tombstone

    before = reopened.get("turn")
    persisted = reopened.update_status(
        turn_id="turn",
        status="cancelled",
        failure_reason=None,
    )
    assert persisted == before
    assert reopened.get("turn") == before

    persisted = reopened.update_status(turn_id="turn", status="completed")
    assert persisted is not None
    assert persisted.status == "cancelled"
    assert reopened.get("turn") == before


def test_launch_then_cancel_uses_reserved_run_identity(tmp_path) -> None:
    path = tmp_path / "state.db"
    launched, _ = RunIdempotencyStore(path).reserve(
        turn_id="turn",
        run_id="run-launched",
        request_fingerprint="launch",
        session_id="session",
        owner_profile=None,
        managed_identity=MANAGED,
    )
    cancelled, is_new = RunIdempotencyStore(path).cancel_managed(
        turn_id="turn",
        run_id="ignored",
        session_id="session",
        owner_profile=None,
        identity=MANAGED,
    )

    assert is_new is False
    assert cancelled.run_id == launched.run_id
    assert cancelled.cancel_requested is True
    assert cancelled.status == "cancelled"


def test_running_cancel_cannot_be_resurrected_by_completion(tmp_path) -> None:
    store = RunIdempotencyStore(tmp_path / "state.db")
    store.reserve(
        turn_id="turn",
        run_id="run-launched",
        request_fingerprint="launch",
        session_id="session",
        owner_profile=None,
        managed_identity=MANAGED,
    )
    store.update_status(turn_id="turn", status="running")

    stopping, _ = store.cancel_managed(
        turn_id="turn",
        run_id="ignored",
        session_id="session",
        owner_profile=None,
        identity=MANAGED,
    )
    after_completion = store.update_status(turn_id="turn", status="completed")
    cancelled = store.update_status(turn_id="turn", status="cancelled")

    assert stopping.status == "stopping"
    assert after_completion is not None
    assert after_completion.status == "stopping"
    assert cancelled is not None
    assert cancelled.status == "cancelled"


def test_concurrent_launch_and_cancel_share_one_cancelled_identity(tmp_path) -> None:
    path = tmp_path / "state.db"

    def launch():
        return RunIdempotencyStore(path).reserve(
            turn_id="turn-race",
            run_id="run-launch",
            request_fingerprint="launch",
            session_id="session",
            owner_profile=None,
            managed_identity=MANAGED,
        )[0]

    def cancel():
        return RunIdempotencyStore(path).cancel_managed(
            turn_id="turn-race",
            run_id="run-cancel",
            session_id="session",
            owner_profile=None,
            identity=MANAGED,
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        launch_future = pool.submit(launch)
        cancel_future = pool.submit(cancel)
        launch_record = launch_future.result()
        cancel_record = cancel_future.result()

    final = RunIdempotencyStore(path).get("turn-race")
    assert final is not None
    assert launch_record.run_id == cancel_record.run_id == final.run_id
    assert final.cancel_requested is True
    assert final.status == "cancelled"
