"""Durable /v1/runs turn-id admission invariants."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3

import pytest

from gateway.run_idempotency import (
    RunIdempotencyMismatch,
    RunIdempotencyStore,
    request_fingerprint,
)


def test_reservation_survives_store_reopen_and_rejects_changed_request(
    tmp_path,
) -> None:
    path = tmp_path / "runs_idempotency.db"
    fingerprint = request_fingerprint({"input": "hello", "session_id": "s"})
    with closing(RunIdempotencyStore(path)) as store:
        first, is_new = store.reserve(
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

    with closing(RunIdempotencyStore(path)) as reopened:
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
    path = tmp_path / "runs_idempotency.db"
    fingerprint = request_fingerprint({"input": "hello"})

    def reserve(index: int):
        with closing(RunIdempotencyStore(path)) as store:
            return store.reserve(
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


def test_steer_outcome_survives_store_reopen_and_keeps_the_first_outcome(tmp_path) -> None:
    path = tmp_path / "runs_idempotency.db"
    with closing(RunIdempotencyStore(path)) as store:
        assert store.steer_outcome("run-1", "subagent:d1") is None
        assert store.record_steer(run_id="run-1", steer_id="subagent:d1", outcome="queued") == "queued"
        assert (
            store.record_steer(run_id="run-1", steer_id="subagent:d1", outcome="redirected")
            == "queued"
        )

    with closing(RunIdempotencyStore(path)) as reopened:
        assert reopened.steer_outcome("run-1", "subagent:d1") == "queued"
        assert reopened.steer_outcome("run-1", "subagent:d2") is None
        assert reopened.steer_outcome("run-2", "subagent:d1") is None


def test_has_run_knows_a_reserved_run_after_reopen(tmp_path) -> None:
    path = tmp_path / "runs_idempotency.db"
    with closing(RunIdempotencyStore(path)) as store:
        store.reserve(
            turn_id="turn-1",
            run_id="run-1",
            request_fingerprint=request_fingerprint({"input": "hello"}),
            session_id="s",
            owner_profile="default",
        )

    with closing(RunIdempotencyStore(path)) as reopened:
        assert reopened.has_run("run-1") is True
        assert reopened.has_run("run-2") is False
