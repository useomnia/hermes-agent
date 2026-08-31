"""Durable /v1/runs turn-id admission invariants."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from gateway.run_idempotency import (
    RunIdempotencyMismatch,
    RunIdempotencyStore,
    request_fingerprint,
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
