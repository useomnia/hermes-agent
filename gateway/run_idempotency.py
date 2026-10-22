"""Durable Turn reservations isolated from transcript and tool-accounting writes.

Adapted from upstream gateway/platforms/api_server_run_idempotency.py: one
profile-local WAL database, one serialized connection, explicit transactions
and close. Omnia retains scalar Turn identity for the lifetime of the runtime;
there is no memory fallback, request-body storage, or terminal-row expiry.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home
from hermes_state import apply_wal_with_fallback
from hermes_state_errors import is_sqlite_lock_error


_TABLE = "api_run_idempotency"
# Steers a run accepted, by caller steer id. Kept as long as the run's row, so
# a caller can learn after a gateway restart whether a steer reached the run.
_STEER_TABLE = "api_run_steers"


class RunIdempotencyMismatch(ValueError):
    """A reused Turn ID carried different immutable request semantics."""


class RunReservationBusy(sqlite3.OperationalError):
    """The reservation write rolled back and a read confirmed no accepted ID."""


@dataclass(frozen=True, slots=True)
class RunIdempotencyRecord:
    turn_id: str
    run_id: str
    request_fingerprint: str
    session_id: str
    owner_profile: str | None
    status: str
    failure_reason: str | None
    created_at: float
    updated_at: float


def request_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash canonical request semantics without retaining the request itself."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunIdempotencyStore:
    """One durable admission fence per profile, independent of state.db."""

    def __init__(self, db_path: str | Path | None = None, *, timeout: float = 1.0):
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else get_hermes_home() / "runs_idempotency.db"
        )
        if db_path is None:
            self._require_fresh_reservation_storage()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=timeout,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        try:
            apply_wal_with_fallback(self._conn, db_label="runs_idempotency.db")
            self._conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
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
            self._conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_STEER_TABLE} (
                    run_id TEXT NOT NULL,
                    steer_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    accepted_at REAL NOT NULL,
                    PRIMARY KEY (run_id, steer_id)
                )"""
            )
            for suffix in ("", "-wal", "-shm"):
                path = Path(str(self.db_path) + suffix)
                if path.exists():
                    path.chmod(0o600)
        except BaseException:
            self._conn.close()
            raise

    def _require_fresh_reservation_storage(self) -> None:
        """Refuse an in-place upgrade that would forget accepted Turn IDs."""
        legacy = self.db_path.with_name("state.db")
        if not legacy.exists():
            return
        conn = sqlite3.connect(legacy.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (_TABLE,),
            ).fetchone()
            if table and conn.execute(f"SELECT 1 FROM {_TABLE} LIMIT 1").fetchone():
                raise RuntimeError(
                    "Legacy Turn reservations exist in state.db. This release requires "
                    "a fresh reprovisioned runtime; keep the source runtime for its owned Turns."
                )
        finally:
            conn.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    @staticmethod
    def _record(row: sqlite3.Row) -> RunIdempotencyRecord:
        return RunIdempotencyRecord(**dict(row))

    @staticmethod
    def _replay(
        record: RunIdempotencyRecord,
        fingerprint: str,
        owner_profile: str | None,
    ) -> tuple[RunIdempotencyRecord, bool]:
        if (
            record.request_fingerprint != fingerprint
            or record.owner_profile != owner_profile
        ):
            raise RunIdempotencyMismatch(
                "turn_id was already used with different request semantics"
            )
        return record, False

    def reserve(
        self,
        *,
        turn_id: str,
        run_id: str,
        request_fingerprint: str,
        session_id: str,
        owner_profile: str | None,
    ) -> tuple[RunIdempotencyRecord, bool]:
        now = time.time()
        try:
            with self._transaction() as conn:
                row = conn.execute(
                    f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is not None:
                    return self._replay(self._record(row), request_fingerprint, owner_profile)
                conn.execute(
                    f"""INSERT INTO {_TABLE} (
                        turn_id, run_id, request_fingerprint, session_id,
                        owner_profile, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
                    (
                        turn_id,
                        run_id,
                        request_fingerprint,
                        session_id,
                        owner_profile,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    f"SELECT * FROM {_TABLE} WHERE turn_id = ?", (turn_id,)
                ).fetchone()
                return self._record(row), True
        except sqlite3.OperationalError as exc:
            if not is_sqlite_lock_error(exc):
                raise
            # A failed write does not prove a prior submission never ran. Read
            # after rollback; an unreadable store must remain an unknown outcome.
            record = self.get(turn_id)
            if record is not None:
                return self._replay(record, request_fingerprint, owner_profile)
            raise RunReservationBusy("Turn reservation storage is busy") from exc

    def update_status(
        self,
        *,
        turn_id: str,
        status: str,
        failure_reason: str | None = None,
        updated_at: float | None = None,
    ) -> RunIdempotencyRecord | None:
        with self._transaction() as conn:
            conn.execute(
                f"""UPDATE {_TABLE} SET status = ?, failure_reason = ?, updated_at = ?
                    WHERE turn_id = ? AND (status != ? OR failure_reason IS NOT ?)""",
                (
                    status,
                    failure_reason,
                    updated_at if updated_at is not None else time.time(),
                    turn_id,
                    status,
                    failure_reason,
                ),
            )
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            return self._record(row) if row is not None else None

    def get(self, turn_id: str) -> RunIdempotencyRecord | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        return self._record(row) if row is not None else None

    def has_run(self, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {_TABLE} WHERE run_id = ?", (run_id,)
            ).fetchone()
        return row is not None

    def record_steer(self, *, run_id: str, steer_id: str, outcome: str) -> str:
        """Keep the first outcome a run gave a steer id, and return it."""
        with self._transaction() as conn:
            conn.execute(
                f"""INSERT OR IGNORE INTO {_STEER_TABLE} (run_id, steer_id, outcome, accepted_at)
                    VALUES (?, ?, ?, ?)""",
                (run_id, steer_id, outcome, time.time()),
            )
            row = conn.execute(
                f"SELECT outcome FROM {_STEER_TABLE} WHERE run_id = ? AND steer_id = ?",
                (run_id, steer_id),
            ).fetchone()
        return row["outcome"]

    def steer_outcome(self, run_id: str, steer_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT outcome FROM {_STEER_TABLE} WHERE run_id = ? AND steer_id = ?",
                (run_id, steer_id),
            ).fetchone()
        return row["outcome"] if row is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
