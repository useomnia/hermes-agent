"""Durable idempotency records for the structured ``/v1/runs`` API.

Only scalar identity, a request fingerprint, and lifecycle metadata are kept.
The request body (including model input) never enters this table.  SQLite's
write transaction is the admission fence: one process wins a ``turn_id`` and
all concurrent/retried callers observe that same run identity.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home


_TABLE = "api_run_idempotency"


class RunIdempotencyMismatch(ValueError):
    """A reused ``turn_id`` carried different immutable request semantics."""


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
    """Small state.db-backed relation for ``turn_id`` to run identity."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_hermes_home() / "state.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
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
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_TABLE}_updated_idx "
                f"ON {_TABLE}(updated_at)"
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> RunIdempotencyRecord:
        return RunIdempotencyRecord(
            turn_id=str(row["turn_id"]),
            run_id=str(row["run_id"]),
            request_fingerprint=str(row["request_fingerprint"]),
            session_id=str(row["session_id"]),
            owner_profile=row["owner_profile"],
            status=str(row["status"]),
            failure_reason=row["failure_reason"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def reserve(
        self,
        *,
        turn_id: str,
        run_id: str,
        request_fingerprint: str,
        session_id: str,
        owner_profile: str | None,
    ) -> tuple[RunIdempotencyRecord, bool]:
        """Atomically reserve a turn, returning ``(record, is_new)``."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is not None:
                record = self._record(row)
                if record.request_fingerprint != request_fingerprint:
                    raise RunIdempotencyMismatch(
                        "turn_id was already used with different request semantics"
                    )
                return record, False
            conn.execute(
                f"""INSERT INTO {_TABLE}(
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
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            assert row is not None
            return self._record(row), True

    def update_status(
        self,
        *,
        turn_id: str,
        status: str,
        failure_reason: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Persist scalar lifecycle state; unknown rows are ignored."""
        with self._connect() as conn:
            conn.execute(
                f"""UPDATE {_TABLE}
                    SET status = ?, failure_reason = ?, updated_at = ?
                    WHERE turn_id = ?""",
                (status, failure_reason, updated_at or time.time(), turn_id),
            )

    def get(self, turn_id: str) -> RunIdempotencyRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        return self._record(row) if row is not None else None


__all__ = [
    "RunIdempotencyMismatch",
    "RunIdempotencyRecord",
    "RunIdempotencyStore",
    "request_fingerprint",
]
