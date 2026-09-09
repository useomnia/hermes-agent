"""Durable idempotency records for the structured ``/v1/runs`` API.

Only scalar identity, request fingerprints, and lifecycle metadata are kept.
The request body (including model input) never enters this table. SQLite's
write transaction is the admission fence: one process wins a ``turn_id`` and
all concurrent or retried callers observe that same run identity.
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
_CANCELLED_BEFORE_START = "cancelled_before_start"
_ACTIVE_STATUSES = {"queued", "running", "waiting_for_approval", "stopping"}


class RunIdempotencyMismatch(ValueError):
    """A reused ``turn_id`` carried different immutable request semantics."""


@dataclass(frozen=True, slots=True)
class ManagedRunIdentity:
    submission_id: str
    execution_fingerprint: str


@dataclass(frozen=True, slots=True)
class RunIdempotencyRecord:
    turn_id: str
    run_id: str
    request_fingerprint: str
    session_id: str
    owner_profile: str | None
    status: str
    failure_reason: str | None
    managed_submission_id: str | None
    managed_execution_fingerprint: str | None
    cancel_requested: bool
    created_at: float
    updated_at: float

    @property
    def is_cancel_tombstone(self) -> bool:
        return (
            self.cancel_requested
            and self.status == "cancelled"
            and self.failure_reason == _CANCELLED_BEFORE_START
        )


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


def managed_cancel_fingerprint(
    *,
    turn_id: str,
    session_id: str,
    owner_profile: str | None,
    identity: ManagedRunIdentity,
) -> str:
    """Hash cancellation identity for a tombstone that has no launch request."""
    return request_fingerprint(
        {
            "domain": "hermes.managed-run.cancel-tombstone.v1",
            "turn_id": turn_id,
            "session_id": session_id,
            "owner_profile": owner_profile,
            "submission_id": identity.submission_id,
            "execution_fingerprint": identity.execution_fingerprint,
        }
    )


class RunIdempotencyStore:
    """Small state.db-backed relation for ``turn_id`` to run identity."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else get_hermes_home() / "state.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    turn_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    request_fingerprint TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    owner_profile TEXT,
                    status TEXT NOT NULL,
                    failure_reason TEXT,
                    managed_submission_id TEXT,
                    managed_execution_fingerprint TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            columns = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({_TABLE})")
            }
            additions = {
                "managed_submission_id": "TEXT",
                "managed_execution_fingerprint": "TEXT",
                "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in additions.items():
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE {_TABLE} ADD COLUMN {column} {declaration}"
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
            managed_submission_id=row["managed_submission_id"],
            managed_execution_fingerprint=row[
                "managed_execution_fingerprint"
            ],
            cancel_requested=bool(row["cancel_requested"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _assert_identity(
        record: RunIdempotencyRecord,
        *,
        session_id: str,
        owner_profile: str | None,
        managed_identity: ManagedRunIdentity | None,
    ) -> None:
        stored_managed = (
            record.managed_submission_id is not None
            or record.managed_execution_fingerprint is not None
        )
        if managed_identity is None:
            if stored_managed:
                raise RunIdempotencyMismatch(
                    "managed and legacy reservations cannot adopt each other"
                )
            return
        if (
            not stored_managed
            or record.session_id != session_id
            or record.owner_profile != owner_profile
            or record.managed_submission_id != managed_identity.submission_id
            or record.managed_execution_fingerprint
            != managed_identity.execution_fingerprint
        ):
            raise RunIdempotencyMismatch(
                "managed run identity does not match the reserved turn"
            )

    def reserve(
        self,
        *,
        turn_id: str,
        run_id: str,
        request_fingerprint: str,
        session_id: str,
        owner_profile: str | None,
        managed_identity: ManagedRunIdentity | None = None,
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
                self._assert_identity(
                    record,
                    session_id=session_id,
                    owner_profile=owner_profile,
                    managed_identity=managed_identity,
                )
                if (
                    not record.is_cancel_tombstone
                    and record.request_fingerprint != request_fingerprint
                ):
                    raise RunIdempotencyMismatch(
                        "turn_id was already used with different request semantics"
                    )
                return record, False
            conn.execute(
                f"""INSERT INTO {_TABLE}(
                    turn_id, run_id, request_fingerprint, session_id,
                    owner_profile, status, managed_submission_id,
                    managed_execution_fingerprint, cancel_requested,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, 0, ?, ?)""",
                (
                    turn_id,
                    run_id,
                    request_fingerprint,
                    session_id,
                    owner_profile,
                    managed_identity.submission_id if managed_identity else None,
                    (
                        managed_identity.execution_fingerprint
                        if managed_identity
                        else None
                    ),
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

    def reconcile_managed(
        self,
        *,
        turn_id: str,
        session_id: str,
        owner_profile: str | None,
        identity: ManagedRunIdentity,
    ) -> RunIdempotencyRecord | None:
        """Read one exact managed mapping without creating or changing it."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        if row is None:
            return None
        record = self._record(row)
        self._assert_identity(
            record,
            session_id=session_id,
            owner_profile=owner_profile,
            managed_identity=identity,
        )
        return record

    def cancel_managed(
        self,
        *,
        turn_id: str,
        run_id: str,
        session_id: str,
        owner_profile: str | None,
        identity: ManagedRunIdentity,
    ) -> tuple[RunIdempotencyRecord, bool]:
        """Atomically latch cancellation, creating a terminal tombstone if absent."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    f"""INSERT INTO {_TABLE}(
                        turn_id, run_id, request_fingerprint, session_id,
                        owner_profile, status, failure_reason,
                        managed_submission_id, managed_execution_fingerprint,
                        cancel_requested, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'cancelled', ?, ?, ?, 1, ?, ?)""",
                    (
                        turn_id,
                        run_id,
                        managed_cancel_fingerprint(
                            turn_id=turn_id,
                            session_id=session_id,
                            owner_profile=owner_profile,
                            identity=identity,
                        ),
                        session_id,
                        owner_profile,
                        _CANCELLED_BEFORE_START,
                        identity.submission_id,
                        identity.execution_fingerprint,
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

            record = self._record(row)
            self._assert_identity(
                record,
                session_id=session_id,
                owner_profile=owner_profile,
                managed_identity=identity,
            )
            if record.status not in _ACTIVE_STATUSES:
                return record, False
            next_status = "cancelled" if record.status == "queued" else "stopping"
            conn.execute(
                f"""UPDATE {_TABLE}
                    SET status = ?, cancel_requested = 1, updated_at = ?
                    WHERE turn_id = ?""",
                (next_status, now, turn_id),
            )
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            assert row is not None
            return self._record(row), False

    def update_status(
        self,
        *,
        turn_id: str,
        status: str,
        failure_reason: str | None = None,
        updated_at: float | None = None,
    ) -> RunIdempotencyRecord | None:
        """Persist lifecycle state without allowing cancellation resurrection."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                return None
            record = self._record(row)
            if record.is_cancel_tombstone or (
                record.cancel_requested and status != "cancelled"
            ):
                return record
            conn.execute(
                f"""UPDATE {_TABLE}
                    SET status = ?, failure_reason = ?, updated_at = ?
                    WHERE turn_id = ?""",
                (status, failure_reason, updated_at or time.time(), turn_id),
            )
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            assert row is not None
            return self._record(row)

    def get(self, turn_id: str) -> RunIdempotencyRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        return self._record(row) if row is not None else None


__all__ = [
    "ManagedRunIdentity",
    "RunIdempotencyMismatch",
    "RunIdempotencyRecord",
    "RunIdempotencyStore",
    "managed_cancel_fingerprint",
    "request_fingerprint",
]
