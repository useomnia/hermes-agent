"""Cold-start and opening-order regressions for the shared state database."""

import sqlite3
from contextlib import closing

import pytest

from hermes_state import SCHEMA_SQL, SessionDB
from tools.async_delegation import _initialize_schema


def _delegation_shape(connection):
    return [
        tuple(row[1:])
        for row in connection.execute("PRAGMA table_info(async_delegations)")
    ]


def test_delegation_writer_should_initialize_session_cost_tables_when_store_is_fresh(
    tmp_path,
):
    with closing(sqlite3.connect(tmp_path / "state.db")) as connection, connection:
        _initialize_schema(connection)

        cost = connection.execute(
            "SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) "
            "FROM sessions WHERE id = ?",
            ("first-turn",),
        ).fetchone()[0]

    assert cost == 0


@pytest.mark.parametrize(
    "opening_order", ["session-tool", "tool-session", "tool-tool-session"]
)
def test_delegation_schema_should_match_canonical_shape_in_every_opening_order(
    tmp_path, opening_order
):
    database_path = tmp_path / "state.db"
    for opener in opening_order.split("-"):
        if opener == "session":
            SessionDB(db_path=database_path).close()
        else:
            with closing(sqlite3.connect(database_path)) as connection, connection:
                _initialize_schema(connection)

    with closing(sqlite3.connect(":memory:")) as reference, reference:
        reference.executescript(SCHEMA_SQL)
        expected = _delegation_shape(reference)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        actual = _delegation_shape(connection)

    assert actual == expected
    assert ("origin_session_id", "TEXT", 1, "''", 0) in actual


@pytest.mark.parametrize("opener", ["session", "tool"])
def test_schema_should_preserve_legacy_delegation_when_origin_column_is_missing(
    tmp_path, opener
):
    database_path = tmp_path / "state.db"
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.executescript(SCHEMA_SQL)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(async_delegations)")
        }
        if "origin_session_id" in columns:
            connection.execute(
                "ALTER TABLE async_delegations DROP COLUMN origin_session_id"
            )
        connection.execute(
            "INSERT INTO async_delegations "
            "(delegation_id, origin_session, state, dispatched_at, updated_at) "
            "VALUES ('retained', 'conversation', 'completed', 1, 1)"
        )

    if opener == "session":
        SessionDB(db_path=database_path).close()
    else:
        with closing(sqlite3.connect(database_path)) as connection, connection:
            _initialize_schema(connection)

    with closing(sqlite3.connect(database_path)) as connection, connection:
        row = connection.execute(
            "SELECT delegation_id, origin_session_id FROM async_delegations"
        ).fetchone()

    assert row == ("retained", "")
