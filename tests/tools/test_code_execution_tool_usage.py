#!/usr/bin/env python3
"""Tests for execute_code's off-transcript tool-usage accounting.

A sandbox RPC call dispatches through the normal tool handler but leaves no
``role='tool'`` message behind, so anything counting tool usage from the
transcript sees a whole script as one ``execute_code`` call. ``session_tool_usage``
is the record that keeps those calls countable — for usage analytics, and (in the
Omnio deployment) for metering paid tools a script drives, such as web_search or
a connector.

Invariants guarded here:
  - counts come from EXECUTED calls, and survive every script exit path;
  - repeat runs ACCUMULATE rather than overwrite;
  - accounting failure never breaks the tool (best-effort contract);
  - the informational ``inner_tool_calls`` payload matches what was recorded.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror the sibling execute_code suites — pin the local backend under xdist."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from contextlib import contextmanager

from gateway.session_context import set_current_session_id
from hermes_state import SessionDB
from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    _flush_inner_tool_usage,
    _inner_tool_counts,
    execute_code,
)

SESSION_ID = "sess-ptc-accounting"


@contextmanager
def _active_session(session_id: str):
    """Make *session_id* the process's current session, then restore.

    Uses the real setter rather than patching ``get_session_env``: that helper
    serves every session env var the sandbox setup reads (cwd, HERMES_HOME, ...),
    so a blanket patch returning a session id breaks script execution itself.
    """
    previous = os.environ.get("HERMES_SESSION_ID", "")
    set_current_session_id(session_id)
    try:
        yield
    finally:
        set_current_session_id(previous)


def _mock_handle_function_call(function_name, function_args, task_id=None, **kwargs):
    """Minimal dispatcher: succeed for terminal/read_file, error otherwise."""
    if function_name == "terminal":
        return json.dumps({"output": "mock", "exit_code": 0})
    if function_name == "read_file":
        return json.dumps({"content": "line1\n", "total_lines": 1})
    return json.dumps({"error": f"Unknown tool: {function_name}"})


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

class TestInnerToolCounts(unittest.TestCase):
    """_inner_tool_counts — the log is the source of truth for what executed."""

    def test_empty_log(self):
        self.assertEqual(_inner_tool_counts([]), {})

    def test_counts_repeats_per_tool(self):
        log = [
            {"tool": "terminal", "args_preview": "", "duration": 0.1},
            {"tool": "terminal", "args_preview": "", "duration": 0.1},
            {"tool": "read_file", "args_preview": "", "duration": 0.1},
        ]
        self.assertEqual(_inner_tool_counts(log), {"terminal": 2, "read_file": 1})

    def test_ignores_entries_without_a_tool_name(self):
        """A malformed entry must not become a phantom '' tool row."""
        log = [{"tool": ""}, {"tool": None}, {}, {"tool": "web_search"}]
        self.assertEqual(_inner_tool_counts(log), {"web_search": 1})


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestRecordToolUsage(unittest.TestCase):
    """SessionDB.record_tool_usage — accumulating upsert, per (tool, source)."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.db = SessionDB(db_path=Path(self._tmp.name) / "state.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_records_and_reads_back(self):
        self.db.record_tool_usage(
            SESSION_ID, {"web_search": 3, "terminal": 1}, source="execute_code"
        )
        self.assertEqual(
            self.db.read_tool_usage(SESSION_ID),
            {"web_search": 3, "terminal": 1},
        )

    def test_second_run_accumulates(self):
        """Two scripts in one session must sum, not clobber — the whole point of
        a cumulative counter is that a re-read can't lose earlier calls."""
        self.db.record_tool_usage(SESSION_ID, {"web_search": 2}, source="execute_code")
        self.db.record_tool_usage(SESSION_ID, {"web_search": 5}, source="execute_code")
        self.assertEqual(self.db.read_tool_usage(SESSION_ID), {"web_search": 7})

    def test_source_scoping(self):
        self.db.record_tool_usage(SESSION_ID, {"web_search": 2}, source="execute_code")
        self.db.record_tool_usage(SESSION_ID, {"web_search": 4}, source="other")
        self.assertEqual(self.db.read_tool_usage(SESSION_ID), {"web_search": 6})
        self.assertEqual(
            self.db.read_tool_usage(SESSION_ID, source="execute_code"),
            {"web_search": 2},
        )

    def test_zero_and_empty_are_noops(self):
        self.db.record_tool_usage(SESSION_ID, {}, source="execute_code")
        self.db.record_tool_usage(SESSION_ID, {"terminal": 0}, source="execute_code")
        self.assertEqual(self.db.read_tool_usage(SESSION_ID), {})

    def test_no_session_id_is_a_noop(self):
        self.db.record_tool_usage("", {"terminal": 1}, source="execute_code")
        self.assertEqual(self.db.read_tool_usage(SESSION_ID), {})

    def test_sessions_row_is_created_for_the_fk(self):
        """The FK to sessions(id) is enforced (PRAGMA foreign_keys=ON), so the
        writer must materialise the session row rather than fail the insert."""
        self.db.record_tool_usage(SESSION_ID, {"terminal": 1}, source="execute_code")
        row = self.db._conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (SESSION_ID,)
        ).fetchone()
        self.assertIsNotNone(row)


class TestFlushInnerToolUsage(unittest.TestCase):
    """_flush_inner_tool_usage — resolves the session, writes, never raises."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "state.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _flush(self, log, session_id=SESSION_ID):
        with patch("hermes_state.DEFAULT_DB_PATH", self.db_path):
            with _active_session(session_id):
                _flush_inner_tool_usage(log)

    def test_writes_counts_for_the_current_session(self):
        self._flush([{"tool": "web_search"}, {"tool": "web_search"}, {"tool": "terminal"}])
        db = SessionDB(db_path=self.db_path)
        self.assertEqual(
            db.read_tool_usage(SESSION_ID, source="execute_code"),
            {"web_search": 2, "terminal": 1},
        )

    def test_empty_log_writes_nothing(self):
        self._flush([])
        # No DB file needs to exist for a no-op flush.
        self.assertFalse(self.db_path.exists())

    def test_no_session_identity_is_not_an_error(self):
        """Bare CLI / harness runs have no session to attribute calls to."""
        self._flush([{"tool": "terminal"}], session_id="")
        self.assertFalse(self.db_path.exists())

    def test_write_failure_is_swallowed(self):
        """Best-effort contract: a broken state.db must not fail the script."""
        with _active_session(SESSION_ID):
            with patch(
                "hermes_state.SessionDB.record_tool_usage",
                side_effect=RuntimeError("db exploded"),
            ):
                _flush_inner_tool_usage([{"tool": "terminal"}])  # must not raise


# ---------------------------------------------------------------------------
# End to end through execute_code
# ---------------------------------------------------------------------------

@unittest.skipIf(
    sys.platform == "win32",
    "Sandbox integration tests are POSIX-only in this suite (see sibling suites).",
)
class TestExecuteCodeAccountsInnerCalls(unittest.TestCase):
    """The calls a real script makes end up in state.db and in the payload."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "state.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, code):
        with patch("hermes_state.DEFAULT_DB_PATH", self.db_path):
            with _active_session(SESSION_ID):
                with patch(
                    "model_tools.handle_function_call",
                    side_effect=_mock_handle_function_call,
                ):
                    raw = execute_code(
                        code=code,
                        task_id="test-ptc-accounting",
                        enabled_tools=list(SANDBOX_ALLOWED_TOOLS),
                    )
        return json.loads(raw)

    def _recorded(self):
        return SessionDB(db_path=self.db_path).read_tool_usage(
            SESSION_ID, source="execute_code"
        )

    def test_successful_script_records_every_call(self):
        result = self._run(
            "from hermes_tools import terminal, read_file\n"
            "terminal('echo one')\n"
            "terminal('echo two')\n"
            "read_file(path='/etc/hostname')\n"
            "print('done')\n"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_calls_made"], 3)
        self.assertEqual(result["inner_tool_calls"], {"terminal": 2, "read_file": 1})
        self.assertEqual(self._recorded(), {"terminal": 2, "read_file": 1})

    def test_script_making_no_calls_records_nothing(self):
        result = self._run("print(2 + 2)")
        self.assertEqual(result["status"], "success")
        self.assertNotIn("inner_tool_calls", result)
        self.assertEqual(self._recorded(), {})

    def test_crashing_script_still_records_calls_it_made(self):
        """The flush lives in `finally` precisely so a script that dies after
        spending money still accounts for what it spent."""
        result = self._run(
            "from hermes_tools import terminal\n"
            "terminal('echo one')\n"
            "raise SystemExit('boom')\n"
        )
        self.assertNotEqual(result["status"], "success")
        self.assertEqual(self._recorded(), {"terminal": 1})
