"""SQLite result-code classification, adapted from upstream 8ac45786bf95."""

import sqlite3


def is_sqlite_lock_error(exc_or_str) -> bool:
    """Result codes win over text: FTS can report BUSY as a constructor failure."""
    code = getattr(exc_or_str, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    text = str(exc_or_str).lower()
    return "locked" in text or "busy" in text
