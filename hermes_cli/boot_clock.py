"""Process-start clock, so a slow boot can be attributed rather than guessed.

`Starting Hermes Gateway...` is the first thing the gateway logs, and everything
before it — interpreter startup, the CLI's import graph, argument parsing, the
lazy import of the gateway and agent modules, building the gateway itself — is
invisible. On a freshly provisioned sandbox that window measures ~9.9 s, of which
only ~0.5 s is reaching the CLI entry point, so the cost is in the preamble that
follows rather than in import weight; a cold read of a 12 MB shared object on the
same box runs at 1090 MB/s, so it is not first-read cost either.

Checkpoints stamped along the way turn one opaque number into a breakdown. Each
records the process's age when it was reached, so consecutive differences are the
phases between them and the last is the gateway's own start.
"""

from __future__ import annotations

import os

# Ordered (name, process age in seconds) pairs, appended by `mark`. A list rather
# than a dict so a checkpoint reached twice stays visible instead of overwriting.
_checkpoints: list[tuple[str, float]] = []


def process_elapsed_seconds() -> float | None:
    """Seconds since THIS process was forked, or None when unavailable.

    Read from procfs rather than a module-import timestamp: by the time any of
    our code runs, interpreter startup and part of the import graph have already
    happened, and those are exactly what we are trying to measure.
    """
    try:
        with open("/proc/self/stat", encoding="ascii") as handle:
            # The comm field can itself contain spaces and parentheses, so split
            # on the LAST ')' — the remaining fields are positional from `state`,
            # which puts starttime (field 22 overall) at index 19.
            fields = handle.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/uptime", encoding="ascii") as handle:
            uptime_seconds = float(handle.read().split()[0])
    except (OSError, IndexError, ValueError):
        return None
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    if not ticks_per_second:
        return None
    return uptime_seconds - start_ticks / ticks_per_second


def mark(name: str) -> None:
    """Record the process's age on reaching a named point in the boot."""
    elapsed = process_elapsed_seconds()
    if elapsed is not None:
        _checkpoints.append((name, elapsed))


def format_preamble() -> str:
    """`to_<name>_ms=…` for every checkpoint plus `to_start_ms`, or ''.

    Values are ages rather than durations, so a phase is the difference between
    two neighbours. That keeps every field meaningful even when a checkpoint in
    the middle was never reached, which a list of durations could not.
    """
    to_start = process_elapsed_seconds()
    if to_start is None:
        return ""
    fields = [f"to_{name}_ms={elapsed * 1000:.0f}" for name, elapsed in _checkpoints]
    fields.append(f"to_start_ms={to_start * 1000:.0f}")
    return " ".join(fields)
