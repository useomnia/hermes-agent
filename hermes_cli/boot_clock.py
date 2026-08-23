"""Process-start clock, so a slow boot can be attributed rather than guessed.

`Starting Hermes Gateway...` is the first thing the gateway logs, and everything
before it — interpreter startup, the CLI's import graph, argument parsing, the
lazy import of the gateway and agent modules, building the gateway itself — is
invisible. On a freshly provisioned sandbox that window measures ~10-11 s.

Import weight is not the cause: reaching the CLI entry point is ~0.5-1.0 s, the
lazy `gateway.run` import ~20 ms, and a cold read of a 12 MB shared object on the
same box runs at 1090 MB/s, so it is not first-read cost either. Measured with
the checkpoints below, the window is dominated by work rather than loading —
~2.5 s in the cold bundled-skills sync, and ~7.3 s inside `start_gateway` before
it logs anything.

Checkpoints stamped along the way turn one opaque number into a breakdown. Each
records the process's age when it was reached, so consecutive differences are the
phases between them and the last is the gateway's own start.
"""

from __future__ import annotations

import os
import re

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


def format_timeline(end_name: str) -> str:
    """`to_<name>_ms=…` for every checkpoint plus the named end, or ''.

    Values are ages rather than durations, so a phase is the difference between
    two neighbours. That keeps every field meaningful even when a checkpoint in
    the middle was never reached, which a list of durations could not.
    """
    if re.fullmatch(r"[a-z][a-z0-9_]*", end_name) is None:
        raise ValueError("boot timeline end name must be a lowercase identifier")
    elapsed_at_end = process_elapsed_seconds()
    if elapsed_at_end is None:
        return ""
    fields = [f"to_{name}_ms={elapsed * 1000:.0f}" for name, elapsed in _checkpoints]
    fields.append(f"to_{end_name}_ms={elapsed_at_end * 1000:.0f}")
    return " ".join(fields)


def format_preamble() -> str:
    """The historical first-log timeline, ending in ``to_start_ms``."""
    return format_timeline("start")
