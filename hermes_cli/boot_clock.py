"""Process-start clock, so a slow boot can be attributed rather than guessed.

`Starting Hermes Gateway...` is the first thing the gateway logs, and everything
before it — interpreter startup, the CLI's import graph, argument parsing, the
lazy import of the gateway and agent modules — is invisible. On a freshly
provisioned sandbox that invisible window has been measured at ~10 s against
~1.8 s for the same imports once the process is warm, and no log line could say
which part was responsible.

Two numbers close that gap: how long the process took to reach `main()` (the
interpreter plus every module imported at CLI import time), and how long it took
to reach the gateway's own start. Their difference is the work the CLI does after
its imports.
"""

from __future__ import annotations

import os

# Seconds from process fork to the CLI entry point, stamped once by `mark_main`.
# None until the entry point runs, and on any platform without procfs.
elapsed_at_main: float | None = None


def process_elapsed_seconds() -> float | None:
    """Seconds since THIS process was forked, or None when unavailable.

    Read from procfs rather than a module-import timestamp: by the time any of
    our code runs, the interpreter startup and part of the import graph have
    already happened, and those are exactly what we are trying to measure.
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


def mark_main() -> None:
    """Record how long the process took to reach the CLI entry point."""
    global elapsed_at_main
    elapsed_at_main = process_elapsed_seconds()


def format_preamble() -> str:
    """`key=value` boot-phase fields for the gateway's start log, or ''.

    `to_main_ms` is interpreter startup plus the CLI import graph; `to_start_ms`
    is the whole preamble up to the gateway starting, so the difference is the
    CLI's own post-import work (argument parsing, config, lazy imports).
    """
    to_start = process_elapsed_seconds()
    if to_start is None:
        return ""
    fields = [f"to_start_ms={to_start * 1000:.0f}"]
    if elapsed_at_main is not None:
        fields.insert(0, f"to_main_ms={elapsed_at_main * 1000:.0f}")
    return " ".join(fields)
