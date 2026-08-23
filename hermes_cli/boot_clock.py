"""Process-start clock, so a slow boot can be attributed rather than guessed.

`Starting Hermes Gateway...` is the first thing the gateway logs, and everything
before it — interpreter startup, the CLI's import graph, argument parsing, the
lazy import of the gateway and agent modules, building the gateway itself — is
invisible. The original instrumented baseline on a freshly provisioned sandbox
was roughly 10–11 s (about 2.5 s in the cold bundled-skills sync and 7.3 s in
`start_gateway`). Those figures are historical, not a current startup target:
platform discovery, skills preparation, and approval setup now overlap or defer
work, so use the emitted checkpoints to attribute the current path.

Checkpoints stamped along the way turn one opaque number into a breakdown. Each
records the process's age when it was reached, so consecutive differences are the
phases between them and the last is the gateway's own start.
"""

from __future__ import annotations

import os
import re

# This is the monitoring contract for the gateway startup timeline. Keep it
# declarative rather than inspecting source text at test time: the names are
# shared by the CLI, runner, and API readiness phases, and a typo should be
# visible in review while instrumentation remains safe to call from any
# embedding. ``mark`` still accepts caller-defined names for compatibility.
BOOT_CHECKPOINT_NAMES = frozenset(
    {
        "main",
        "skills",
        "cli_import",
        "dispatch",
        "pre_start",
        "plugins",
        "relay",
        "session_recovery",
        "platforms",
        "runtime_ready",
        "fingerprint",
        "dup_guard",
        "skills_resync",
        "logging",
        "audit",
        "runner_init",
        "pid_lock",
        "mcp_discovery_start",
        "api_connect",
        "api_routes",
        "api_bound",
        "api_ready",
        "api_approvals_start",
    }
)

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
