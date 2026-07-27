"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from typing import Optional

_mcp_discovery_lock = threading.Lock()
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        return isinstance(mcp_servers, dict) and len(mcp_servers) > 0
    except Exception:
        # Be conservative: if config probing fails, try discovery in the
        # background so startup still can't block.
        return True


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one shared background MCP discovery thread for this process."""
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            return
        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        def _discover() -> None:
            try:
                from tools.mcp_tool import discover_mcp_tools

                discover_mcp_tools()
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def mcp_discovery_was_started() -> bool:
    """True if a background discovery thread exists for this process.

    ``start_background_mcp_discovery`` is a no-op when the cheap config probe
    finds no ``mcp_servers``, so a caller that *depends* on discovery having run
    (rather than merely benefiting from it) needs to know the difference between
    "already finished" and "never started". Mirrors
    ``tui_gateway.entry.mcp_discovery_in_flight``, but reports whether the thread
    was ever created rather than whether it is still alive.
    """
    return _mcp_discovery_thread is not None


# Join bound for callers that must not lose tools. Sized ABOVE
# ``discover_mcp_tools``'s own internal 120s per-server wait rather than at
# ``mcp_discovery_timeout`` (default 1.5s): the CLI/TUI accept a degraded first
# prompt and repair it with the late-binding refresh, but the socket gateway used
# to wait for discovery in full before serving at all, and its first turn
# routinely needs every tool (an Omnio sandbox registers 221, of which 152 come
# from a connectors server that takes ~2-3.5s to enumerate). Waiting long enough
# to preserve that exactly is the point; the bound exists only so a wedged thread
# can't hang a turn forever.
AGENT_BUILD_JOIN_SECONDS = 130.0

# Set once if a join ever exhausts its bound with discovery still running. That
# bound is a backstop for a WEDGED discovery thread, and a wedged thread does not
# recover — so charging it again to the next caller would turn one stuck server
# into *every* later request paying the full bound and still getting a partial
# registry. After the first exhaustion the process gives up for good and falls
# back to the late-binding refresh, the same degradation the CLI and TUI have
# always accepted. Written once, from whichever thread got there first; a benign
# double-write only repeats the log line.
_join_abandoned = False


def _abandon_join() -> None:
    global _join_abandoned

    if _join_abandoned:
        return
    _join_abandoned = True
    import logging

    logging.getLogger(__name__).warning(
        "MCP discovery did not finish within %.0fs; giving up for this process. "
        "Agents built from here on may start with an incomplete MCP tool registry "
        "and rely on the late-binding refresh.",
        AGENT_BUILD_JOIN_SECONDS,
    )


def ensure_mcp_discovery_complete(timeout: "float | None" = None) -> None:
    """Block until MCP discovery has run, for callers that need the full registry.

    ``AIAgent`` reads the tool registry ONCE at construction and never re-reads
    it (see ``tools.mcp_tool.refresh_agent_mcp_tools``), so an agent built while
    discovery is still in flight cannot call the missing tools for its whole
    lifetime. Callers that would rather wait than lose tools use this;
    ``wait_for_mcp_discovery`` remains the bounded, best-effort variant.

    **Blocking — call it only from a worker thread.** Every current caller sits
    inside an agent-building closure already dispatched through
    ``run_in_executor``, so it never touches the loop thread carrying platform
    heartbeats; putting it back on the loop would re-create #16856.

    Never raises: a timed-out or failed join degrades to the existing
    late-binding refresh rather than killing the turn. And it only ever times out
    ONCE per process — see ``_join_abandoned``, which stops a permanently wedged
    discovery thread from re-charging the full bound to every later agent build.
    """
    try:
        if mcp_discovery_was_started():
            if _join_abandoned:
                # An earlier caller already waited out the full bound on this
                # thread and it never finished. Don't re-charge that wait.
                return
            wait_for_mcp_discovery(
                timeout=AGENT_BUILD_JOIN_SECONDS if timeout is None else timeout
            )
            thread = _mcp_discovery_thread
            if timeout is None and thread is not None and thread.is_alive():
                # Only the default bound retires the join. An explicit (smaller)
                # timeout is a caller saying "wait this long", not evidence that
                # discovery is wedged.
                _abandon_join()
            return
        # No thread was started — ``start_background_mcp_discovery`` skips when
        # its cheap ``read_raw_config`` probe finds no ``mcp_servers``. That probe
        # reads the raw file while discovery itself reads the migrated/merged
        # config, so the two can in principle disagree. Discovering inline here
        # means a caller cannot lose a server that a blocking discovery would
        # have connected; it is idempotent, and a no-op for the overwhelmingly
        # common "genuinely no MCP servers" case.
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "MCP discovery join before agent build failed", exc_info=True
        )


def _resolve_discovery_timeout(explicit: "float | None") -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config > default.

    Reads ``mcp_discovery_timeout`` from config.yaml, defaulting to the value in
    ``DEFAULT_CONFIG`` (single source of truth) when the key is absent. Kept lazy
    and fail-safe — a missing/invalid value or a broken config falls back to a
    short safe bound so startup can never hang or crash.
    """
    if explicit is not None:
        return explicit
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get("mcp_discovery_timeout", 1.5))
        raw = (load_config() or {}).get("mcp_discovery_timeout", default)
        val = float(raw)
        return val if val > 0 else default
    except Exception:
        return 1.5


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.
    """
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout))
