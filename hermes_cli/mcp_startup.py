"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from contextlib import nullcontext
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
    """Spawn one shared background MCP discovery thread for this process.

    If the first background discovery run exits without connecting any MCP
    server (for example after startup cancellation / OOM restart), later calls
    are allowed to retry instead of permanently pinning the process in a
    "discovery already started" state with zero MCP tools.
    """
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            thread = _mcp_discovery_thread
            if thread is not None and thread.is_alive():
                return
            try:
                from tools.mcp_tool import get_mcp_status

                status = get_mcp_status() or []
                if any(entry.get("connected") for entry in status):
                    return
            except Exception:
                return
            logger.warning(
                "Background MCP discovery previously exited with no connected "
                "servers; retrying discovery thread"
            )
            _mcp_discovery_started = False
            _mcp_discovery_thread = None

        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        # Capture the caller's context-local HERMES_HOME override (profile
        # scoping in multi-profile processes like the dashboard/desktop
        # backend) and re-install it inside the discovery thread. ContextVars
        # do not propagate into bare threads, so without this a session
        # "switched" to profile X would discover the LAUNCH profile's
        # mcp_servers instead (#67605). The config gate above already runs on
        # the caller's thread, so it sees the same override.
        try:
            from hermes_constants import get_hermes_home_override

            home_override = get_hermes_home_override()
        except Exception:
            home_override = None

        def _discover() -> None:
            token = None
            try:
                from hermes_constants import set_hermes_home_override

                token = set_hermes_home_override(home_override)
            except Exception:
                token = None
            try:
                _discover_mcp_tools_without_interactive_oauth()
                try:
                    from tools.mcp_tool import get_mcp_status
                    status = get_mcp_status() or []
                    if not any(entry.get("connected") for entry in status):
                        logger.warning(
                            "Background MCP discovery completed with zero connected servers"
                        )
                except Exception:
                    logger.debug("Failed to inspect MCP status after background discovery", exc_info=True)
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)
            finally:
                if token is not None:
                    try:
                        from hermes_constants import reset_hermes_home_override

                        reset_hermes_home_override(token)
                    except Exception:
                        pass
                # Keep the completed Thread object as the single-flight
                # marker. Clearing it here made the first agent fall through
                # to ``discover_mcp_tools()`` and repeat the entire discovery
                # run after the background worker had already finished. A
                # later explicit startup call can still replace this marker
                # and retry when no server connected (see the status check
                # above).

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def mcp_discovery_was_started() -> bool:
    """True if a background discovery attempt exists for this process.

    ``start_background_mcp_discovery`` is a no-op when the cheap config probe
    finds no ``mcp_servers``, so a caller that *depends* on discovery having run
    (rather than merely benefiting from it) needs to know the difference between
    "already finished" and "never started". The completed Thread object is
    retained as that marker; ``mcp_discovery_in_flight`` remains the API for
    checking whether it is still alive.
    """
    with _mcp_discovery_lock:
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
        if _join_startup_discovery(timeout):
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


def _join_startup_discovery(timeout: "float | None") -> bool:
    """Join the startup discovery thread. False if there was never one.

    Blocking; see ``ensure_mcp_discovery_complete`` for the thread-safety rules.
    """
    if not mcp_discovery_was_started():
        return False
    if _join_abandoned:
        # An earlier caller already waited out the full bound on this thread and
        # it never finished. Don't re-charge that wait.
        return True
    wait_for_mcp_discovery(timeout=AGENT_BUILD_JOIN_SECONDS if timeout is None else timeout)
    with _mcp_discovery_lock:
        thread = _mcp_discovery_thread
    if timeout is None and thread is not None and thread.is_alive():
        # Only the default bound retires the join. An explicit (smaller) timeout
        # is a caller saying "wait this long", not evidence discovery is wedged.
        _abandon_join()
    return True


def wait_for_startup_mcp_discovery(timeout: "float | None" = None) -> None:
    """Join an in-flight startup discovery WITHOUT ever discovering inline.

    For callers that are about to run their own ``discover_mcp_tools()`` and only
    need the startup thread to be out of the way first — the MCP reload endpoint,
    which must not overlap startup discovery because ``register_mcp_servers``
    dedupes against connected servers but not against in-flight connects.

    The distinction from ``ensure_mcp_discovery_complete`` matters: that one
    discovers inline when no thread was started, which for a caller like reload
    would connect servers BEFORE it snapshots the previous set — reporting an
    empty ``added`` list and then tearing down and rebuilding what it just
    connected.

    **Blocking — call it only from a worker thread.**  Never raises.
    """
    try:
        _join_startup_discovery(timeout)
    except Exception:
        import logging

        logging.getLogger(__name__).debug("MCP startup discovery join failed", exc_info=True)


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


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.
    """
    with _mcp_discovery_lock:
        thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout))


def mcp_discovery_in_flight() -> bool:
    """Return True if THIS module's background discovery thread is still running.

    Mirrors ``tui_gateway.entry.mcp_discovery_in_flight`` for the surfaces that
    start discovery through ``start_background_mcp_discovery`` here (the desktop
    app + dashboard WebSocket sidecar via ``tui_gateway/ws.py``, and
    ``hermes dashboard``).  Those processes populate THIS module's
    ``_mcp_discovery_thread``, not ``tui_gateway.entry``'s, so the late-refresh
    scheduler must consult both to decide whether a slow server's tools are
    still pending (see #51587).
    """
    with _mcp_discovery_lock:
        thread = _mcp_discovery_thread
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """Block until THIS module's background discovery finishes, up to ``timeout``.

    Returns True if discovery has completed (thread absent or no longer alive),
    False if it is still running after the timeout.  Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and reports
    the outcome, for the off-critical-path late-refresh waiter.
    """
    with _mcp_discovery_lock:
        thread = _mcp_discovery_thread
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()
