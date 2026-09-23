"""Per-thread interrupt signaling for all tools.

Provides thread-scoped interrupt tracking so that interrupting one agent
session does not kill tools running in other sessions.  This is critical
in the gateway where multiple agents run concurrently in the same process.

The agent stores its execution thread ID at the start of run_conversation()
and passes it to set_interrupt()/clear_interrupt().  Tools call
is_interrupted() which checks the CURRENT thread — no argument needed.

Usage in tools:
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Opt-in debug tracing — pairs with HERMES_DEBUG_INTERRUPT in
# tools/environments/base.py.  Enables per-call logging of set/check so the
# caller thread, target thread, and current state are visible when
# diagnosing "interrupt signaled but tool never saw it" reports.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode path forces `tools` logger to ERROR on CLI startup.
    # Force our own logger back to INFO so the trace is visible in agent.log.
    logger.setLevel(logging.INFO)

# Set of thread idents that have been interrupted.
_interrupted_threads: set[int] = set()
_lock = threading.Lock()

_execution_scope: ContextVar["ToolExecutionScope | None"] = ContextVar(
    "tool_execution_scope", default=None
)


class ToolExecutionScope:
    """Fence nested dispatch when its owning script stops or expires.

    Created on the script's execution thread, then shared with its RPC worker.
    Admission is the boundary after which an external call may already be in
    flight; cancellation never claims to roll that operation back.
    """

    def __init__(self, stop_event: threading.Event):
        self._stop = stop_event
        self._owner = threading.get_ident()
        self._parent = _execution_scope.get()
        self._deadline: float | None = None
        self._lock = threading.RLock()

    def start(self, timeout: float) -> None:
        with self._lock:
            self._deadline = time.monotonic() + timeout

    def cancel(self) -> None:
        with self._lock:
            self._stop.set()

    def is_cancelled(self) -> bool:
        with self._lock:
            with _lock:
                interrupted = self._owner in _interrupted_threads
            return (
                self._stop.is_set()
                or interrupted
                or (self._deadline is not None and time.monotonic() >= self._deadline)
                or (self._parent is not None and self._parent.is_cancelled())
            )

    def admit(self) -> bool:
        with self._lock:
            return not self.is_cancelled()


def current_execution_scope() -> ToolExecutionScope | None:
    return _execution_scope.get()


@contextmanager
def bind_execution_scope(scope: ToolExecutionScope):
    token = _execution_scope.set(scope)
    try:
        yield
    finally:
        _execution_scope.reset(token)


def set_interrupt(active: bool, thread_id: int | None = None) -> None:
    """Set or clear interrupt for a specific thread.

    Args:
        active: True to signal interrupt, False to clear it.
        thread_id: Target thread ident.  When None, targets the
                   current thread (backward compat for CLI/tests).
    """
    tid = thread_id if thread_id is not None else threading.current_thread().ident
    with _lock:
        if active:
            _interrupted_threads.add(tid)
        else:
            _interrupted_threads.discard(tid)
        _snapshot = set(_interrupted_threads) if _DEBUG_INTERRUPT else None
    if _DEBUG_INTERRUPT:
        logger.info(
            "[interrupt-debug] set_interrupt(active=%s, target_tid=%s) "
            "called_from_tid=%s current_set=%s",
            active, tid, threading.current_thread().ident, _snapshot,
        )


def is_interrupted() -> bool:
    """Check if an interrupt has been requested for the current thread.

    Safe to call from any thread — each thread only sees its own
    interrupt state.
    """
    tid = threading.current_thread().ident
    with _lock:
        interrupted = tid in _interrupted_threads
    scope = _execution_scope.get()
    return interrupted or (scope is not None and scope.is_cancelled())


def clear_current_thread_interrupt() -> None:
    """Clear any interrupt bit on the CURRENT thread.

    Gives a user-approved command a clean interrupt slate immediately before
    it spawns its child process, so a stale bit that landed on this thread
    during the blocking approval-wait cannot SIGINT the just-approved run
    (exit 130 + "[Command interrupted]").  Single-thread ordering on this tid
    keeps the DO-NOT-BREAK invariant intact: a *genuine* interrupt arriving
    after this call re-sets the bit on the same thread and is still observed by
    the executor's poll loop.  Call this directly, never via the
    _interrupt_event proxy (its .clear() binds to whatever thread runs it).
    """
    set_interrupt(False)  # thread_id=None -> current thread (see set_interrupt)


# ---------------------------------------------------------------------------
# Backward-compatible _interrupt_event proxy
# ---------------------------------------------------------------------------
# Some legacy call sites (code_execution_tool, process_registry, tests)
# import _interrupt_event directly and call .is_set() / .set() / .clear().
# This shim maps those calls to the per-thread functions above so existing
# code keeps working while the underlying mechanism is thread-scoped.

class _ThreadAwareEventProxy:
    """Drop-in proxy that maps threading.Event methods to per-thread state."""

    def is_set(self) -> bool:
        return is_interrupted()

    def set(self) -> None:  # noqa: A003
        set_interrupt(True)

    def clear(self) -> None:
        set_interrupt(False)

    def wait(self, timeout: float | None = None) -> bool:
        """Not truly supported — returns current state immediately."""
        return self.is_set()


_interrupt_event = _ThreadAwareEventProxy()
