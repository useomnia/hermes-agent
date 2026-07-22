"""Shared blocking-wait mechanics for interactive tool gates.

Each gate owns a separate :class:`BlockingWaitRegistry` instance so its
surfaces, waiters, and completion reasons remain isolated.  The registry owns
only the mechanical layer: surface lifetime, waiter queues, blocking with
heartbeats and interrupt checks, resolution ordering, and reset cleanup.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

ResultT = TypeVar("ResultT")
SurfaceT = TypeVar("SurfaceT")
PayloadT = TypeVar("PayloadT")


@dataclass(slots=True)
class BlockingWaitEntry(Generic[ResultT, PayloadT]):
    """One parked call and the gate-specific payload associated with it."""

    event: threading.Event
    tool_call_id: str
    payload: PayloadT | None = None
    result: ResultT | None = None


@dataclass(frozen=True, slots=True)
class _Surface(Generic[SurfaceT]):
    token: object
    value: SurfaceT | None


class BlockingWaitRegistry(Generic[ResultT, SurfaceT, PayloadT]):
    """Own session-scoped surfaces and FIFO queues of blocking waiters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._surfaces: dict[str, _Surface[SurfaceT]] = {}
        self._waits: dict[
            str, list[BlockingWaitEntry[ResultT, PayloadT]]
        ] = {}
        self._completion_reasons: dict[tuple[str, str], str] = {}

    def register_surface(self, session_key: str, value: SurfaceT | None = None) -> object:
        """Register a surface and return the fresh token that owns it."""
        token = object()
        with self._lock:
            self._surfaces[session_key] = _Surface(token=token, value=value)
        return token

    def has_surface(self, session_key: str) -> bool:
        """Return whether a surface is registered for *session_key*."""
        with self._lock:
            return session_key in self._surfaces

    def surface_value(self, session_key: str) -> SurfaceT | None:
        """Return a registered surface's value, or ``None`` when absent.

        Use :meth:`has_surface` when a registered ``None`` value must be
        distinguished from an absent surface.
        """
        with self._lock:
            surface = self._surfaces.get(session_key)
            return surface.value if surface is not None else None

    def unregister_surface(
        self,
        session_key: str,
        token: object,
        *,
        on_unregister: Callable[[], None] | None = None,
    ) -> bool:
        """Drop an owned surface and release its waiters.

        A stale token is a complete no-op.  ``on_unregister`` runs under the
        registry lock before waiter events are signalled, allowing a gate to
        clear related semantic state without racing a resumed worker.
        """
        waiters: list[BlockingWaitEntry[ResultT, PayloadT]] = []
        try:
            with self._lock:
                surface = self._surfaces.get(session_key)
                if surface is None or surface.token is not token:
                    return False
                self._surfaces.pop(session_key, None)
                self._drop_completion_reasons_locked(session_key)
                waiters = self._waits.pop(session_key, [])
                if on_unregister is not None:
                    on_unregister()
        finally:
            self._signal(waiters)
        return True

    def wait(
        self,
        session_key: str,
        tool_call_id: str,
        timeout_s: float,
        activity_label: str,
        *,
        payload: PayloadT | None = None,
        on_parked: Callable[[SurfaceT | None], None] | None = None,
    ) -> tuple[ResultT | None, str | None]:
        """Park a call until it resolves, expires, or is interrupted.

        The waiter is enqueued before ``on_parked`` runs, so a gate may notify
        a surface that resolves synchronously.  The returned reason is
        ``"no_surface"`` when no interactive surface is registered,
        ``"expired"`` on deadline expiry, ``"cancelled"`` on interruption or
        external release, and ``None`` for a resolved value.  Only expired and
        cancelled waits are recorded for later consumption.

        If ``on_parked`` raises, the waiter is removed and the exception is
        propagated; no completion reason is recorded because the interaction
        was never successfully presented.
        """
        call_id = tool_call_id or ""
        with self._lock:
            surface = self._surfaces.get(session_key)
            if surface is None:
                return None, "no_surface"
            self._completion_reasons.pop((session_key, call_id), None)
            entry = BlockingWaitEntry[ResultT, PayloadT](
                event=threading.Event(),
                tool_call_id=call_id,
                payload=payload,
            )
            self._waits.setdefault(session_key, []).append(entry)
            surface_value = surface.value

        try:
            if on_parked is not None:
                on_parked(surface_value)
        except Exception:
            self._drop_wait(session_key, entry)
            raise

        touch_activity_if_due: Callable[..., None] | None = None
        try:
            from tools.environments.base import (
                touch_activity_if_due as _touch_activity_if_due,
            )

            touch_activity_if_due = _touch_activity_if_due
        except Exception:  # pragma: no cover - optional in stripped runtimes
            pass

        is_interrupted: Callable[[], bool] | None = None
        try:
            from tools.interrupt import is_interrupted as _is_interrupted

            is_interrupted = _is_interrupted
        except Exception:  # pragma: no cover - optional in stripped runtimes
            pass

        now = time.monotonic()
        deadline = now + max(timeout_s, 0)
        activity = {"last_touch": now, "start": now}
        expired = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                expired = True
                break
            if entry.event.wait(timeout=min(1.0, remaining)):
                break
            if is_interrupted is not None and is_interrupted():
                break
            if touch_activity_if_due is not None:
                touch_activity_if_due(activity, activity_label)

        self._drop_wait(session_key, entry)
        reason = None
        if entry.result is None:
            reason = "expired" if expired else "cancelled"
            with self._lock:
                self._completion_reasons[(session_key, call_id)] = reason
        return entry.result, reason

    def resolve(
        self,
        session_key: str,
        tool_call_id: str,
        value: ResultT,
        on_release: (
            Callable[[BlockingWaitEntry[ResultT, PayloadT]], None] | None
        ) = None,
        *,
        fallback_on_miss: bool = False,
    ) -> bool:
        """Resolve one matching waiter, or the FIFO head when no id is given.

        ``on_release`` runs under the lock after the result is assigned and
        before the event is signalled.  This preserves commit-before-release
        ordering for gate state consumed immediately by the resumed worker.
        ``fallback_on_miss`` exists for a gate whose established single-waiter
        contract treats an unknown defensive call id as a FIFO resolution.
        """
        entry: BlockingWaitEntry[ResultT, PayloadT] | None = None
        try:
            with self._lock:
                queue = self._waits.get(session_key)
                entry = self._pop_waiter_locked(
                    queue,
                    tool_call_id or "",
                    fallback_on_miss=fallback_on_miss,
                )
                if queue is not None and not queue:
                    self._waits.pop(session_key, None)
                if entry is None:
                    return False
                entry.result = value
                if on_release is not None:
                    on_release(entry)
        finally:
            if entry is not None:
                entry.event.set()
        return True

    def waiter_payload(
        self, session_key: str, tool_call_id: str = ""
    ) -> PayloadT | None:
        """Return the matching waiter's payload without removing it."""
        with self._lock:
            queue = self._waits.get(session_key) or []
            if tool_call_id:
                entry = next(
                    (
                        candidate
                        for candidate in queue
                        if candidate.tool_call_id == tool_call_id
                    ),
                    None,
                )
            else:
                entry = queue[0] if queue else None
            return entry.payload if entry is not None else None

    def consume_completion_reason(
        self, session_key: str, tool_call_id: str = ""
    ) -> str | None:
        """Return and clear one unresolved wait's completion reason."""
        with self._lock:
            return self._completion_reasons.pop(
                (session_key, tool_call_id or ""), None
            )

    def consume_session_completion_reason(self, session_key: str) -> str | None:
        """Return and clear the sole unresolved reason for a session.

        This supports gates that guarantee at most one concurrent wait per
        session and therefore expose a per-session completion API without a
        call id.
        """
        with self._lock:
            key = next(
                (key for key in self._completion_reasons if key[0] == session_key),
                None,
            )
            return self._completion_reasons.pop(key, None) if key is not None else None

    def clear(self, session_key: str) -> None:
        """Blindly clear a session and release every parked waiter."""
        with self._lock:
            self._surfaces.pop(session_key, None)
            self._drop_completion_reasons_locked(session_key)
            waiters = self._waits.pop(session_key, [])
        self._signal(waiters)

    def pending_count(self, session_key: str) -> int:
        """Return the number of currently parked calls for diagnostics/tests."""
        with self._lock:
            return len(self._waits.get(session_key, ()))

    def _drop_wait(
        self,
        session_key: str,
        entry: BlockingWaitEntry[ResultT, PayloadT],
    ) -> None:
        with self._lock:
            queue = self._waits.get(session_key)
            if queue and entry in queue:
                queue.remove(entry)
            if queue is not None and not queue:
                self._waits.pop(session_key, None)

    @staticmethod
    def _pop_waiter_locked(
        queue: list[BlockingWaitEntry[ResultT, PayloadT]] | None,
        tool_call_id: str,
        *,
        fallback_on_miss: bool,
    ) -> BlockingWaitEntry[ResultT, PayloadT] | None:
        if not queue:
            return None
        if tool_call_id:
            for index, candidate in enumerate(queue):
                if candidate.tool_call_id == tool_call_id:
                    return queue.pop(index)
            if not fallback_on_miss:
                return None
        return queue.pop(0)

    def _drop_completion_reasons_locked(self, session_key: str) -> None:
        for key in [key for key in self._completion_reasons if key[0] == session_key]:
            self._completion_reasons.pop(key, None)

    @staticmethod
    def _signal(waiters: list[BlockingWaitEntry[ResultT, PayloadT]]) -> None:
        for entry in waiters:
            entry.event.set()
