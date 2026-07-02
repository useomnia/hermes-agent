"""Blocking user-input gate for the ``request_user_input`` tool (Omnia / Omnio).

The agent's ``request_user_input`` tool (shipped by the omnio_interaction plugin)
asks the user a structured question — choice / confirm / multi_choice / secret /
form — rendered as a card in the Omnia chat. This BLOCKS the agent worker thread
until the user answers (or a timeout) and returns the answer as the tool's
result, so the answer becomes part of the SAME turn — the way the write-tool
approval gate (``tools/tool_approval.py``) resumes a gated call inline, rather
than the older "end the turn, the answer is the next message" shape. That
inline-resume shape holds only when the user answers before the timeout: a
timeout now ENDS the turn (mirroring the "presented" sentinel below) instead
of letting the agent keep working with a "no_response" result — the card
stays open and answerable in the chat, and a late answer arrives as the next
turn's user message.

The card itself rides the tool's ``running`` lifecycle event, emitted by the
api_server seam (``_on_tool_start``) with the tool's args under ``interaction``,
so this module owns only the BLOCK and its release: a per-session FIFO of
waiters, each parked on a ``threading.Event``, released by ``resolve_user_input``
when the user's answer arrives on the loopback endpoint
(``POST /v1/omnio/user-input``). Exactly one ``request_user_input`` blocks per
session at a time — the worker is parked on it — so a single FIFO per session
key is sufficient (``tool_call_id`` only refines the match defensively).

State is module-level, keyed by the session key (== the conversation's
``X-Hermes-Session-Id``), matching ``tools/tool_approval.py`` so a session reset
clears it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

from utils import env_var_enabled

logger = logging.getLogger(__name__)

# Killswitch: when truthy, request_user_input reverts to the OLD non-blocking
# behavior (the tool returns immediately, the api_server ends the turn, and the
# user's answer is the next message). Frozen at import — same rationale as
# tools/tool_approval.py's _DISABLED_FROZEN: reading it live would let in-process
# plugin code flip it mid-run. The env is process-constant per gateway spawn.
_ENV_DISABLED = "OMNIO_USER_INPUT_BLOCKING_DISABLED"
BLOCKING_DISABLED: bool = env_var_enabled(_ENV_DISABLED)

# How long (seconds) the agent blocks waiting for the user to answer. Matches
# the approval gate's 300s default. On timeout the turn ends (see api_server's
# _on_tool_complete) rather than letting the agent keep working with a
# "no_response" result — the card stays open and answerable in the chat, and
# the user's late answer arrives as the next turn's user message. The chat
# keepalive holds the SSE open while the worker is parked.
_ENV_TIMEOUT = "OMNIO_USER_INPUT_TIMEOUT"
_DEFAULT_TIMEOUT_S = 300

_lock = threading.Lock()
# session_key -> FIFO of blocked input waiters.
_waits: dict[str, list["_InputWait"]] = {}
# Sessions with an interactive chat surface that can render the card and POST an
# answer back. The api_server chat path registers the session here; a
# non-interactive caller (e.g. a proactive /v1/runs task) is absent, so the gate
# fails fast instead of parking the worker for the full timeout with no one to
# answer. Mirrors tools/tool_approval.py's "no notify cb → no surface" check.
_active_sessions: set[str] = set()


def register_user_input_session(session_key: str) -> None:
    """Mark a session as having an interactive chat surface (can answer cards)."""
    if not session_key:
        return
    with _lock:
        _active_sessions.add(session_key)


def unregister_user_input_session(session_key: str) -> None:
    """Drop the interactive-surface mark AND release any still-blocked waiters for
    this session, so a finished/interrupted run can't leave a worker parked."""
    if not session_key:
        return
    with _lock:
        _active_sessions.discard(session_key)
        waiters = _waits.pop(session_key, [])
    for entry in waiters:
        entry.event.set()  # answer stays None → caller maps to "no answer"


class _InputWait:
    """One blocked ``request_user_input`` call awaiting the user's answer."""

    __slots__ = ("event", "tool_call_id", "answer")

    def __init__(self, tool_call_id: str = "") -> None:
        self.event = threading.Event()
        self.tool_call_id = tool_call_id
        self.answer: Optional[str] = None  # the user's submitted answer text


def _input_timeout() -> int:
    try:
        return int(os.environ.get(_ENV_TIMEOUT, "") or _DEFAULT_TIMEOUT_S)
    except (ValueError, TypeError):
        return _DEFAULT_TIMEOUT_S


def _drop_wait(session_key: str, entry: "_InputWait") -> None:
    with _lock:
        queue = _waits.get(session_key)
        if queue and entry in queue:
            queue.remove(entry)
        if queue is not None and not queue:
            _waits.pop(session_key, None)


def await_user_input(session_key: str, tool_call_id: str = "") -> Optional[str]:
    """Block until the user answers the ``request_user_input`` card, or a timeout.

    Returns the user's answer string, or ``None`` when the wait times out or the
    agent is interrupted (the user stopped the turn, or the chat disconnected).
    The caller (the plugin) maps ``None`` to a "no answer" tool result so the
    model continues gracefully rather than confabulating an answer.
    """
    if not session_key:
        # No conversation surface to receive an answer on — don't park forever.
        return None
    with _lock:
        if session_key not in _active_sessions:
            # No interactive chat surface for this session (e.g. a proactive
            # /v1/runs task): fail fast rather than park for the full timeout
            # with no one to answer. Mirrors the approval gate's no-surface deny.
            return None
        entry = _InputWait(tool_call_id)
        _waits.setdefault(session_key, []).append(entry)

    # Block in short slices so we can heartbeat the inactivity tracker — without
    # it the gateway watchdog would kill the agent while the user is answering —
    # and observe the interrupt bit so a stop/disconnect releases the worker
    # promptly (agent.interrupt() fans the bit out to this tool-worker thread).
    # Mirrors tools/tool_approval.await_tool_approval.
    # Soft-optional imports: a stripped runtime may not ship these helpers. Import
    # into a temp and assign to an optional-typed local so the None fallback is a
    # type-clean assignment (importing the name directly would declare it as the
    # function type, which `None` then can't be assigned to).
    touch_activity_if_due: Callable[..., None] | None = None
    try:
        from tools.environments.base import (
            touch_activity_if_due as _touch_activity_if_due,
        )

        touch_activity_if_due = _touch_activity_if_due
    except Exception:  # pragma: no cover
        pass
    is_interrupted: Callable[[], bool] | None = None
    try:
        from tools.interrupt import is_interrupted as _is_interrupted

        is_interrupted = _is_interrupted
    except Exception:  # pragma: no cover
        pass

    now = time.monotonic()
    deadline = now + max(_input_timeout(), 0)
    activity = {"last_touch": now, "start": now}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if entry.event.wait(timeout=min(1.0, remaining)):
            break
        if is_interrupted is not None and is_interrupted():
            break  # answer stays None → caller maps to "no answer"
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity, "waiting for user input")

    _drop_wait(session_key, entry)
    return entry.answer


def resolve_user_input(session_key: str, answer: str, tool_call_id: str = "") -> bool:
    """Apply the user's answer posted from the Omnia chat: unblock the waiting
    ``request_user_input`` call.

    When ``tool_call_id`` is given the MATCHING waiter is released; otherwise (or
    if no match) the queue head (FIFO). Since exactly one input blocks per
    session, FIFO is the normal path. Returns ``False`` only for a malformed
    request (no session key) or when no call is currently waiting.
    """
    if not session_key:
        return False
    with _lock:
        queue = _waits.get(session_key)
        entry = None
        if queue:
            if tool_call_id:
                for index, candidate in enumerate(queue):
                    if candidate.tool_call_id == tool_call_id:
                        entry = queue.pop(index)
                        break
            if entry is None:
                entry = queue.pop(0)
        if queue is not None and not queue:
            _waits.pop(session_key, None)
    if entry is None:
        return False
    entry.answer = answer if answer is not None else ""
    entry.event.set()
    return True


def clear_session(session_key: str) -> None:
    """Release any blocked input waiters for a session (run end / reset), so a
    finished or interrupted run never leaves a worker thread parked. The released
    waiter's answer stays ``None`` → the caller maps it to a "no answer" result."""
    if not session_key:
        return
    with _lock:
        waiters = _waits.pop(session_key, [])
    for entry in waiters:
        entry.event.set()
