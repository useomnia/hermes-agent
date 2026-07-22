"""Blocking user-input gate for the ``request_user_input`` tool (Omnia / Omnio).

The agent's ``request_user_input`` tool (shipped by the omnio_interaction plugin)
asks the user a structured question — choice / confirm / multi_choice / secret /
form — rendered as a card in the Omnia chat. This BLOCKS the agent worker thread
until the user answers (or a timeout) and returns the answer as the tool's
result, so the answer becomes part of the SAME turn — the way the write-tool
approval gate (``tools/tool_approval.py``) resumes a gated call inline, rather
than the older "end the turn, the answer is the next message" shape. That
inline-resume shape holds only when the user answers before the timeout: a
timeout now ENDS the turn (mirroring the plugin's "presented" sentinel) instead
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
from typing import Optional

from tools.blocking_wait import BlockingWaitRegistry
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

# This gate owns a separate instance from tool_approval, so answer and approval
# state cannot cross-talk. User-input surfaces carry no gate-specific value.
_wait_registry: BlockingWaitRegistry[str, None, None] = BlockingWaitRegistry()


def register_user_input_session(session_key: str) -> object | None:
    """Mark an interactive chat surface and return its ownership token."""
    if not session_key:
        return None
    return _wait_registry.register_surface(session_key)


def unregister_user_input_session(session_key: str, token: object) -> None:
    """Drop the interactive-surface mark AND release any still-blocked waiters for
    this session when *token* still owns its registration."""
    if not session_key:
        return
    _wait_registry.unregister_surface(session_key, token)


def _input_timeout() -> int:
    try:
        return int(os.environ.get(_ENV_TIMEOUT, "") or _DEFAULT_TIMEOUT_S)
    except (ValueError, TypeError):
        return _DEFAULT_TIMEOUT_S


def consume_user_input_completion_reason(session_key: str) -> Optional[str]:
    """Return and clear the sole unresolved wait reason for a session.

    ``request_user_input`` never runs concurrently within one session, so its
    public contract intentionally needs no tool-call id.
    """
    if not session_key:
        return None
    return _wait_registry.consume_session_completion_reason(session_key)


def await_user_input(session_key: str, tool_call_id: str = "") -> Optional[str]:
    """Block until the user answers the ``request_user_input`` card, or a timeout.

    Returns the user's answer string, or ``None`` when the wait times out or the
    agent is interrupted (the user stopped the turn, or the chat disconnected).
    The caller (the plugin) maps ``None`` to a "no_response" tool result, which
    the api_server seam treats as turn-ending — the card stays answerable in the
    chat and a late answer arrives as the next turn's user message.
    """
    if not session_key:
        # No conversation surface to receive an answer on — don't park forever.
        return None
    # The public completion contract is one slot per session. Clear a prior
    # sequential call's unconsumed reason before parking the next request.
    _wait_registry.consume_session_completion_reason(session_key)
    answer, _reason = _wait_registry.wait(
        session_key,
        tool_call_id,
        _input_timeout(),
        "waiting for user input",
    )
    return answer


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
    return _wait_registry.resolve(
        session_key,
        tool_call_id,
        answer if answer is not None else "",
        fallback_on_miss=True,
    )


def clear_session(session_key: str) -> None:
    """Release blocked input waiters during an explicit conversation reset."""
    if not session_key:
        return
    _wait_registry.clear(session_key)
