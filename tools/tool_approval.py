"""Per-call user approval for connector WRITE tools (Omnia / Omnio).

Reads run ungated. WRITE actions on a customer's connected third-party SaaS
(e.g. drafting an email, creating a doc) require an explicit per-call user
approval rendered as a control in the Omnia chat.

A tool is gated when the connectors MCP route advertised it as NOT read-only
(MCP ``readOnlyHint=False``). The route derives that from its own write allowlist
and stamps it per tool, so the gated set tracks exactly what the route exposes —
it can't drift from a provision-time snapshot. The gate FAILS CLOSED: a
connectors tool with no read-only hint (e.g. an old route that hasn't advertised
it yet) is gated rather than run as an ungated write.

This gate is **blocking** — like ``tools.approval`` (the dangerous-shell-command
gate). When the agent calls a gated write, the guard surfaces the approval card
on the chat stream and BLOCKS the agent worker thread until the user resolves it
(or a timeout). On approval the SAME tool call proceeds inline and the agent
gets the real result; on deny it gets a denial. This is deliberately not the
non-blocking "return a prompt, ask the agent to re-issue" pattern: an MCP tool
result is wrapped as untrusted data the model is told to ignore, so it cannot be
used to instruct a re-call — and the agent would confabulate success instead of
re-issuing. Blocking keeps the result trustworthy and the agent honest.

The chat surface registers a per-session notify callback
(``register_tool_approval_notify``) that pushes the interaction onto the chat
stream; the resolve endpoint (``resolve_tool_approval``) unblocks the waiter.
State is module-level, keyed by the approval session key (stable per
conversation), matching ``tools.approval``'s shape so a session reset clears it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Optional

from tools.approval import get_current_session_key
from tools.mcp_tool import mcp_tool_is_read_only
from utils import env_var_enabled

logger = logging.getLogger(__name__)

# Killswitch: when truthy, no tool is gated (every write runs ungated).
_ENV_DISABLED = "OMNIO_TOOL_APPROVAL_DISABLED"
# How long (seconds) the agent blocks waiting for the user to resolve. Mirrors
# tools/approval.py's gateway_timeout default; the chat keepalive holds the SSE.
_ENV_TIMEOUT = "OMNIO_TOOL_APPROVAL_TIMEOUT"
_DEFAULT_TIMEOUT_S = 300

# Registered MCP names for the per-brand connectors server are
# ``mcp_connectors_<slug>``. The write gate is scoped to this prefix so only
# connector tools are gated through the connector-approval card.
_CONNECTORS_TOOL_PREFIX = "mcp_connectors_"

# User-facing option labels and the scope each one grants. Index-aligned so the
# Omnia frontend can map a chosen label back to its scope.
APPROVAL_OPTIONS = ["Allow once", "Allow for this chat", "Allow always", "Deny"]
APPROVAL_OPTION_SCOPES = ["once", "session", "always", "deny"]
APPROVAL_SCOPES = frozenset({"once", "session", "always", "deny"})

_lock = threading.Lock()
# session_key -> tool names approved for the whole conversation.
_session_approved: dict[str, set[str]] = {}
# Tool names approved for EVERY conversation on this gateway (the `always` scope).
# Gateway-wide, not session-keyed, so it spans chats — and deliberately NOT cleared
# by clear_session: it lives for the gateway's life and resets only on reprovision/
# restart. (A durable cross-reprovision grant would need a persistent store.)
_always_approved: set[str] = set()
# session_key -> per-session notify callback (bridges guard thread → chat stream).
_notify_cbs: dict[str, Callable[[dict], None]] = {}
# session_key -> FIFO of blocked approval waiters.
_waits: dict[str, list["_ApprovalWait"]] = {}


class _ApprovalWait:
    """One blocked tool call awaiting the user's decision."""

    __slots__ = ("event", "tool", "tool_call_id", "result")

    def __init__(self, tool: str, tool_call_id: str = "") -> None:
        self.event = threading.Event()
        self.tool = tool
        self.tool_call_id = tool_call_id
        self.result: Optional[str] = None  # "once" | "session" | "always" | "deny"


# Frozen at import — same rationale as tools/approval.py's _YOLO_MODE_FROZEN:
# reading the killswitch live on every call would let in-process skill/plugin
# code flip it mid-run and bypass the gate (a prompt-injection escalation path).
# The env is process-constant (set per gateway at spawn), so a snapshot loses
# nothing. Tests patch this module attr.
_DISABLED_FROZEN: bool = env_var_enabled(_ENV_DISABLED)


def _approval_timeout() -> int:
    try:
        return int(os.environ.get(_ENV_TIMEOUT, "") or _DEFAULT_TIMEOUT_S)
    except (ValueError, TypeError):
        return _DEFAULT_TIMEOUT_S


def is_gated_tool(function_name: str) -> bool:
    """Whether *function_name* is a connector WRITE that needs approval.

    Gates a connectors-server tool (``mcp_connectors_<slug>``) unless the route
    advertised it as read-only (MCP ``readOnlyHint=True``). The route stamps that
    per tool from its write allowlist, so the gated set tracks exactly what the
    route exposes — no provision-time snapshot to drift. FAILS CLOSED: a
    connectors tool with no read-only hint (e.g. a route that hasn't advertised
    it yet) is gated rather than run as an ungated write.
    """
    if _DISABLED_FROZEN:
        return False
    if not function_name or not function_name.startswith(_CONNECTORS_TOOL_PREFIX):
        return False
    return not mcp_tool_is_read_only(function_name)


def is_tool_approved(session_key: str, function_name: str) -> bool:
    """True when the tool is approved for the whole session (`session` scope)."""
    with _lock:
        return function_name in _session_approved.get(session_key, set())


def record_session_approval(session_key: str, function_name: str) -> None:
    """Grant a tool for the rest of the session (the `session` scope)."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(function_name)


def is_always_approved(function_name: str) -> bool:
    """True when the tool is approved for every conversation on this gateway
    (the `always` scope), until the gateway restarts/reprovisions."""
    with _lock:
        return function_name in _always_approved


def record_always_approval(function_name: str) -> None:
    """Grant a tool for every conversation on this gateway (the `always` scope)."""
    with _lock:
        _always_approved.add(function_name)


def register_tool_approval_notify(session_key: str, cb: Callable[[dict], None]) -> None:
    """Register the chat surface's per-session callback that pushes an approval
    interaction onto the stream. Called once per chat run around the agent."""
    if not session_key:
        return
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_tool_approval_notify(session_key: str) -> None:
    """Drop the notify callback AND release any still-blocked waiters for this
    session (so a finished/interrupted run can't leave a thread parked)."""
    if not session_key:
        return
    with _lock:
        _notify_cbs.pop(session_key, None)
        waiters = _waits.pop(session_key, [])
    for entry in waiters:
        entry.event.set()  # result stays None → guard fails closed (deny)


def _drop_wait(session_key: str, entry: "_ApprovalWait") -> None:
    with _lock:
        queue = _waits.get(session_key)
        if queue and entry in queue:
            queue.remove(entry)
        if queue is not None and not queue:
            _waits.pop(session_key, None)


def await_tool_approval(
    session_key: str,
    function_name: str,
    interaction_event: dict,
    tool_call_id: str = "",
) -> Optional[str]:
    """Surface the approval card and block until the user resolves it.

    Returns the chosen scope (``once`` / ``session`` / ``deny``), or ``None``
    when there is no chat surface registered (non-interactive caller), the wait
    times out, or the agent is interrupted (the user stopped the turn, or the
    chat disconnected). In every ``None`` case the caller MUST fail closed (not
    execute the write).
    """
    with _lock:
        cb = _notify_cbs.get(session_key)
        if cb is None:
            return None  # no interactive surface → caller denies the write
        entry = _ApprovalWait(function_name, tool_call_id)
        _waits.setdefault(session_key, []).append(entry)

    try:
        cb(interaction_event)
    except Exception:
        logger.warning("tool-approval notify failed", exc_info=True)
        _drop_wait(session_key, entry)
        return None

    # Block in short slices so we can heartbeat the inactivity tracker — without
    # it the gateway watchdog would kill the agent while the user is deciding.
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover
        touch_activity_if_due = None
    # A disconnect/stop reaches this worker as a thread interrupt: on SSE
    # disconnect the gateway calls agent.interrupt(), which fans the interrupt
    # bit out to the tool-worker thread running this guard. Observe it and
    # release — otherwise the worker parks here for the full timeout while the
    # run can't unwind (its notify cleanup only runs once this returns).
    try:
        from tools.interrupt import is_interrupted
    except Exception:  # pragma: no cover
        is_interrupted = None

    now = time.monotonic()
    deadline = now + max(_approval_timeout(), 0)
    activity = {"last_touch": now, "start": now}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if entry.event.wait(timeout=min(1.0, remaining)):
            break
        if is_interrupted is not None and is_interrupted():
            break  # result stays None → fail closed (deny)
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity, "waiting for tool approval")

    _drop_wait(session_key, entry)
    return entry.result


def resolve_tool_approval(
    session_key: str,
    function_name: str,
    scope: str,
    tool_call_id: str = "",
) -> bool:
    """Apply a decision posted from the Omnia chat: unblock the waiting tool
    call and, for ``session`` scope, remember it for the rest of the chat.

    When ``tool_call_id`` is given, the MATCHING waiter is resolved — not the
    queue head — so two writes blocked in one turn can't cross-talk (the user's
    decision on one card releasing the other). Falls back to FIFO only when no
    id is supplied (a legacy / single-call caller).

    Returns False only for a malformed request. A decision that arrives before
    the guard blocked (or after it timed out) still records the grant so the
    next call sees it — so the result is True as long as the input is valid.
    """
    if not session_key or scope not in APPROVAL_SCOPES:
        return False
    if scope == "session" and function_name:
        record_session_approval(session_key, function_name)
    if scope == "always" and function_name:
        record_always_approval(function_name)

    with _lock:
        queue = _waits.get(session_key)
        entry = None
        if queue:
            if tool_call_id:
                for index, candidate in enumerate(queue):
                    if candidate.tool_call_id == tool_call_id:
                        entry = queue.pop(index)
                        break
            else:
                entry = queue.pop(0)
        if queue is not None and not queue:
            _waits.pop(session_key, None)
    if entry is not None:
        entry.result = scope
        entry.event.set()
    return True


def clear_session(session_key: str) -> None:
    """Drop all approval state for a session (called on conversation reset)."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _notify_cbs.pop(session_key, None)
        waiters = _waits.pop(session_key, [])
    for entry in waiters:
        entry.event.set()


def _readable_tool(function_name: str) -> str:
    """A human label for the prompt, e.g. mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT
    -> 'Gmail create email draft'."""
    name = function_name
    marker = "_connectors_"
    if marker in name:
        name = name.split(marker, 1)[1]
    elif name.startswith("mcp_"):
        # Drop the mcp_<server>_ prefix generically (two leading components).
        parts = name.split("_", 2)
        name = parts[2] if len(parts) == 3 else name
    return name.replace("_", " ").strip().capitalize() or function_name


def _denial_result(choice: Optional[str]) -> str:
    """A tool result telling the agent the write did NOT happen. This is status
    data (not an instruction), so it's safe even wrapped as an untrusted result."""
    if choice == "deny":
        reason = "The user declined this action."
    else:
        reason = "This action needs the user's approval, which wasn't granted (no response)."
    return json.dumps(
        {
            "error": (
                f"{reason} It was NOT performed. Do not retry it or tell the user it "
                "succeeded; let them know it needs their approval."
            )
        },
        ensure_ascii=False,
    )


def fail_closed_denial(function_name: str) -> Optional[str]:
    """Denial result for a call site whose approval guard raised — so a gated
    write never runs unapproved on a guard error. Returns ``None`` for an
    ungated tool (it may proceed). Treats an unclassifiable tool as gated, so the
    write gate stays fail-closed even when classification itself fails."""
    try:
        gated = is_gated_tool(function_name)
    except Exception:
        gated = True  # can't classify → a write gate fails closed
    return _denial_result(None) if gated else None


def maybe_require_tool_approval(
    function_name: str,
    tool_call_id: str = "",
) -> Optional[str]:
    """Gate a connector write tool behind a blocking user approval.

    Returns ``None`` when the call may proceed (read tool, gating disabled,
    already approved for the session, or just approved). Otherwise BLOCKS until
    the user resolves the prompt and returns a denial tool-result string (the
    write must not execute) on deny / timeout / no-surface.
    """
    if not is_gated_tool(function_name):
        return None
    session_key = get_current_session_key()
    if is_always_approved(function_name):
        return None  # granted for every conversation on this gateway
    if is_tool_approved(session_key, function_name):
        return None  # approved for the whole session earlier

    interaction_event = {
        "tool": function_name,
        "toolCallId": tool_call_id or "",
        "status": "running",
        "interaction": {
            "kind": "approval",
            "question": (
                f'Allow Omnio to use "{_readable_tool(function_name)}"? '
                "It will act on your connected account."
            ),
            "options": list(APPROVAL_OPTIONS),
            "approval": {
                "tool": function_name,
                "tool_call_id": tool_call_id or "",
                "option_scopes": list(APPROVAL_OPTION_SCOPES),
            },
        },
    }

    choice = await_tool_approval(
        session_key, function_name, interaction_event, tool_call_id or ""
    )
    if choice in ("session", "always"):
        # resolve_tool_approval already recorded the grant; proceed.
        return None
    if choice == "once":
        return None  # this call proceeds; the next one prompts again
    # deny / timeout / no interactive surface → fail closed.
    return _denial_result(choice)
