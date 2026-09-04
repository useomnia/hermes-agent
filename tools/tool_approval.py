"""Per-call user approval for connector writes and MCP credit spends.

Reads run ungated. WRITE actions on a customer's connected third-party SaaS
(e.g. drafting an email, creating a doc) require an explicit per-call user
approval rendered as a control in the Omnia chat. MCP tools advertising an
``_meta["omnia/credits"]`` descriptor require approval for every call.

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
re-issuing. Blocking keeps the result trustworthy and the agent honest. A skip
also fails closed when the user sends a new message instead of deciding.

The chat surface registers a per-run notify callback
(``register_tool_approval_notify``) that pushes the interaction onto the chat
stream; the resolve endpoint (``resolve_tool_approval``) unblocks the waiter.
Mechanical surface/wait ownership is run-scoped so concurrent turns cannot
replace or release each other, while session grants remain conversation-scoped.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
from typing import Callable, Optional

from tools.approval import get_current_session_key
from tools.blocking_wait import BlockingWaitEntry, BlockingWaitRegistry
from tools.mcp_tool import (
    mcp_tool_credits_meta,
    mcp_tool_has_read_only_hint,
    mcp_tool_is_read_only,
)
from tools.omnio_approval_state import (
    CONNECTORS_TOOL_PREFIXES,
    _always_approved,
    _injected_always_approved,
    _injected_always_approved_slugs,
    connector_tool_slug,
    is_always_approved,
    record_always_approval as _record_always_approval_state,
    register_always_approval_authority as _register_always_approval_authority_state,
    replace_injected_always_approvals as _replace_injected_always_approvals_state,
)
from utils import env_var_enabled

logger = logging.getLogger(__name__)

# Killswitch: when truthy, no tool is gated (every write runs ungated).
_ENV_DISABLED = "OMNIO_TOOL_APPROVAL_DISABLED"
# How long (seconds) the agent blocks waiting for the user to resolve. Mirrors
# tools/approval.py's gateway_timeout default; the chat keepalive holds the SSE.
_ENV_TIMEOUT = "OMNIO_TOOL_APPROVAL_TIMEOUT"
_DEFAULT_TIMEOUT_S = 300
# User-facing option labels and the scope each one grants. Index-aligned so the
# Omnia frontend can map a chosen label back to its scope.
APPROVAL_OPTIONS = ["Allow once", "Allow for this chat", "Allow always", "Deny"]
APPROVAL_OPTION_SCOPES = ["once", "session", "always", "deny"]
APPROVAL_SCOPES = frozenset({"once", "session", "always", "deny", "skip"})
CREDIT_APPROVAL_OPTIONS = ["Approve", "Deny"]
CREDIT_APPROVAL_OPTION_SCOPES = ["once", "deny"]

# Sentinel returned by ``await_tool_approval`` when there is no interactive
# surface registered at all (e.g. a proactive /v1/runs headless task with
# nobody to show the card to). Distinct from a genuine timeout (``None``) so
# ``maybe_require_tool_approval`` can route it to a non-turn-ending denial:
# interrupting a headless task on its first gated write would be wrong, since
# there was never anyone who could have answered in time.
_NO_SURFACE = "__no_surface__"


class ToolApprovalDenial(str):
    """A trusted tool result produced when approval prevents dispatch.

    The JSON body is intentionally still a plain string for existing callers,
    but the type carries control-flow provenance that the executor can persist
    without inspecting attacker-controlled tool-result content.
    """

    effect_disposition = "none"


_lock = threading.Lock()
# session_key -> tool names approved for the whole conversation.
_session_approved: dict[str, set[str]] = {}
# (session_key, tool_call_id, tool_name, canonical args) grants consumed by
# exactly one re-dispatch. Kept process-local like the existing session grant
# store: the durable interaction is the dangling SessionDB tool call, not a
# second grant record with a separate lifecycle.
_once_approved: set[tuple[str, str, str, str]] = set()
# Mechanical surface/wait state stays isolated from the user-input gate by this
# module's own registry instance. The waiter payload is the gated tool name.
_wait_registry: BlockingWaitRegistry[
    str, Callable[[dict], None], str
] = BlockingWaitRegistry()
# surface_key -> (ownership token, conversation grant key). This lets the
# conversation-scoped resolve endpoint locate the exact run-owned waiter.
_surface_grant_sessions: dict[str, tuple[object, str]] = {}
# (surface_key, tool_call_id) -> the resolved decision
# (once/session/always/deny/skip) for a released waiter, consumed by the
# gateway's tool-complete callback to echo `interaction.answered` on the gated
# call's completed event.
_decisions: dict[tuple[str, str], str] = {}
_tool_approval_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tool_approval_session_key",
    default="",
)
_tool_approval_surface_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tool_approval_surface_key",
    default="",
)


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


def set_current_tool_approval_session_key(
    session_key: str,
) -> contextvars.Token[str]:
    """Bind the conversation namespace used by connector approval grants."""
    return _tool_approval_session_key.set(session_key or "")


def reset_current_tool_approval_session_key(
    token: contextvars.Token[str],
) -> None:
    """Restore the prior connector-approval conversation namespace."""
    _tool_approval_session_key.reset(token)


def get_current_tool_approval_session_key(default: str = "default") -> str:
    """Return the connector grant namespace, falling back for legacy callers."""
    return _tool_approval_session_key.get() or get_current_session_key(default)


def set_current_tool_approval_surface_key(
    surface_key: str,
) -> contextvars.Token[str]:
    """Bind the run-owned surface namespace used by connector approval waits."""
    return _tool_approval_surface_key.set(surface_key or "")


def reset_current_tool_approval_surface_key(
    token: contextvars.Token[str],
) -> None:
    """Restore the prior connector-approval surface namespace."""
    _tool_approval_surface_key.reset(token)


def get_current_tool_approval_surface_key(default: str = "default") -> str:
    """Return the run-owned surface key, falling back for legacy callers."""
    return _tool_approval_surface_key.get() or get_current_tool_approval_session_key(
        default
    )


def is_gated_tool(function_name: str) -> bool:
    """Whether *function_name* needs connector-write or credit approval.

    Gates a connectors-server tool unless the route advertised it as read-only
    (MCP ``readOnlyHint=True``). The route stamps that per tool from its write
    allowlist, so the gated set tracks exactly what the route exposes — no
    provision-time snapshot to drift. FAILS CLOSED: a connectors tool with no
    read-only hint (e.g. a route that hasn't advertised it yet) is gated rather
    than run as an ungated write. Any MCP tool with an Omnia credit-spend
    descriptor is gated regardless of its server prefix.
    """
    if _DISABLED_FROZEN:
        return False
    if mcp_tool_credits_meta(function_name) is not None:
        return True
    if not function_name or not function_name.startswith(CONNECTORS_TOOL_PREFIXES):
        return False
    return not mcp_tool_is_read_only(function_name)


def is_credit_gated_tool(function_name: str) -> bool:
    """Whether *function_name* advertises a per-call credit spend."""
    if _DISABLED_FROZEN:
        return False
    return mcp_tool_credits_meta(function_name) is not None


def is_tool_approved(session_key: str, function_name: str) -> bool:
    """True when the tool is approved for the whole session (`session` scope)."""
    with _lock:
        return function_name in _session_approved.get(session_key, set())


def record_session_approval(session_key: str, function_name: str) -> None:
    """Grant a tool for the rest of the session (the `session` scope)."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(function_name)


def _canonical_args(function_args: Optional[dict]) -> str:
    """Stable, non-secret-bearing key material for an exact tool call."""
    try:
        canonical = json.dumps(
            function_args if isinstance(function_args, dict) else {},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        return hashlib.sha256(b"{}").hexdigest()


def record_once_approval(
    session_key: str,
    tool_call_id: str,
    function_name: str,
    function_args: Optional[dict],
) -> None:
    """Grant exactly one matching resumed tool-call identity."""
    with _lock:
        _once_approved.add(
            (session_key, tool_call_id, function_name, _canonical_args(function_args))
        )


def consume_once_approval(
    session_key: str,
    tool_call_id: str,
    function_name: str,
    function_args: Optional[dict],
) -> bool:
    """Atomically consume the exact one-shot tool-call grant, if present."""
    key = (session_key, tool_call_id, function_name, _canonical_args(function_args))
    with _lock:
        if key not in _once_approved:
            return False
        _once_approved.remove(key)
        return True


def resolve_approval_target(
    function_name: str,
    function_args: Optional[dict],
) -> tuple[str, dict]:
    """Return the tool identity that the approval gate actually protects.

    Progressive disclosure presents deferred MCP tools to the model through
    the ``tool_call`` bridge.  The persisted assistant call therefore names
    the bridge even though ``model_tools.handle_function_call`` unwraps it
    before applying this approval gate.  Durable grants must follow that same
    unwrapping or they approve the inert bridge and the resumed underlying
    call asks the user a second time.

    Resolution is deliberately fail-closed: malformed or unavailable bridge
    calls keep their outer identity, which cannot approve an underlying write.
    """
    args = function_args if isinstance(function_args, dict) else {}
    if function_name != "tool_call":
        return function_name, args
    try:
        from tools.tool_search import resolve_underlying_call

        underlying_name, underlying_args, error = resolve_underlying_call(args)
    except Exception:
        return function_name, args
    if (
        error
        or not isinstance(underlying_name, str)
        or not underlying_name
        or not isinstance(underlying_args, dict)
    ):
        return function_name, args
    return underlying_name, underlying_args


def rehydrate_resolved_approval(
    session_key: str,
    tool_call_id: str,
    function_name: str,
    raw_arguments: str,
    durable_grant: object,
) -> bool:
    """Restore one accepted durable approval after a gateway restart.

    The assistant message remains the canonical record. Its tool-call identity
    must exactly match the metadata captured by SessionDB before any transient
    grant is restored. Credit-spend tools deliberately demote standing scopes
    to one exact call, matching the live approval resolver.
    """
    if not isinstance(durable_grant, dict):
        return False
    scope = durable_grant.get("scope")
    if scope not in {"once", "session", "always"}:
        return False
    if durable_grant.get("tool_name") != function_name:
        return False
    if durable_grant.get("arguments") != raw_arguments:
        return False
    try:
        function_args = json.loads(raw_arguments or "{}")
    except (TypeError, ValueError):
        function_args = {}
    if not isinstance(function_args, dict):
        function_args = {}
    approval_name, approval_args = resolve_approval_target(
        function_name,
        function_args,
    )
    try:
        credit_gated = is_credit_gated_tool(approval_name)
    except Exception:
        credit_gated = True
    if scope == "once" or credit_gated:
        record_once_approval(
            session_key,
            tool_call_id,
            approval_name,
            approval_args,
        )
    elif scope == "session":
        record_session_approval(session_key, approval_name)
    else:
        record_always_approval(approval_name)
    return True


def register_always_approval_authority(
    cb: Callable[[str], bool] | None,
) -> None:
    """Set the server-authoritative checker used for standing grant candidates."""
    _register_always_approval_authority_state(cb)


def record_always_approval(function_name: str) -> None:
    """Grant a tool for every conversation on this gateway (the `always` scope)."""
    _record_always_approval_state(function_name)


def replace_injected_always_approvals(
    function_names: list[str],
    tool_slugs: list[str] | None = None,
) -> None:
    """Replace the durable Omnia grant snapshot and clear local bridge grants.

    Local ``always`` approvals are only a bridge between the user's click and the
    next authoritative DB refresh. A reload means the DB snapshot has spoken; on
    failure the caller passes an empty list so stale grants fail closed.

    ``tool_slugs`` is the harness-agnostic form of the same grants. Older Omnia
    payloads carry only prefixed names, so when it is absent the slugs are
    derived from those names — a grant published under either spelling means
    the toolkit's tool slug is granted, whatever prefix this harness mints.
    """
    exact_names = {
        name.strip()
        for name in function_names
        if isinstance(name, str)
        and _is_recordable_tool_name(name.strip(), require_hint=False)
    }
    _replace_injected_always_approvals_state(
        list(exact_names),
        tool_slugs=tool_slugs,
    )


def register_tool_approval_notify(
    surface_key: str,
    cb: Callable[[dict], None],
    *,
    grant_session_key: str | None = None,
) -> object | None:
    """Register one run-owned chat surface and its conversation grant scope."""
    if not surface_key:
        return None
    token = _wait_registry.register_surface(surface_key, cb)
    with _lock:
        _surface_grant_sessions[surface_key] = (
            token,
            grant_session_key or surface_key,
        )
    return token


def unregister_tool_approval_notify(surface_key: str, token: object) -> None:
    """Drop an owned surface and release any still-blocked waiters."""
    if not surface_key:
        return

    def clear_decisions() -> None:
        with _lock:
            owner = _surface_grant_sessions.get(surface_key)
            if owner is not None and owner[0] is token:
                _surface_grant_sessions.pop(surface_key, None)
            for key in [key for key in _decisions if key[0] == surface_key]:
                _decisions.pop(key, None)

    _wait_registry.unregister_surface(
        surface_key,
        token,
        on_unregister=clear_decisions,
    )


def consume_tool_approval_completion_reason(
    surface_key: str,
    tool_call_id: str,
) -> Optional[str]:
    """Return and clear the unresolved wait reason for one approval call."""
    if not surface_key:
        return None
    return _wait_registry.consume_completion_reason(surface_key, tool_call_id)


def await_tool_approval(
    session_key: str,
    function_name: str,
    interaction_event: dict,
    tool_call_id: str = "",
) -> Optional[str]:
    """Surface the approval card and block until the user resolves it.

    Returns the chosen scope (``once`` / ``session`` / ``always`` / ``deny`` /
    ``skip``); ``None`` when the wait genuinely timed out or the agent was
    interrupted (the user stopped the turn, or the chat disconnected) while a
    real surface was registered; or the ``_NO_SURFACE`` sentinel when there was
    no chat surface registered at all (non-interactive caller). The caller MUST
    fail closed (not execute the write) for both ``None`` and ``_NO_SURFACE`` —
    the two are kept distinct only so it can pick the right (turn-ending vs not)
    denial status.
    """
    def notify(cb: Callable[[dict], None] | None) -> None:
        if cb is None:
            raise RuntimeError("tool-approval surface has no notify callback")
        cb(interaction_event)

    try:
        result, reason = _wait_registry.wait(
            session_key,
            tool_call_id,
            _approval_timeout(),
            "waiting for tool approval",
            payload=function_name,
            on_parked=notify,
        )
    except Exception:
        # The notify callback raising is a plumbing malfunction (e.g. the chat
        # stream write failed) — the card was never actually shown, so the user
        # may still be present and simply never got a chance to answer. Treat
        # this the same as "no interactive surface": _NO_SURFACE routes to the
        # non-turn-ending `approval_error` status, not a genuine `None` timeout
        # (which would end the turn via `approval_no_response`).
        logger.warning("tool-approval notify failed", exc_info=True)
        return _NO_SURFACE
    if reason == "no_surface":
        return _NO_SURFACE
    return result


def _legacy_surface_key_for_waiter(
    grant_session_key: str,
    tool_call_id: str,
) -> str:
    """Find the matching or FIFO run surface for a pre-surface-id client."""
    with _lock:
        surface_keys = [
            surface_key
            for surface_key, (_, owner_grant_session_key) in (
                _surface_grant_sessions.items()
            )
            if owner_grant_session_key == grant_session_key
        ]
    for surface_key in surface_keys:
        if _wait_registry.waiter_payload(surface_key, tool_call_id) is not None:
            return surface_key
    return grant_session_key


def resolve_tool_approval(
    session_key: str,
    function_name: str,
    scope: str,
    tool_call_id: str = "",
    tools: list[str] | None = None,
    *,
    surface_key: str | None = None,
) -> bool:
    """Apply a decision posted from the Omnia chat: unblock the waiting tool
    call and, for ``session`` scope, remember it for the rest of the chat.

    New interactive callers provide both ``surface_key`` and ``tool_call_id``
    from the approval interaction. The surface must belong to this
    conversation's grant namespace, and only the matching waiter on that exact
    surface can be released. Older clients omit ``surface_key``; for them,
    search this conversation's run surfaces for the first matching call id, or
    the first surface with a pending waiter when the id is also omitted. That
    fallback deliberately preserves the legacy call-id/FIFO ambiguity while
    never crossing conversation/profile grants.

    Returns True only when a blocked waiter was actually found and released —
    i.e. the tool call that showed the card is still live to receive the
    decision. When no waiter matches (the wait already timed out and the guard
    moved on to its own denial), the write already didn't happen: returning
    True here would make the client show the card as granted while nothing
    ran. `session`/`always` grants are still recorded in that case — they
    legitimately help the NEXT call of the same tool skip the prompt — but the
    call itself reports False so the caller can tell the user their decision
    arrived too late to affect this write. Credit-gated calls are the exception:
    their `session`/`always` decisions are reduced to `once` and never recorded.
    """
    if not session_key or scope not in APPROVAL_SCOPES:
        return False
    resolved_surface_key = (
        surface_key
        if surface_key is not None
        else _legacy_surface_key_for_waiter(session_key, tool_call_id)
    )
    if surface_key is not None:
        if not surface_key or not tool_call_id:
            return False
        with _lock:
            owner = _surface_grant_sessions.get(surface_key)
        if owner is not None and owner[1] != session_key:
            return False
        if owner is not None:
            waiting_tool = _wait_registry.waiter_payload(surface_key, tool_call_id)
            if waiting_tool is not None and waiting_tool != function_name:
                return False
        elif scope not in {"session", "always"}:
            # Run cleanup removes the surface after a timeout. A late durable
            # grant still belongs to the authenticated conversation namespace,
            # but no one-shot/denial decision can target a vanished waiter.
            return False
    if scope in {"session", "always"}:
        approval_tool = (
            _wait_registry.waiter_payload(resolved_surface_key, tool_call_id)
            or function_name
        )
        try:
            credit_gated = is_credit_gated_tool(approval_tool)
        except Exception:
            credit_gated = True
        if credit_gated:
            scope = "once"
    approved_tools = _approved_tool_names(function_name, tools)
    if scope == "session":
        for tool_name in approved_tools:
            record_session_approval(session_key, tool_name)
    if scope == "always":
        for tool_name in approved_tools:
            record_always_approval(tool_name)

    def commit_decision(entry: BlockingWaitEntry[str, str]) -> None:
        # Commit the completed-event outcome before releasing the executor
        # worker. Once signalled, that worker can finish the tool and consume
        # this value immediately.
        with _lock:
            _decisions[(resolved_surface_key, entry.tool_call_id)] = scope

    return _wait_registry.resolve(
        resolved_surface_key,
        tool_call_id,
        scope,
        on_release=commit_decision,
    )


def consume_tool_approval_decision(
    surface_key: str, tool_call_id: str = ""
) -> Optional[str]:
    """Pop the recorded decision (once/session/always/deny/skip) for a released
    gated call, or None when the wait ended without one (timeout/interrupt)."""
    if not surface_key:
        return None
    with _lock:
        return _decisions.pop((surface_key, tool_call_id or ""), None)


def clear_session(session_key: str) -> None:
    """Drop all approval state for a session (called on conversation reset)."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        for key in [key for key in _once_approved if key[0] == session_key]:
            _once_approved.discard(key)
        surface_keys = {
            surface_key
            for surface_key, (_, grant_session_key) in _surface_grant_sessions.items()
            if grant_session_key == session_key
        }
        surface_keys.add(session_key)
        for surface_key in surface_keys:
            _surface_grant_sessions.pop(surface_key, None)
        for key in [key for key in _decisions if key[0] in surface_keys]:
            _decisions.pop(key, None)
    for surface_key in surface_keys:
        _wait_registry.clear(surface_key)


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


def _approved_tool_names(function_name: str, tools: list[str] | None) -> set[str]:
    approved: set[str] = set()
    current_name = function_name.strip() if isinstance(function_name, str) else ""
    if current_name and _is_recordable_tool_name(current_name, require_hint=True):
        approved.add(current_name)

    for name in tools or []:
        if not isinstance(name, str):
            continue
        candidate = name.strip()
        if not candidate or candidate == current_name:
            continue
        # A client may spell a sibling tool with a prefix this registry does
        # not use (older clients predate the native-name convention), so try
        # every spelling of the same slug and keep whichever is registered.
        for spelling in _spelling_candidates(candidate):
            if spelling != current_name and _is_recordable_tool_name(
                spelling, require_hint=True
            ):
                approved.add(spelling)
    return approved


def _spelling_candidates(name: str) -> list[str]:
    slug = connector_tool_slug(name)
    if slug is None:
        return [name]
    return [f"{prefix}{slug}" for prefix in CONNECTORS_TOOL_PREFIXES]


def _is_recordable_tool_name(function_name: str, *, require_hint: bool) -> bool:
    if not function_name.startswith(CONNECTORS_TOOL_PREFIXES):
        return False
    if require_hint and not mcp_tool_has_read_only_hint(function_name):
        return False
    try:
        return is_gated_tool(function_name)
    except Exception:
        return False


def _denial_result(
    choice: Optional[str], *, status: Optional[str] = None
) -> ToolApprovalDenial:
    """A tool result telling the agent the write did NOT happen. This is status
    data (not an instruction), so it's safe even wrapped as an untrusted result.

    Also carries a machine-readable ``status`` field for the api_server seam to
    key its turn-ending decision on, alongside the human-readable ``error``
    text:
      - ``choice == "deny"`` -> ``"approval_denied"`` (explicit denial; NOT
        turn-ending, the agent continues and reports it inline).
      - ``choice == "skip"`` -> ``"approval_skipped"`` (the user sent a new
        message instead; NOT turn-ending, so it arrives as the next user turn).
      - otherwise (the wait ended unresolved with a real approval surface — a
        timeout, or an interrupt/stop releasing the waiter) ->
        ``"approval_no_response"`` (turn-ending; for the interrupt case the turn
        is already ending, so the extra interrupt is a no-op).
      - ``status`` overrides these defaults — used for paths that are none of
        the above and must NOT be turn-ending: a guard error
        (``fail_closed_denial``) or no interactive surface at all
        (``maybe_require_tool_approval``'s no-surface branch).
    """
    if choice == "deny":
        reason = "The user declined this action."
        default_status = "approval_denied"
        tail = "let them know it needs their approval."
    elif choice == "skip":
        reason = (
            "The user skipped this request by sending a new message instead; "
            "their message arrives as the next user turn."
        )
        default_status = "approval_skipped"
        tail = "address their new message instead."
    else:
        reason = (
            "This action needs the user's approval, which wasn't granted (no response)."
        )
        default_status = "approval_no_response"
        tail = "let them know it needs their approval."
    return ToolApprovalDenial(
        json.dumps(
            {
                "status": status or default_status,
                "error": (
                    f"{reason} It was NOT performed. Do not retry it or tell the user it "
                    f"succeeded; {tail}"
                ),
            },
            ensure_ascii=False,
        )
    )


def fail_closed_denial(function_name: str) -> Optional[str]:
    """Denial result for a call site whose approval guard raised — so a gated
    write never runs unapproved on a guard error. Returns ``None`` for an
    ungated tool (it may proceed). Treats an unclassifiable tool as gated, so the
    write gate stays fail-closed even when classification itself fails.

    Status is ``"approval_error"``, NOT ``"approval_no_response"``: the user may
    well be present, this is a guard malfunction, so the turn must not end —
    the agent should continue and report the error inline.
    """
    try:
        gated = is_gated_tool(function_name)
    except Exception:
        gated = True  # can't classify → a write gate fails closed
    return _denial_result(None, status="approval_error") if gated else None


def _credit_cost_data(descriptor: dict, function_args: Optional[dict]) -> dict:
    strategy = descriptor.get("strategy")
    unit = descriptor.get("unit")
    credits_per_unit = descriptor.get("creditsPerUnit")
    if not (
        isinstance(credits_per_unit, (int, float))
        and not isinstance(credits_per_unit, bool)
    ):
        credits_per_unit = None

    raw_engines = (
        function_args.get("engines") if isinstance(function_args, dict) else None
    )
    engines = (
        list(dict.fromkeys(str(engine) for engine in raw_engines))
        if isinstance(raw_engines, list)
        else None
    )
    engine_count = len(engines) if engines is not None else None
    credits = None
    if (
        strategy == "fixed"
        and unit == "per_engine"
        and credits_per_unit is not None
        and engine_count is not None
    ):
        credits = credits_per_unit * engine_count

    return {
        "credits": credits,
        "creditsPerUnit": credits_per_unit,
        "unit": unit if isinstance(unit, str) else None,
        "engineCount": engine_count,
        "engines": engines,
    }


def _credit_cost_sentence(descriptor: dict, cost: dict) -> str:
    strategy = descriptor.get("strategy")
    credits_per_unit = cost["creditsPerUnit"]
    if strategy == "fixed" and cost["unit"] == "per_engine" and credits_per_unit is not None:
        engine_count = cost["engineCount"]
        if engine_count is not None:
            return (
                f"This call spends {cost['credits']:g} credits "
                f"({credits_per_unit:g} x {engine_count} engines)."
            )
        return f"This call spends {credits_per_unit:g} credits per engine."
    if strategy == "real-cost":
        return "This call spends credits based on its actual cost."
    return "This call spends credits."


def maybe_require_tool_approval(
    function_name: str,
    tool_call_id: str = "",
    function_args: Optional[dict] = None,
) -> Optional[str]:
    """Gate a connector write or MCP credit spend behind user approval.

    Returns ``None`` when the call may proceed (read tool, gating disabled,
    connector write already approved for the session, or just approved).
    Otherwise BLOCKS until the user resolves the prompt and returns a denial
    tool-result string (the tool must not execute) on deny / skip / timeout /
    no-surface.
    """
    if not is_gated_tool(function_name):
        return None
    credits_descriptor = mcp_tool_credits_meta(function_name)
    grant_session_key = get_current_tool_approval_session_key()
    surface_key = get_current_tool_approval_surface_key()
    if consume_once_approval(
        grant_session_key, tool_call_id, function_name, function_args
    ):
        return None
    if credits_descriptor is None:
        if is_always_approved(function_name):
            return None  # granted for every conversation on this gateway
        if is_tool_approved(grant_session_key, function_name):
            return None  # approved for the whole session earlier
        options = APPROVAL_OPTIONS
        option_scopes = APPROVAL_OPTION_SCOPES
        question = (
            f'Allow Omnio to use "{_readable_tool(function_name)}"? '
            "It will act on your connected account."
        )
    else:
        options = CREDIT_APPROVAL_OPTIONS
        option_scopes = CREDIT_APPROVAL_OPTION_SCOPES
        cost = _credit_cost_data(credits_descriptor, function_args)
        question = (
            f'Allow Omnio to use "{_readable_tool(function_name)}"? '
            f"{_credit_cost_sentence(credits_descriptor, cost)}"
        )

    approval: dict[str, object] = {
        "tool": function_name,
        "tool_call_id": tool_call_id or "",
        "surface_id": surface_key,
        "option_scopes": list(option_scopes),
        "skip_scope": True,
    }
    if credits_descriptor is not None:
        approval["cost"] = cost

    interaction_event = {
        "tool": function_name,
        "toolCallId": tool_call_id or "",
        "status": "running",
        "interaction": {
            "kind": "approval",
            "question": question,
            "options": list(options),
            "approval": approval,
        },
    }

    choice = await_tool_approval(
        surface_key, function_name, interaction_event, tool_call_id or ""
    )
    if choice in ("session", "always"):
        # resolve_tool_approval already recorded the grant; proceed.
        return None
    if choice == "once":
        return None  # this call proceeds; the next one prompts again
    if choice in ("deny", "skip"):
        return _denial_result(choice)
    if choice == _NO_SURFACE:
        # No interactive surface at all (e.g. a proactive /v1/runs task) — fail
        # closed, but NOT turn-ending: there was never anyone who could have
        # answered, so interrupting a headless run would be wrong.
        return _denial_result(None, status="approval_error")
    # An unresolved wait with a real surface — timeout or interrupt (choice is
    # None) — fails closed and ends the turn via "approval_no_response".
    return _denial_result(None)
