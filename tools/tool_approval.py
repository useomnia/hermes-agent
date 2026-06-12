"""Per-call user approval for connector WRITE tools (Omnia / Omnio).

Reads run ungated. WRITE actions on a customer's connected third-party SaaS
(e.g. drafting an email, creating a doc) require an explicit per-call user
approval rendered as a control in the Omnia chat.

The gated set comes from ``OMNIO_CONNECTORS_WRITE_TOOLS`` — a JSON list of
Composio action slugs injected by Omnia from its allowlist (so the gated set
never drifts from what the connectors MCP route actually exposes). A tool is
gated when its registered MCP name (``mcp_<server>_<slug>``) resolves to one of
those slugs.

Unlike ``tools.approval`` (the dangerous-shell-command gate, which BLOCKS the
agent thread), this gate is NON-BLOCKING and turn-ending: the guard returns an
``approval_required`` result, the api_server seam renders the prompt and ends
the turn, and the user's choice is recorded here via ``resolve_tool_approval``
before the agent re-issues the call. State is module-level and keyed by the
approval session key (stable per conversation), matching ``tools.approval``'s
shape so a session reset clears both.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from tools.approval import get_current_session_key
from tools.mcp_tool import sanitize_mcp_name_component
from utils import env_var_enabled

logger = logging.getLogger(__name__)

# Env carrying the JSON list of gated write-action slugs (Omnia → sprite).
_ENV_WRITE_TOOLS = "OMNIO_CONNECTORS_WRITE_TOOLS"
# Killswitch: when truthy, no tool is gated (every write runs ungated).
_ENV_DISABLED = "OMNIO_TOOL_APPROVAL_DISABLED"

# User-facing option labels and the scope each one grants. Index-aligned so the
# Omnia frontend can map a chosen label back to its scope.
APPROVAL_OPTIONS = ["Allow once", "Allow for this chat", "Deny"]
APPROVAL_OPTION_SCOPES = ["once", "session", "deny"]
APPROVAL_SCOPES = frozenset({"once", "session", "deny"})

_lock = threading.Lock()
# session_key -> tool names approved for the whole conversation.
_session_approved: dict[str, set[str]] = {}
# session_key -> tool names with a pending single-use ("once") grant.
_once_approved: dict[str, set[str]] = {}


def _parse_gated_slugs(raw: str) -> frozenset[str]:
    """Sanitized, lower-cased write-action slugs from the env JSON list."""
    if not raw:
        return frozenset()
    try:
        slugs = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Malformed %s; gating no tools", _ENV_WRITE_TOOLS)
        return frozenset()
    if not isinstance(slugs, list):
        return frozenset()
    return frozenset(
        sanitize_mcp_name_component(s).lower() for s in slugs if isinstance(s, str) and s
    )


# Frozen at import — same rationale as tools/approval.py's _YOLO_MODE_FROZEN:
# reading these live on every call would let in-process skill/plugin code flip
# the killswitch or clear the gated set mid-run and bypass the gate (a
# prompt-injection escalation path). The env is process-constant (set per gateway
# at spawn), so a snapshot loses nothing. Tests patch these module attrs.
_DISABLED_FROZEN: bool = env_var_enabled(_ENV_DISABLED)
_GATED_SLUGS_FROZEN: frozenset[str] = _parse_gated_slugs(os.environ.get(_ENV_WRITE_TOOLS, ""))


def is_gated_tool(function_name: str) -> bool:
    """Whether *function_name* is a connector write action that needs approval.

    Matches the sanitized slug as a suffix of the registered MCP name
    (``mcp_<server>_<slug>``) so the check is independent of the connectors
    server name. The match is case-insensitive: the gated set is the Composio
    action slug (upper-case) but the advertised tool name could differ in case,
    and a security gate must not fail OPEN on a casing divergence. Slugs are
    specific action names, so a suffix match cannot collide with an unrelated tool.
    """
    if _DISABLED_FROZEN or not _GATED_SLUGS_FROZEN:
        return False
    if not function_name or not function_name.startswith("mcp_"):
        return False
    name = function_name.lower()
    for slug in _GATED_SLUGS_FROZEN:
        if name == slug or name.endswith("_" + slug):
            return True
    return False


def is_tool_approved(session_key: str, function_name: str) -> bool:
    """True when the tool is approved for this session, consuming a once-grant."""
    with _lock:
        if function_name in _session_approved.get(session_key, set()):
            return True
        once = _once_approved.get(session_key)
        if once and function_name in once:
            once.discard(function_name)
            if not once:
                _once_approved.pop(session_key, None)
            return True
    return False


def record_tool_approval(session_key: str, function_name: str, scope: str) -> None:
    """Record the user's choice. 'deny' records nothing (the agent was told)."""
    with _lock:
        if scope == "session":
            _session_approved.setdefault(session_key, set()).add(function_name)
        elif scope == "once":
            _once_approved.setdefault(session_key, set()).add(function_name)


def resolve_tool_approval(session_key: str, function_name: str, scope: str) -> bool:
    """Apply a resolution posted from the Omnia chat. Returns False if invalid."""
    if not session_key or not function_name or scope not in APPROVAL_SCOPES:
        return False
    record_tool_approval(session_key, function_name, scope)
    return True


def clear_session(session_key: str) -> None:
    """Drop all approvals for a session (called on conversation reset)."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _once_approved.pop(session_key, None)


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


def maybe_require_tool_approval(
    function_name: str,
    tool_call_id: str = "",
) -> Optional[str]:
    """Gate a connector write tool behind user approval.

    Returns ``None`` when the call may proceed (read tool, gating disabled, or
    already approved). Otherwise returns a JSON ``approval_required`` result
    carrying the interaction the api_server seam renders; execution is skipped
    and the turn ends until the user responds.
    """
    if not is_gated_tool(function_name):
        return None
    session_key = get_current_session_key()
    if is_tool_approved(session_key, function_name):
        return None

    interaction = {
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
    }
    return json.dumps(
        {
            "status": "approval_required",
            "interaction": interaction,
            "message": (
                "This action needs your approval. I've asked you in the chat — "
                "approve it there and I'll continue."
            ),
        },
        ensure_ascii=False,
    )
