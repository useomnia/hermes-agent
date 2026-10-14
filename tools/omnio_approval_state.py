"""Process-local state for Omnia's durable connector approvals.

This module intentionally has no Hermes/tool imports.  The API gateway needs
to clear and populate the durable approval candidate snapshot while binding
its listener; importing the full tool-approval gate there would also import
the MCP registry and make listener readiness wait on tool discovery.

The gateway owns approval state, as upstream Hermes does: Omnia's durable
grants are loaded into these sets at gateway start and on a connector reload,
and an in-chat "Allow always" is recorded here directly.  A grant or revoke
made elsewhere reaches a warm gateway on its next start or reload.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Native MCP names use ``mcp__<server>__<tool>``. Existing durable approval
# records may still contain the earlier flattened connector names, so both
# forms remain in the connector-write trust boundary.
CONNECTORS_TOOL_PREFIXES = ("mcp__connectors__", "mcp_connectors_")


def connector_tool_slug(function_name: str) -> Optional[str]:
    """Return the stable connector slug behind a wire name, or ``None``.

    The prefix is transport dressing owned by this harness and may change;
    durable grants are therefore matched by the harness-agnostic slug as well
    as by the exact wire name.
    """
    if not isinstance(function_name, str):
        return None
    for prefix in CONNECTORS_TOOL_PREFIXES:
        if function_name.startswith(prefix):
            return function_name[len(prefix) :]
    return None


_lock = threading.Lock()

# Tool names approved for every conversation on this gateway by an in-chat
# click.  Omnia saves the grant before releasing the call, so the next
# snapshot carries it too; replacing a snapshot clears these.
_always_approved: set[str] = set()

# Exact wire names injected from Omnia's durable per-toolkit grant snapshot.
_injected_always_approved: set[str] = set()

# Harness-agnostic slugs injected from the same snapshot.  Keeping this beside
# the exact-name index preserves grants across the native/legacy prefix rename.
_injected_always_approved_slugs: set[str] = set()

# Joins the gateway's startup grant snapshot (bounded). Called on the first
# candidate lookup instead of before every agent build, so a Turn that never
# reaches a gated write does not wait on the snapshot fetch.
_always_approval_snapshot_waiter: Callable[[], None] | None = None


# Loads the current conversation's durable "Allow for this chat" grants as
# ``(tools, slugs)``; ``None`` when the conversation cannot be identified yet.
ConversationGrantLoader = Callable[[], Optional[tuple[list[str], Optional[list[str]]]]]
_conversation_grant_loader: ConversationGrantLoader | None = None


def register_always_approval_snapshot_waiter(cb: Callable[[], None] | None) -> None:
    global _always_approval_snapshot_waiter
    _always_approval_snapshot_waiter = cb


def register_conversation_grant_loader(cb: ConversationGrantLoader | None) -> None:
    """Set how the current conversation's durable chat grants are loaded."""
    global _conversation_grant_loader
    _conversation_grant_loader = cb


def conversation_grant_loader() -> ConversationGrantLoader | None:
    return _conversation_grant_loader


def is_always_approved(function_name: str) -> bool:
    """Return whether *function_name* holds a standing grant on this gateway."""
    waiter = _always_approval_snapshot_waiter
    if waiter is not None:
        try:
            waiter()
        except Exception:  # noqa: BLE001 — a failed join leaves the snapshot empty
            logger.debug("approval snapshot join failed", exc_info=True)
    slug = connector_tool_slug(function_name)
    with _lock:
        return (
            function_name in _always_approved
            or function_name in _injected_always_approved
            or (slug is not None and slug in _injected_always_approved_slugs)
        )


def record_always_approval(function_name: str) -> None:
    """Record a gateway-wide local ``always`` approval bridge grant."""
    with _lock:
        _always_approved.add(function_name)


def replace_injected_always_approvals(
    function_names: list[str],
    tool_slugs: list[str] | None = None,
) -> None:
    """Replace the durable Omnia grant snapshot.

    Replacing a snapshot always clears local in-chat grants.  Callers that
    cannot load the snapshot should pass empty lists so stale grants fail
    closed.  Exact names are limited to connector wire names;
    the API gateway cannot import the MCP registry merely to perform this
    process-global bookkeeping.

    ``tool_slugs`` is the harness-agnostic form of the same grants.  Older
    Omnia payloads carry only prefixed names, so absent slugs are derived from
    those names.
    """
    exact_names = {
        name.strip()
        for name in function_names
        if isinstance(name, str)
        and name.strip()
        and connector_tool_slug(name.strip()) is not None
    }
    if tool_slugs is None:
        slugs = {
            slug for slug in (connector_tool_slug(name) for name in exact_names) if slug
        }
    else:
        slugs = {
            slug.strip()
            for slug in tool_slugs
            if isinstance(slug, str) and slug.strip()
        }
    with _lock:
        _always_approved.clear()
        _injected_always_approved.clear()
        _injected_always_approved.update(exact_names)
        _injected_always_approved_slugs.clear()
        _injected_always_approved_slugs.update(slugs)


__all__ = [
    "CONNECTORS_TOOL_PREFIXES",
    "connector_tool_slug",
    "is_always_approved",
    "record_always_approval",
    "register_always_approval_snapshot_waiter",
    "register_conversation_grant_loader",
    "replace_injected_always_approvals",
]
