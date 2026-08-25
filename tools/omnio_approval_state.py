"""Process-local state for Omnia's durable connector approvals.

This module intentionally has no Hermes/tool imports.  The API gateway needs
to clear and populate the durable approval candidate snapshot while binding
its listener; importing the full tool-approval gate there would also import
the MCP registry and make listener readiness wait on tool discovery.

The sets in this module are only candidate indexes.  A candidate is never
enough to authorize a write: :func:`is_always_approved` always asks the
server-authoritative callback before returning ``True``.  This keeps a warm
gateway fail-closed when a shared Omnia grant is revoked or the authority is
unavailable.
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

# Tool names approved for every conversation on this gateway by a recent
# in-chat click.  These local grants are a bridge until the next durable
# snapshot refresh; replacing a snapshot clears them.
_always_approved: set[str] = set()

# Exact wire names injected from Omnia's durable per-toolkit grant snapshot.
_injected_always_approved: set[str] = set()

# Harness-agnostic slugs injected from the same snapshot.  Keeping this beside
# the exact-name index preserves grants across the native/legacy prefix rename.
_injected_always_approved_slugs: set[str] = set()

# Fresh server-authoritative check for one exact standing grant.  The callback
# is deliberately not cached: revocation must take effect on every call in a
# warm gateway.
_always_approval_authority: Callable[[str], bool] | None = None


def is_always_approved(function_name: str) -> bool:
    """Return whether the authority currently grants a candidate tool.

    Local and injected names are only candidate indexes.  Missing authority,
    an authority exception, and any non-``True`` response all fail closed.
    """
    slug = connector_tool_slug(function_name)
    with _lock:
        candidate = (
            function_name in _always_approved
            or function_name in _injected_always_approved
            or (slug is not None and slug in _injected_always_approved_slugs)
        )
        authority = _always_approval_authority
    if not candidate:
        return False
    if authority is None:
        logger.warning(
            "standing tool approval authority unavailable; prompting for %s",
            function_name,
        )
        return False
    try:
        return authority(function_name) is True
    except Exception:
        logger.warning(
            "standing tool approval check failed; prompting for %s",
            function_name,
            exc_info=True,
        )
        return False


def register_always_approval_authority(
    cb: Callable[[str], bool] | None,
) -> None:
    """Set the server-authoritative checker for standing-grant candidates."""
    global _always_approval_authority
    with _lock:
        _always_approval_authority = cb


def record_always_approval(function_name: str) -> None:
    """Record a gateway-wide local ``always`` approval bridge grant."""
    with _lock:
        _always_approved.add(function_name)


def replace_injected_always_approvals(
    function_names: list[str],
    tool_slugs: list[str] | None = None,
) -> None:
    """Replace the durable Omnia candidate snapshot.

    Replacing a snapshot always clears local bridge grants.  Callers that
    cannot load the authoritative snapshot should pass empty lists so stale
    candidates fail closed.  Exact names are limited to connector wire names;
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
    "register_always_approval_authority",
    "replace_injected_always_approvals",
]
