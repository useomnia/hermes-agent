"""Reasoning floor for auxiliary requests: when a route refuses to switch reasoning OFF, step it UP.

Aux lanes that want a concise answer over a reasoning trace (compression:
``auxiliary.compression.reasoning_effort: none``; title generation:
``reasoning_config={"enabled": False}``) send the provider's thinking-off encoding — top-level
``reasoning_effort: "none"`` on the custom profile, ``extra_body.reasoning: {"enabled": false}`` on
OpenRouter-shaped relays, ``_reasoning_config`` on the Anthropic Messages adapters. Some endpoints
understand the field but refuse the disable ("Reasoning is mandatory for this endpoint and cannot be
disabled" — OpenRouter on ``openai/gpt-6.1-sol``, the Nous Portal on gpt-6-astra); before this module
every such call 400'd, so compaction never produced a summary on those models.

The recovery is a *step up*, not a strip: the same request goes out again at the lowest effort every
reasoning wire accepts (``low``; ``minimal`` is rejected by o-series and Claude), and the (route, model)
pair is memoised so the next disabled-reasoning aux call on it starts at the floor instead of burning
the guaranteed 400 first. Dropping the field would also succeed once, but it says nothing about the
next call and hands the effort choice back to the provider default (often ``medium`` or higher, the
opposite of what a thinking-off caller asked for).

Ported from upstream NousResearch/hermes-agent 2768f31628. Upstream floors ``reasoning_config`` before
the profile projection; this fork's compression lane carries its disable in ``extra_body.reasoning``
rather than ``reasoning_config``, so the up-front floor here lifts the built kwargs with the same
``with_reasoning_floor`` the retry uses.
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REASONING_FLOOR_EFFORT = "low"

# (route key, model) pairs that refused a reasoning disable in this process.
_FLOORED_ROUTES: set[tuple[str, str]] = set()

_DISABLED_EFFORTS = {"none", "off", "disabled", "false", "0"}


def _route_key(provider: Optional[str], base_url: Optional[str]) -> str:
    """Endpoint host:port when known (a base_url override turns a named provider into ``custom``), else
    the provider name."""
    return (urlparse(base_url or "").netloc or "").lower() or str(provider or "").strip().lower()


def _is_disabled(reasoning_config: Any) -> bool:
    return isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False


def floor_reasoning_config(reasoning_config: Any) -> Any:
    """The caller's disabled ``reasoning_config`` lifted to the floor; anything else returned as-is."""
    if _is_disabled(reasoning_config):
        return {"enabled": True, "effort": REASONING_FLOOR_EFFORT}
    return reasoning_config


def with_reasoning_floor(kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Copy of *kwargs* with every thinking-OFF encoding lifted to ``REASONING_FLOOR_EFFORT``:
    top-level ``reasoning_effort``, ``extra_body.reasoning`` (OpenRouter shape) and the adapter's private
    ``_reasoning_config``. ``None`` when nothing was disabled, so a retry never re-sends an unchanged
    request."""
    changed = False
    retry = dict(kwargs)
    if str(retry.get("reasoning_effort", "")).strip().lower() in _DISABLED_EFFORTS:
        retry["reasoning_effort"] = REASONING_FLOOR_EFFORT
        changed = True
    if _is_disabled(retry.get("_reasoning_config")):
        retry["_reasoning_config"] = floor_reasoning_config(retry["_reasoning_config"])
        changed = True
    extra_body = retry.get("extra_body")
    if isinstance(extra_body, dict):
        reasoning = extra_body.get("reasoning")
        if _is_disabled(reasoning) or (
            isinstance(reasoning, dict) and str(reasoning.get("effort", "")).strip().lower() in _DISABLED_EFFORTS
        ):
            retry["extra_body"] = {**extra_body, "reasoning": {"enabled": True, "effort": REASONING_FLOOR_EFFORT}}
            changed = True
    return retry if changed else None


def remember_reasoning_floor(provider: Optional[str], base_url: Optional[str], rejected_kwargs: dict[str, Any]) -> None:
    """Record that this route's ``rejected_kwargs["model"]`` refuses to disable reasoning (called after
    the stepped-up retry succeeded)."""
    _FLOORED_ROUTES.add((_route_key(provider, base_url), str(rejected_kwargs.get("model") or "")))


def known_reasoning_floor(
    kwargs: dict[str, Any], provider: Optional[str], base_url: Optional[str], task: Optional[str] = None,
) -> dict[str, Any]:
    """*kwargs* lifted to the floor when this route+model is known to refuse a disable; unchanged
    otherwise."""
    if (_route_key(provider, base_url), str(kwargs.get("model") or "")) not in _FLOORED_ROUTES:
        return kwargs
    floored = with_reasoning_floor(kwargs)
    if floored is None:
        return kwargs
    logger.info(
        "Auxiliary %s: %s (%s) cannot disable reasoning; sending effort=%s up front",
        task or "call", _route_key(provider, base_url) or "provider", kwargs.get("model") or "model",
        REASONING_FLOOR_EFFORT,
    )
    return floored
