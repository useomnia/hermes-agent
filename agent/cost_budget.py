"""Per-turn cost budget — an estimated-USD ceiling on a single turn.

Three things in this repo are called a "budget"; they are unrelated and this
module is only the third:

- :class:`agent.iteration_budget.IterationBudget` counts loop iterations.
- ``tools/budget_config.py`` caps the *characters* a tool result may add to a
  turn, scaled to the model's context window.
- :class:`CostBudget` (here) caps the *money* one turn may spend.

The quantity measured is ``agent.session_estimated_cost_usd`` — the runtime's
own running estimate, in USD, from :mod:`agent.usage_pricing`. Estimated rather
than provider-reported on purpose: the estimate is available the instant a call
returns and only ever grows, so a threshold crossed stays crossed. A reported
cost can land later and revise the number downward, which would let an already
tripped budget un-trip.

Scope is the agent's *own* spend. A ``delegate`` subagent runs as a separate
agent with its own session and its own counter, so its spend is invisible here
by construction. Callers that need tree-wide enforcement have to measure
outside the process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# The exact keys a ``/v1/runs`` request may put in ``budget``. Anything else is
# rejected rather than ignored: a typo like ``maxCostUsd`` would otherwise mean
# "no cap" while reading like a cap.
BUDGET_REQUEST_FIELDS: Tuple[str, ...] = ("max_cost_usd", "max_iterations")

# Exit reason recorded on the turn when the cap is hit. Deliberately NOT
# ``budget_exhausted`` (the iteration budget's reason), because that one is
# eligible for the toolless-summary fallback in ``agent.turn_finalizer`` and
# this one must not be — see ``handle_max_iterations``.
COST_BUDGET_EXIT_REASON = "cost_budget_exhausted"


@dataclass(frozen=True)
class CostBudget:
    """An immutable per-turn ceiling in estimated USD."""

    max_cost_usd: float

    def exceeded(self, spent_usd: float) -> bool:
        """True when ``spent_usd`` has reached the ceiling.

        Inclusive: spending exactly the cap is spent. A non-finite or
        unreadable figure is treated as *not* exceeded so a bad number can
        never end a turn on its own.
        """
        try:
            spent = float(spent_usd)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(spent):
            return False
        return spent >= self.max_cost_usd


def _positive_number(value: Any) -> Optional[float]:
    """Coerce a JSON number to a positive finite float, else None.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, so ``True``
    would otherwise pass as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _positive_int(value: Any) -> Optional[int]:
    """Coerce a JSON number to a positive int, else None. Rejects ``bool``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def parse_budget_request(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate the ``budget`` object from a ``/v1/runs`` request body.

    Returns ``(parsed, error)``. ``error`` is a caller-facing message; when it
    is set, ``parsed`` is meaningless. An absent or null ``budget`` is valid and
    yields an empty dict.

    Validation is fail-CLOSED — a malformed policy is a 400, not a silently
    uncapped run. That is the opposite of how the cap behaves at runtime, where
    an unreadable cost figure fails open, and the asymmetry is deliberate: a
    malformed request is a caller bug detectable at the boundary, while an
    unreadable cost mid-turn is a degradation that must not kill live work.
    """
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return {}, "budget must be an object"

    unknown = sorted(set(raw) - set(BUDGET_REQUEST_FIELDS))
    if unknown:
        return {}, (
            "budget has unknown field(s): "
            + ", ".join(unknown)
            + "; expected any of "
            + ", ".join(BUDGET_REQUEST_FIELDS)
        )

    parsed: Dict[str, Any] = {}

    if "max_cost_usd" in raw:
        max_cost = _positive_number(raw["max_cost_usd"])
        if max_cost is None:
            return {}, "budget.max_cost_usd must be a positive number"
        parsed["max_cost_usd"] = max_cost

    if "max_iterations" in raw:
        max_iterations = _positive_int(raw["max_iterations"])
        if max_iterations is None:
            return {}, "budget.max_iterations must be a positive integer"
        parsed["max_iterations"] = max_iterations

    return parsed, None


def clamp_max_iterations(requested: Optional[int], ceiling: int) -> int:
    """Resolve a request's ``max_iterations`` against the operator's ceiling.

    A request may only ever *tighten* the limit. ``None`` (unset) keeps the
    ceiling, and a value above it is clamped down rather than rejected — the
    caller asked for at least that much headroom and the operator's number is
    the authority on how much exists.
    """
    if requested is None or requested > ceiling:
        return ceiling
    return requested


def cost_budget_from_request(parsed: Dict[str, Any]) -> Optional[CostBudget]:
    """Build the :class:`CostBudget` for a parsed request, or None if uncapped."""
    max_cost_usd = parsed.get("max_cost_usd")
    if max_cost_usd is None:
        return None
    return CostBudget(max_cost_usd=float(max_cost_usd))


def turn_spend_usd(agent: Any) -> float:
    """Estimated USD this turn has spent, relative to its starting baseline.

    Reads the session-cumulative counter and subtracts the value latched when
    the turn began, so a long-lived session's earlier turns don't consume this
    turn's budget. Fails open (``0.0``) if either figure is unreadable.
    """
    try:
        total = float(getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0)
        baseline = float(getattr(agent, "_cost_budget_turn_start_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(total) or not math.isfinite(baseline):
        return 0.0
    return max(0.0, total - baseline)


__all__ = [
    "BUDGET_REQUEST_FIELDS",
    "COST_BUDGET_EXIT_REASON",
    "CostBudget",
    "clamp_max_iterations",
    "cost_budget_from_request",
    "parse_budget_request",
    "turn_spend_usd",
]
