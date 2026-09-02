"""Per-turn cost budget: request parsing, the ceiling, and the finalizer seam.

Covers the three behaviours the feature rests on:

1. request validation is fail-CLOSED (a typo is a 400, not an uncapped run);
2. the ceiling itself fails OPEN on an unreadable figure;
3. a cost exit never routes through the toolless-summary fallback, which would
   spend one more full-context call on the turn the ceiling exists to stop.
"""

from types import SimpleNamespace

import pytest

from agent.cost_budget import (
    BUDGET_REQUEST_FIELDS,
    COST_BUDGET_EXIT_REASON,
    CostBudget,
    clamp_max_iterations,
    cost_budget_from_request,
    parse_budget_request,
    turn_spend_usd,
)


class TestParseBudgetRequest:
    """Validation is fail-closed: anything malformed is an error, never a default."""

    def test_absent_and_empty_are_valid_and_uncapped(self):
        assert parse_budget_request(None) == ({}, None)
        assert parse_budget_request({}) == ({}, None)

    def test_accepts_both_fields(self):
        parsed, error = parse_budget_request({"max_cost_usd": 2.5, "max_iterations": 40})
        assert error is None
        assert parsed == {"max_cost_usd": 2.5, "max_iterations": 40}

    def test_integer_cost_is_accepted_as_a_float(self):
        parsed, error = parse_budget_request({"max_cost_usd": 3})
        assert error is None
        assert parsed == {"max_cost_usd": 3.0}
        assert isinstance(parsed["max_cost_usd"], float)

    def test_non_object_is_rejected(self):
        _, error = parse_budget_request("2.50")
        assert error and "must be an object" in error

    def test_unknown_field_is_rejected_not_ignored(self):
        # The whole point: `maxCostUsd` would otherwise read as a cap and mean
        # "no cap". The message has to name the offender and the valid set.
        _, error = parse_budget_request({"maxCostUsd": 2.5})
        assert error
        assert "maxCostUsd" in error
        for field in BUDGET_REQUEST_FIELDS:
            assert field in error

    def test_unknown_fields_are_reported_together(self):
        _, error = parse_budget_request({"nope": 1, "alsoNope": 2})
        assert error and "alsoNope, nope" in error

    @pytest.mark.parametrize("value", [0, -1, -0.01, float("inf"), float("nan"), True, "2.5", None])
    def test_non_positive_or_non_numeric_cost_is_rejected(self, value):
        _, error = parse_budget_request({"max_cost_usd": value})
        assert error and "max_cost_usd" in error

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "40", None])
    def test_non_positive_or_non_integer_iterations_are_rejected(self, value):
        _, error = parse_budget_request({"max_iterations": value})
        assert error and "max_iterations" in error

    def test_bool_is_not_a_number(self):
        # bool subclasses int, so `True` would pass a naive isinstance check
        # and silently become a $1.00 cap.
        _, cost_error = parse_budget_request({"max_cost_usd": True})
        _, iter_error = parse_budget_request({"max_iterations": True})
        assert cost_error and iter_error


class TestClampMaxIterations:
    """A request may tighten the operator's ceiling, never raise it."""

    def test_unset_keeps_the_ceiling(self):
        assert clamp_max_iterations(None, 500) == 500

    def test_lower_request_tightens(self):
        assert clamp_max_iterations(40, 500) == 40

    def test_higher_request_is_clamped_down_not_rejected(self):
        assert clamp_max_iterations(9000, 500) == 500

    def test_equal_request_is_unchanged(self):
        assert clamp_max_iterations(500, 500) == 500


class TestCostBudgetCeiling:
    def test_inclusive_at_the_cap(self):
        budget = CostBudget(max_cost_usd=1.0)
        assert not budget.exceeded(0.999999)
        assert budget.exceeded(1.0)
        assert budget.exceeded(1.5)

    @pytest.mark.parametrize("value", [None, "oops", float("nan"), float("inf")])
    def test_unreadable_spend_fails_open(self, value):
        # A bad figure must never end a live turn on its own — that is the
        # runtime half of the fail-open/fail-closed split.
        assert not CostBudget(max_cost_usd=0.01).exceeded(value)

    def test_from_request_requires_a_cost_field(self):
        assert cost_budget_from_request({}) is None
        assert cost_budget_from_request({"max_iterations": 5}) is None
        assert cost_budget_from_request({"max_cost_usd": 3}) == CostBudget(3.0)


class TestTurnSpend:
    def test_measures_only_this_turn(self):
        agent = SimpleNamespace(
            session_estimated_cost_usd=5.0, _cost_budget_turn_start_usd=4.25
        )
        assert turn_spend_usd(agent) == pytest.approx(0.75)

    def test_missing_attributes_fail_open_to_zero(self):
        assert turn_spend_usd(SimpleNamespace()) == 0.0

    def test_never_negative(self):
        # A session counter that moved backwards (a revised estimate) must not
        # hand the turn free budget, nor a negative reading.
        agent = SimpleNamespace(
            session_estimated_cost_usd=1.0, _cost_budget_turn_start_usd=3.0
        )
        assert turn_spend_usd(agent) == 0.0

    def test_unreadable_values_fail_open_to_zero(self):
        agent = SimpleNamespace(
            session_estimated_cost_usd="lots", _cost_budget_turn_start_usd=0.0
        )
        assert turn_spend_usd(agent) == 0.0


class TestFinalizerDoesNotPayForASummary:
    def test_cost_exit_reason_is_not_fallback_eligible(self):
        """The exit reason must stay outside the finalizer's eligible set.

        ``agent/turn_finalizer.py`` gates its toolless-summary call on
        ``_turn_exit_reason in {"unknown", "budget_exhausted"}``. Reading the
        set from the source keeps this test honest if someone edits it.
        """
        import inspect

        from agent import turn_finalizer

        source = inspect.getsource(turn_finalizer)
        assert '{"unknown", "budget_exhausted"}' in source, (
            "the finalizer's eligible-reason set moved; re-verify that "
            f"{COST_BUDGET_EXIT_REASON!r} is still excluded from it"
        )
        assert COST_BUDGET_EXIT_REASON not in {"unknown", "budget_exhausted"}

    def test_exit_reason_has_a_user_facing_explanation(self):
        from run_agent import AIAgent

        explanation = AIAgent._format_turn_completion_explanation(
            COST_BUDGET_EXIT_REASON
        )
        assert explanation, "an abnormal turn ending must never be silent"
        assert "spend ceiling" in explanation
