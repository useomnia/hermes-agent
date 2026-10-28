"""Reported dollars bypass pricing; absent or malformed cost keeps route-aware estimation."""
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.usage_pricing import CanonicalUsage, CostResult, resolve_usage_cost


@pytest.mark.parametrize("cost", [0, 0.000001, 1.25])
@pytest.mark.parametrize("shape", [dict, SimpleNamespace])
def test_reported_cost_does_not_estimate(monkeypatch, cost, shape):
    estimate = Mock(side_effect=AssertionError("pricing must not run"))
    monkeypatch.setattr("agent.usage_pricing.estimate_usage_cost", estimate)

    result = resolve_usage_cost(
        "model@preset/example", CanonicalUsage(input_tokens=10),
        response_usage=shape(cost=cost), provider="openrouter",
    )

    estimate.assert_not_called()
    assert result.amount_usd == Decimal(str(float(cost)))
    assert result.status == "actual"
    assert result.source == "provider_cost_api"


@pytest.mark.parametrize("raw", [
    None, {}, {"cost": None}, {"cost": True}, {"cost": "0.1"},
    {"cost": -1}, {"cost": float("nan")}, {"cost": float("inf")},
])
@pytest.mark.parametrize("fallback", [
    CostResult(Decimal("0.001"), "estimated", "provider_models_api", "~$0.001"),
    CostResult(None, "unknown", "none", "n/a"),
    CostResult(Decimal(0), "included", "none", "included"),
])
def test_unusable_cost_preserves_existing_fallback(monkeypatch, raw, fallback):
    usage = CanonicalUsage(input_tokens=10)
    estimate = Mock(return_value=fallback)
    monkeypatch.setattr("agent.usage_pricing.estimate_usage_cost", estimate)

    result = resolve_usage_cost(
        "route-model", usage, response_usage=raw, provider="custom",
        base_url="https://example.invalid/v1", api_key="test-key",
    )

    assert result is fallback
    estimate.assert_called_once_with(
        "route-model", usage, provider="custom",
        base_url="https://example.invalid/v1", api_key="test-key",
    )
