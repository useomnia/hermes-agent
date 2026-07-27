"""Regression coverage for Omnio auxiliary-LLM cost accounting."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.aux_accounting import (
    record_aux_usage,
    reset_accounting_context,
    set_accounting_context,
)


def _response(*, reported_cost=None):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    if reported_cost is not None:
        usage.cost = reported_cost
    return SimpleNamespace(model="google/gemini-3.1-flash-lite", usage=usage)


def test_should_persist_estimate_as_complete_billable_cost_when_provider_omits_cost(
    monkeypatch,
):
    db = MagicMock()
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=0.0012, status="estimated", source="catalog"
        ),
    )
    token = set_accounting_context(db, "sess-1")
    try:
        record_aux_usage(_response(), "vision", provider="openrouter")
    finally:
        reset_accounting_context(token)

    call = db.record_auxiliary_usage.call_args
    assert call.kwargs["estimated_cost_usd"] == pytest.approx(0.0012)
    assert call.kwargs["actual_cost_usd"] == pytest.approx(0.0012)
    assert call.kwargs["cost_status"] == "estimated"


def test_should_prefer_provider_reported_cost_for_complete_billable_cost(monkeypatch):
    db = MagicMock()
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=0.0012, status="estimated", source="catalog"
        ),
    )
    token = set_accounting_context(db, "sess-1")
    try:
        record_aux_usage(
            _response(reported_cost=0.0033),
            "vision",
            provider="openrouter",
        )
    finally:
        reset_accounting_context(token)

    call = db.record_auxiliary_usage.call_args
    assert call.kwargs["estimated_cost_usd"] == pytest.approx(0.0012)
    assert call.kwargs["actual_cost_usd"] == pytest.approx(0.0033)
    assert call.kwargs["cost_status"] == "actual"
    assert call.kwargs["cost_source"] == "provider_cost_api"


def test_should_swallow_storage_errors_so_accounting_cannot_break_the_call():
    db = MagicMock()
    db.record_auxiliary_usage.side_effect = RuntimeError("disk full")
    token = set_accounting_context(db, "sess-1")
    try:
        record_aux_usage(_response(), "vision", provider="openrouter")
    finally:
        reset_accounting_context(token)
