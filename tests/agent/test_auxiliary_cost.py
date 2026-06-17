"""Tests for auxiliary-LLM cost accounting (Omnio).

Auxiliary calls (vision, web_extract summarizer, compression, title-gen, ...) are
priced and added to the CURRENT session's `estimated_cost_usd` in state.db so a
reader summing the session tree sees agent + auxiliary cost together. These pin:
  - the happy path writes the priced cost (+ tokens) to the active session;
  - every no-op guard (no context, no session, no usage, unknown/zero price);
  - accounting never raises;
  - `_validate_llm_response` (the single return chokepoint) triggers accounting.
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as ac


def _response(*, model="google/gemini-3.1-flash-lite", usage=True):
    """A minimal OpenAI-shaped response with the .choices[0].message shape the
    validator requires, plus optional .usage / .model for pricing."""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        model=model,
    )
    if usage:
        resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return resp


def _canonical():
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )


@pytest.fixture(autouse=True)
def _reset_ctx():
    """ContextVars persist within a thread across tests — reset to the default so
    each test controls the in-flight request identity deterministically."""
    ac._AUX_COST_CTX.set(None)
    yield
    ac._AUX_COST_CTX.set(None)


def _patch_pricing(amount_usd):
    """Patch the lazily-imported pricing helpers to deterministic values."""
    cost = SimpleNamespace(amount_usd=amount_usd, status="estimated", source="catalog")
    return (
        patch("agent.usage_pricing.normalize_usage", return_value=_canonical()),
        patch("agent.usage_pricing.estimate_usage_cost", return_value=cost),
    )


def test_should_add_priced_cost_to_the_current_session():
    ac._AUX_COST_CTX.set(("openrouter", "google/gemini-3.1-flash-lite", "https://openrouter.ai/api/v1", "k"))
    db = MagicMock()
    norm, est = _patch_pricing(Decimal("0.0012"))
    with norm, est, \
        patch("gateway.session_context.get_session_env", return_value="sess-1"), \
        patch.object(ac, "_get_aux_cost_session_db", return_value=db):
        ac._account_auxiliary_cost(_response())

    db.update_token_counts.assert_called_once()
    call = db.update_token_counts.call_args
    assert call.args[0] == "sess-1"  # attributed to the active session
    assert call.kwargs["estimated_cost_usd"] == pytest.approx(0.0012)
    assert call.kwargs["input_tokens"] == 10
    assert call.kwargs["output_tokens"] == 5
    assert call.kwargs["api_call_count"] == 1
    assert call.kwargs["model"] == "google/gemini-3.1-flash-lite"


def test_should_noop_when_no_request_context_is_set():
    # _AUX_COST_CTX defaults to None (reset by the fixture) — nothing to price.
    db = MagicMock()
    with patch.object(ac, "_get_aux_cost_session_db", return_value=db):
        ac._account_auxiliary_cost(_response())
    db.update_token_counts.assert_not_called()


def test_should_noop_when_there_is_no_active_session():
    ac._AUX_COST_CTX.set(("openrouter", "m", "u", "k"))
    db = MagicMock()
    norm, est = _patch_pricing(Decimal("0.0012"))
    with norm, est, \
        patch("gateway.session_context.get_session_env", return_value=""), \
        patch.object(ac, "_get_aux_cost_session_db", return_value=db):
        ac._account_auxiliary_cost(_response())
    db.update_token_counts.assert_not_called()


def test_should_noop_when_the_response_has_no_usage():
    ac._AUX_COST_CTX.set(("openrouter", "m", "u", "k"))
    db = MagicMock()
    with patch("gateway.session_context.get_session_env", return_value="sess-1"), \
        patch.object(ac, "_get_aux_cost_session_db", return_value=db):
        ac._account_auxiliary_cost(_response(usage=False))
    db.update_token_counts.assert_not_called()


@pytest.mark.parametrize("amount", [None, Decimal("0"), Decimal("-0.5")])
def test_should_noop_when_price_is_unknown_or_nonpositive(amount):
    # Unknown price (None) or subscription-included / zero ⇒ nothing to add.
    ac._AUX_COST_CTX.set(("openrouter", "m", "u", "k"))
    db = MagicMock()
    norm, est = _patch_pricing(amount)
    with norm, est, \
        patch("gateway.session_context.get_session_env", return_value="sess-1"), \
        patch.object(ac, "_get_aux_cost_session_db", return_value=db):
        ac._account_auxiliary_cost(_response())
    db.update_token_counts.assert_not_called()


def test_should_swallow_errors_so_accounting_never_breaks_a_call():
    ac._AUX_COST_CTX.set(("openrouter", "m", "u", "k"))
    norm = patch("agent.usage_pricing.normalize_usage", side_effect=RuntimeError("boom"))
    with norm, patch("gateway.session_context.get_session_env", return_value="sess-1"):
        # Must not raise.
        ac._account_auxiliary_cost(_response())


def test_validate_llm_response_should_trigger_accounting_on_success():
    # The single return chokepoint accounts the cost of every validated response.
    with patch.object(ac, "_account_auxiliary_cost") as spy:
        resp = _response()
        assert ac._validate_llm_response(resp, "vision") is resp
        spy.assert_called_once_with(resp)


def test_validate_llm_response_should_not_account_an_invalid_response():
    with patch.object(ac, "_account_auxiliary_cost") as spy:
        with pytest.raises(RuntimeError):
            ac._validate_llm_response(SimpleNamespace(), "vision")  # no .choices
        spy.assert_not_called()
