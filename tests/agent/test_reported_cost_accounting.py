"""Reported-cost responses do not consult unavailable model prices."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.aux_accounting import record_aux_usage, reset_accounting_context, set_accounting_context
from agent.usage_pricing import CanonicalUsage
from hermes_state import SessionDB


def response(cost):
    return SimpleNamespace(
        model="cost-test-model",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="done", tool_calls=[]), finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=cost,
            prompt_tokens_details=SimpleNamespace(cached_tokens=30),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )


@pytest.mark.parametrize("cost", [0.0033, 0.0])
def test_auxiliary_cost_skips_unavailable_prices(monkeypatch, tmp_path, cost):
    prices = Mock(return_value={})
    monkeypatch.setattr("agent.usage_pricing.fetch_model_metadata", prices)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("aux-cost", source="api_server")
    token = set_accounting_context(db, "aux-cost")
    try:
        record_aux_usage(response(cost), "compression", provider="openrouter")
    finally:
        reset_accounting_context(token)
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ?", ("aux-cost",),
        ).fetchall()
    db.close()

    prices.assert_not_called()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["actual_cost_usd"] == pytest.approx(cost)
    assert row["estimated_cost_usd"] == pytest.approx(cost)
    assert row["cost_status"] == "actual"
    assert row["input_tokens"] == 70
    assert row["output_tokens"] == 20
    assert row["cache_read_tokens"] == 30
    assert row["reasoning_tokens"] == 5
    assert row["api_call_count"] == 1


@pytest.mark.parametrize("cost", [0.0033, 0.0, None])
@pytest.mark.parametrize("advisor_cost", [None, 0.0044])
def test_main_loop_skips_unavailable_prices(monkeypatch, tmp_path, cost, advisor_cost):
    from run_agent import AIAgent

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 128000)
    monkeypatch.setattr("agent.agent_init.fetch_model_metadata", lambda: {})
    monkeypatch.setattr("agent.output_budget.get_model_output_limit", lambda *a, **k: 8192)
    db = SessionDB(tmp_path / "state.db")
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", provider="openrouter",
        api_mode="chat_completions", model="cost-test-model", max_iterations=1,
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        session_id="main-cost", session_db=db,
    )
    reply = response(cost)
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda *a, **k: reply)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda *a, **k: reply)
    if advisor_cost is not None:
        monkeypatch.setattr(
            agent.client, "consume_reference_usage",
            lambda: (CanonicalUsage(input_tokens=7, output_tokens=3), advisor_cost),
            raising=False,
        )
    prices = Mock(return_value={})
    monkeypatch.setattr("agent.usage_pricing.fetch_model_metadata", prices)

    result = agent.run_conversation("Reply with done.")
    row = db.get_session("main-cost")
    db.close()

    assert result["final_response"] == "done"
    if cost is None:
        prices.assert_called_once()
    else:
        prices.assert_not_called()
    expected = None if cost is None and advisor_cost is None else (cost or 0) + (advisor_cost or 0)
    if expected is None:
        assert row["actual_cost_usd"] is None
    else:
        assert row["actual_cost_usd"] == pytest.approx(expected)
    # Hermes' legacy-named result counter carries total spend for displays and
    # delegated-child rollups. Its persisted counterpart serves saved reports.
    assert result["estimated_cost_usd"] == pytest.approx(expected or 0)
    assert agent.session_estimated_cost_usd == pytest.approx(expected or 0)
    assert row["estimated_cost_usd"] == pytest.approx(expected or 0)
    assert row["cost_status"] == ("unknown" if cost is None else "actual")
    assert row["input_tokens"] == 70 + (7 if advisor_cost is not None else 0)
    assert row["output_tokens"] == 20 + (3 if advisor_cost is not None else 0)
    assert row["cache_read_tokens"] == 30
    assert row["reasoning_tokens"] == 5
    assert row["api_call_count"] == 1
