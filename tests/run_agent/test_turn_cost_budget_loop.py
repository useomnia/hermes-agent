"""The per-turn cost ceiling, exercised against the real conversation loop.

The module-level tests in ``tests/agent/test_cost_budget.py`` cover parsing and
the comparison. These drive ``run_conversation`` so the three properties that
only exist in the loop are actually observed:

- the check runs BEFORE the next provider call, so a breach costs nothing more;
- the exit does not buy a summary call (the finalizer's fallback is skipped);
- the budget measures one turn, so a session's earlier spend is not charged to
  the next turn's ceiling.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.cost_budget import COST_BUDGET_EXIT_REASON, CostBudget
from run_agent import AIAgent


def _tool_defs() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "web_search tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _text_response(content="done"):
    msg = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, reasoning=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _tool_call_response():
    """A response that forces the loop around for another iteration."""
    call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    msg = SimpleNamespace(
        content=None, tool_calls=[call], reasoning_content=None, reasoning=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


def _make_agent(cost_budget=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            cost_budget=cost_budget,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run(agent, message="go"):
    """Drive one turn with tool execution and persistence stubbed out."""

    def _fake_tool_exec(assistant_message, messages, _task_id, _count):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "web_search",
                "content": "tool result",
            }
        )
        return None

    with (
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_tool_exec),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


class TestCeilingStopsTheLoop:
    def test_breach_ends_the_turn_without_another_provider_call(self):
        """One call spends past the cap; the loop must not make a second."""
        agent = _make_agent(CostBudget(max_cost_usd=0.10))
        calls = {"n": 0}

        def _spend_then_ask_for_a_tool(*_a, **_k):
            calls["n"] += 1
            agent.session_estimated_cost_usd += 0.25  # blows the $0.10 cap
            return _tool_call_response()

        agent.client.chat.completions.create.side_effect = _spend_then_ask_for_a_tool
        result = _run(agent)

        assert calls["n"] == 1, "the ceiling must be read before the next call"
        assert result["turn_exit_reason"] == COST_BUDGET_EXIT_REASON

    def test_breach_does_not_buy_a_summary_call(self):
        """The finalizer's toolless summary is the most expensive call there is."""
        agent = _make_agent(CostBudget(max_cost_usd=0.10))

        def _spend_then_ask_for_a_tool(*_a, **_k):
            agent.session_estimated_cost_usd += 0.25
            return _tool_call_response()

        agent.client.chat.completions.create.side_effect = _spend_then_ask_for_a_tool

        with patch.object(agent, "_handle_max_iterations") as summary:
            result = _run(agent)

        summary.assert_not_called()
        assert result["turn_exit_reason"] == COST_BUDGET_EXIT_REASON

    def test_spend_under_the_cap_completes_normally(self):
        agent = _make_agent(CostBudget(max_cost_usd=10.0))

        def _cheap(*_a, **_k):
            agent.session_estimated_cost_usd += 0.01
            return _text_response("all done")

        agent.client.chat.completions.create.side_effect = _cheap
        result = _run(agent)

        assert result["final_response"] == "all done"
        assert result["turn_exit_reason"] != COST_BUDGET_EXIT_REASON

    def test_no_budget_never_stops_the_loop_on_cost(self):
        agent = _make_agent(cost_budget=None)

        def _expensive(*_a, **_k):
            agent.session_estimated_cost_usd += 99.0
            return _text_response("expensive but allowed")

        agent.client.chat.completions.create.side_effect = _expensive
        result = _run(agent)

        assert result["final_response"] == "expensive but allowed"
        assert result["turn_exit_reason"] != COST_BUDGET_EXIT_REASON


class TestBudgetIsPerTurn:
    def test_earlier_turns_do_not_consume_this_turns_ceiling(self):
        """A session already past the cap still gets a full budget next turn."""
        agent = _make_agent(CostBudget(max_cost_usd=0.10))
        agent.session_estimated_cost_usd = 5.00  # spent by previous turns

        def _cheap(*_a, **_k):
            agent.session_estimated_cost_usd += 0.01
            return _text_response("fresh turn")

        agent.client.chat.completions.create.side_effect = _cheap
        result = _run(agent)

        assert result["final_response"] == "fresh turn"
        assert agent._cost_budget_turn_start_usd == pytest.approx(5.00)

    def test_baseline_is_relatched_on_each_turn(self):
        agent = _make_agent(CostBudget(max_cost_usd=10.0))

        def _cheap(*_a, **_k):
            agent.session_estimated_cost_usd += 1.0
            return _text_response("ok")

        agent.client.chat.completions.create.side_effect = _cheap

        _run(agent, "first")
        assert agent._cost_budget_turn_start_usd == pytest.approx(0.0)
        _run(agent, "second")
        assert agent._cost_budget_turn_start_usd == pytest.approx(1.0)
