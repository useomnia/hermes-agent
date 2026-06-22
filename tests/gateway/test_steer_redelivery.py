"""Tests for _run_with_steer_redelivery — a /steer that landed after the agent's
final tool batch (no tool result to absorb it) comes back as
result["pending_steer"] and must be re-delivered as a follow-up user turn
instead of being silently dropped. Mirrors the gateway/CLI leftover-steer
handling (gateway/run.py, cli.py), adapted to the OpenAI chat-completions path.
"""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def _turn(*, final="ok", messages=None, pending_steer=None, interrupted=False, usage=None):
    """Build a (result, usage) pair like _run_agent returns."""
    result = {
        "final_response": final,
        "messages": messages if messages is not None else [{"role": "assistant", "content": final}],
        "interrupted": interrupted,
    }
    if pending_steer is not None:
        result["pending_steer"] = pending_steer
    return result, (usage or {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})


class _Scripted:
    """An async one_turn that yields scripted turns and records its calls.

    Once the script runs out, the last turn repeats — so a single
    always-pending-steer turn drives the cap test.
    """

    def __init__(self, turns):
        self._turns = turns
        self.calls = []

    async def __call__(self, user_message, conversation_history):
        self.calls.append((user_message, conversation_history))
        return self._turns[min(len(self.calls) - 1, len(self._turns) - 1)]


@pytest.mark.asyncio
async def test_no_pending_steer_runs_a_single_turn(adapter):
    one_turn = _Scripted([_turn(final="done")])
    result, usage = await adapter._run_with_steer_redelivery(
        one_turn, "hello", [], session_id="s1"
    )
    assert len(one_turn.calls) == 1
    assert one_turn.calls[0] == ("hello", [])
    assert result["final_response"] == "done"
    assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


@pytest.mark.asyncio
async def test_leftover_steer_is_redelivered_as_a_follow_up_turn(adapter):
    transcript = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "first"}]
    one_turn = _Scripted([
        _turn(final="first", messages=transcript, pending_steer="make it formal"),
        _turn(final="second"),
    ])

    result, usage = await adapter._run_with_steer_redelivery(
        one_turn, "hello", [], session_id="s1"
    )

    # Two turns: the original, then the leftover steer re-delivered.
    assert len(one_turn.calls) == 2
    assert one_turn.calls[0] == ("hello", [])
    # The follow-up runs with the steer text as the user message and the
    # just-finished transcript as history (so the re-run persists only the new turn).
    assert one_turn.calls[1] == ("make it formal", transcript)
    assert result["final_response"] == "second"
    # Usage is summed across both turns.
    assert usage == {"input_tokens": 2, "output_tokens": 4, "total_tokens": 6}


@pytest.mark.asyncio
async def test_interrupt_supersedes_a_pending_steer(adapter):
    # A stop / request_input interrupt drops the steer; never re-deliver it.
    one_turn = _Scripted([
        _turn(final="first", pending_steer="too late", interrupted=True),
    ])
    result, _ = await adapter._run_with_steer_redelivery(one_turn, "hello", [], session_id="s1")
    assert len(one_turn.calls) == 1
    assert result["interrupted"] is True


@pytest.mark.asyncio
async def test_redelivery_is_bounded_by_the_cap(adapter):
    # A turn that always ends on a fresh steer must not spin forever.
    one_turn = _Scripted([_turn(final="loop", pending_steer="again")])
    result, usage = await adapter._run_with_steer_redelivery(
        one_turn, "hello", [], session_id="s1"
    )
    # Initial turn + _MAX_STEER_REDELIVERY follow-ups, then it stops.
    assert len(one_turn.calls) == adapter._MAX_STEER_REDELIVERY + 1
    # Usage is summed across every turn that actually ran.
    assert usage["total_tokens"] == 3 * (adapter._MAX_STEER_REDELIVERY + 1)
    # The still-pending steer is surfaced in the result but dropped (logged), not retried.
    assert result["pending_steer"] == "again"


@pytest.mark.asyncio
async def test_blank_or_missing_usage_does_not_break_summing(adapter):
    one_turn = _Scripted([
        _turn(final="first", pending_steer="more", usage=None),
        (
            {"final_response": "second", "messages": [], "interrupted": False},
            None,  # a turn with no usage dict
        ),
    ])
    result, usage = await adapter._run_with_steer_redelivery(one_turn, "hi", [], session_id="s1")
    assert len(one_turn.calls) == 2
    # The None usage on the second turn is ignored, not summed.
    assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert result["final_response"] == "second"
