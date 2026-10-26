"""Reasoning-required rejection → step the aux effort up to the floor and remember the route.

Compression asks for thinking off (``auxiliary.compression.reasoning_effort: none`` →
``extra_body.reasoning: {"enabled": false}``). OpenRouter on ``openai/gpt-6.1-sol`` understands the
field but refuses the disable (400 "Reasoning is mandatory for this endpoint and cannot be disabled"),
so every compaction on that model failed. The recovery is a step UP (``low``), memoised per
(route, model) so the next thinking-off aux call on that route starts at the floor without the
guaranteed 400. Ported from upstream NousResearch/hermes-agent 2768f31628.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_reasoning_floor
from agent.auxiliary_client import async_call_llm, call_llm
from agent.error_classifier import is_reasoning_required_rejection

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_SOL = "openai/gpt-6.1-sol@preset/omnio-internal"
_LUNA = "openai/gpt-6-luna@preset/omnio"
_FLOOR = {"enabled": True, "effort": auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT}


class _ProviderError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _mandatory_400():
    return _ProviderError(
        "Error code: 400 - {'error': {'message': 'Reasoning is mandatory for this endpoint and cannot be "
        "disabled.', 'code': 400, 'metadata': {'provider_name': None}}}"
    )


@pytest.fixture(autouse=True)
def _fresh_memo():
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()
    yield
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()


def _patches(client, model):
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openrouter", model, _OPENROUTER_BASE, "sk-x", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, model)),
        patch("agent.auxiliary_client._get_task_extra_body", return_value={"reasoning": {"enabled": False}}),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
    )


def _compress(client, model=_SOL):
    p1, p2, p3, p4, p5 = _patches(client, model)
    with p1, p2, p3, p4, p5:
        return call_llm(task="compression", messages=[{"role": "user", "content": "summarise"}])


def _sync_client():
    client = MagicMock()
    client.base_url = _OPENROUTER_BASE
    return client


def test_reasoning_required_400_steps_effort_up_to_the_floor_and_remembers_the_route():
    """First call: disable → 400 → retry at the floor with everything else intact. Second call on the
    same route+model: the floor goes out up front, no 400 round-trip."""
    client = _sync_client()
    client.chat.completions.create.side_effect = [_mandatory_400(), {"ok": 1}, {"ok": 2}]

    assert _compress(client) == {"ok": 1}
    first, retry = (c.kwargs for c in client.chat.completions.create.call_args_list[:2])
    assert first["extra_body"]["reasoning"] == {"enabled": False}
    assert retry["extra_body"]["reasoning"] == _FLOOR
    assert {k: v for k, v in retry["extra_body"].items() if k != "reasoning"} == {
        k: v for k, v in first["extra_body"].items() if k != "reasoning"
    }
    assert retry["model"] == first["model"] == _SOL
    assert retry["messages"] == first["messages"]

    assert _compress(client) == {"ok": 2}
    assert client.chat.completions.create.call_count == 3
    upfront = client.chat.completions.create.call_args_list[2].kwargs
    assert upfront["extra_body"]["reasoning"] == _FLOOR


def test_memo_is_per_model_so_a_model_that_allows_off_keeps_it():
    client = _sync_client()
    client.chat.completions.create.side_effect = [_mandatory_400(), {"ok": 1}, {"ok": 2}]
    _compress(client, _SOL)

    assert _compress(client, _LUNA) == {"ok": 2}
    luna = client.chat.completions.create.call_args_list[2].kwargs
    assert luna["extra_body"]["reasoning"] == {"enabled": False}


def test_a_floor_that_still_fails_surfaces_the_error_and_remembers_nothing():
    client = _sync_client()
    client.chat.completions.create.side_effect = [_mandatory_400(), _ProviderError("Error code: 400 - bad request")]

    with pytest.raises(_ProviderError, match="bad request"):
        _compress(client)
    assert client.chat.completions.create.call_count == 2
    assert not auxiliary_reasoning_floor._FLOORED_ROUTES


def test_other_400s_are_not_stepped_up():
    client = _sync_client()
    client.chat.completions.create.side_effect = [_ProviderError("Error code: 400 - context length exceeded")]

    with pytest.raises(_ProviderError, match="context length"):
        _compress(client)
    assert client.chat.completions.create.call_count == 1
    assert not auxiliary_reasoning_floor._FLOORED_ROUTES


def test_async_reasoning_required_400_steps_up_and_remembers_the_route():
    client = MagicMock()
    client.base_url = _OPENROUTER_BASE
    client.chat.completions.create = AsyncMock(side_effect=[_mandatory_400(), {"ok": 1}, {"ok": 2}])

    async def _run():
        p1, p2, p3, p4, p5 = _patches(client, _SOL)
        with p1, p2, p3, p4, p5:
            first = await async_call_llm(task="compression", messages=[{"role": "user", "content": "x"}])
            second = await async_call_llm(task="compression", messages=[{"role": "user", "content": "x"}])
        return first, second

    assert asyncio.run(_run()) == ({"ok": 1}, {"ok": 2})
    calls = [c.kwargs for c in client.chat.completions.create.call_args_list]
    assert [c["extra_body"]["reasoning"] for c in calls] == [{"enabled": False}, _FLOOR, _FLOOR]


@pytest.mark.parametrize("message, expected", [
    ("Reasoning is mandatory for this endpoint and cannot be disabled.", True),
    ("thinking cannot be turned off for this model", True),
    ("Unrecognized request argument supplied: reasoning_effort", False),
    ("The model kimi-k2-thinking is mandatory for this plan", False),
    ("max_tokens is required", False),
])
def test_is_reasoning_required_rejection(message, expected):
    assert is_reasoning_required_rejection(message) is expected


def test_with_reasoning_floor_lifts_every_disabled_encoding_and_skips_enabled_requests():
    lifted = auxiliary_reasoning_floor.with_reasoning_floor({
        "reasoning_effort": "none",
        "_reasoning_config": {"enabled": False},
        "extra_body": {"reasoning": {"enabled": False}, "usage": {"include": True}},
    })
    assert lifted == {
        "reasoning_effort": "low",
        "_reasoning_config": _FLOOR,
        "extra_body": {"reasoning": _FLOOR, "usage": {"include": True}},
    }
    assert auxiliary_reasoning_floor.with_reasoning_floor({"extra_body": {"reasoning": {"effort": "high"}}}) is None
    assert auxiliary_reasoning_floor.with_reasoning_floor({"model": "m"}) is None
