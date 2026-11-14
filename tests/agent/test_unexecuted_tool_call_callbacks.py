"""The conversation loop marks a refused tool call while its callbacks fire."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import List

from agent.conversation_loop import _emit_unexecuted_tool_call_callbacks


def _call(call_id: str = "call-1", name: str = "request_user_input") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps({"question": "?"})),
    )


def _agent(seen: List[bool]) -> SimpleNamespace:
    agent = SimpleNamespace(unexecuted_tool_call_ids=set(), tool_progress_callback=None)
    agent.tool_start_callback = lambda call_id, *_: seen.append(
        call_id in agent.unexecuted_tool_call_ids
    )
    agent.tool_complete_callback = lambda call_id, *_: seen.append(
        call_id in agent.unexecuted_tool_call_ids
    )
    return agent


def test_lists_the_call_as_unexecuted_while_its_callbacks_fire() -> None:
    seen: List[bool] = []

    _emit_unexecuted_tool_call_callbacks(_agent(seen), _call(), "refused")

    assert seen == [True, True]


def test_forgets_the_call_once_its_callbacks_fired() -> None:
    agent = _agent([])

    _emit_unexecuted_tool_call_callbacks(agent, _call(), "refused")

    assert agent.unexecuted_tool_call_ids == set()


def test_forgets_the_call_when_a_callback_raises() -> None:
    agent = _agent([])

    def explode(*_args: object) -> None:
        raise RuntimeError("callback failed")

    agent.tool_start_callback = explode

    _emit_unexecuted_tool_call_callbacks(agent, _call(), "refused")

    assert agent.unexecuted_tool_call_ids == set()


def test_fires_callbacks_for_an_agent_without_the_marker() -> None:
    seen: List[str] = []
    agent = SimpleNamespace(
        tool_progress_callback=None,
        tool_start_callback=lambda call_id, *_: seen.append(call_id),
        tool_complete_callback=None,
    )

    _emit_unexecuted_tool_call_callbacks(agent, _call(), "refused")

    assert seen == ["call-1"]
