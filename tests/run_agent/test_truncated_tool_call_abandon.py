"""A tool call dropped for unrepairable arguments must release its item.

A Responses consumer opens an in-progress function-call item as soon as the
provider streams a call's name. When the arguments then arrive truncated
beyond repair the call never reaches execution, so no execution boundary
closes that item — it survives to the run's terminal sweep, and a client
renders a tool card that runs for the rest of the turn.

The streaming assembly path therefore abandons such a call explicitly, using
the same tentative signal a dropped stream attempt uses, so a retry reusing
the deterministic provider call ID can still reclaim the item.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import json
import pytest


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
        model=None,
        usage=None,
    )


def _tc(index=0, tc_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _run(stream_factory):
    """Drive one streaming call, returning (response, lifecycle)."""
    with patch("run_agent.AIAgent._replace_primary_openai_client"), \
         patch("run_agent.AIAgent._close_request_openai_client"), \
         patch("run_agent.AIAgent._create_request_openai_client") as mock_create:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: stream_factory()
        )
        mock_create.return_value = mock_client

        agent = _agent()
        lifecycle: list[tuple] = []
        agent.tool_gen_event_callback = (
            lambda name, call_id: lifecycle.append(("started", name, call_id))
        )
        agent.tool_gen_event_aborted_callback = (
            lambda call_id: lifecycle.append(("aborted", call_id))
        )
        response = agent._interruptible_streaming_api_call({})
    return response, lifecycle


class TestTruncatedToolCallAbandon:

    def test_unrepairable_arguments_abandon_the_open_item(self):
        """The item opened while the name streamed is released, not leaked."""
        def _stream():
            yield _chunk(tool_calls=[_tc(tc_id="call-trunc", name="write_file")])
            yield _chunk(tool_calls=[
                _tc(arguments='{"path":"/tmp/a.md","content":"the first half of the fi'),
            ])
            yield _chunk(finish_reason="tool_calls")

        response, lifecycle = _run(_stream)

        assert ("started", "write_file", "call-trunc") in lifecycle
        assert ("aborted", "call-trunc") in lifecycle
        assert lifecycle.index(("started", "write_file", "call-trunc")) < lifecycle.index(
            ("aborted", "call-trunc")
        )

    def test_truncated_arguments_are_not_handed_to_execution(self):
        """The dropped bytes must not reach the tool as if they were whole."""
        def _stream():
            yield _chunk(tool_calls=[_tc(tc_id="call-trunc", name="write_file")])
            yield _chunk(tool_calls=[
                _tc(arguments='{"path":"/tmp/a.md","content":"the first half of the fi'),
            ])
            yield _chunk(finish_reason="tool_calls")

        response, _ = _run(_stream)

        call = response.choices[0].message.tool_calls[0]
        # The lossy recovery never reaches the executable payload: the raw
        # fragment is carried through as-is rather than silently completed
        # into parseable-but-short JSON a tool would act on.
        with pytest.raises(json.JSONDecodeError):
            json.loads(call.function.arguments)
        # Flagged as truncated so the turn retries rather than ending clean.
        assert response.choices[0].finish_reason == "length"

    def test_repairable_arguments_are_kept_and_not_abandoned(self):
        """A recoverable malformation still executes — no false abandonment."""
        def _stream():
            yield _chunk(tool_calls=[_tc(tc_id="call-ok", name="terminal")])
            yield _chunk(tool_calls=[_tc(arguments='{"command": "ls -la"')])
            yield _chunk(finish_reason="tool_calls")

        response, lifecycle = _run(_stream)

        call = response.choices[0].message.tool_calls[0]
        assert json.loads(call.function.arguments) == {"command": "ls -la"}
        assert ("aborted", "call-ok") not in lifecycle

    def test_intact_arguments_are_untouched(self):
        def _stream():
            yield _chunk(tool_calls=[_tc(tc_id="call-fine", name="terminal")])
            yield _chunk(tool_calls=[_tc(arguments='{"command": "pwd"}')])
            yield _chunk(finish_reason="tool_calls")

        response, lifecycle = _run(_stream)

        call = response.choices[0].message.tool_calls[0]
        assert json.loads(call.function.arguments) == {"command": "pwd"}
        assert ("aborted", "call-fine") not in lifecycle
        assert response.choices[0].finish_reason == "tool_calls"
