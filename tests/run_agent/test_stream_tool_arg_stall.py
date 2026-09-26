"""Tool-call argument stalls and slow-stream diagnostics in the streaming path.

A provider can keep a stream alive with chunks while a tool call's arguments
stop growing, so the chunk-based stale detector never fires and the turn
waits minutes before the attempt ends truncated. The argument watchdog
reconnects instead, and slow or truncated attempts leave diagnostics.
"""
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.stream_diag import stream_diag_summary
from tests.run_agent.test_streaming import _make_stream_chunk, _make_tool_call_delta


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class _Stream:
    response = SimpleNamespace(headers={"x-openrouter-provider": "OpenAI"})

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


def _write_file_call(arguments, tc_id="call_1"):
    return _make_tool_call_delta(index=0, tc_id=tc_id, name="write_file", arguments=arguments)


class TestToolArgumentStallWatchdog:
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_reconnect_when_tool_arguments_stop_growing_while_chunks_keep_arriving(
        self, mock_close, mock_create, mock_abort, mock_replace, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "0.2")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        class StalledArguments:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"# He')])
                # Chunks keep the stream alive, but the arguments never grow.
                for _ in range(40):
                    time.sleep(0.05)
                    yield _make_stream_chunk(content="")
                raise httpx.RemoteProtocolError("peer closed connection")

        complete = '{"path":"/tmp/a.md","content":"# Heading"}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            StalledArguments(),
            _Stream(
                [
                    _make_stream_chunk(tool_calls=[_write_file_call(complete, tc_id="call_2")]),
                    _make_stream_chunk(finish_reason="tool_calls", model="test/model"),
                ]
            ),
        ]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            response = agent._interruptible_streaming_api_call({})

        assert mock_abort.called
        assert "Tool call arguments stalled" in caplog.text
        assert response.choices[0].message.tool_calls[0].function.arguments == complete

    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_not_reconnect_while_tool_arguments_keep_growing(
        self, mock_close, mock_create, mock_abort, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "0.2")

        class GrowingArguments:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"')])
                for _ in range(12):
                    time.sleep(0.05)
                    yield _make_stream_chunk(tool_calls=[_write_file_call("x", tc_id=None)])
                yield _make_stream_chunk(tool_calls=[_write_file_call('"}', tc_id=None)])
                yield _make_stream_chunk(finish_reason="tool_calls", model="test/model")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [GrowingArguments()]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            response = agent._interruptible_streaming_api_call({})

        assert not mock_abort.called
        assert "Tool call arguments stalled" not in caplog.text
        assert response.choices[0].message.tool_calls[0].function.arguments.endswith('"}')

    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_not_watch_arguments_after_the_finish_reason_arrives(
        self, mock_close, mock_create, mock_abort, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "0.2")

        class TrailingUsage:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"ok"}')])
                yield _make_stream_chunk(finish_reason="tool_calls", model="test/model")
                time.sleep(0.5)
                yield _make_stream_chunk(content="")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [TrailingUsage()]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            agent._interruptible_streaming_api_call({})

        assert not mock_abort.called
        assert "Tool call arguments stalled" not in caplog.text


class TestSlowOrTruncatedStreamDiagnostics:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_log_finish_reason_and_token_spend_when_tool_arguments_arrive_truncated(
        self, mock_close, mock_create, monkeypatch, caplog
    ):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4000,
            total_tokens=4010,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3900),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _Stream(
                [
                    _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"# He')]),
                    _make_stream_chunk(finish_reason="length", model="test/model", usage=usage),
                ]
            ),
        ]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            agent._interruptible_streaming_api_call({})

        assert "Stream attempt truncated: finish_reason=length" in caplog.text
        assert "completion_tokens=4000 reasoning_tokens=3900" in caplog.text
        assert "tools=['write_file']" in caplog.text

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_stay_quiet_for_a_fast_complete_stream(self, mock_close, mock_create, caplog):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _Stream([_make_stream_chunk(content="done"), _make_stream_chunk(finish_reason="stop")])
        ]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            agent._interruptible_streaming_api_call({})

        assert "Stream attempt" not in caplog.text


class TestStreamDiagSummary:
    def test_should_summarize_timing_volume_and_provider_headers(self):
        diag = {
            "started_at": 100.0,
            "first_chunk_at": 102.5,
            "chunks": 42,
            "bytes": 9000,
            "max_chunk_gap_s": 61.25,
            "http_status": 200,
            "headers": {"x-openrouter-id": "gen-1", "x-openrouter-provider": "OpenAI"},
        }

        summary = stream_diag_summary(diag, now=160.0)

        assert summary == (
            "elapsed=60.0s first_chunk=2.5s chunks=42 bytes=9000 max_gap=61.2s http=200 "
            "x-openrouter-id=gen-1 x-openrouter-provider=OpenAI"
        )

    def test_should_report_no_first_chunk_and_tolerate_missing_diag(self):
        assert "first_chunk=none" in stream_diag_summary({"started_at": 1.0}, now=2.0)
        assert stream_diag_summary(None) == "none"
