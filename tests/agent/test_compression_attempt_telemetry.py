import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

from agent.conversation_compression import compress_context
from agent.context_compressor import ContextCompressor


class _TodoStore:
    def format_for_injection(self):
        return ""


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "session-telemetry-test"
        self.platform = "cli"
        self.model = "test/main-model"
        self.provider = "test-provider"
        self.tools = []
        self._compression_feasibility_checked = True
        self.compression_in_place = False
        self._memory_manager = None
        self._session_db = None
        self._todo_store = _TodoStore()
        self._cached_system_prompt = None

    def _emit_status(self, _message):
        pass

    def _emit_warning(self, _message):
        pass

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, system_message):
        return system_message

    def commit_memory_session(self, _messages):
        pass


def _messages(secret_text="TOPSECRET_TRANSCRIPT_TEXT"):
    msgs = [{"role": "system", "content": "system prompt"}]
    for idx in range(10):
        msgs.append({"role": "user", "content": f"user message {idx} {secret_text}"})
        msgs.append({"role": "assistant", "content": f"assistant reply {idx} {secret_text}"})
    return msgs


def _extract_telemetry(caplog):
    records = [
        record.getMessage()
        for record in caplog.records
        if "context compression attempt telemetry:" in record.getMessage()
    ]
    assert len(records) == 1
    return json.loads(records[0].split("context compression attempt telemetry: ", 1)[1])


def test_compression_attempt_telemetry_is_metadata_only(caplog):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model",
            provider="test-provider",
            threshold_percent=0.50,
            quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)

    with patch.object(compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compressed, system_prompt = compress_context(
                agent,
                _messages(),
                "system prompt",
                approx_tokens=75_000,
                force=True,
            )

    assert system_prompt == "system prompt"
    assert compressed is not None
    payload = _extract_telemetry(caplog)

    assert payload["event"] == "compression_attempt"
    assert payload["attempt_id"]
    assert payload["session_id"] == "session-telemetry-test"
    assert payload["trigger_source"] == "manual"
    assert payload["main_model"] == "test/main-model"
    assert payload["main_context_limit"] == 100_000
    assert payload["current_estimated_tokens"] == 75_000
    assert payload["effective_threshold"] == compressor.threshold_tokens
    assert payload["protected_head_tokens"] is not None
    assert payload["protected_tail_tokens"] is not None
    assert payload["middle_window_tokens"] is not None
    assert payload["chunking"] is False
    assert payload["chunk_count"] in {0, 1}
    assert payload["commit_status"] == "committed"
    assert payload["split_status"] == "not_applicable"
    assert payload["fallback_used"] is False
    assert isinstance(payload["total_duration_ms"], int)

    raw_log = json.dumps(payload)
    assert "TOPSECRET_TRANSCRIPT_TEXT" not in raw_log
    assert "SANITIZED SUMMARY" not in raw_log
    assert "user message" not in raw_log
    assert "assistant reply" not in raw_log


def test_aux_call_telemetry_records_durations_without_content(caplog):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model",
            provider="test-provider",
            threshold_percent=0.50,
            quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="SANITIZED SUMMARY"))]
    )

    with patch("agent.context_compressor.call_llm", return_value=response):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compress_context(
                agent,
                _messages(),
                "system prompt",
                approx_tokens=75_000,
            )

    payload = _extract_telemetry(caplog)
    assert payload["aux_prompt_tokens"] is not None
    # Current main intentionally omits max_tokens from the aux summary call
    # (the summary budget is prompt-level guidance only), so no output
    # reservation is recorded.
    assert payload["aux_output_reservation"] is None
    assert isinstance(payload["aux_call_duration_ms"], int)
    assert payload["aux_provider"]
    assert payload["aux_model"]

    raw_log = json.dumps(payload)
    assert "TOPSECRET_TRANSCRIPT_TEXT" not in raw_log
    assert "SANITIZED SUMMARY" not in raw_log


async def _compress_with_turn_log(summary):
    import asyncio
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.turn_event_log import TurnEventEmitter, TurnEventLogStore

    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(model="test/main-model", provider="test-provider", threshold_percent=0.5, quiet_mode=True, config_context_length=100_000)
    compressor.tail_token_budget = 10
    compressor.abort_on_summary_failure = True
    agent = _Agent(compressor)
    store = TurnEventLogStore()
    log = store.create_run("run_compaction", agent.session_id)
    emitter = TurnEventEmitter(store, "run_compaction", agent.session_id)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": ""}))
    agent.tool_progress_callback = adapter._make_run_custom_event_callback(emitter, asyncio.get_running_loop())
    original = _messages()
    with patch.object(compressor, "_generate_summary", return_value=summary):
        compressed, _ = compress_context(agent, original, "system prompt", approx_tokens=75000, force=True)
    await asyncio.sleep(0)
    events = [json.loads(stored.frame.removeprefix(b"data: ").strip()) for stored in log.events]
    return original, compressed, events


def test_compaction_should_write_one_event_through_real_turn_bridge():
    import asyncio
    from agent.compaction_snapshot import capture_compaction, project_compaction
    from agent.redact import redact_sensitive_text
    from gateway.turn_event_log import _bounded_utf8

    original, compressed, events = asyncio.run(_compress_with_turn_log("[CONTEXT COMPACTION] summary"))
    assert len(compressed) < len(original)
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "response.omnio.compaction"
    snapshot = capture_compaction(compressed, previous_count=len(original) - 1, session_id="session-telemetry-test")
    expected = project_compaction(snapshot, redact=lambda text: redact_sensitive_text(text, force=True), bound=_bounded_utf8)
    assert {key: value for key, value in event.items() if key not in {"type", "sequence_number"}} == expected
    assert "_compressed_summary" not in json.dumps(event)
    assert "TOPSECRET_TRANSCRIPT_TEXT" not in json.dumps(event)
    assert event["retained_tail_refs"]


def test_compaction_should_emit_no_snapshot_on_summary_abort():
    import asyncio
    original, compressed, events = asyncio.run(_compress_with_turn_log(""))
    assert compressed == original
    assert events == []
