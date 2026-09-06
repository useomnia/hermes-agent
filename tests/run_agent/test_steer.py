"""Tests for AIAgent.steer() — mid-run user message injection.

/steer appends a durable user message after the current tool batch without
interrupting the tool call. Live requests and resumed history share that row.
"""
from __future__ import annotations

import threading

import pytest

from agent.prompt_builder import STEER_MARKER_OPEN, format_steer_marker
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__, then install the steer
    state manually — matches the existing object.__new__ stub pattern
    used elsewhere in the test suite.
    """
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = False
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._current_streamed_reasoning_text = ""
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = False
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    return agent


class TestSteerAcceptance:
    def test_accepts_non_empty_text(self):
        agent = _bare_agent()
        assert agent.steer("go ahead and check the logs") is True
        assert agent._pending_steer == "go ahead and check the logs"

    def test_rejects_empty_string(self):
        agent = _bare_agent()
        assert agent.steer("") is False
        assert agent._pending_steer is None

    def test_rejects_whitespace_only(self):
        agent = _bare_agent()
        assert agent.steer("   \n\t  ") is False
        assert agent._pending_steer is None

    def test_rejects_none(self):
        agent = _bare_agent()
        assert agent.steer(None) is False  # type: ignore[arg-type]
        assert agent._pending_steer is None

    def test_strips_surrounding_whitespace(self):
        agent = _bare_agent()
        assert agent.steer("  hello world  \n") is True
        assert agent._pending_steer == "hello world"

    def test_concatenates_multiple_steers_with_newlines(self):
        agent = _bare_agent()
        agent.steer("first note")
        agent.steer("second note")
        agent.steer("third note")
        assert agent._pending_steer == "first note\nsecond note\nthird note"


class TestSteerDrain:
    def test_drain_returns_and_clears(self):
        agent = _bare_agent()
        agent.steer("hello")
        assert agent._drain_pending_steer() == "hello"
        assert agent._pending_steer is None

    def test_drain_on_empty_returns_none(self):
        agent = _bare_agent()
        assert agent._drain_pending_steer() is None


class TestActiveTurnRedirect:
    def test_rejects_when_no_turn_is_active(self):
        agent = _bare_agent()
        assert agent.redirect("change course") is False
        assert agent._pending_redirect is None

    def test_cancels_only_an_active_model_request(self):
        agent = _bare_agent()
        agent._model_request_active.set()

        assert agent.redirect("use Postgres") is True
        assert agent._pending_redirect == "use Postgres"
        assert agent._interrupt_requested is True
        assert agent._interrupt_message is None

    def test_multiple_redirects_preserve_message_boundaries(self):
        agent = _bare_agent()
        agent._model_request_active.set()

        assert agent.redirect("first correction") is True
        assert agent.redirect("second correction") is True
        assert agent._pending_redirect == (
            "first correction\n\n"
            "[Additional user correction]\n"
            "second correction"
        )

    def test_hard_interrupt_wins_over_new_redirect(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        agent._interrupt_requested = True

        assert agent.redirect("too late") is False
        assert agent._pending_redirect is None

    def test_hidden_reasoning_is_not_checkpointed(self):
        agent = _bare_agent()
        agent.reasoning_callback = None
        agent._current_streamed_reasoning_text = ""

        agent._fire_reasoning_delta("private provider thinking")

        assert agent._current_streamed_reasoning_text == ""

    def test_response_completion_before_redirect_lock_rejects_correction(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        started = threading.Event()
        outcome = {}

        def redirect():
            started.set()
            outcome["accepted"] = agent.redirect("late correction")

        with agent._pending_redirect_lock:
            worker = threading.Thread(target=redirect)
            worker.start()
            assert started.wait(timeout=1)
            # Mirrors conversation_loop clearing the request-active marker
            # under this same lock before redirect can commit its slot.
            agent._model_request_active.clear()
        worker.join(timeout=1)

        assert outcome["accepted"] is False
        assert agent._pending_redirect is None

    def test_hard_stop_wins_concurrent_redirect(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        start = threading.Barrier(3)
        outcome = {}

        def redirect():
            start.wait()
            outcome["redirect"] = agent.redirect("change course")

        def hard_stop():
            start.wait()
            agent.interrupt("stop requested")

        redirect_thread = threading.Thread(target=redirect)
        stop_thread = threading.Thread(target=hard_stop)
        redirect_thread.start()
        stop_thread.start()
        start.wait()
        redirect_thread.join(timeout=1)
        stop_thread.join(timeout=1)

        assert redirect_thread.is_alive() is False
        assert stop_thread.is_alive() is False
        assert agent._interrupt_requested is True
        assert agent._interrupt_message == "stop requested"
        assert agent._pending_redirect is None

    def test_codex_app_server_hard_stop_reaches_native_session(self):
        agent = _bare_agent()
        calls = []
        agent.api_mode = "codex_app_server"
        agent._codex_session = type(
            "_CodexSession",
            (),
            {"request_interrupt": lambda self: calls.append("interrupt")},
        )()

        agent.interrupt()

        assert calls == ["interrupt"]

    def test_codex_app_server_redirect_rejects_after_hard_stop(self):
        agent = _bare_agent()
        calls = []
        agent.api_mode = "codex_app_server"
        agent._interrupt_requested = True
        agent._codex_session = type(
            "_CodexSession",
            (),
            {"request_steer": lambda self, text: calls.append(text) or True},
        )()

        assert agent.redirect("too late") is False
        assert calls == []

    def test_redirect_during_tool_execution_uses_safe_steer_boundary(self):
        agent = _bare_agent()
        agent._executing_tools = True

        assert agent.redirect("also check migrations") is True
        assert agent._pending_redirect is None
        assert agent._pending_steer == "also check migrations"
        assert agent._interrupt_requested is False


class TestActiveTurnRedirectCheckpoint:
    def test_assistant_tail_puts_correction_last(self):
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _bare_agent()
        agent._current_streamed_reasoning_text = "Shown reasoning."
        agent._current_streamed_assistant_text = "Visible draft."
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "committed assistant item"},
        ]

        _apply_active_turn_redirect(agent, messages, "Use Postgres instead.")

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].endswith("Use Postgres instead.")
        assert sum(1 for m in messages if m["role"] == "assistant") == 1
        assert "Shown reasoning." in messages[-1]["api_content"]
        assert "Visible draft." in messages[-1]["api_content"]
        assert "Context from the interrupted assistant response" in messages[-1]["api_content"]

    def test_tool_tail_scaffold_never_on_assistant_placeholder(self):
        """Mid-tool redirects keep scaffold bytes on the user sidecar only."""
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]

        _apply_active_turn_redirect(agent, messages, "Stop and do X instead.")

        placeholder = messages[-2]
        correction = messages[-1]
        assert placeholder["role"] == "assistant"
        assert placeholder.get("display_kind") == "hidden"
        assert placeholder.get("content") == ""
        assert not placeholder.get("api_content")
        assert correction["role"] == "user"
        assert correction["content"] == "Stop and do X instead."
        assert correction["api_content"].startswith(
            "[Context from the interrupted assistant response]\n"
            "[This response was interrupted by a user correction.]"
        )


class TestSteerInjection:
    def test_appends_standalone_user_message_after_tool_results(self):
        agent = _bare_agent()
        agent.steer("please also check auth.log")
        messages = [
            {"role": "user", "content": "what's in /var/log?"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "ls output A", "tool_call_id": "a"},
            {"role": "tool", "content": "ls output B", "tool_call_id": "b"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=2)
        # Existing tool rows are untouched (append-only persistence contract);
        # the steer becomes a NEW user message at the tail.
        assert messages[2]["content"] == "ls output A"
        assert messages[3]["content"] == "ls output B"
        assert messages[-1]["role"] == "user"
        assert STEER_MARKER_OPEN in messages[-1]["content"]
        assert "please also check auth.log" in messages[-1]["content"]
        # Role-alternation pattern: assistant(tool_calls) → tool → user is
        # the documented legal "user jumped in mid-run" shape.
        # And pending_steer is consumed.
        assert agent._pending_steer is None

    def test_appended_user_message_is_persistable(self):
        """The appended user dict carries no _DB_PERSISTED_MARKER yet, so the
        next _flush_messages_to_session_db writes it to state.db — the steer
        text lands in the durable transcript (messages.content, role=user)."""
        from agent.context_compressor import _DB_PERSISTED_MARKER

        agent = _bare_agent()
        agent.steer("remember this decision")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "output", "tool_call_id": "a"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["role"] == "user"
        assert _DB_PERSISTED_MARKER not in messages[-1]

    def test_no_op_when_no_steer_pending(self):
        agent = _bare_agent()
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "output", "tool_call_id": "a"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["content"] == "output"  # unchanged


    def test_marker_labels_text_as_out_of_band_user_message(self):
        """The injection marker must attribute the appended text to the user
        via the explicit out-of-band marker (which the system prompt tells the
        model to trust) — otherwise the model reads it as untrusted tool output
        and refuses it as suspected prompt injection.  Cache-safe: the marker
        is delivered as a NEW user message, never by rewriting existing tool
        content, so the persisted transcript matches the wire bytes.
        """
        agent = _bare_agent()
        agent.steer("stop after next step")
        messages = [{"role": "tool", "content": "x", "tool_call_id": "1"}]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["role"] == "user"
        content = messages[-1]["content"]
        assert STEER_MARKER_OPEN in content
        assert "stop after next step" in content
        # The tool row itself is untouched.
        assert messages[0]["content"] == "x"

    def test_multimodal_content_list_preserved(self):
        """Anthropic-style list content on tool results is left untouched —
        the steer is appended as a standalone user message instead of being
        merged into the content blocks."""
        agent = _bare_agent()
        agent.steer("extra note")
        original_blocks = [{"type": "text", "text": "existing output"}]
        messages = [
            {"role": "tool", "content": list(original_blocks), "tool_call_id": "1"}
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[0]["content"] == original_blocks       # untouched
        assert messages[-1]["role"] == "user"
        assert "extra note" in messages[-1]["content"]



class TestSteerThreadSafety:
    def test_concurrent_steer_calls_preserve_all_text(self):
        agent = _bare_agent()
        N = 200

        def worker(idx: int) -> None:
            agent.steer(f"note-{idx}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        text = agent._drain_pending_steer()
        assert text is not None
        # Every single note must be preserved — none dropped by the lock.
        lines = text.split("\n")
        assert len(lines) == N
        assert set(lines) == {f"note-{i}" for i in range(N)}


class TestSteerClearedOnInterrupt:
    def test_clear_interrupt_drops_pending_steer(self):
        """A hard interrupt supersedes any pending steer — the agent's
        next tool iteration won't happen, so delivering the steer later
        would be surprising."""
        agent = _bare_agent()
        # Minimal surface needed by clear_interrupt()
        agent._interrupt_requested = True
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = None
        agent._tool_worker_threads_lock = None

        agent.steer("will be dropped")
        agent._pending_redirect = "also drop this"
        assert agent._pending_steer == "will be dropped"

        agent.clear_interrupt()
        assert agent._pending_steer is None
        assert agent._pending_redirect is None


class TestPreApiCallSteerDrain:
    def test_should_append_user_row_when_steer_arrives_between_batches(self):
        agent = _bare_agent()
        messages = [
            {"role": "tool", "content": "output", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "checking"},
        ]
        agent.steer("change approach")
        agent._apply_pending_steer_to_tool_results(messages, len(messages))
        assert messages[-1] == {"role": "user", "content": format_steer_marker("change approach")}
        assert messages[0]["content"] == "output"

    def test_should_keep_steer_pending_when_no_batch_exists(self):
        agent = _bare_agent()
        agent.steer("early steer")
        messages = [{"role": "user", "content": "hello"}]
        agent._apply_pending_steer_to_tool_results(messages, len(messages))
        assert messages == [{"role": "user", "content": "hello"}]
        assert agent._drain_pending_steer() == "early steer"

    def test_should_persist_steer_once_after_previously_flushed_tool(self, tmp_path):
        from hermes_state import SessionDB
        from tests.agent.test_session_rotation_flush_cold_resume_68454 import _make_flush_agent

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("steer", source="cli")
        agent = _bare_agent()
        flush_agent = _make_flush_agent(db, "steer")
        messages = [
            {"role": "user", "content": "check logs"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "output", "tool_call_id": "tc1"},
        ]
        flush_agent._flush_messages_to_session_db(messages)
        agent.steer("remember this")
        agent._apply_pending_steer_to_tool_results(messages, len(messages))
        agent._repair_message_sequence(messages)
        flush_agent._flush_messages_to_session_db(messages)
        flush_agent._flush_messages_to_session_db(messages)
        saved = db.get_messages("steer")
        assert [m["role"] for m in saved] == ["user", "assistant", "tool", "user"]
        assert saved[-1]["content"] == messages[-1]["content"] == format_steer_marker("remember this")
        assert saved[-2]["content"] == "output"
        db.close()


class TestSteerMarkerContract:
    def test_system_prompt_note_describes_the_real_marker(self):
        """The system-prompt note tells the model which marker to trust; it
        must reference the exact open/close the injector emits, or the model
        trusts a marker that never appears (and vice-versa)."""
        from agent.prompt_builder import STEER_CHANNEL_NOTE, STEER_MARKER_CLOSE

        emitted = format_steer_marker("hi")
        assert STEER_MARKER_OPEN in emitted and STEER_MARKER_CLOSE in emitted
        assert STEER_MARKER_OPEN in STEER_CHANNEL_NOTE and STEER_MARKER_CLOSE in STEER_CHANNEL_NOTE

    def test_system_prompt_scopes_freshness_to_unanswered_marker(self):
        """A delivered marker remains in immutable history on later API calls.

        The prompt contract must distinguish the unanswered tail occurrence
        from one followed by an assistant response, or a model can interpret a
        historical steer as newly delivered and repeat non-idempotent work.
        """
        from agent.prompt_builder import STEER_CHANNEL_NOTE

        assert "latest tool-result batch" in STEER_CHANNEL_NOTE
        assert "no later assistant message follows it" in STEER_CHANNEL_NOTE
        assert "do not treat it as a new message" in STEER_CHANNEL_NOTE
        assert "repeat completed work" in STEER_CHANNEL_NOTE

        emitted = format_steer_marker("deploy once")
        assert "delivered once at this position" in emitted
        assert "not a new delivery when replayed" in emitted

    def test_marker_no_longer_uses_the_distrusted_label(self):
        """Regression: the bare 'User guidance:' line read as tool content and
        got refused as injection — it must not come back."""
        assert "User guidance:" not in format_steer_marker("hi")


class TestSteerCommandRegistry:
    def test_steer_in_command_registry(self):
        """The /steer slash command must be registered so it reaches all
        platforms (CLI, gateway, TUI autocomplete, Telegram/Slack menus).
        """
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("steer")
        assert cmd is not None
        assert cmd.name == "steer"
        assert cmd.category == "Session"
        assert cmd.args_hint == "<prompt>"

    def test_steer_in_bypass_set(self):
        """When the agent is running, /steer MUST bypass the Level-1
        base-adapter queue so it reaches the gateway runner's /steer
        handler. Otherwise it would be queued as user text and only
        delivered at turn end — defeating the whole point.
        """
        from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, should_bypass_active_session

        assert "steer" in ACTIVE_SESSION_BYPASS_COMMANDS
        assert should_bypass_active_session("steer") is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
