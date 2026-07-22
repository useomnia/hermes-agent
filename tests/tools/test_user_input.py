"""Tests for tools/user_input.py — blocking answer delivery for request_user_input."""

import threading
import time

import pytest

import tools.user_input as user_input
from tools.interrupt import set_interrupt
from tools.user_input import (
    await_user_input,
    clear_session,
    consume_user_input_completion_reason,
    register_user_input_session,
    resolve_user_input,
    unregister_user_input_session,
)

SESSION = "sess-1"


@pytest.fixture(autouse=True)
def _clean_state():
    clear_session(SESSION)
    register_user_input_session(SESSION)  # an interactive chat surface is present
    yield
    clear_session(SESSION)


def _wait_until_blocked(session_key: str = SESSION, timeout: float = 3.0) -> bool:
    """Poll until a waiter is registered for the session (the worker is parked)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if user_input._wait_registry.pending_count(session_key):
            return True
        time.sleep(0.01)
    return False


def _start_wait(
    call_id: str, session_key: str = SESSION
) -> tuple[threading.Thread, dict[str, object]]:
    result: dict[str, object] = {}
    previous_count = user_input._wait_registry.pending_count(session_key)

    def worker():
        result["answer"] = await_user_input(session_key, call_id)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if user_input._wait_registry.pending_count(session_key) > previous_count:
            break
        time.sleep(0.01)
    else:
        pytest.fail("the new waiter should be parked")
    return thread, result


class TestAwaitAndResolve:
    def test_resolve_delivers_the_answer_to_a_blocked_await(self):
        result: dict[str, object] = {}

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked(), "the worker should be parked on the wait"

        assert resolve_user_input(SESSION, "Tuesday at 3pm", "call-1") is True
        thread.join(timeout=3)
        assert not thread.is_alive(), "resolve must release the wait"
        assert result["answer"] == "Tuesday at 3pm"

    def test_delivers_an_empty_string_answer(self):
        # An empty answer (e.g. a skipped optional field) is a real answer, not a
        # "no response" — it must be delivered, not mapped to None.
        result: dict[str, object] = {}

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked()

        assert resolve_user_input(SESSION, "", "call-1") is True
        thread.join(timeout=3)
        assert result["answer"] == ""

    def test_returns_none_on_timeout(self, monkeypatch):
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "0")
        assert await_user_input(SESSION, "call-1") is None
        assert consume_user_input_completion_reason(SESSION) == "expired"

    def test_timeout_drops_the_waiter(self, monkeypatch):
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "0")
        await_user_input(SESSION, "call-1")
        assert user_input._wait_registry.pending_count(SESSION) == 0

    def test_next_wait_clears_the_previous_session_completion_reason(
        self, monkeypatch
    ):
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "0")
        assert await_user_input(SESSION, "call-old") is None
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "5")

        waiter, result = _start_wait("call-new")
        assert resolve_user_input(SESSION, "answer", "call-new") is True
        waiter.join(timeout=3)

        assert result["answer"] == "answer"
        assert consume_user_input_completion_reason(SESSION) is None

    def test_returns_none_without_a_session_key_and_does_not_park(self):
        # No conversation surface to receive an answer → return immediately.
        assert await_user_input("", "call-1") is None
        assert user_input._wait_registry.pending_count("") == 0

    def test_returns_none_without_an_interactive_surface_and_does_not_park(self):
        # A session with no registered chat surface (e.g. a proactive /v1/runs
        # task) must fail fast, not park for the full timeout with no one to answer.
        clear_session(SESSION)
        assert await_user_input(SESSION, "call-1") is None
        assert user_input._wait_registry.pending_count(SESSION) == 0

    def test_interrupt_releases_a_blocked_wait_and_returns_none(self):
        result: dict[str, object] = {}

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked(), "the worker should be parked on the wait"

        set_interrupt(True, thread.ident)
        try:
            thread.join(timeout=5)
            assert not thread.is_alive(), (
                "interrupt must release the wait, not park for the timeout"
            )
            assert result["answer"] is None, "an interrupted wait yields no answer"
            assert consume_user_input_completion_reason(SESSION) == "cancelled"
        finally:
            set_interrupt(False, thread.ident)


class TestResolveUserInput:
    def test_returns_false_without_a_session_key(self):
        assert resolve_user_input("", "hi", "call-1") is False

    def test_returns_false_when_no_call_is_waiting(self):
        # Answer arrived but nothing is parked (stale card / already answered).
        assert resolve_user_input(SESSION, "hi", "call-1") is False

    def test_resolves_the_matching_call_id_not_the_queue_head(self):
        first, first_result = _start_wait("call-A")
        second, second_result = _start_wait("call-B")

        assert resolve_user_input(SESSION, "answer-B", "call-B") is True

        second.join(timeout=3)
        assert second_result["answer"] == "answer-B"
        assert first.is_alive()
        assert resolve_user_input(SESSION, "answer-A", "call-A") is True
        first.join(timeout=3)
        assert first_result["answer"] == "answer-A"

    def test_falls_back_to_fifo_head_when_no_call_id(self):
        first, first_result = _start_wait("call-A")
        second, second_result = _start_wait("call-B")

        assert resolve_user_input(SESSION, "answer", "") is True

        first.join(timeout=3)
        assert first_result["answer"] == "answer"
        assert second.is_alive()
        assert resolve_user_input(SESSION, "answer-B", "call-B") is True
        second.join(timeout=3)
        assert second_result["answer"] == "answer-B"

    def test_falls_back_to_fifo_when_call_id_does_not_match(self):
        only, result = _start_wait("call-A")

        assert resolve_user_input(SESSION, "answer", "call-NOPE") is True
        only.join(timeout=3)
        assert result["answer"] == "answer"

    def test_scopes_waiters_per_session(self):
        mine, result = _start_wait("call-A")

        assert resolve_user_input("other-session", "answer") is False
        assert mine.is_alive()
        assert resolve_user_input(SESSION, "mine", "call-A") is True
        mine.join(timeout=3)
        assert result["answer"] == "mine"


class TestClearSession:
    def test_releases_a_blocked_wait_with_no_answer(self):
        result: dict[str, object] = {}

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked()

        clear_session(SESSION)
        thread.join(timeout=3)
        assert not thread.is_alive(), "clear_session must release parked waiters"
        assert result["answer"] is None, "a cleared wait yields no answer"
        assert user_input._wait_registry.pending_count(SESSION) == 0

    def test_is_a_noop_without_a_session_key(self):
        # Should not raise and should not touch state.
        clear_session("")


class TestSurfaceRegistry:
    def test_unregister_drops_the_surface_and_releases_waiters(self):
        result: dict[str, object] = {}
        token = register_user_input_session(SESSION)
        assert token is not None

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked()

        unregister_user_input_session(SESSION, token)
        thread.join(timeout=3)
        assert not thread.is_alive(), "unregister must release parked waiters"
        assert result["answer"] is None
        assert user_input._wait_registry.has_surface(SESSION) is False
        # A subsequent input on the now-unregistered session fails fast.
        assert await_user_input(SESSION, "call-2") is None

    def test_stale_unregister_preserves_the_new_run_state(self):
        first_token = register_user_input_session(SESSION)
        second_token = register_user_input_session(SESSION)
        assert first_token is not None
        assert second_token is not None
        assert first_token is not second_token
        waiter, result = _start_wait("call-new")

        unregister_user_input_session(SESSION, first_token)

        assert user_input._wait_registry.has_surface(SESSION) is True
        assert user_input._wait_registry.pending_count(SESSION) == 1
        assert waiter.is_alive()

        unregister_user_input_session(SESSION, second_token)

        waiter.join(timeout=3)
        assert not waiter.is_alive()
        assert result["answer"] is None
        assert user_input._wait_registry.has_surface(SESSION) is False
        assert user_input._wait_registry.pending_count(SESSION) == 0


class TestKillswitch:
    def test_blocking_disabled_is_a_module_bool(self):
        # Frozen at import (hardening against in-process flips); the api_server
        # seam reads it to fall back to the legacy non-blocking turn-ending path.
        assert isinstance(user_input.BLOCKING_DISABLED, bool)
