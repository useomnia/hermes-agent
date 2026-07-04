"""Tests for tools/user_input.py — blocking answer delivery for request_user_input."""

import threading
import time

import pytest

import tools.user_input as user_input
from tools.interrupt import set_interrupt
from tools.user_input import (
    _InputWait,
    _active_sessions,
    _waits,
    await_user_input,
    clear_session,
    register_user_input_session,
    resolve_user_input,
    unregister_user_input_session,
)

SESSION = "sess-1"


@pytest.fixture(autouse=True)
def _clean_state():
    _waits.clear()
    _active_sessions.clear()
    register_user_input_session(SESSION)  # an interactive chat surface is present
    yield
    _waits.clear()
    _active_sessions.clear()


def _wait_until_blocked(session_key: str = SESSION, timeout: float = 3.0) -> bool:
    """Poll until a waiter is registered for the session (the worker is parked)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with user_input._lock:
            if _waits.get(session_key):
                return True
        time.sleep(0.01)
    return False


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

    def test_timeout_drops_the_waiter(self, monkeypatch):
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "0")
        await_user_input(SESSION, "call-1")
        assert SESSION not in _waits, "a timed-out wait must not leak a waiter"

    def test_returns_none_without_a_session_key_and_does_not_park(self):
        # No conversation surface to receive an answer → return immediately.
        assert await_user_input("", "call-1") is None
        assert not _waits

    def test_returns_none_without_an_interactive_surface_and_does_not_park(self):
        # A session with no registered chat surface (e.g. a proactive /v1/runs
        # task) must fail fast, not park for the full timeout with no one to answer.
        _active_sessions.discard(SESSION)
        assert await_user_input(SESSION, "call-1") is None
        assert SESSION not in _waits

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
        finally:
            set_interrupt(False, thread.ident)


class TestTimeoutOverride:
    def test_timeout_s_overrides_the_env_default(self, monkeypatch):
        # A short explicit budget must win over a long env/default timeout.
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "600")
        start = time.monotonic()
        assert await_user_input(SESSION, "call-1", timeout_s=0) is None
        assert time.monotonic() - start < 2.0, "timeout_s=0 must not park for the env timeout"
        assert SESSION not in _waits, "a timed-out wait must not leak a waiter"

    def test_timeout_s_none_keeps_the_env_default(self, monkeypatch):
        monkeypatch.setenv("OMNIO_USER_INPUT_TIMEOUT", "0")
        assert await_user_input(SESSION, "call-1", timeout_s=None) is None

    def test_resolve_delivers_the_answer_within_a_timeout_s_window(self):
        result: dict[str, object] = {}

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1", timeout_s=30)

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked(), "the worker should be parked on the wait"

        assert resolve_user_input(SESSION, '{"route": "/monitor"}', "call-1") is True
        thread.join(timeout=3)
        assert not thread.is_alive(), "resolve must release the wait"
        assert result["answer"] == '{"route": "/monitor"}'


class TestResolveUserInput:
    def test_returns_false_without_a_session_key(self):
        assert resolve_user_input("", "hi", "call-1") is False

    def test_returns_false_when_no_call_is_waiting(self):
        # Answer arrived but nothing is parked (stale card / already answered).
        assert resolve_user_input(SESSION, "hi", "call-1") is False

    def test_resolves_the_matching_call_id_not_the_queue_head(self):
        first = _InputWait("call-A")
        second = _InputWait("call-B")
        _waits[SESSION] = [first, second]

        assert resolve_user_input(SESSION, "answer-B", "call-B") is True

        assert second.answer == "answer-B" and second.event.is_set()
        assert first.answer is None and not first.event.is_set()

    def test_falls_back_to_fifo_head_when_no_call_id(self):
        first = _InputWait("call-A")
        second = _InputWait("call-B")
        _waits[SESSION] = [first, second]

        assert resolve_user_input(SESSION, "answer", "") is True

        assert first.answer == "answer" and first.event.is_set()
        assert second.answer is None and not second.event.is_set()

    def test_falls_back_to_fifo_when_call_id_does_not_match(self):
        only = _InputWait("call-A")
        _waits[SESSION] = [only]

        assert resolve_user_input(SESSION, "answer", "call-NOPE") is True
        assert only.answer == "answer" and only.event.is_set()

    def test_scopes_waiters_per_session(self):
        mine = _InputWait("call-A")
        _waits[SESSION] = [mine]

        assert resolve_user_input("other-session", "answer") is False
        assert mine.answer is None and not mine.event.is_set()


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
        assert SESSION not in _waits

    def test_is_a_noop_without_a_session_key(self):
        # Should not raise and should not touch state.
        clear_session("")


class TestSurfaceRegistry:
    def test_unregister_drops_the_surface_and_releases_waiters(self):
        result: dict[str, object] = {}

        def worker():
            result["answer"] = await_user_input(SESSION, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert _wait_until_blocked()

        unregister_user_input_session(SESSION)
        thread.join(timeout=3)
        assert not thread.is_alive(), "unregister must release parked waiters"
        assert result["answer"] is None
        assert SESSION not in _active_sessions
        # A subsequent input on the now-unregistered session fails fast.
        assert await_user_input(SESSION, "call-2") is None


class TestKillswitch:
    def test_blocking_disabled_is_a_module_bool(self):
        # Frozen at import (hardening against in-process flips); the api_server
        # seam reads it to fall back to the legacy non-blocking turn-ending path.
        assert isinstance(user_input.BLOCKING_DISABLED, bool)
