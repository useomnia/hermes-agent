"""Tests for tools/tool_approval.py — blocking per-call approval for WRITE tools."""

import threading

import pytest

import tools.tool_approval as tool_approval
from tools.approval import reset_current_session_key, set_current_session_key
from tools.interrupt import set_interrupt
from tools.tool_approval import (
    APPROVAL_OPTION_SCOPES,
    APPROVAL_OPTIONS,
    _ApprovalWait,
    _parse_gated_slugs,
    _notify_cbs,
    _session_approved,
    _waits,
    clear_session,
    fail_closed_denial,
    is_gated_tool,
    is_tool_approved,
    maybe_require_tool_approval,
    register_tool_approval_notify,
    resolve_tool_approval,
    unregister_tool_approval_notify,
)

GATED = "mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT"
READ = "mcp_connectors_GOOGLE_ANALYTICS_RUN_REPORT"
SESSION = "sess-1"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # The gated set + killswitch are frozen at import (hardening against
    # in-process bypass), so patch the frozen module attrs rather than os.environ.
    monkeypatch.setattr(tool_approval, "_GATED_SLUGS_FROZEN", frozenset({"gmail_create_email_draft"}))
    monkeypatch.setattr(tool_approval, "_DISABLED_FROZEN", False)
    token = set_current_session_key(SESSION)
    _session_approved.clear()
    _notify_cbs.clear()
    _waits.clear()
    yield
    _session_approved.clear()
    _notify_cbs.clear()
    _waits.clear()
    reset_current_session_key(token)


def _resolving_notify(scope):
    """A notify that resolves the just-enqueued wait synchronously, so the guard
    unblocks in-thread — exercises the full block→resolve path without a thread."""

    def cb(event):
        resolve_tool_approval(SESSION, event["interaction"]["approval"]["tool"], scope)

    return cb


class TestIsGatedTool:
    def test_should_gate_a_connector_write_tool(self):
        assert is_gated_tool(GATED) is True

    def test_should_gate_case_insensitively_when_the_name_is_lower_cased(self):
        assert is_gated_tool("mcp_connectors_gmail_create_email_draft") is True

    def test_should_not_gate_a_connector_read_tool(self):
        assert is_gated_tool(READ) is False

    def test_should_not_gate_when_the_write_list_is_empty(self, monkeypatch):
        monkeypatch.setattr(tool_approval, "_GATED_SLUGS_FROZEN", frozenset())
        assert is_gated_tool(GATED) is False

    def test_should_not_gate_when_the_killswitch_is_set(self, monkeypatch):
        monkeypatch.setattr(tool_approval, "_DISABLED_FROZEN", True)
        assert is_gated_tool(GATED) is False

    def test_should_not_gate_a_non_mcp_tool(self):
        assert is_gated_tool("terminal") is False

    def test_should_parse_no_slugs_from_malformed_env(self):
        assert _parse_gated_slugs("not json") == frozenset()
        assert _parse_gated_slugs("") == frozenset()
        assert _parse_gated_slugs('{"not": "a list"}') == frozenset()


class TestMaybeRequireToolApproval:
    def test_should_allow_a_read_tool_without_prompting(self):
        assert maybe_require_tool_approval(READ) is None

    def test_should_proceed_when_the_user_allows_once(self):
        register_tool_approval_notify(SESSION, _resolving_notify("once"))
        assert maybe_require_tool_approval(GATED, "call-1") is None

    def test_should_deny_when_the_user_denies(self):
        register_tool_approval_notify(SESSION, _resolving_notify("deny"))
        result = maybe_require_tool_approval(GATED, "call-1")
        assert result is not None
        assert "not performed" in result.lower()

    def test_should_remember_a_session_grant_so_the_next_call_doesnt_prompt(self):
        register_tool_approval_notify(SESSION, _resolving_notify("session"))
        assert maybe_require_tool_approval(GATED) is None
        # Second call: even with the notify gone, the session grant lets it proceed.
        unregister_tool_approval_notify(SESSION)
        assert maybe_require_tool_approval(GATED) is None

    def test_once_grant_does_not_carry_to_the_next_call(self):
        register_tool_approval_notify(SESSION, _resolving_notify("once"))
        assert maybe_require_tool_approval(GATED) is None
        # The once-grant was for that one call; without the notify the next call
        # has no interactive surface and fails closed.
        unregister_tool_approval_notify(SESSION)
        assert maybe_require_tool_approval(GATED) is not None

    def test_should_fail_closed_with_no_interactive_surface(self):
        # No notify registered (e.g. a proactive /v1/runs task): deny, don't hang.
        result = maybe_require_tool_approval(GATED, "call-1")
        assert result is not None
        assert "approval" in result.lower()

    def test_should_fail_closed_on_timeout(self, monkeypatch):
        monkeypatch.setenv("OMNIO_TOOL_APPROVAL_TIMEOUT", "0")
        register_tool_approval_notify(SESSION, lambda event: None)  # never resolves
        result = maybe_require_tool_approval(GATED, "call-1")
        assert result is not None

    def test_should_surface_the_interaction_with_options_and_scopes(self):
        captured = {}
        register_tool_approval_notify(
            SESSION,
            lambda event: (captured.update(event), resolve_tool_approval(SESSION, GATED, "once")),
        )
        maybe_require_tool_approval(GATED, "call-9")

        it = captured["interaction"]
        assert captured["toolCallId"] == "call-9"
        assert it["kind"] == "approval"
        assert it["options"] == APPROVAL_OPTIONS
        assert it["approval"]["tool"] == GATED
        assert it["approval"]["option_scopes"] == APPROVAL_OPTION_SCOPES


class TestResolveToolApproval:
    def test_should_reject_an_invalid_scope(self):
        assert resolve_tool_approval(SESSION, GATED, "forever") is False

    def test_should_record_a_session_grant_even_with_no_pending_wait(self):
        # Resolve arriving before the guard blocked still records the grant.
        assert resolve_tool_approval(SESSION, GATED, "session") is True
        assert is_tool_approved(SESSION, GATED) is True

    def test_should_scope_grants_per_session(self):
        resolve_tool_approval(SESSION, GATED, "session")
        assert is_tool_approved("other-session", GATED) is False

    def test_should_resolve_the_matching_call_id_not_the_queue_head(self):
        # Two writes blocked in one turn: resolving by call id must release the
        # one the user decided on, not whichever is first in the FIFO.
        first = _ApprovalWait(GATED, "call-A")
        second = _ApprovalWait(GATED, "call-B")
        _waits[SESSION] = [first, second]

        assert resolve_tool_approval(SESSION, GATED, "once", "call-B") is True

        assert second.result == "once" and second.event.is_set()
        assert first.result is None and not first.event.is_set()

    def test_should_fall_back_to_fifo_head_when_no_call_id_is_given(self):
        first = _ApprovalWait(GATED, "call-A")
        second = _ApprovalWait(GATED, "call-B")
        _waits[SESSION] = [first, second]

        assert resolve_tool_approval(SESSION, GATED, "deny") is True

        assert first.result == "deny" and first.event.is_set()
        assert second.result is None and not second.event.is_set()


class TestConcurrentApproval:
    """Two gated writes blocked in one turn must resolve independently — a
    decision on one card must not release the other (the cross-talk bug)."""

    def test_a_decision_on_one_call_does_not_release_a_different_pending_call(self):
        results: dict[str, object] = {}
        both_blocked = threading.Event()
        captured: list[str] = []

        def notify(event):
            captured.append(event["interaction"]["approval"]["tool_call_id"])
            if len(captured) == 2:
                both_blocked.set()

        register_tool_approval_notify(SESSION, notify)

        def worker(call_id):
            # A raw thread doesn't inherit the session-key contextvar (the real
            # tool executor propagates it); bind it so the guard scopes to SESSION.
            set_current_session_key(SESSION)
            results[call_id] = maybe_require_tool_approval(GATED, call_id)

        thread_a = threading.Thread(target=worker, args=("call-A",))
        thread_b = threading.Thread(target=worker, args=("call-B",))
        thread_a.start()
        thread_b.start()
        assert both_blocked.wait(timeout=3), "both writes should be blocked on their cards"

        # Approve ONLY call-B for the session; call-A must stay blocked.
        resolve_tool_approval(SESSION, GATED, "session", "call-B")
        thread_b.join(timeout=3)

        assert results.get("call-B") is None, "approved call proceeds"
        assert thread_a.is_alive(), "call-A must NOT be released by call-B's decision"

        # Release call-A so the test doesn't leak a thread.
        resolve_tool_approval(SESSION, GATED, "deny", "call-A")
        thread_a.join(timeout=3)
        assert results.get("call-A") is not None, "denied call returns a denial"


class TestInterruptRelease:
    """An interrupt (user stop / chat disconnect) must release a blocked wait
    and fail closed — not park the worker for the full timeout."""

    def test_interrupt_releases_a_blocked_wait_and_denies(self):
        result: dict[str, object] = {}
        blocked = threading.Event()
        register_tool_approval_notify(SESSION, lambda event: blocked.set())

        def worker():
            set_current_session_key(SESSION)  # raw thread: bind the session-key contextvar
            result["choice"] = maybe_require_tool_approval(GATED, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert blocked.wait(timeout=3), "the card surfaced; the guard is blocking"

        # The gateway interrupts this worker thread on SSE disconnect / stop.
        set_interrupt(True, thread.ident)
        try:
            thread.join(timeout=5)
            assert not thread.is_alive(), "interrupt must release the wait, not park for the timeout"
            assert result["choice"] is not None, "interrupted wait fails closed (denial)"
        finally:
            set_interrupt(False, thread.ident)


class TestFailClosedDenial:
    def test_should_deny_a_gated_write(self):
        result = fail_closed_denial(GATED)
        assert result is not None
        assert "approval" in result.lower()

    def test_should_allow_an_ungated_read(self):
        assert fail_closed_denial(READ) is None

    def test_should_deny_when_classification_itself_raises(self, monkeypatch):
        def boom(_name):
            raise RuntimeError("cannot classify")

        monkeypatch.setattr(tool_approval, "is_gated_tool", boom)
        assert fail_closed_denial(GATED) is not None


class TestClearSession:
    def test_should_drop_session_grants(self):
        resolve_tool_approval(SESSION, GATED, "session")
        clear_session(SESSION)
        assert is_tool_approved(SESSION, GATED) is False
