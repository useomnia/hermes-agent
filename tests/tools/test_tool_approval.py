"""Tests for tools/tool_approval.py — blocking per-call approval for WRITE tools."""

import pytest

import tools.tool_approval as tool_approval
from tools.approval import reset_current_session_key, set_current_session_key
from tools.tool_approval import (
    APPROVAL_OPTION_SCOPES,
    APPROVAL_OPTIONS,
    _parse_gated_slugs,
    _notify_cbs,
    _session_approved,
    _waits,
    clear_session,
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


class TestClearSession:
    def test_should_drop_session_grants(self):
        resolve_tool_approval(SESSION, GATED, "session")
        clear_session(SESSION)
        assert is_tool_approved(SESSION, GATED) is False
