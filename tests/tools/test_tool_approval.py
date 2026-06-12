"""Tests for tools/tool_approval.py — per-call approval for connector WRITE tools."""

import json

import pytest

from tools.approval import reset_current_session_key, set_current_session_key
from tools.tool_approval import (
    APPROVAL_OPTION_SCOPES,
    APPROVAL_OPTIONS,
    _session_approved,
    _once_approved,
    clear_session,
    is_gated_tool,
    is_tool_approved,
    maybe_require_tool_approval,
    record_tool_approval,
    resolve_tool_approval,
)

WRITE_ENV = "OMNIO_CONNECTORS_WRITE_TOOLS"
GATED = "mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT"
READ = "mcp_connectors_GOOGLE_ANALYTICS_RUN_REPORT"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # A single connector write is gated for every test unless overridden.
    monkeypatch.setenv(WRITE_ENV, json.dumps(["GMAIL_CREATE_EMAIL_DRAFT"]))
    monkeypatch.delenv("OMNIO_TOOL_APPROVAL_DISABLED", raising=False)
    token = set_current_session_key("sess-1")
    _session_approved.clear()
    _once_approved.clear()
    yield
    _session_approved.clear()
    _once_approved.clear()
    reset_current_session_key(token)


class TestIsGatedTool:
    def test_should_gate_a_connector_write_tool_when_its_slug_is_in_the_env_list(self):
        assert is_gated_tool(GATED) is True

    def test_should_not_gate_a_connector_read_tool(self):
        assert is_gated_tool(READ) is False

    def test_should_not_gate_when_the_write_list_is_empty(self, monkeypatch):
        monkeypatch.setenv(WRITE_ENV, "[]")
        assert is_gated_tool(GATED) is False

    def test_should_not_gate_when_the_killswitch_is_set(self, monkeypatch):
        monkeypatch.setenv("OMNIO_TOOL_APPROVAL_DISABLED", "1")
        assert is_gated_tool(GATED) is False

    def test_should_not_gate_a_non_mcp_tool(self):
        assert is_gated_tool("terminal") is False

    def test_should_not_gate_when_the_env_is_malformed(self, monkeypatch):
        monkeypatch.setenv(WRITE_ENV, "not json")
        assert is_gated_tool(GATED) is False


class TestMaybeRequireToolApproval:
    def test_should_allow_a_read_tool_without_a_prompt(self):
        assert maybe_require_tool_approval(READ) is None

    def test_should_require_approval_for_an_unapproved_gated_write(self):
        result = maybe_require_tool_approval(GATED, tool_call_id="call-1")

        assert result is not None
        payload = json.loads(result)
        assert payload["status"] == "approval_required"
        interaction = payload["interaction"]
        assert interaction["kind"] == "approval"
        assert interaction["options"] == APPROVAL_OPTIONS
        assert interaction["approval"]["tool"] == GATED
        assert interaction["approval"]["tool_call_id"] == "call-1"
        assert interaction["approval"]["option_scopes"] == APPROVAL_OPTION_SCOPES

    def test_should_allow_after_a_session_approval_is_recorded(self):
        record_tool_approval("sess-1", GATED, "session")

        assert maybe_require_tool_approval(GATED) is None

    def test_should_re_prompt_after_a_once_grant_is_consumed(self):
        record_tool_approval("sess-1", GATED, "once")

        # First call consumes the once-grant and runs ungated.
        assert maybe_require_tool_approval(GATED) is None
        # The next call must prompt again.
        assert maybe_require_tool_approval(GATED) is not None


class TestResolveToolApproval:
    def test_should_record_a_valid_session_scope(self):
        assert resolve_tool_approval("sess-1", GATED, "session") is True
        assert is_tool_approved("sess-1", GATED) is True

    def test_should_reject_an_invalid_scope(self):
        assert resolve_tool_approval("sess-1", GATED, "forever") is False

    def test_should_record_nothing_for_deny(self):
        assert resolve_tool_approval("sess-1", GATED, "deny") is True
        assert is_tool_approved("sess-1", GATED) is False

    def test_should_scope_approvals_per_session(self):
        record_tool_approval("sess-1", GATED, "session")

        assert is_tool_approved("sess-2", GATED) is False


class TestClearSession:
    def test_should_drop_all_approvals_for_a_session(self):
        record_tool_approval("sess-1", GATED, "session")
        clear_session("sess-1")

        assert is_tool_approved("sess-1", GATED) is False
