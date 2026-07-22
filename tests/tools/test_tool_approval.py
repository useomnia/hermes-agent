"""Tests for tools/tool_approval.py — blocking per-call approval for WRITE tools."""

import json
import threading
import time

import pytest

import tools.mcp_tool as mcp_tool
import tools.tool_approval as tool_approval
from tools.approval import reset_current_session_key, set_current_session_key
from tools.interrupt import set_interrupt
from tools.tool_approval import (
    APPROVAL_OPTION_SCOPES,
    APPROVAL_OPTIONS,
    CREDIT_APPROVAL_OPTION_SCOPES,
    CREDIT_APPROVAL_OPTIONS,
    _always_approved,
    _decisions,
    _injected_always_approved,
    _session_approved,
    await_tool_approval,
    clear_session,
    consume_tool_approval_completion_reason,
    consume_tool_approval_decision,
    fail_closed_denial,
    is_always_approved,
    is_credit_gated_tool,
    is_gated_tool,
    is_tool_approved,
    maybe_require_tool_approval,
    record_always_approval,
    record_session_approval,
    register_always_approval_authority,
    register_tool_approval_notify,
    replace_injected_always_approvals,
    resolve_tool_approval,
    unregister_tool_approval_notify,
)

GATED = "mcp_connectors_GMAIL_CREATE_EMAIL_DRAFT"
SIBLING = "mcp_connectors_GMAIL_SEND_EMAIL"
READ = "mcp_connectors_GOOGLE_ANALYTICS_RUN_REPORT"
CREDIT_GATED = "mcp_omnia_create_prompts_insights"
CREDIT_DESCRIPTOR = {
    "workflow": "insights",
    "strategy": "fixed",
    "unit": "per_engine",
    "creditsPerUnit": 30,
}
SESSION = "sess-1"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # The killswitch is frozen at import (hardening against in-process bypass), so
    # patch the module attr rather than os.environ.
    monkeypatch.setattr(tool_approval, "_DISABLED_FROZEN", False)
    token = set_current_session_key(SESSION)
    _session_approved.clear()
    _always_approved.clear()
    _injected_always_approved.clear()
    clear_session(SESSION)
    register_always_approval_authority(lambda _function_name: True)
    mcp_tool._mcp_tool_read_only_hints.clear()
    mcp_tool._mcp_tool_credits_meta.clear()
    # Model the connectors route having advertised its tools: the write is NOT
    # read-only (gated), the read IS (ungated). Gating reads the live annotation.
    mcp_tool._track_mcp_tool_read_only(GATED, False)
    mcp_tool._track_mcp_tool_read_only(SIBLING, False)
    mcp_tool._track_mcp_tool_read_only(READ, True)
    mcp_tool._track_mcp_tool_credits(CREDIT_GATED, CREDIT_DESCRIPTOR)
    yield
    _session_approved.clear()
    _always_approved.clear()
    _injected_always_approved.clear()
    clear_session(SESSION)
    register_always_approval_authority(None)
    mcp_tool._mcp_tool_read_only_hints.clear()
    mcp_tool._mcp_tool_credits_meta.clear()
    reset_current_session_key(token)


def _resolving_notify(scope):
    """A notify that resolves the just-enqueued wait synchronously, so the guard
    unblocks in-thread — exercises the full block→resolve path without a thread."""

    def cb(event):
        resolve_tool_approval(SESSION, event["interaction"]["approval"]["tool"], scope)

    return cb


def _start_approval_wait(
    function_name: str, call_id: str
) -> tuple[threading.Thread, dict[str, str | None]]:
    result: dict[str, str | None] = {}
    previous_count = tool_approval._wait_registry.pending_count(SESSION)

    def worker() -> None:
        result["choice"] = await_tool_approval(SESSION, function_name, {}, call_id)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if tool_approval._wait_registry.pending_count(SESSION) > previous_count:
            break
        time.sleep(0.01)
    else:
        pytest.fail("the approval waiter should be parked")
    return thread, result


class TestIsGatedTool:
    """Gating is driven by the live MCP ``readOnlyHint`` the route advertised: a
    connectors tool is gated unless it's explicitly read-only (fail closed)."""

    def test_should_gate_a_connector_write_tool(self):
        assert is_gated_tool(GATED) is True

    def test_should_gate_any_mcp_tool_with_a_credit_descriptor(self):
        assert is_credit_gated_tool(CREDIT_GATED) is True
        assert is_gated_tool(CREDIT_GATED) is True

    def test_should_not_gate_a_connector_read_tool(self):
        assert is_gated_tool(READ) is False

    def test_should_gate_an_unadvertised_connectors_tool_fail_closed(self):
        # No readOnlyHint recorded (e.g. the route hasn't advertised it yet) →
        # gate rather than risk running a write ungated.
        assert is_gated_tool("mcp_connectors_SOME_NEW_ACTION") is True

    def test_should_not_gate_when_the_killswitch_is_set(self, monkeypatch):
        monkeypatch.setattr(tool_approval, "_DISABLED_FROZEN", True)
        assert is_gated_tool(GATED) is False
        assert is_gated_tool(CREDIT_GATED) is False

    def test_should_not_gate_a_non_connectors_tool(self):
        assert is_gated_tool("terminal") is False
        # Another MCP server's write hint is NOT routed through the connector gate.
        mcp_tool._track_mcp_tool_read_only("mcp_other_DO_THING", False)
        assert is_gated_tool("mcp_other_DO_THING") is False
        assert is_gated_tool("mcp_omnia_unadvertised_tool") is False


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
        # Machine-readable status: an explicit deny is NOT turn-ending — the
        # agent continues and reports the denial inline.
        assert json.loads(result)["status"] == "approval_denied"

    def test_should_fail_closed_with_skip_semantics_when_the_user_types_past(self):
        register_tool_approval_notify(SESSION, _resolving_notify("skip"))

        result = maybe_require_tool_approval(GATED, "call-1")

        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] == "approval_skipped"
        assert "sending a new message instead" in parsed["error"]
        assert "arrives as the next user turn" in parsed["error"]
        assert "address their new message instead" in parsed["error"]
        assert "let them know it needs their approval" not in parsed["error"]

    def test_should_remember_a_session_grant_so_the_next_call_doesnt_prompt(self):
        notify = _resolving_notify("session")
        notify_token = register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(GATED) is None
        # Second call: even with the notify gone, the session grant lets it proceed.
        unregister_tool_approval_notify(SESSION, notify_token)
        assert maybe_require_tool_approval(GATED) is None

    def test_once_grant_does_not_carry_to_the_next_call(self):
        notify = _resolving_notify("once")
        notify_token = register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(GATED) is None
        # The once-grant was for that one call; without the notify the next call
        # has no interactive surface and fails closed.
        unregister_tool_approval_notify(SESSION, notify_token)
        assert maybe_require_tool_approval(GATED) is not None

    def test_should_fail_closed_with_no_interactive_surface(self):
        # No notify registered (e.g. a proactive /v1/runs task): deny, don't hang.
        result = maybe_require_tool_approval(GATED, "call-1")
        assert result is not None
        assert "approval" in result.lower()
        # No surface at all → NOT the turn-ending status: interrupting a
        # headless run on its first gated write would be wrong (there was
        # never anyone who could have answered in time).
        assert json.loads(result)["status"] != "approval_no_response"

    def test_should_fail_closed_on_timeout(self, monkeypatch):
        monkeypatch.setenv("OMNIO_TOOL_APPROVAL_TIMEOUT", "0")
        register_tool_approval_notify(SESSION, lambda event: None)  # never resolves
        result = maybe_require_tool_approval(GATED, "call-1")
        assert result is not None
        # A genuine timeout with a real interactive surface IS turn-ending.
        assert json.loads(result)["status"] == "approval_no_response"
        assert (
            consume_tool_approval_completion_reason(SESSION, "call-1") == "expired"
        )

    def test_notify_raising_is_a_plumbing_error_not_a_user_timeout(self):
        # The notify callback raising means the card was never actually shown
        # (chat stream write failed etc.) — the user may still be present, so
        # this must NOT look like a genuine no-response timeout.
        def raising_notify(event):
            raise RuntimeError("stream write failed")

        register_tool_approval_notify(SESSION, raising_notify)
        result = maybe_require_tool_approval(GATED, "call-1")
        assert result is not None
        assert json.loads(result)["status"] == "approval_error"

    def test_should_surface_the_interaction_with_options_and_scopes(self):
        captured = {}
        register_tool_approval_notify(
            SESSION,
            lambda event: (
                captured.update(event),
                resolve_tool_approval(SESSION, GATED, "once"),
            ),
        )
        maybe_require_tool_approval(GATED, "call-9")

        it = captured["interaction"]
        assert captured["toolCallId"] == "call-9"
        assert it["kind"] == "approval"
        assert it["options"] == APPROVAL_OPTIONS
        assert it["approval"]["tool"] == GATED
        assert it["approval"]["option_scopes"] == APPROVAL_OPTION_SCOPES
        assert it["approval"]["skip_scope"] is True
        assert "skip" not in it["approval"]["option_scopes"]
        assert "cost" not in it["approval"]


class TestCreditApproval:
    def test_should_surface_only_per_call_approve_and_deny_options(self):
        captured = {}

        def notify(event):
            captured.update(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once")

        register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(CREDIT_GATED, "call-credit") is None

        interaction = captured["interaction"]
        assert interaction["kind"] == "approval"
        assert interaction["options"] == ["Approve", "Deny"]
        assert interaction["options"] == CREDIT_APPROVAL_OPTIONS
        assert interaction["approval"]["tool"] == CREDIT_GATED
        assert interaction["approval"]["tool_call_id"] == "call-credit"
        assert interaction["approval"]["option_scopes"] == ["once", "deny"]
        assert (
            interaction["approval"]["option_scopes"]
            == CREDIT_APPROVAL_OPTION_SCOPES
        )
        assert interaction["approval"]["skip_scope"] is True
        assert "skip" not in interaction["approval"]["option_scopes"]

    def test_should_show_the_total_for_fixed_per_engine_spend(self):
        captured = {}

        def notify(event):
            captured.update(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once")

        register_tool_approval_notify(SESSION, notify)
        assert (
            maybe_require_tool_approval(
                CREDIT_GATED,
                "call-credit",
                function_args={"engines": ["google", 7, "perplexity"]},
            )
            is None
        )

        assert (
            "This call spends 90 credits (30 x 3 engines)."
            in captured["interaction"]["question"]
        )
        assert captured["interaction"]["approval"]["cost"] == {
            "credits": 90,
            "creditsPerUnit": 30,
            "unit": "per_engine",
            "engineCount": 3,
            "engines": ["google", "7", "perplexity"],
        }

    def test_should_deduplicate_engines_in_the_fixed_spend_total(self):
        captured = {}

        def notify(event):
            captured.update(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once")

        register_tool_approval_notify(SESSION, notify)
        assert (
            maybe_require_tool_approval(
                CREDIT_GATED,
                "call-credit",
                function_args={"engines": ["openai", "openai", "perplexity"]},
            )
            is None
        )

        question = captured["interaction"]["question"]
        assert "This call spends 60 credits (30 x 2 engines)." in question
        assert "90 credits" not in question
        assert captured["interaction"]["approval"]["cost"] == {
            "credits": 60,
            "creditsPerUnit": 30,
            "unit": "per_engine",
            "engineCount": 2,
            "engines": ["openai", "perplexity"],
        }

    @pytest.mark.parametrize(
        "function_args",
        [None, {"engines": "google"}],
        ids=["missing", "non-list"],
    )
    def test_should_show_the_per_engine_price_without_an_engine_list(
        self, function_args
    ):
        captured = {}

        def notify(event):
            captured.update(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once")

        register_tool_approval_notify(SESSION, notify)
        assert (
            maybe_require_tool_approval(
                CREDIT_GATED, "call-credit", function_args=function_args
            )
            is None
        )

        assert (
            "This call spends 30 credits per engine."
            in captured["interaction"]["question"]
        )
        assert captured["interaction"]["approval"]["cost"] == {
            "credits": None,
            "creditsPerUnit": 30,
            "unit": "per_engine",
            "engineCount": None,
            "engines": None,
        }

    def test_should_not_claim_a_total_for_real_cost_spend(self):
        captured = {}
        mcp_tool._track_mcp_tool_credits(
            CREDIT_GATED,
            {"workflow": "insights", "strategy": "real-cost", "unit": "run"},
        )

        def notify(event):
            captured.update(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once")

        register_tool_approval_notify(SESSION, notify)
        assert (
            maybe_require_tool_approval(
                CREDIT_GATED,
                "call-credit",
                function_args={"engines": ["google", "chatgpt", "perplexity"]},
            )
            is None
        )

        question = captured["interaction"]["question"]
        assert "This call spends credits based on its actual cost." in question
        assert "90 credits" not in question
        assert "30 x 3" not in question
        assert captured["interaction"]["approval"]["cost"] == {
            "credits": None,
            "creditsPerUnit": None,
            "unit": "run",
            "engineCount": 3,
            "engines": ["google", "chatgpt", "perplexity"],
        }

    def test_should_not_consult_preexisting_session_or_always_grants(self):
        record_session_approval(SESSION, CREDIT_GATED)
        record_always_approval(CREDIT_GATED)
        prompts = []

        def notify(event):
            prompts.append(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once")

        register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(CREDIT_GATED, "call-credit") is None
        assert len(prompts) == 1

    @pytest.mark.parametrize("scope", ["session", "always"])
    def test_should_treat_standing_scope_resolutions_as_once(self, scope):
        notify = _resolving_notify(scope)
        notify_token = register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(CREDIT_GATED, "call-credit") is None

        assert consume_tool_approval_decision(SESSION, "call-credit") == "once"
        assert is_tool_approved(SESSION, CREDIT_GATED) is False
        assert is_always_approved(CREDIT_GATED) is False

        unregister_tool_approval_notify(SESSION, notify_token)
        denial = maybe_require_tool_approval(CREDIT_GATED, "call-next")
        assert denial is not None
        assert json.loads(denial)["status"] == "approval_error"

    def test_should_enforce_once_from_the_blocked_call_identity(self):
        def notify(_event):
            assert (
                resolve_tool_approval(
                    SESSION,
                    GATED,
                    "always",
                    tool_call_id="call-credit",
                )
                is True
            )

        register_tool_approval_notify(SESSION, notify)
        choice = await_tool_approval(SESSION, CREDIT_GATED, {}, "call-credit")

        assert choice == "once"
        assert is_always_approved(GATED) is False

    def test_should_preserve_skip_scope_for_a_credit_gated_call(self):
        def notify(_event):
            assert (
                resolve_tool_approval(
                    SESSION,
                    CREDIT_GATED,
                    "skip",
                    tool_call_id="call-credit",
                )
                is True
            )

        register_tool_approval_notify(SESSION, notify)
        choice = await_tool_approval(SESSION, CREDIT_GATED, {}, "call-credit")

        assert choice == "skip"
        assert consume_tool_approval_decision(SESSION, "call-credit") == "skip"
        assert is_tool_approved(SESSION, CREDIT_GATED) is False
        assert is_always_approved(CREDIT_GATED) is False


class TestCreditApprovalDispatch:
    def test_should_execute_the_same_call_inline_when_approved(self, monkeypatch):
        import model_tools

        dispatched = []
        captured = {}

        def dispatch(function_name, function_args, **kwargs):
            dispatched.append((function_name, function_args))
            return json.dumps({"success": True})

        def notify(event):
            captured.update(event)
            resolve_tool_approval(SESSION, CREDIT_GATED, "once", "call-credit")

        monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
        register_tool_approval_notify(SESSION, notify)
        function_args = {"engines": ["google", "chatgpt", "perplexity"]}

        result = model_tools.handle_function_call(
            CREDIT_GATED,
            function_args,
            tool_call_id="call-credit",
            skip_pre_tool_call_hook=True,
        )

        assert json.loads(result) == {"success": True}
        assert dispatched == [(CREDIT_GATED, function_args)]
        assert "90 credits" in captured["interaction"]["question"]

    def test_should_not_execute_the_call_when_denied(self, monkeypatch):
        import model_tools

        dispatched = []

        def dispatch(function_name, function_args, **kwargs):
            dispatched.append((function_name, function_args))
            return json.dumps({"success": True})

        monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
        register_tool_approval_notify(SESSION, _resolving_notify("deny"))

        result = model_tools.handle_function_call(
            CREDIT_GATED,
            {"engines": ["google"]},
            tool_call_id="call-credit",
            skip_pre_tool_call_hook=True,
        )

        assert json.loads(result)["status"] == "approval_denied"
        assert dispatched == []

    def test_should_not_execute_the_call_when_skipped(self, monkeypatch):
        import model_tools

        dispatched = []

        def dispatch(function_name, function_args, **kwargs):
            dispatched.append((function_name, function_args))
            return json.dumps({"success": True})

        monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
        register_tool_approval_notify(SESSION, _resolving_notify("skip"))

        result = model_tools.handle_function_call(
            CREDIT_GATED,
            {"engines": ["google"]},
            tool_call_id="call-credit",
            skip_pre_tool_call_hook=True,
        )

        assert json.loads(result)["status"] == "approval_skipped"
        assert dispatched == []

    def test_should_fail_closed_without_dispatch_when_the_gate_raises(
        self, monkeypatch
    ):
        import model_tools

        dispatched = []

        def dispatch(function_name, function_args, **kwargs):
            dispatched.append((function_name, function_args))
            return json.dumps({"success": True})

        def raise_gate_error(*args, **kwargs):
            raise RuntimeError("credit gate failed")

        monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
        monkeypatch.setattr(
            tool_approval, "maybe_require_tool_approval", raise_gate_error
        )

        result = model_tools.handle_function_call(
            CREDIT_GATED,
            {"engines": ["google"]},
            tool_call_id="call-credit",
            skip_pre_tool_call_hook=True,
        )

        assert json.loads(result)["status"] == "approval_error"
        assert dispatched == []

class TestAlwaysScope:
    """`always` grants a tool for EVERY conversation on this gateway (not just the
    current chat) and survives clear_session until the next Omnia refresh."""

    def test_should_record_an_always_grant(self):
        # No waiter pending here, so the call itself returns False (nothing
        # released) even though the always grant is recorded for next time.
        assert resolve_tool_approval(SESSION, GATED, "always") is False
        assert is_always_approved(GATED) is True

    def test_should_record_all_supplied_tools_for_always(self):
        assert resolve_tool_approval(SESSION, GATED, "always", tools=[SIBLING]) is False
        assert is_always_approved(GATED) is True
        assert is_always_approved(SIBLING) is True

    def test_always_is_not_a_session_grant(self):
        # Stored gateway-wide, so another session sees no SESSION grant for it,
        # yet the always grant still lets that session's call proceed.
        resolve_tool_approval(SESSION, GATED, "always")
        assert is_tool_approved("another-session", GATED) is False
        assert is_always_approved(GATED) is True

    def test_always_grant_survives_clear_session(self):
        resolve_tool_approval(SESSION, GATED, "always")
        clear_session(SESSION)
        assert is_always_approved(GATED) is True
        assert maybe_require_tool_approval(GATED) is None

    def test_should_proceed_and_persist_when_the_user_allows_always(self):
        notify = _resolving_notify("always")
        notify_token = register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(GATED, "call-1") is None
        # The grant carried to the next call without any interactive surface.
        unregister_tool_approval_notify(SESSION, notify_token)
        assert maybe_require_tool_approval(GATED) is None

    def test_first_always_call_proceeds_but_later_call_waits_for_persistence(self):
        register_always_approval_authority(None)
        notify = _resolving_notify("always")
        notify_token = register_tool_approval_notify(SESSION, notify)

        assert maybe_require_tool_approval(GATED, "call-1") is None

        unregister_tool_approval_notify(SESSION, notify_token)
        result = maybe_require_tool_approval(GATED, "call-2")
        assert json.loads(result)["status"] == "approval_error"

    def test_warm_gateway_rechecks_and_prompts_after_authoritative_revoke(self):
        replace_injected_always_approvals([GATED])
        authority_results = iter([True, False])
        checked: list[str] = []

        def authority(function_name: str) -> bool:
            checked.append(function_name)
            return next(authority_results)

        register_always_approval_authority(authority)
        assert maybe_require_tool_approval(GATED, "call-1") is None

        prompts: list[dict] = []

        def deny_prompt(event: dict) -> None:
            prompts.append(event)
            resolve_tool_approval(SESSION, GATED, "deny", "call-2")

        register_tool_approval_notify(SESSION, deny_prompt)
        result = maybe_require_tool_approval(GATED, "call-2")

        assert json.loads(result)["status"] == "approval_denied"
        assert prompts[0]["interaction"]["approval"]["tool"] == GATED
        assert checked == [GATED, GATED]

    def test_authority_outage_prompts_instead_of_using_stale_grant(self):
        replace_injected_always_approvals([GATED])

        def unavailable(_function_name: str) -> bool:
            raise TimeoutError("omnia timed out")

        register_always_approval_authority(unavailable)
        register_tool_approval_notify(SESSION, _resolving_notify("deny"))

        result = maybe_require_tool_approval(GATED, "call-1")
        assert json.loads(result)["status"] == "approval_denied"

    def test_injected_always_refresh_replaces_local_always(self):
        resolve_tool_approval(SESSION, GATED, "always")
        replace_injected_always_approvals([SIBLING])

        assert is_always_approved(GATED) is False
        assert is_always_approved(SIBLING) is True

        replace_injected_always_approvals([])
        assert is_always_approved(GATED) is False
        assert is_always_approved(SIBLING) is False

    def test_revoke_reload_after_in_chat_always_grant_forces_the_next_call_to_prompt(
        self,
    ):
        notify = _resolving_notify("always")
        notify_token = register_tool_approval_notify(SESSION, notify)
        assert maybe_require_tool_approval(GATED, "call-1") is None
        unregister_tool_approval_notify(SESSION, notify_token)
        assert maybe_require_tool_approval(GATED, "call-2") is None

        replace_injected_always_approvals([])

        assert maybe_require_tool_approval(GATED, "call-3") is not None


class TestResolveToolApproval:
    def test_should_reject_an_invalid_scope(self):
        assert resolve_tool_approval(SESSION, GATED, "forever") is False

    def test_should_record_a_session_grant_even_with_no_pending_wait(self):
        # Resolve arriving before the guard blocked (or after it timed out)
        # still records the grant for the NEXT call — but returns False since
        # no waiter was actually released for THIS decision.
        assert resolve_tool_approval(SESSION, GATED, "session") is False
        assert is_tool_approved(SESSION, GATED) is True

    def test_should_record_all_supplied_tools_for_session(self):
        assert (
            resolve_tool_approval(SESSION, GATED, "session", tools=[SIBLING]) is False
        )
        assert is_tool_approved(SESSION, GATED) is True
        assert is_tool_approved(SESSION, SIBLING) is True

    def test_should_ignore_read_unknown_and_non_connector_tools_from_the_client_list(
        self,
    ):
        assert (
            resolve_tool_approval(
                SESSION,
                GATED,
                "session",
                tools=[SIBLING, READ, "mcp_connectors_UNKNOWN_WRITE", "terminal"],
            )
            is False
        )

        assert is_tool_approved(SESSION, GATED) is True
        assert is_tool_approved(SESSION, SIBLING) is True
        assert is_tool_approved(SESSION, READ) is False
        assert is_tool_approved(SESSION, "mcp_connectors_UNKNOWN_WRITE") is False
        assert is_tool_approved(SESSION, "terminal") is False

    def test_should_not_record_an_unknown_current_tool_name(self):
        assert (
            resolve_tool_approval(SESSION, "mcp_connectors_UNKNOWN_WRITE", "session")
            is False
        )

        assert is_tool_approved(SESSION, "mcp_connectors_UNKNOWN_WRITE") is False

    def test_no_waiter_once_records_nothing_and_returns_false(self):
        assert resolve_tool_approval(SESSION, GATED, "once") is False
        assert is_tool_approved(SESSION, GATED) is False
        assert is_always_approved(GATED) is False

    def test_no_waiter_session_records_grant_but_returns_false(self):
        assert resolve_tool_approval(SESSION, GATED, "session") is False
        assert is_tool_approved(SESSION, GATED) is True

    def test_waiter_present_once_returns_true(self):
        register_tool_approval_notify(SESSION, lambda _event: None)
        waiter, result = _start_approval_wait(GATED, "call-A")
        assert resolve_tool_approval(SESSION, GATED, "once", "call-A") is True
        waiter.join(timeout=3)
        assert not waiter.is_alive()
        assert result["choice"] == "once"

    def test_should_release_a_waiter_with_skip_without_recording_a_grant(self):
        def notify(_event):
            assert resolve_tool_approval(SESSION, GATED, "skip", "call-A") is True

        register_tool_approval_notify(SESSION, notify)

        choice = await_tool_approval(SESSION, GATED, {}, "call-A")

        assert choice == "skip"
        assert consume_tool_approval_decision(SESSION, "call-A") == "skip"
        assert is_tool_approved(SESSION, GATED) is False
        assert is_always_approved(GATED) is False

    def test_should_record_the_decision_for_the_completed_event_echo(self):
        # A released waiter's decision is consumed once by the gateway's
        # tool-complete callback (interaction.answered on the completed event).
        register_tool_approval_notify(SESSION, lambda _event: None)
        waiter, _result = _start_approval_wait(GATED, "call-A")
        resolve_tool_approval(SESSION, GATED, "deny", "call-A")
        waiter.join(timeout=3)

        assert consume_tool_approval_decision(SESSION, "call-A") == "deny"
        assert consume_tool_approval_decision(SESSION, "call-A") is None

    def test_should_store_the_decision_before_releasing_the_waiter(self, monkeypatch):
        write_started = threading.Event()
        allow_write = threading.Event()

        class BlockingDecisions(dict):
            def __setitem__(self, key, value):
                write_started.set()
                assert allow_write.wait(timeout=3)
                super().__setitem__(key, value)

        monkeypatch.setattr(tool_approval, "_decisions", BlockingDecisions())
        register_tool_approval_notify(SESSION, lambda _event: None)
        waiter, result = _start_approval_wait(GATED, "call-A")
        resolver = threading.Thread(
            target=lambda: resolve_tool_approval(
                SESSION, GATED, "once", "call-A"
            )
        )
        resolver.start()
        assert write_started.wait(timeout=3)
        waiter.join(timeout=0.1)
        assert waiter.is_alive(), "decision storage must finish before release"

        allow_write.set()
        resolver.join(timeout=3)
        waiter.join(timeout=3)
        assert not resolver.is_alive()
        assert not waiter.is_alive()
        assert result["choice"] == "once"
        assert consume_tool_approval_decision(SESSION, "call-A") == "once"

    def test_should_not_record_a_decision_when_no_waiter_was_released(self):
        # A late decision (the wait already timed out) records the grant but
        # must not echo answered on a call that already failed closed.
        assert resolve_tool_approval(SESSION, GATED, "session", "call-A") is False
        assert consume_tool_approval_decision(SESSION, "call-A") is None

    def test_should_scope_grants_per_session(self):
        resolve_tool_approval(SESSION, GATED, "session")
        assert is_tool_approved("other-session", GATED) is False

    def test_should_resolve_the_matching_call_id_not_the_queue_head(self):
        # Two writes blocked in one turn: resolving by call id must release the
        # one the user decided on, not whichever is first in the FIFO.
        register_tool_approval_notify(SESSION, lambda _event: None)
        first, first_result = _start_approval_wait(GATED, "call-A")
        second, second_result = _start_approval_wait(GATED, "call-B")

        assert resolve_tool_approval(SESSION, GATED, "once", "call-B") is True

        second.join(timeout=3)
        assert second_result["choice"] == "once"
        assert first.is_alive()
        assert resolve_tool_approval(SESSION, GATED, "deny", "call-A") is True
        first.join(timeout=3)
        assert first_result["choice"] == "deny"

    def test_should_fall_back_to_fifo_head_when_no_call_id_is_given(self):
        register_tool_approval_notify(SESSION, lambda _event: None)
        first, first_result = _start_approval_wait(GATED, "call-A")
        second, second_result = _start_approval_wait(GATED, "call-B")

        assert resolve_tool_approval(SESSION, GATED, "deny") is True

        first.join(timeout=3)
        assert first_result["choice"] == "deny"
        assert second.is_alive()
        assert resolve_tool_approval(SESSION, GATED, "once", "call-B") is True
        second.join(timeout=3)
        assert second_result["choice"] == "once"


class TestSurfaceRegistry:
    def test_stale_unregister_preserves_the_new_run_state(self):
        def first_notify(_event):
            pass

        def second_notify(_event):
            pass

        first_token = register_tool_approval_notify(SESSION, first_notify)
        second_token = register_tool_approval_notify(SESSION, second_notify)
        _wait_registry = tool_approval._wait_registry
        assert _wait_registry.wait(
            SESSION, "call-old", 0, "waiting", payload=GATED
        ) == (None, "expired")
        waiter, result = _start_approval_wait(GATED, "call-new")
        _decisions[(SESSION, "call-new")] = "once"

        unregister_tool_approval_notify(SESSION, first_token)

        assert _wait_registry.surface_value(SESSION) is second_notify
        assert _wait_registry.pending_count(SESSION) == 1
        assert waiter.is_alive()
        assert consume_tool_approval_completion_reason(SESSION, "call-old") == "expired"
        assert _decisions[(SESSION, "call-new")] == "once"

        assert _wait_registry.wait(
            SESSION, "call-old", 0, "waiting", payload=GATED
        ) == (None, "expired")
        unregister_tool_approval_notify(SESSION, second_token)

        waiter.join(timeout=3)
        assert not waiter.is_alive()
        assert result["choice"] is None
        assert _wait_registry.has_surface(SESSION) is False
        assert _wait_registry.pending_count(SESSION) == 0
        assert consume_tool_approval_completion_reason(SESSION, "call-old") is None
        assert (SESSION, "call-new") not in _decisions


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
        assert both_blocked.wait(timeout=3), (
            "both writes should be blocked on their cards"
        )

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
            set_current_session_key(
                SESSION
            )  # raw thread: bind the session-key contextvar
            result["choice"] = maybe_require_tool_approval(GATED, "call-1")

        thread = threading.Thread(target=worker)
        thread.start()
        assert blocked.wait(timeout=3), "the card surfaced; the guard is blocking"

        # The gateway interrupts this worker thread on SSE disconnect / stop.
        set_interrupt(True, thread.ident)
        try:
            thread.join(timeout=5)
            assert not thread.is_alive(), (
                "interrupt must release the wait, not park for the timeout"
            )
            assert result["choice"] is not None, (
                "interrupted wait fails closed (denial)"
            )
            assert (
                consume_tool_approval_completion_reason(SESSION, "call-1")
                == "cancelled"
            )
        finally:
            set_interrupt(False, thread.ident)


class TestFailClosedDenial:
    def test_should_deny_a_gated_write(self):
        result = fail_closed_denial(GATED)
        assert result is not None
        assert "approval" in result.lower()
        # A guard-error path: the user may well be present, so this must NOT
        # be the turn-ending status — the agent continues and reports it.
        assert json.loads(result)["status"] != "approval_no_response"

    def test_should_allow_an_ungated_read(self):
        assert fail_closed_denial(READ) is None

    def test_should_deny_when_classification_itself_raises(self, monkeypatch):
        def boom(_name):
            raise RuntimeError("cannot classify")

        monkeypatch.setattr(tool_approval, "is_gated_tool", boom)
        result = fail_closed_denial(GATED)
        assert result is not None
        assert json.loads(result)["status"] != "approval_no_response"


class TestClearSession:
    def test_should_drop_session_grants(self):
        resolve_tool_approval(SESSION, GATED, "session")
        clear_session(SESSION)
        assert is_tool_approved(SESSION, GATED) is False
