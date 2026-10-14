"""Behavioral tests for the lightweight durable Omnia approval state."""

import pytest

import tools.omnio_approval_state as approval_state


NATIVE = "mcp__connectors__GMAIL_SEND_EMAIL"
LEGACY = "mcp_connectors_GMAIL_SEND_EMAIL"


@pytest.fixture(autouse=True)
def _clean_state():
    approval_state._always_approved.clear()
    approval_state._injected_always_approved.clear()
    approval_state._injected_always_approved_slugs.clear()
    yield
    approval_state._always_approved.clear()
    approval_state._injected_always_approved.clear()
    approval_state._injected_always_approved_slugs.clear()


def test_snapshot_replacement_clears_local_grants_and_derives_slugs():
    approval_state.record_always_approval(LEGACY)

    approval_state.replace_injected_always_approvals([LEGACY, "terminal"])

    assert approval_state._always_approved == set()
    assert approval_state._injected_always_approved == {LEGACY}
    assert approval_state._injected_always_approved_slugs == {"GMAIL_SEND_EMAIL"}
    assert approval_state.is_always_approved(NATIVE) is True


def test_snapshot_slug_grants_without_asking_omnia_again():
    approval_state.replace_injected_always_approvals(
        [], tool_slugs=["GMAIL_SEND_EMAIL"]
    )

    assert approval_state.is_always_approved(NATIVE) is True
    assert approval_state.is_always_approved(LEGACY) is True


def test_local_in_chat_grant_counts_until_the_next_snapshot():
    approval_state.record_always_approval(NATIVE)
    assert approval_state.is_always_approved(NATIVE) is True

    approval_state.replace_injected_always_approvals([])

    assert approval_state.is_always_approved(NATIVE) is False


def test_tool_without_any_grant_is_not_approved():
    approval_state.replace_injected_always_approvals([LEGACY])

    assert approval_state.is_always_approved("mcp__connectors__SLACK_SEND") is False
