"""Behavioral tests for the lightweight durable Omnia approval state."""

import pytest

import tools.omnio_approval_state as approval_state


NATIVE = "mcp__connectors__GMAIL_SEND_EMAIL"
LEGACY = "mcp_connectors_GMAIL_SEND_EMAIL"


@pytest.fixture(autouse=True)
def _clean_state():
    approval_state.register_always_approval_authority(None)
    approval_state._always_approved.clear()
    approval_state._injected_always_approved.clear()
    approval_state._injected_always_approved_slugs.clear()
    yield
    approval_state.register_always_approval_authority(None)
    approval_state._always_approved.clear()
    approval_state._injected_always_approved.clear()
    approval_state._injected_always_approved_slugs.clear()


def test_snapshot_replacement_clears_local_bridge_and_derives_slugs():
    approval_state.register_always_approval_authority(lambda _tool: True)
    approval_state.record_always_approval(LEGACY)

    approval_state.replace_injected_always_approvals([LEGACY, "terminal"])

    assert approval_state._always_approved == set()
    assert approval_state._injected_always_approved == {LEGACY}
    assert approval_state._injected_always_approved_slugs == {"GMAIL_SEND_EMAIL"}
    assert approval_state.is_always_approved(NATIVE) is True


def test_snapshot_slug_matching_still_requires_fresh_authority():
    approval_state.replace_injected_always_approvals(
        [], tool_slugs=["GMAIL_SEND_EMAIL"]
    )
    decisions = iter([True, False])
    approval_state.register_always_approval_authority(lambda _tool: next(decisions))

    assert approval_state.is_always_approved(NATIVE) is True
    assert approval_state.is_always_approved(NATIVE) is False


@pytest.mark.parametrize(
    "authority",
    [None, lambda _tool: False, lambda _tool: (_ for _ in ()).throw(RuntimeError())],
)
def test_missing_or_failed_authority_fails_closed(authority):
    approval_state.replace_injected_always_approvals([NATIVE])
    approval_state.register_always_approval_authority(authority)

    assert approval_state.is_always_approved(NATIVE) is False
