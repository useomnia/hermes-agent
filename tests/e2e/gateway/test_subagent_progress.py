"""Subagent-progress probe: the ``hermes.tool.progress`` SSE channel.

Covers the gateway change that forwards a delegated child's activity onto the
chat-completions stream. When the agent runs a ``delegate_task`` batch, each
child's ``subagent.*`` events (relayed up while the parent blocks) are emitted
as a custom ``event: hermes.tool.progress`` — so a multi-minute delegation
isn't silent on the wire, and a client disconnect (the stop button) is noticed
promptly instead of only at the 30s SSE keepalive.

This also covers the live per-subagent trace: each child streams its reasoning
and its response up as ``subagent.reasoning`` / ``subagent.response`` events,
whose ``preview`` carries an incremental delta (the new text since the last
event, not the running cumulative string); the client appends them.

Like the reasoning probe, this validates *shape when present* and skips when the
backend produces no subagent activity: whether a model chooses to delegate for a
given prompt — and whether ``delegate_task`` is in the image's toolset — is
model- and config-dependent, so its absence is never a hard failure.
"""

from __future__ import annotations

import pytest

from .constants import MODEL
from .http_client import TOOL_PROGRESS_EVENT, tool_progress

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]

# Make delegation the obvious path: two independent trivial subtasks with an
# explicit nudge to fan them out. Whether the model actually calls delegate_task
# is up to it (and the image's toolset) — the probe skips if it doesn't.
_DELEGATE_PROMPT = (
    "Use your delegate_task tool to fan this out to two subagents running in "
    "parallel: one must reply with exactly the word ALPHA, the other with "
    "exactly the word BETA. Wait for both, then report the two words."
)

# Child-identity fields the gateway copies onto a subagent progress event,
# with the type each must have *if present* — the gateway only includes the
# ones the underlying event carried, so every check is presence-gated.
_INT_FIELDS = ("depth", "task_index", "task_count", "tool_count")
_STR_FIELDS = ("subagent_id", "parent_id", "goal")


def _collect_subagent_progress(gateway) -> list[dict]:
    """Stream the delegation prompt; return the ``subagent.*`` progress payloads.

    The progress channel legitimately also carries the parent's own structured
    tool lifecycle (``status: running``/``completed``), so we keep only the
    ``subagent``-prefixed events — the ones this patch put on the wire.
    """
    progress: list[dict] = []
    for event, data in gateway.stream_events(
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": _DELEGATE_PROMPT}]},
    ):
        if event != TOOL_PROGRESS_EVENT:
            continue
        payload = tool_progress(data)
        assert payload is not None, (
            f"non-JSON {TOOL_PROGRESS_EVENT} payload: {data[:200]!r}"
        )
        if str(payload.get("status", "")).startswith("subagent"):
            progress.append(payload)
    return progress


def test_subagent_progress_shape(gateway):
    progress = _collect_subagent_progress(gateway)
    if not progress:
        pytest.skip(
            f"{gateway.provider.id} streamed no subagent.* progress for this prompt "
            "(model didn't delegate, or delegate_task isn't in the image's toolset)"
        )

    for ev in progress:
        status = ev.get("status")
        assert isinstance(status, str) and status.startswith("subagent"), (
            f"subagent progress event needs a 'subagent.*' status, got {status!r}"
        )
        # Always set by the gateway, even when the underlying event names no
        # tool (then it's None) — assert presence, not truthiness.
        assert "tool" in ev, f"progress event missing 'tool' key: {ev}"
        # preview is always set too, but is None for events that carry no text
        # (e.g. subagent.start); when present it's the streamed string.
        if ev.get("preview") is not None:
            assert isinstance(ev["preview"], str), (
                f"preview must be a str when set, got {ev['preview']!r}"
            )
        for key in _INT_FIELDS:
            if key in ev:
                assert isinstance(ev[key], int) and not isinstance(ev[key], bool), (
                    f"{key} must be an int, got {ev[key]!r}"
                )
        for key in _STR_FIELDS:
            if key in ev:
                assert isinstance(ev[key], str), f"{key} must be a str, got {ev[key]!r}"


def test_subagent_reasoning_and_response_streamed(gateway):
    """Each subagent streams its reasoning / response up as a live trace.

    ``subagent.reasoning`` and ``subagent.response`` carry the child's text as
    incremental deltas in ``preview`` (new chars since the last event, not the
    cumulative string). Like the rest of this probe it's presence-gated: a model
    that delegates but emits no reasoning/response stream (or no delegation at
    all) skips rather than fails.
    """
    progress = _collect_subagent_progress(gateway)
    if not progress:
        pytest.skip(
            f"{gateway.provider.id} streamed no subagent.* progress for this prompt "
            "(model didn't delegate, or delegate_task isn't in the image's toolset)"
        )

    streamed = [
        ev for ev in progress if ev.get("status") in ("subagent.reasoning", "subagent.response")
    ]
    if not streamed:
        pytest.skip(
            f"{gateway.provider.id} delegated but streamed no subagent.reasoning/"
            "subagent.response events (model emitted no reasoning/response stream)"
        )

    for ev in streamed:
        preview = ev.get("preview")
        assert isinstance(preview, str) and preview.strip(), (
            f"{ev['status']} must carry non-empty preview text, got {preview!r}"
        )
        # The trace is keyed per child by subagent_id — present whenever the
        # delegation assigned one (presence-gated, like the other identity fields).
        if "subagent_id" in ev:
            assert isinstance(ev["subagent_id"], str), (
                f"subagent_id must be a str, got {ev['subagent_id']!r}"
            )
