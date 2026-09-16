import json

import pytest

from agent.compaction_snapshot import capture_compaction, project_compaction
from gateway.turn_event_log import _bounded_utf8


def _summary(content="[CONTEXT COMPACTION] summary"):
    return {"role": "user", "content": content, "_compressed_summary": True}


def test_snapshot_should_omit_outputs_and_arguments():
    snapshot = capture_compaction(
        [
            {"role": "system", "content": "private system"},
            {"role": "user", "content": "head"},
            _summary(),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "terminal",
                            "arguments": "private arguments",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "private output"},
            {"role": "user", "content": "latest request"},
        ],
        previous_count=100,
        session_id="child",
    )
    event = project_compaction(snapshot, redact=lambda text: text, bound=_bounded_utf8)
    assert event["retained_head"] == 1
    assert event["retained_tail_refs"][0] == {
        "role": "assistant",
        "tool_call_ids": ["call-1"],
    }
    assert event["retained_tail_refs"][1] == {"role": "tool", "tool_call_id": "call-1"}
    assert event["compacted_messages"] == 96
    serialized = json.dumps(event)
    for forbidden in ("private", "latest request", "_compressed_summary", "arguments"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "no summary"}],
        [_summary(), _summary()],
        [_summary()] + [{"role": "user", "content": "tail"}] * 513,
    ],
)
def test_snapshot_should_skip_unresolvable_summary(messages):
    assert capture_compaction(messages, previous_count=1000, session_id="s") is None


def test_snapshot_should_preserve_merged_summary_without_duplicating_user_message():
    merged = "[CONTEXT COMPACTION] summary\n[summary ends]\nretained request"
    snapshot = capture_compaction(
        [_summary(merged), {"role": "assistant", "content": "tail"}],
        previous_count=10,
        session_id="s",
    )
    assert snapshot["summary"] == merged
    assert snapshot["retained_tail_messages"] == [
        {"role": "assistant", "content": "tail"}
    ]


def test_snapshot_should_mark_utf8_truncation_for_safe_restore_fallback():
    snapshot = capture_compaction(
        [_summary("[CONTEXT COMPACTION]" + "😀" * 20000)],
        previous_count=100,
        session_id="s",
    )
    event = project_compaction(snapshot, redact=lambda text: text, bound=_bounded_utf8)
    assert event["summary_truncated"] is True
    assert len(event["summary"].encode()) <= 65536


def test_snapshot_should_skip_multimodal_retained_content():
    snapshot = capture_compaction(
        [
            _summary(),
            {"role": "user", "content": [{"type": "image_url", "image_url": "image"}]},
        ],
        previous_count=10,
        session_id="s",
    )
    assert (
        project_compaction(snapshot, redact=lambda text: text, bound=_bounded_utf8)
        is None
    )
