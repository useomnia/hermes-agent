"""Capture a compacted transcript without copying tool results or call arguments."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from agent.context_compressor import is_compaction_summary_message

MAX_RETAINED_REFERENCES = 512


def _visible_message(message: dict) -> dict:
    role = message.get("role")
    if role == "tool":
        return {"role": role, "tool_call_id": message.get("tool_call_id")}
    calls = message.get("tool_calls")
    if calls:
        return {"role": role, "tool_call_ids": [call.get("id") for call in calls]}
    return {"role": role, "content": message.get("content") or ""}


def capture_compaction(messages: list[dict], *, previous_count: int, session_id: str) -> dict | None:
    visible = [message for message in messages if message.get("role") in {"user", "assistant", "tool"}]
    summaries = [index for index, message in enumerate(visible) if is_compaction_summary_message(message)]
    if len(summaries) != 1 or len(visible) - 1 > MAX_RETAINED_REFERENCES:
        return None
    index = summaries[0]
    summary = visible[index]
    # A merged summary may carry multimodal content or calls. Its non-text
    # context cannot be represented by a single durable summary message.
    if not isinstance(summary.get("content"), str) or summary.get("tool_calls"):
        return None
    return {
        "summary": summary["content"],
        "summary_role": summary["role"],
        "retained_head_messages": [_visible_message(message) for message in visible[:index]],
        "retained_tail_messages": [_visible_message(message) for message in visible[index + 1:]],
        "compacted_messages": max(0, previous_count - (len(visible) - 1)),
        "session_id": session_id,
        "retained_tail_from": None,
    }


def _message_reference(message: dict, redact: Callable[[str], str]) -> dict | None:
    if "content" not in message:
        return dict(message)
    content = message["content"]
    # Multimodal histories need a durable attachment identity before they can
    # participate in text-only restore manifests.
    if not isinstance(content, str):
        return None
    canonical = redact(content).strip()
    return {"role": message["role"], "text_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def project_compaction(snapshot: dict, *, redact: Callable[[str], str], bound: Callable[[Any, int], str]) -> dict | None:
    head = [_message_reference(message, redact) for message in snapshot["retained_head_messages"]]
    tail = [_message_reference(message, redact) for message in snapshot["retained_tail_messages"]]
    if None in head or None in tail:
        return None
    summary = redact(snapshot["summary"])
    bounded = bound(summary, 64 * 1024)
    return {
        "summary": bounded,
        "summary_role": snapshot["summary_role"],
        "summary_truncated": bounded != summary,
        "retained_head": len(head),
        "retained_head_refs": head,
        "retained_tail_refs": tail,
        "retained_tail_from": snapshot["retained_tail_from"],
        "compacted_messages": snapshot["compacted_messages"],
        "session_id": snapshot["session_id"],
    }
