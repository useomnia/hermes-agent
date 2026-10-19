"""Late answers to calls that are no longer the session's tail.

An Omnio question or approval whose wait expired stays open in SessionDB, and
the conversation may move on before the User answers it. The answer cannot be
stored as the call's tool row: rows replay in insertion order, so a result
written now would land after later messages instead of next to its call. The
assistant row that holds the call records the answer instead, and every
request built from history places it back as that call's result.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Key in an assistant row's display metadata: tool call id -> late close.
LATE_CLOSES_KEY = "_omnio_late_closes"


def late_close_results(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The tool results an assistant message's late answers stand for, in call order."""
    if message.get("role") != "assistant":
        return []
    metadata = message.get("display_metadata")
    closes = metadata.get(LATE_CLOSES_KEY) if isinstance(metadata, dict) else None
    if not isinstance(closes, dict) or not closes:
        return []
    results: List[Dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        call_id = call.get("id") if isinstance(call, dict) else None
        close = closes.get(call_id) if isinstance(call_id, str) else None
        if not isinstance(close, dict) or not isinstance(close.get("content"), str):
            continue
        results.append({
            "role": "tool",
            "name": close.get("tool_name") or "",
            "content": close["content"],
            "tool_call_id": call_id,
        })
    return results


# The result an approved late call stands for: the call itself never ran.
LATE_APPROVAL_RESULT = (
    '{"status": "approved_late", "note": "The user approved this call after its '
    'approval request timed out. It has not run. Call it again if it is still '
    'needed; the approval covers the same arguments."}'
)
