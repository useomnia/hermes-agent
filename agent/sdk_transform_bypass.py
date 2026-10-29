"""Route bulk request payloads around the OpenAI SDK's request transform (#93650).

``responses.create`` and ``chat.completions.create`` both re-walk the whole request
body against their TypedDict/union param graph client-side, with the GIL held,
before any byte leaves the process. #93650 documents that walk wedging for 12+
hours on a ~1.4 MB conversation — starving the TTFB/stale watchdogs whose job is
to rescue this exact call; the hang is pre-network, so no socket kill helps.

Hermes assembles these payloads from JSON round-trips, so they are already wire
format and the walk has nothing to convert. The SDK merges ``extra_body`` into
the JSON body *after* the transform (``_base_client._build_request``), so moving
the bulk fields there skips the walk and yields the same request bytes.
"""

from __future__ import annotations

from typing import Any

# Bulk request fields carrying the conversation payload, per API family. Everything
# else is scalar configuration the SDK transform handles in microseconds.
RESPONSES_BYPASS_FIELDS = ("input", "tools")
CHAT_COMPLETIONS_BYPASS_FIELDS = ("messages", "tools")


def _is_plain_json_data(value: Any) -> bool:
    """True when ``value`` is purely JSON wire types; pydantic models / generators must keep the typed SDK path."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_plain_json_data(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_is_plain_json_data(item) for item in value)
    return False


def bypass_sdk_request_transform(
    request_kwargs: dict,
    fields: tuple[str, ...] = RESPONSES_BYPASS_FIELDS,
    *,
    keep_slots: bool = False,
) -> dict:
    """Move wire-format bulk ``fields`` into ``extra_body``.

    Returns ``request_kwargs`` itself when nothing is safe to move, so callers use
    the result unconditionally. ``keep_slots`` leaves an empty-list placeholder in
    the typed kwargs for each moved field: it satisfies ``@required_args``
    (``messages`` on chat.completions) and keeps the field's position in the JSON
    body, so the bytes — and therefore any byte-keyed prompt cache — are unchanged.
    """
    if request_kwargs.get("extra_body") is not None and not isinstance(request_kwargs["extra_body"], dict):
        return request_kwargs
    moved = {f: request_kwargs[f] for f in fields
             if isinstance(request_kwargs.get(f), (dict, list)) and _is_plain_json_data(request_kwargs[f])}
    if not moved:
        return request_kwargs
    bypassed = {key: ([] if keep_slots else value) if key in moved else value
                for key, value in request_kwargs.items() if keep_slots or key not in moved}
    extra_body = bypassed.get("extra_body")
    merged = dict(extra_body) if isinstance(extra_body, dict) else {}
    # An explicit caller-provided extra_body entry keeps precedence (SDK post-transform merge).
    bypassed["extra_body"] = {**merged, **{f: v for f, v in moved.items() if f not in merged}}
    return bypassed


def bypass_chat_sdk_request_transform(request_kwargs: dict, client: Any) -> dict:
    """Chat-completions bypass, gated on the real OpenAI SDK.

    Only the SDK performs the transform and only the SDK merges ``extra_body``
    afterwards. Hermes also drives chat-shaped facades that are NOT the SDK (the
    in-process MoA aggregator, test stand-ins); handing those an ``extra_body`` they
    never merge would silently send an empty conversation.
    """
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is None or not type(completions).__module__.startswith("openai."):
        return request_kwargs
    return bypass_sdk_request_transform(request_kwargs, CHAT_COMPLETIONS_BYPASS_FIELDS, keep_slots=True)


def bypass_responses_sdk_request_transform(request_kwargs: dict, client: Any) -> dict:
    """Preserve field positions and non-SDK facades on the Responses path too."""
    responses = getattr(client, "responses", None)
    if responses is None or not type(responses).__module__.startswith("openai."):
        return request_kwargs
    return bypass_sdk_request_transform(request_kwargs, keep_slots=True)
