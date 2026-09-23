# Scripted provider testkit

`hermes_testkit.scripted_provider` is a dependency-free, deterministic
OpenAI-compatible HTTP provider used by Hermes and Omnio conformance runs. It
is packaged deliberately because a Sprite can launch it with the same Python
environment as the agent:

```bash
python -m hermes_testkit.scripted_provider \
  --script ./fixture.json \
  --control-token "$CONTROL_TOKEN"
```

## Version 1 contract

A script contains `schema_version: 1`, an optional model, and ordered `steps`.
Text responses use `text` for backwards compatibility. They may instead add
`chunks: [string, ...]` (or provide both with matching concatenation) to
prescribe exact SSE delta boundaries. Empty and Unicode strings are valid;
non-streaming content, usage, and held text use the concatenated value. Chunks
are rejected on tool-call, error, and connection-close responses.

Text and tool-call responses may provide an optional strict `usage` override.
It is emitted on the non-streaming response and on the final streamed chunk;
held text/tool-call responses preserve it until release:

```json
{
  "response": {
    "type": "text",
    "text": "fixture response",
    "usage": {
      "prompt_tokens": 17,
      "completion_tokens": 5,
      "total_tokens": 22
    }
  }
}
```

All three fields are required non-negative JSON integers and
`total_tokens` must equal `prompt_tokens + completion_tokens`. Unknown or
missing fields, booleans, floats, negative values, and `null` are rejected.
Usage is rejected on HTTP-error and connection-close responses because those
responses never emit a completion usage object. Omitting `usage` retains the
legacy deterministic token derivation exactly.

Scripts may also provide optional per-model pricing metadata. The map keys must
be model IDs listed by `models` (or the primary `model` when `models` is
omitted), and pricing values are exact, finite, non-negative per-token decimal
strings:

```json
{
  "schema_version": 1,
  "model": "omnio-conformance-scripted",
  "model_metadata": {
    "omnio-conformance-scripted": {
      "pricing": {
        "prompt": "0.000001",
        "completion": "0.000002",
        "cache_read": "0",
        "cache_write": "0"
      }
    }
  },
  "steps": []
}
```

The supported pricing fields are `prompt`, `completion`, `request`,
`cache_read`, and `cache_write`; all are optional per model but a `pricing`
object must contain at least one field. Unknown fields, JSON numbers, negative
values, non-finite values, and metadata for an unlisted model are rejected.
When present, each model's metadata is merged directly into its item in
`GET /v1/models`, so generic OpenAI-compatible clients (including Hermes) can
discover and price the model. Omitting `model_metadata` preserves the original
model-list response exactly.

Other response kinds are `tool_calls`, `http_error`, `connection_close`, and
`hold`; `hold` wraps one text, tool-call, or HTTP-error response and is released
through the authenticated control endpoint.

When independent requests can race, one top-level step may be an unordered
group:

```json
{
  "unordered": [
    {
      "request": { "model": "parent-model" },
      "response": { "type": "text", "text": "parent complete" }
    },
    {
      "request": { "model": "child-model" },
      "response": { "type": "text", "text": "child complete" }
    }
  ]
}
```

Every branch must have an explicit request predicate. Remaining branches may
arrive in any order and each is consumed exactly once. A request matching no
remaining branch or more than one branch fails with `409` and consumes
nothing. The next top-level step is unavailable until every branch is
consumed. Holds remain pending independently, so a held branch does not block
a sibling request; `complete` still waits for every hold to settle. Arm and
reset clear all unordered progress. Omitting unordered groups preserves the
original ordered-step behavior and state shape.

The provider recognizes Hermes' deterministic local-server capability probes:
`POST /api/show`, `GET /api/v1/models`, `/api/tags`, `/v1/props`, `/props`,
`/version`, and `/v1/models/<model>`. They intentionally return the same 404
probe-down response as an unsupported local server, but do not consume scripted
chat steps or enter `unexpected_requests`; other endpoints remain 404 and are
recorded there.

## Security and lifecycle

The server binds to loopback by default. A non-loopback bind is refused unless
an inference API key is configured (`--api-key` or
`HERMES_SCRIPTED_PROVIDER_API_KEY`); inference requests accept Bearer or
`X-API-Key`. Control endpoints remain loopback-only and require the control
token. Captured headers omit authorization, API keys, cookies, and other
credential-bearing fields, and request bodies are never logged.

Use explicit JSON objects for control reset/release calls. `{}` means reset
with defaults or release all pending holds; an empty, malformed, or non-object
body is rejected. The state endpoint reports `consumed` separately from
`complete`; `complete` remains false while a hold is pending.
