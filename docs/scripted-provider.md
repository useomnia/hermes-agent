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

Other response kinds are `tool_calls`, `http_error`, `connection_close`, and
`hold`; `hold` wraps one text, tool-call, or HTTP-error response and is released
through the authenticated control endpoint.

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
