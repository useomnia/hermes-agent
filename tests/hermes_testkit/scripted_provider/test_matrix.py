"""Requirements-derived behavior matrix for the scripted provider.

This is deliberately an HTTP-level suite.  The same requests are made by a
Hermes process in a Sprite, so testing only helper functions would miss wire
compatibility, authentication, EOF behavior, and control-plane races.
"""

from __future__ import annotations

import json
import http.client
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from hermes_testkit.scripted_provider import (
    SCRIPT_SCHEMA_VERSION,
    ResponseStep,
    Script,
    ScriptValidationError,
    ScriptedProviderServer,
    matches_request,
    parse_script,
)
from hermes_testkit.scripted_provider.cli import main as scripted_provider_main


@pytest.fixture(autouse=True)
def _clear_inference_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default-auth tests deterministic; env opt-in is tested explicitly."""

    monkeypatch.delenv("HERMES_SCRIPTED_PROVIDER_API_KEY", raising=False)


def _request(
    server: ScriptedProviderServer,
    method: str,
    path: str,
    payload: Any | None = None,
    *,
    token: str | None = None,
    api_key: str | None = None,
    api_key_header: str = "Authorization",
    timeout: float = 2,
) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
    headers: dict[str, str] = {}
    data = None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if api_key is not None:
        if api_key_header.lower() == "authorization":
            headers[api_key_header] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        server.url + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            decoded: dict[str, Any] | str
            if "json" in content_type:
                decoded = json.loads(raw)
            else:
                decoded = raw.decode()
            return response.status, decoded, dict(response.headers.items())
    except urllib.error.HTTPError as error:
        raw = error.read()
        return error.code, json.loads(raw), dict(error.headers.items())


def _chat(
    server: ScriptedProviderServer,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    api_key_header: str = "Authorization",
    timeout: float = 2,
) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
    return _request(
        server,
        "POST",
        "/v1/chat/completions",
        payload,
        api_key=api_key,
        api_key_header=api_key_header,
        timeout=timeout,
    )


def _sse_payloads(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _script(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "model": "matrix-model",
        "steps": list(steps),
    }


def _text(text: str, *, request: dict[str, Any] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"response": {"type": "text", "text": text}}
    if request is not None:
        item["request"] = request
    return item


def _tool(
    name: str = "echo", arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "response": {
            "type": "tool_calls",
            "tool_calls": [{"name": name, "arguments": arguments or {}}],
        }
    }


def test_health_models_and_stable_metadata() -> None:
    with ScriptedProviderServer(
        _script(_text("ok")), control_token="matrix-token"
    ) as server:
        status, health, _ = _request(server, "GET", "/healthz")
        assert status == 200
        assert health == {"status": "ok", "schema_version": SCRIPT_SCHEMA_VERSION}
        status, models, _ = _request(server, "GET", "/v1/models")
        assert status == 200
        assert models["object"] == "list"
        model = models["data"][0]
        assert model["id"] == "matrix-model"
        assert model["created"] == 1_700_000_000
        assert model["owned_by"] == "hermes-testkit"


def test_models_emit_optional_per_model_pricing_without_consuming_script() -> None:
    script = _script(_text("ok"))
    script["models"] = ["matrix-model", "second-model"]
    script["model_metadata"] = {
        "matrix-model": {
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "request": "0",
            }
        }
    }
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        status, models, _ = _request(server, "GET", "/v1/models")
        assert status == 200
        assert models["data"] == [
            {
                "id": "matrix-model",
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "hermes-testkit",
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                    "request": "0",
                },
            },
            {
                "id": "second-model",
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "hermes-testkit",
            },
        ]
        assert server.state["step_index"] == 0
        assert server.state["request_count"] == 0
        # Discovery is safe to retry: repeated listing is identical and does
        # not consume the first scripted chat response.
        status, repeated, _ = _request(server, "GET", "/v1/models")
        assert status == 200
        assert repeated == models
        assert server.state["step_index"] == 0
        assert server.state["request_count"] == 0


def test_hermes_discovers_scripted_pricing_and_estimates_exact_cost() -> None:
    from agent.model_metadata import fetch_endpoint_model_metadata
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    script = _script(_text("priced"))
    script["model_metadata"] = {
        "matrix-model": {
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
            }
        }
    }
    with ScriptedProviderServer(
        script,
        control_token="matrix-token",
        api_key="inference-secret",
    ) as server:
        discovered = fetch_endpoint_model_metadata(
            server.base_url,
            api_key="inference-secret",
            force_refresh=True,
        )
        assert discovered["matrix-model"]["pricing"] == {
            "prompt": "0.000001",
            "completion": "0.000002",
        }
        cost = estimate_usage_cost(
            "matrix-model",
            CanonicalUsage(input_tokens=1_000, output_tokens=500),
            provider="custom",
            base_url=server.base_url,
            api_key="inference-secret",
        )
        assert cost.amount_usd == Decimal("0.002000")
        assert cost.amount_usd is not None
        assert cost.amount_usd > 0
        assert cost.source == "provider_models_api"
        assert server.state["step_index"] == 0
        assert server.state["request_count"] == 0


def test_known_model_capability_probes_are_probe_down_and_not_unexpected() -> None:
    with ScriptedProviderServer(
        _script(_text("chat")), control_token="matrix-token"
    ) as server:
        probes = [
            ("POST", "/api/show", {"name": "matrix-model"}),
            ("GET", "/api/v1/models", None),
            ("GET", "/api/tags", None),
            ("GET", "/v1/props", None),
            ("GET", "/props", None),
            ("GET", "/version", None),
            ("GET", "/v1/models/matrix-model", None),
            ("GET", "/v1/models/matrix-model", None),
        ]
        for method, path, payload in probes:
            status, body, _ = _request(server, method, path, payload)
            assert status == 404
            assert isinstance(body, dict)
            assert body["error"]["message"] == "not found"

        assert server.state["step_index"] == 0
        assert server.state["request_count"] == 0
        assert server.state["unexpected_requests"] == []

        status, _, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 200
        assert server.state["step_index"] == 1
        assert server.state["request_count"] == 1
        assert server.state["unexpected_requests"] == []

        status, _, _ = _request(server, "GET", "/genuinely-unknown")
        assert status == 404
        assert server.state["unexpected_requests"][-1]["path"] == "/genuinely-unknown"
        status, _, _ = _request(server, "POST", "/genuinely-unknown")
        assert status == 404
        assert server.state["unexpected_requests"][-1]["method"] == "POST"


def test_capability_probes_honor_inference_auth_without_consuming_steps() -> None:
    script = _script(_text("auth"))
    script["model_metadata"] = {"matrix-model": {"pricing": {"prompt": "0.000001"}}}
    with ScriptedProviderServer(
        script,
        control_token="matrix-token",
        api_key="probe-key",
    ) as server:
        status, _, _ = _request(server, "GET", "/api/tags")
        assert status == 401
        status, _, _ = _request(server, "POST", "/api/show", {"name": "matrix-model"})
        assert status == 401
        status, body, _ = _request(server, "GET", "/v1/models")
        assert status == 401
        assert "pricing" not in json.dumps(body)
        assert server.state["step_index"] == 0
        assert server.state["request_count"] == 0
        assert server.state["unexpected_requests"] == []


def test_inference_api_key_auth_is_optional_and_redacted_from_captures() -> None:
    script = _script(_text("authenticated"))
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        status, _, _ = _request(server, "GET", "/v1/models")
        assert status == 200
        status, body, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "authenticated"

    with ScriptedProviderServer(
        _script(_text("authenticated")),
        control_token="matrix-token",
        api_key="inference-secret",
    ) as server:
        status, body, _ = _request(server, "GET", "/v1/models")
        assert status == 401
        assert body["error"]["message"] == "inference authentication required"
        assert server.state["step_index"] == 0

        status, _, _ = _request(server, "GET", "/v1/models", api_key="wrong-secret")
        assert status == 401
        assert server.state["step_index"] == 0

        status, _, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 401
        status, _, _ = _chat(
            server,
            {"model": "matrix-model", "messages": []},
            api_key="wrong-secret",
        )
        assert status == 401
        assert server.state["step_index"] == 0

        status, models, _ = _request(
            server, "GET", "/v1/models", api_key="inference-secret"
        )
        assert status == 200
        assert models["data"][0]["id"] == "matrix-model"

        status, body, _ = _chat(
            server,
            {"model": "matrix-model", "messages": []},
            api_key="inference-secret",
            api_key_header="X-API-Key",
        )
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "authenticated"
        capture = server.requests[0]
        assert "authorization" not in capture.get("headers", {})
        assert "x-api-key" not in capture.get("headers", {})


def test_inference_api_key_can_be_configured_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_SCRIPTED_PROVIDER_API_KEY", "env-inference-secret")
    with ScriptedProviderServer(
        _script(_text("environment-auth")), control_token="matrix-token"
    ) as server:
        status, _, _ = _request(server, "GET", "/v1/models")
        assert status == 401
        status, _, _ = _chat(
            server,
            {"model": "matrix-model", "messages": []},
            api_key="env-inference-secret",
        )
        assert status == 200


def test_non_loopback_bind_requires_inference_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_SCRIPTED_PROVIDER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="non-loopback"):
        ScriptedProviderServer(_script(_text("public")), host="0.0.0.0")
    assert scripted_provider_main(["--host", "0.0.0.0", "--quiet"]) == 2

    server = ScriptedProviderServer(
        _script(_text("public")), host="0.0.0.0", api_key="inference-secret"
    )
    assert server.api_key == "inference-secret"


def test_text_non_streaming_has_openai_shape_and_usage() -> None:
    with ScriptedProviderServer(
        _script(_text("hello")), control_token="matrix-token"
    ) as server:
        status, body, _ = _chat(
            server,
            {"model": "matrix-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 200
        assert body["id"] == "scripted-completion-000001"
        assert body["created"] == 1_700_000_000
        assert body["choices"][0]["message"] == {
            "role": "assistant",
            "content": "hello",
        }
        assert (
            body["usage"]["total_tokens"]
            == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
        )


def test_explicit_text_chunks_concatenate_for_non_streaming_and_usage() -> None:
    chunks = ["first", " ", "second"]
    with ScriptedProviderServer(
        _script({"response": {"type": "text", "chunks": chunks}}),
        control_token="matrix-token",
    ) as server:
        status, body, _ = _chat(
            server, {"model": "matrix-model", "messages": [{"role": "user"}]}
        )
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "".join(chunks)
        assert body["usage"]["completion_tokens"] == max(
            1, (len("".join(chunks)) + 3) // 4
        )


def test_explicit_empty_chunks_keep_empty_non_streaming_content() -> None:
    with ScriptedProviderServer(
        _script({"response": {"type": "text", "chunks": [""]}}),
        control_token="matrix-token",
    ) as server:
        status, body, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 200
        assert body["choices"][0]["message"]["content"] == ""


def test_text_streaming_is_sse_and_terminates_with_done() -> None:
    with ScriptedProviderServer(
        _script(_text("hello world")), control_token="matrix-token"
    ) as server:
        status, body, headers = _chat(
            server, {"model": "matrix-model", "stream": True, "messages": []}
        )
        assert status == 200
        assert "text/event-stream" in headers["Content-Type"]
        assert isinstance(body, str)
        assert "chat.completion.chunk" in body
        assert '"content":"hello' in body
        assert '"content":"hello ' in body
        assert '"content":"world' in body
        assert '"finish_reason":"stop"' in body
        assert body.endswith("data: [DONE]\n\n")


def test_explicit_text_chunks_drive_sse_order_and_preserve_empty_unicode() -> None:
    chunks = ["α", "", "🙂", " tail"]
    with ScriptedProviderServer(
        _script({"response": {"type": "text", "chunks": chunks}}),
        control_token="matrix-token",
    ) as server:
        status, body, _ = _chat(
            server, {"model": "matrix-model", "stream": True, "messages": []}
        )
        assert status == 200
        assert isinstance(body, str)
        payloads = _sse_payloads(body)
        content_deltas = [
            payload["choices"][0]["delta"]["content"]
            for payload in payloads
            if "content" in payload["choices"][0]["delta"]
        ]
        assert content_deltas == chunks
        assert body.endswith("data: [DONE]\n\n")


def test_text_without_explicit_chunks_keeps_legacy_splitter_boundaries() -> None:
    with ScriptedProviderServer(
        _script(_text("hello world")), control_token="matrix-token"
    ) as server:
        status, body, _ = _chat(
            server, {"model": "matrix-model", "stream": True, "messages": []}
        )
        assert status == 200
        assert isinstance(body, str)
        payloads = _sse_payloads(body)
        content_deltas = [
            payload["choices"][0]["delta"]["content"]
            for payload in payloads
            if "content" in payload["choices"][0]["delta"]
        ]
        assert content_deltas == ["hello ", "world"]


def test_stream_headers_are_suppressed_when_reset_wins_header_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reset racing the header write must not leak a stale HTTP response."""

    with ScriptedProviderServer(
        _script(_text("stale")), control_token="matrix-token"
    ) as server:
        entered = threading.Event()
        proceed = threading.Event()
        outcome: list[tuple[str, Any]] = []
        original_run_if_current = server._state.run_if_current  # noqa: SLF001
        first_call = True

        def gated_run_if_current(generation: int, action: Any) -> bool:
            nonlocal first_call
            if first_call:
                first_call = False
                entered.set()
                assert proceed.wait(2)
            return original_run_if_current(generation, action)

        monkeypatch.setattr(
            server._state,  # noqa: SLF001
            "run_if_current",
            gated_run_if_current,
        )

        def request() -> None:
            try:
                outcome.append((
                    "response",
                    _chat(
                        server,
                        {"model": "matrix-model", "stream": True, "messages": []},
                        timeout=3,
                    ),
                ))
            except Exception as exc:  # transport EOF is expected on cancellation
                outcome.append(("error", exc))

        worker = threading.Thread(target=request)
        worker.start()
        try:
            assert entered.wait(2)
            generation = server.state["generation"]
            server.reset()
            assert server.state["generation"] == generation + 1
        finally:
            proceed.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert outcome and outcome[0][0] == "error"


@pytest.mark.parametrize("stream", [False, True])
def test_tool_calls_are_openai_compatible_for_both_transports(stream: bool) -> None:
    with ScriptedProviderServer(
        _script(_tool("lookup", {"id": 7})), control_token="matrix-token"
    ) as server:
        status, body, _ = _chat(
            server, {"model": "matrix-model", "stream": stream, "messages": []}
        )
        assert status == 200
        if stream:
            assert isinstance(body, str)
            assert '"name":"lookup"' in body
            assert '"arguments":"{\\"id\\":7}"' in body
            assert '"finish_reason":"tool_calls"' in body
        else:
            call = body["choices"][0]["message"]["tool_calls"][0]
            assert call["type"] == "function"
            assert call["function"] == {"name": "lookup", "arguments": '{"id":7}'}
            assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_http_error_step_is_returned_without_provider_fallback() -> None:
    step = {
        "response": {
            "type": "http_error",
            "status": 429,
            "error": {"code": "rate_limit", "message": "try later"},
        }
    }
    with ScriptedProviderServer(_script(step), control_token="matrix-token") as server:
        status, body, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 429
        assert body["error"]["message"] == "try later"
        assert body["error"]["code"] == "rate_limit"
        assert server.state["complete"] is True


@pytest.mark.parametrize(
    "response",
    [
        {"type": "connection_close"},
        {"type": "connection_close", "before_headers": False, "text": "partial"},
    ],
)
def test_connection_close_steps_fail_the_client(response: dict[str, Any]) -> None:
    with ScriptedProviderServer(
        _script({"response": response}), control_token="matrix-token"
    ) as server:
        request = urllib.request.Request(
            server.url + "/v1/chat/completions",
            data=json.dumps({
                "model": "matrix-model",
                "stream": True,
                "messages": [],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            body = urllib.request.urlopen(request, timeout=2).read()
        except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionError):
            # Closing before headers is surfaced as a transport failure.
            return
        # Closing after headers is also a valid scripted failure; clients may
        # expose the partial bytes without raising until they parse the body.
        assert b"[DONE]" not in body


def test_held_response_blocks_until_authenticated_release() -> None:
    script = _script({
        "response": {
            "type": "hold",
            "id": "hold-a",
            "response": {"type": "text", "text": "released"},
        }
    })
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        result: list[tuple[int, dict[str, Any] | str, dict[str, str]]] = []

        def request() -> None:
            result.append(
                _chat(server, {"model": "matrix-model", "messages": []}, timeout=3)
            )

        worker = threading.Thread(target=request)
        worker.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not server.state["held"]:
            time.sleep(0.01)
        assert server.state["held"][0]["id"] == "hold-a"
        assert server.state["consumed"] is True
        assert server.state["complete"] is False
        status, _, _ = _request(
            server, "POST", "/__control/release", {"id": "hold-a"}, token="wrong"
        )
        assert status == 401
        assert worker.is_alive()
        status, release, _ = _request(
            server, "POST", "/__control/release", {"id": "hold-a"}, token="matrix-token"
        )
        assert status == 200
        assert release["released"] == ["hold-a"]
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result[0][0] == 200
        assert server.state["complete"] is True


def test_held_text_chunks_are_emitted_after_release_in_script_order() -> None:
    chunks = ["held", " ", "🙂"]
    script = _script({
        "response": {
            "type": "hold",
            "id": "chunk-hold",
            "response": {"type": "text", "chunks": chunks},
        }
    })
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        result: list[tuple[int, dict[str, Any] | str, dict[str, str]]] = []

        def request() -> None:
            result.append(
                _chat(
                    server,
                    {"model": "matrix-model", "stream": True, "messages": []},
                    timeout=3,
                )
            )

        worker = threading.Thread(target=request)
        worker.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not server.state["held"]:
            time.sleep(0.01)
        assert server.state["held"][0]["id"] == "chunk-hold"
        status, _, _ = _request(
            server,
            "POST",
            "/__control/release",
            {"id": "chunk-hold"},
            token="matrix-token",
        )
        assert status == 200
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result[0][0] == 200
        body = result[0][1]
        assert isinstance(body, str)
        payloads = _sse_payloads(body)
        assert [
            payload["choices"][0]["delta"]["content"]
            for payload in payloads
            if "content" in payload["choices"][0]["delta"]
        ] == chunks


def test_control_auth_state_arm_reset_and_strict_unexpected_requests() -> None:
    with ScriptedProviderServer(control_token="matrix-token") as server:
        status, _, _ = _request(server, "GET", "/__control/state")
        assert status == 401
        status, _, _ = _request(server, "GET", "/__control/state", token="wrong")
        assert status == 401
        status, state, _ = _request(
            server, "GET", "/__control/state", token="matrix-token"
        )
        assert status == 200
        assert state["armed"] is False
        status, body, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 409
        assert "no script" in body["error"]["message"]
        assert server.state["unexpected_requests"]

        status, armed, _ = _request(
            server,
            "POST",
            "/__control/arm",
            _script(_text("armed")),
            token="matrix-token",
        )
        assert status == 200
        assert armed["armed"] is True
        status, _, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 200
        status, reset, _ = _request(
            server, "POST", "/__control/reset", {}, token="matrix-token"
        )
        assert status == 200
        assert reset["step_index"] == 0
        status, _, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 200


def test_control_reset_and_release_require_explicit_object_bodies() -> None:
    with ScriptedProviderServer(
        _script(_text("control")), control_token="matrix-token"
    ) as server:
        for path in ("/__control/reset", "/__control/release"):
            status, body, _ = _request(server, "POST", path, token="matrix-token")
            assert status == 400
            assert "empty" in body["error"]["message"]

            status, body, _ = _request(server, "POST", path, [], token="matrix-token")
            assert status == 400
            assert "object" in body["error"]["message"]

            status, body, _ = _request(
                server, "POST", path, "not-an-object", token="matrix-token"
            )
            assert status == 400
            assert "object" in body["error"]["message"]

        status, reset, _ = _request(
            server, "POST", "/__control/reset", {}, token="matrix-token"
        )
        assert status == 200
        assert reset["step_index"] == 0
        status, release, _ = _request(
            server, "POST", "/__control/release", {}, token="matrix-token"
        )
        assert status == 200
        assert release["released"] == []


@pytest.mark.parametrize(
    "request_match",
    [
        {"messages": [{"role": "user"}]},
        {"json": {"messages": [{"role": "user"}]}},
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {"messages": [{"role": "user"}]},
        },
    ],
)
def test_request_matching_capture_is_deterministic_and_secrets_are_not_headers(
    request_match: dict[str, Any],
) -> None:
    script = _script(_text("matched", request=request_match))
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        status, _, _ = _chat(
            server,
            {
                "model": "matrix-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert status == 200
        request = server.requests[0]
        assert request["id"] == "request-g000001-s000001"
        assert request["generation"] == 1
        assert request["timestamp"] == 1_700_000_001
        assert "authorization" not in request.get("headers", {})
        assert request["matched"] is True
        assert server.requests == server.state["requests"]


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ({"enabled": True}, {"enabled": 1}),
        ({"enabled": 1}, {"enabled": True}),
        ({"count": False}, {"count": 0}),
        ({"count": 0}, {"count": False}),
        (
            {"nested": [{"enabled": True}, {"count": 0}]},
            {"nested": [{"enabled": 1}, {"count": False}]},
        ),
    ],
)
def test_request_matching_is_type_sensitive_for_json_booleans_and_numbers(
    expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    assert matches_request(expected, {"json": actual}) is False


def test_request_matching_keeps_equal_json_types_equal() -> None:
    assert (
        matches_request(
            {"nested": [{"enabled": True, "count": 1}]},
            {"json": {"nested": [{"enabled": True, "count": 1.0}]}},
        )
        is True
    )


def test_schema_is_versioned_and_rejects_unknown_response_or_version() -> None:
    with pytest.raises(ScriptValidationError, match="schema_version"):
        parse_script({"schema_version": 99, "steps": []})
    with pytest.raises(ScriptValidationError, match="unsupported"):
        parse_script({"schema_version": 1, "steps": [{"response": {"type": "wat"}}]})


def test_exported_dataclasses_enforce_schema_invariants_and_revalidate() -> None:
    valid = parse_script(_script(_text("ok")))
    assert parse_script(valid).as_dict() == valid.as_dict()

    invalid_cases = [
        lambda: ResponseStep(kind="wat"),
        lambda: ResponseStep(kind="text", text=7),
        lambda: ResponseStep(kind="tool_calls"),
        lambda: ResponseStep(kind="tool_calls", tool_calls=[{"name": "bad"}]),
        lambda: ResponseStep(kind="http_error", status=200),
        lambda: ResponseStep(kind="hold", hold_response_kind="wat"),
        lambda: Script(steps=[{"response": {"type": "text"}}]),
        lambda: Script(steps=(), schema_version=2),
    ]
    for factory in invalid_cases:
        with pytest.raises(ScriptValidationError):
            factory()

    # A frozen dataclass can still be corrupted through object.__setattr__ by
    # an ill-behaved caller; parse_script must not return that object unchanged.
    object.__setattr__(valid, "model", "")
    with pytest.raises(ScriptValidationError, match="model"):
        parse_script(valid)


def test_model_metadata_pricing_is_strict_and_round_trips_exact_strings() -> None:
    original = _script(_text("priced"))
    original["models"] = ["matrix-model", "second-model"]
    original["model_metadata"] = {
        "matrix-model": {
            "pricing": {
                "prompt": "0.000000125",
                "completion": "1e-7",
                "cache_read": "0",
            }
        }
    }
    parsed = parse_script(original)
    assert parsed.model_metadata == {
        "matrix-model": {
            "pricing": {
                "prompt": "0.000000125",
                "completion": "1e-7",
                "cache_read": "0",
            }
        }
    }
    assert parse_script(parsed.as_dict()).as_dict() == parsed.as_dict()

    invalid_values = [
        0.000001,
        1,
        True,
        "",
        " -1",
        "-0.1",
        "NaN",
        "Infinity",
        "1.2.3",
    ]
    for value in invalid_values:
        invalid = _script(_text("invalid"))
        invalid["model_metadata"] = {"matrix-model": {"pricing": {"prompt": value}}}
        with pytest.raises(ScriptValidationError, match="decimal string"):
            parse_script(invalid)

    unknown_pricing = _script(_text("invalid"))
    unknown_pricing["model_metadata"] = {
        "matrix-model": {"pricing": {"prompt": "0.1", "input": "0.2"}}
    }
    with pytest.raises(ScriptValidationError, match="unsupported field"):
        parse_script(unknown_pricing)

    unknown_model = _script(_text("invalid"))
    unknown_model["model_metadata"] = {"not-listed": {"pricing": {"prompt": "0.1"}}}
    with pytest.raises(ScriptValidationError, match="listed in script.models"):
        parse_script(unknown_model)


def test_script_as_dict_round_trips_every_response_variant() -> None:
    original = {
        "schema_version": 1,
        "model": "roundtrip-model",
        "models": ["roundtrip-model", "second-model"],
        "created": 1_700_000_123,
        "metadata": {"nested": {"stable": [1, "two"]}},
        "steps": [
            {
                "request": {"headers": {"content-type": "application/json"}},
                "response": {"type": "text", "text": "hello"},
            },
            {
                "response": {
                    "type": "tool_calls",
                    "text": "working",
                    "tool_calls": [
                        {
                            "id": "call-fixed",
                            "name": "lookup",
                            "arguments": '{"id":7}',
                        }
                    ],
                }
            },
            {
                "response": {
                    "type": "http_error",
                    "status": 429,
                    "error": {"code": "rate_limit", "nested": {"retry": True}},
                }
            },
            {
                "response": {
                    "type": "connection_close",
                    "before_headers": False,
                    "after_chunks": 2,
                    "text": "partial text",
                }
            },
            {
                "response": {
                    "type": "hold",
                    "id": "hold-tool",
                    "timeout_seconds": 3,
                    "response": {
                        "type": "tool_calls",
                        "text": "held",
                        "tool_calls": [
                            {
                                "id": "call-held",
                                "name": "release_me",
                                "arguments": {"ok": True},
                            }
                        ],
                    },
                }
            },
            {
                "response": {
                    "type": "hold",
                    "id": "hold-error",
                    "response": {
                        "type": "http_error",
                        "status": 503,
                        "error": {"message": "held outage"},
                    },
                }
            },
        ],
    }
    parsed = parse_script(original)
    original["metadata"]["nested"]["stable"].append("mutated")
    assert "mutated" not in parsed.as_dict()["metadata"]["nested"]["stable"]
    round_tripped = parse_script(parsed.as_dict())
    assert round_tripped.as_dict() == parsed.as_dict()


def test_explicit_chunks_validate_concatenation_and_round_trip_for_held_text() -> None:
    original = _script(
        {
            "response": {
                "type": "text",
                "chunks": ["a", "", "雪"],
            }
        },
        {
            "response": {
                "type": "hold",
                "id": "chunk-hold",
                "response": {
                    "type": "text",
                    "text": "held text",
                    "chunks": ["held", " ", "text"],
                },
            },
        },
    )
    parsed = parse_script(original)
    assert parsed.steps[0].text == "a雪"
    assert parsed.steps[0].chunks == ("a", "", "雪")
    assert parsed.steps[1].text == "held text"
    assert parsed.steps[1].chunks == ("held", " ", "text")
    assert parse_script(parsed.as_dict()).as_dict() == parsed.as_dict()

    with pytest.raises(ScriptValidationError, match="concatenation"):
        parse_script(
            _script({
                "response": {
                    "type": "text",
                    "text": "wrong",
                    "chunks": ["right"],
                }
            })
        )
    with pytest.raises(ScriptValidationError, match="only strings"):
        parse_script(_script({"response": {"type": "text", "chunks": ["ok", 3]}}))
    with pytest.raises(ScriptValidationError, match="must be an array"):
        parse_script(_script({"response": {"type": "text", "chunks": "not-an-array"}}))
    with pytest.raises(ScriptValidationError, match="only supported for text"):
        parse_script(
            _script({
                "response": {
                    "type": "tool_calls",
                    "chunks": ["not allowed"],
                    "tool_calls": [{"name": "lookup"}],
                }
            })
        )
    with pytest.raises(ScriptValidationError, match="only supported for text"):
        parse_script(_script({"response": {"type": "http_error", "chunks": []}}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_are_rejected_in_script_fields(value: float) -> None:
    script = _script(_text("ok"))
    script["metadata"] = {"value": value}
    with pytest.raises(ScriptValidationError, match="finite"):
        parse_script(script)
    with pytest.raises(ScriptValidationError, match="positive and finite"):
        parse_script(
            _script({
                "response": {
                    "type": "hold",
                    "timeout_seconds": value,
                    "response": {"type": "text", "text": "held"},
                }
            })
        )


def test_connection_close_text_must_be_a_string() -> None:
    with pytest.raises(ScriptValidationError, match="response.text"):
        parse_script(_script({"response": {"type": "connection_close", "text": 3}}))


@pytest.mark.parametrize(
    "tool_call",
    [
        {"function": {"name": "ok", "arguments": '{"x":1}'}},
        {"function": {"name": "ok", "arguments": "null"}},
    ],
)
def test_tool_call_wrapper_and_json_arguments_are_validated(
    tool_call: dict[str, Any],
) -> None:
    parsed = parse_script({
        "steps": [{"response": {"type": "tool_calls", "tool_calls": [tool_call]}}]
    })
    assert parsed.steps[0].tool_calls[0].name == "ok"

    with pytest.raises(ScriptValidationError, match="name"):
        parse_script({
            "steps": [
                {
                    "response": {
                        "type": "tool_calls",
                        "tool_calls": [{"function": {"name": 7, "arguments": "{}"}}],
                    }
                }
            ]
        })
    with pytest.raises(ScriptValidationError, match="valid JSON"):
        parse_script({
            "steps": [
                {
                    "response": {
                        "type": "tool_calls",
                        "tool_calls": [{"name": "bad", "arguments": "not-json"}],
                    }
                }
            ]
        })


def test_headers_are_sanitized_before_matching_and_capture() -> None:
    script = _script(
        _text(
            "header match",
            request={
                "headers": {"content-type": "application/json"},
                "messages": [],
            },
        )
    )
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        status, _, _ = _chat(server, {"model": "matrix-model", "messages": []})
        assert status == 200
        capture = server.requests[0]
        assert capture["headers"]["content-type"] == "application/json"
        assert "authorization" not in capture["headers"]


def test_reset_cancels_old_generation_hold_and_next_arm_is_new_epoch() -> None:
    script = _script({
        "response": {
            "type": "hold",
            "id": "old-hold",
            "response": {"type": "text", "text": "old"},
        }
    })
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        result: list[tuple[str, Any]] = []

        def request() -> None:
            try:
                result.append(("response", _chat(server, {"messages": []}, timeout=3)))
            except Exception as exc:  # transport EOF is expected on cancellation
                result.append(("error", exc))

        worker = threading.Thread(target=request)
        worker.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not server.state["held"]:
            time.sleep(0.01)
        before = server.state["generation"]
        status, reset, _ = _request(
            server, "POST", "/__control/reset", {}, token="matrix-token"
        )
        assert status == 200
        assert reset["generation"] == before + 1
        assert reset["held"] == []
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result and result[0][0] == "error"

        status, armed, _ = _request(
            server,
            "POST",
            "/__control/arm",
            _script(_text("new")),
            token="matrix-token",
        )
        assert status == 200
        assert armed["generation"] == before + 2
        status, body, _ = _chat(server, {"messages": []})
        assert status == 200
        assert body["id"] == "scripted-completion-000002"
        assert server.requests[0]["id"].startswith("request-g000003-s")


def test_release_after_timeout_loses_under_the_state_lock() -> None:
    script = _script({
        "response": {
            "type": "hold",
            "id": "short-hold",
            "timeout_seconds": 0.05,
            "response": {"type": "text", "text": "never"},
        }
    })
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        status, body, _ = _chat(server, {"messages": []}, timeout=2)
        assert status == 504
        assert "timed out" in body["error"]["message"]
        status, _, _ = _request(
            server,
            "POST",
            "/__control/release",
            {"id": "short-hold"},
            token="matrix-token",
        )
        assert status == 404


def test_held_http_error_is_emitted_after_release() -> None:
    script = _script({
        "response": {
            "type": "hold",
            "id": "held-error",
            "response": {
                "type": "http_error",
                "status": 503,
                "error": {"message": "released outage"},
            },
        }
    })
    with ScriptedProviderServer(script, control_token="matrix-token") as server:
        result: list[tuple[int, dict[str, Any] | str, dict[str, str]]] = []

        def request() -> None:
            result.append(_chat(server, {"messages": []}, timeout=3))

        worker = threading.Thread(target=request)
        worker.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not server.state["held"]:
            time.sleep(0.01)
        status, _, _ = _request(
            server,
            "POST",
            "/__control/release",
            {"id": "held-error"},
            token="matrix-token",
        )
        assert status == 200
        worker.join(timeout=2)
        assert result[0][0] == 503
        assert result[0][1]["error"]["message"] == "released outage"


def test_module_command_starts_loopback_server_without_logging_control_secret(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(_script(_text("cli"))), encoding="utf-8")
    token = "cli-argument-token"
    env_token = "cli-environment-token"
    process_env = os.environ.copy()
    process_env["HERMES_SCRIPTED_PROVIDER_CONTROL_TOKEN"] = env_token
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_testkit.scripted_provider",
            "--script",
            str(script_path),
            "--control-token",
            token,
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=process_env,
    )
    try:
        assert process.stdout is not None
        ready = json.loads(process.stdout.readline())
        assert ready["url"].startswith("http://127.0.0.1:")
        assert ready["schema_version"] == SCRIPT_SCHEMA_VERSION
        assert token not in json.dumps(ready)
        assert env_token not in json.dumps(ready)
        status, _, _ = _request_raw(ready["healthz"])
        assert status == 200
        control_request = urllib.request.Request(
            ready["control"] + "/state",
            headers={"Authorization": f"Bearer {env_token}"},
        )
        with urllib.request.urlopen(control_request, timeout=2) as response:
            assert response.status == 200
    finally:
        process.terminate()
        process.wait(timeout=3)


def _request_raw(url: str) -> tuple[int, bytes, dict[str, str]]:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, response.read(), dict(response.headers.items())
