"""A strict, dependency-free OpenAI-compatible scripted inference server.

The implementation uses :mod:`http.server` rather than a framework so it can
run inside the Python-based Sprite.  It is a deterministic conformance
provider, not a production model fallback: chat requests are accepted only
when they match the next step in the armed script.
"""

from __future__ import annotations

import copy
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .schema import (
    DEFAULT_MODEL,
    FIXED_CREATED,
    SCRIPT_SCHEMA_VERSION,
    ResponseStep,
    Script,
    ScriptValidationError,
    ToolCall,
    UnorderedStepGroup,
    matches_request,
    parse_script,
)

MAX_BODY_BYTES = 8 * 1024 * 1024


def _reject_nonfinite_json_constant(constant: str) -> Any:
    raise ValueError(f"non-finite JSON number {constant} is not supported")


class _StaleGeneration(Exception):
    """Internal signal that an arm/reset invalidated an in-flight handler."""


@dataclass(frozen=True)
class CapturedRequest:
    """A sanitized, deterministic record of an incoming chat request."""

    sequence: int
    request_id: str
    timestamp: int
    method: str
    path: str
    body: Mapping[str, Any] | None
    raw_body: str | None
    stream: bool | None
    headers: Mapping[str, str]
    matched: bool
    error: str | None = None
    generation: int = 0

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sequence": self.sequence,
            "id": self.request_id,
            "timestamp": self.timestamp,
            "generation": self.generation,
            "method": self.method,
            "path": self.path,
            "matched": self.matched,
        }
        if self.body is not None:
            result["json"] = copy.deepcopy(dict(self.body))
        elif self.raw_body is not None:
            result["raw_body"] = self.raw_body
        if self.stream is not None:
            result["stream"] = self.stream
        if self.headers:
            result["headers"] = dict(self.headers)
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class _Hold:
    hold_id: str
    event: threading.Event
    request_id: str
    generation: int
    released: bool = False
    cancelled: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class _ConsumeResult:
    step: ResponseStep | None
    request: CapturedRequest
    script: Script | None
    generation: int
    hold: _Hold | None = None
    error_status: int | None = None
    error_message: str | None = None


class _State:
    """Thread-safe script state shared by request and control handlers."""

    def __init__(self, script: Script | None, *, control_token: str):
        self._lock = threading.RLock()
        self.control_token = control_token
        self.script = script
        self.armed = script is not None
        self.generation = 1
        self.step_index = 0
        self.request_index = 0
        self._unordered_consumed: set[int] = set()
        self._sequence_counter = 0
        self._requests: list[CapturedRequest] = []
        self._unexpected: list[dict[str, Any]] = []
        self._holds: dict[str, _Hold] = {}

    def arm(self, script: Script) -> None:
        with self._lock:
            self._advance_generation_locked()
            self.script = script
            self.armed = True
            self.step_index = 0
            self.request_index = 0
            self._unordered_consumed.clear()
            self._requests.clear()
            self._unexpected.clear()

    def reset(self, *, disarm: bool = False) -> None:
        with self._lock:
            self._advance_generation_locked()
            self.step_index = 0
            self.request_index = 0
            self._unordered_consumed.clear()
            self._requests.clear()
            self._unexpected.clear()
            self.armed = not disarm and self.script is not None

    def cancel(self) -> None:
        """Invalidate all in-flight handlers without changing capture state."""

        with self._lock:
            self._advance_generation_locked()
            self.armed = False

    def _advance_generation_locked(self) -> None:
        self.generation += 1
        for hold in self._holds.values():
            hold.cancelled = True
            hold.event.set()
        self._holds.clear()

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self.generation and self.armed

    def run_if_current(self, generation: int, action: Any) -> bool:
        """Run a response write atomically with the generation check."""

        with self._lock:
            if generation != self.generation or not self.armed:
                return False
            action()
            return True

    def record_unexpected(self, item: Mapping[str, Any]) -> None:
        with self._lock:
            self._unexpected.append(dict(item))

    def _record_unexpected_locked(self, request: CapturedRequest) -> None:
        self._unexpected.append({
            "request_id": request.request_id,
            "sequence": request.sequence,
            "generation": request.generation,
            "path": request.path,
            "error": request.error or "unexpected request",
        })

    def consume(
        self,
        *,
        method: str,
        path: str,
        body: str,
        decoded: Any,
        headers: Mapping[str, str],
    ) -> _ConsumeResult:
        with self._lock:
            script = self.script
            generation = self.generation
            self.request_index += 1
            self._sequence_counter += 1
            sequence = self._sequence_counter
            request_id = f"request-g{generation:06d}-s{sequence:06d}"
            timestamp = (script.created if script else FIXED_CREATED) + sequence
            parsed_body = decoded if isinstance(decoded, Mapping) else None
            stream = parsed_body.get("stream") if parsed_body is not None else None
            if not isinstance(stream, bool):
                stream = None

            if not self.armed or script is None:
                request = CapturedRequest(
                    sequence,
                    request_id,
                    timestamp,
                    method,
                    path,
                    parsed_body,
                    None if parsed_body is not None else body,
                    stream,
                    headers,
                    False,
                    "no script is armed",
                    generation,
                )
                self._requests.append(request)
                self._record_unexpected_locked(request)
                return _ConsumeResult(
                    None,
                    request,
                    script,
                    generation,
                    error_status=409,
                    error_message="no script is armed",
                )

            if not isinstance(decoded, Mapping):
                request = CapturedRequest(
                    sequence,
                    request_id,
                    timestamp,
                    method,
                    path,
                    None,
                    body,
                    None,
                    headers,
                    False,
                    "request body must be a JSON object",
                    generation,
                )
                self._requests.append(request)
                self._record_unexpected_locked(request)
                return _ConsumeResult(
                    None,
                    request,
                    script,
                    generation,
                    error_status=400,
                    error_message="request body must be a JSON object",
                )

            if self.step_index >= len(script.steps):
                request = CapturedRequest(
                    sequence,
                    request_id,
                    timestamp,
                    method,
                    path,
                    decoded,
                    None,
                    stream,
                    headers,
                    False,
                    "script exhausted",
                    generation,
                )
                self._requests.append(request)
                self._record_unexpected_locked(request)
                return _ConsumeResult(
                    None,
                    request,
                    script,
                    generation,
                    error_status=409,
                    error_message="script exhausted",
                )

            step_index = self.step_index
            script_step = script.steps[step_index]
            actual = {
                "method": method,
                "path": path,
                "headers": headers,
                "json": decoded,
            }
            step: ResponseStep | None
            branch_index: int | None = None
            match_error: str | None = None
            if isinstance(script_step, UnorderedStepGroup):
                matching_branches = [
                    branch_index
                    for branch_index, branch in enumerate(script_step.steps)
                    if branch_index not in self._unordered_consumed
                    and matches_request(branch.request, actual)
                ]
                if len(matching_branches) == 1:
                    branch_index = matching_branches[0]
                    step = script_step.steps[branch_index]
                elif not matching_branches:
                    step = None
                    match_error = (
                        f"request did not match any remaining branch in unordered "
                        f"step {step_index}"
                    )
                else:
                    step = None
                    branches = ", ".join(str(index) for index in matching_branches)
                    match_error = (
                        f"request matched multiple branches ({branches}) in unordered "
                        f"step {step_index}"
                    )
            else:
                step = script_step
                if not matches_request(step.request, actual):
                    match_error = f"request did not match step {step_index}"
            matched = step is not None
            request = CapturedRequest(
                sequence,
                request_id,
                timestamp,
                method,
                path,
                decoded,
                None,
                stream,
                headers,
                matched,
                match_error,
                generation,
            )
            self._requests.append(request)
            if not matched:
                self._record_unexpected_locked(request)
                return _ConsumeResult(
                    None,
                    request,
                    script,
                    generation,
                    error_status=409,
                    error_message=match_error
                    or f"unexpected request for script step {step_index}",
                )
            assert step is not None
            if isinstance(script_step, UnorderedStepGroup):
                assert branch_index is not None
                self._unordered_consumed.add(branch_index)
                if len(self._unordered_consumed) == len(script_step.steps):
                    self.step_index += 1
                    self._unordered_consumed.clear()
            else:
                self.step_index += 1
            hold = None
            if step.kind == "hold":
                hold_id = step.hold_id or f"hold-g{generation:06d}-s{sequence:06d}"
                while hold_id in self._holds:
                    hold_id = f"{hold_id}-{sequence:06d}"
                hold = _Hold(
                    hold_id=hold_id,
                    event=threading.Event(),
                    request_id=request_id,
                    generation=generation,
                )
                # Registration is part of consume's lock transaction.  A
                # reset cannot happen between accepting the step and creating
                # the hold, so no stale handler can add an unbounded hold.
                self._holds[hold_id] = hold
            return _ConsumeResult(step, request, script, generation, hold=hold)

    def wait_hold(self, hold: _Hold, timeout: float | None) -> str:
        # ``Event.wait`` handles the actual sleep; all state transitions are
        # decided while holding ``_lock`` so release-vs-timeout has one winner.
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if hold.cancelled:
                    self._holds.pop(hold.hold_id, None)
                    return "cancelled"
                if hold.released:
                    self._holds.pop(hold.hold_id, None)
                    return "released"
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    hold.timed_out = True
                    self._holds.pop(hold.hold_id, None)
                    return "timed_out"
            hold.event.wait(remaining)

    def release(self, hold_id: str | None = None) -> list[str]:
        with self._lock:
            if hold_id is None:
                holds = list(self._holds.values())
            else:
                hold = self._holds.get(hold_id)
                holds = [hold] if hold is not None else []
            released: list[str] = []
            for hold in holds:
                hold.released = True
                hold.event.set()
                released.append(hold.hold_id)
            return released

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            script = self.script
            result: dict[str, Any] = {
                "schema_version": SCRIPT_SCHEMA_VERSION,
                "armed": self.armed,
                "generation": self.generation,
                "next_sequence": self._sequence_counter + 1,
                "step_index": self.step_index,
                "step_count": len(script.steps) if script else 0,
                "consumed": bool(script and self.step_index >= len(script.steps)),
                "complete": bool(
                    script and self.step_index >= len(script.steps) and not self._holds
                ),
                "request_count": len(self._requests),
                "requests": [request.as_dict() for request in self._requests],
                "unexpected_requests": list(self._unexpected),
                "held": [
                    {
                        "id": hold.hold_id,
                        "request_id": hold.request_id,
                        "generation": hold.generation,
                        "released": hold.released,
                    }
                    for hold in self._holds.values()
                ],
            }
            if script is not None and self.step_index < len(script.steps):
                current_step = script.steps[self.step_index]
                if isinstance(current_step, UnorderedStepGroup):
                    consumed = sorted(self._unordered_consumed)
                    result["unordered_progress"] = {
                        "step_index": self.step_index,
                        "consumed_branch_indexes": consumed,
                        "remaining_branch_indexes": [
                            index
                            for index in range(len(current_step.steps))
                            if index not in self._unordered_consumed
                        ],
                    }
            if script is not None:
                result["script"] = script.as_dict()
                result["model"] = script.model
                result["created"] = script.created
            else:
                result["model"] = DEFAULT_MODEL
                result["created"] = FIXED_CREATED
            return result


class ScriptedProviderServer:
    """Threaded loopback HTTP server backed by a strict response script.

    Parameters are intentionally stdlib-only.  The default bind address is
    ``127.0.0.1``. Control endpoints still reject non-loopback clients and
    always require the bearer token. Loopback inference routes are
    unauthenticated by default; non-loopback binds require ``api_key`` (or
    ``HERMES_SCRIPTED_PROVIDER_API_KEY``), which gates the OpenAI-compatible
    routes with Bearer or ``X-API-Key`` authentication.
    """

    def __init__(
        self,
        script: Script | Mapping[str, Any] | str | Path | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        control_token: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError("port must be between 0 and 65535")
        if script is not None and isinstance(script, (str, Path)):
            path = Path(script)
            try:
                script = json.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise ScriptValidationError(
                    f"cannot read script {path}: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ScriptValidationError(
                    f"script {path} is not valid JSON: {exc.msg}"
                ) from exc
        parsed = parse_script(script) if script is not None else None
        if control_token is None:
            # A generated token is never printed or included in responses.
            # Programmatic callers can read it from ``control_token``.
            control_token = secrets.token_urlsafe(32)
        if not isinstance(control_token, str) or not control_token:
            raise ValueError("control_token must be a non-empty string")
        if api_key is None:
            # Keep the fixture unauthenticated by default.  A supervised
            # service can opt into inference auth without changing callers
            # that use the desktop/local loopback defaults.
            api_key = os.environ.get("HERMES_SCRIPTED_PROVIDER_API_KEY") or None
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ValueError("api_key must be a non-empty string when provided")
        if not _is_loopback_host(host) and api_key is None:
            raise ValueError("non-loopback binds require an inference API key")
        self.host = host
        self.requested_port = port
        self.control_token = control_token
        self.api_key = api_key
        self._state = _State(parsed, control_token=control_token)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> dict[str, Any]:
        return self._state.snapshot()

    @property
    def requests(self) -> list[dict[str, Any]]:
        # Return a detached list so callers can inspect or filter captures
        # without mutating the server's synchronized state.
        return list(self._state.snapshot()["requests"])

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("server is not started")
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return self.url

    @property
    def port(self) -> int:
        if self._httpd is None or self._httpd.server_address is None:
            return self.requested_port
        return int(self._httpd.server_address[1])

    def start(self) -> "ScriptedProviderServer":
        if self._httpd is not None:
            return self
        owner = self

        class Handler(_RequestHandler):
            provider = owner

        httpd = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        httpd.daemon_threads = True
        httpd.allow_reuse_address = True
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, name="scripted-provider", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        self._state.cancel()
        httpd.shutdown()
        httpd.server_close()
        self._httpd = None
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2)

    close = stop

    def __enter__(self) -> "ScriptedProviderServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    def arm(self, script: Script | Mapping[str, Any] | str | Path) -> dict[str, Any]:
        if isinstance(script, (str, Path)):
            script = json.loads(Path(script).read_text(encoding="utf-8"))
        parsed = parse_script(script)
        self._state.arm(parsed)
        return self.state

    def reset(self, *, disarm: bool = False) -> dict[str, Any]:
        self._state.reset(disarm=disarm)
        return self.state

    def release(self, hold_id: str | None = None) -> tuple[str, ...]:
        return tuple(self._state.release(hold_id))


# Friendly alias used by callers that call the fixture an inference server.
ScriptedInferenceServer = ScriptedProviderServer


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP protocol implementation; instances are short-lived per request."""

    provider: ScriptedProviderServer
    protocol_version = "HTTP/1.1"

    # Do not write access logs: request bodies can contain user prompts and
    # provider credentials.  The authenticated state endpoint is the explicit
    # capture surface for tests.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key"
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(
                HTTPStatus.OK, {"status": "ok", "schema_version": SCRIPT_SCHEMA_VERSION}
            )
            return
        if path in {"/api/v1/models", "/api/tags", "/v1/props", "/props", "/version"}:
            if not self._require_inference_auth():
                return
            self._ignore_capability_probe()
            return
        if path.startswith("/v1/models/") and path.removeprefix("/v1/models/"):
            if not self._require_inference_auth():
                return
            self._ignore_capability_probe()
            return
        if path == "/v1/models":
            if not self._require_inference_auth():
                return
            snapshot = self.provider.state
            model = snapshot.get("model", DEFAULT_MODEL)
            script_snapshot = snapshot.get("script", {})
            ids = script_snapshot.get("models", [model])
            model_metadata = script_snapshot.get("model_metadata", {})
            created = snapshot.get("created", FIXED_CREATED)
            data = []
            for model_id in ids:
                item: dict[str, Any] = {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "hermes-testkit",
                }
                metadata = model_metadata.get(model_id)
                if isinstance(metadata, Mapping):
                    # Script validation limits these to OpenAI-compatible
                    # model fields.  Copy before merging so a caller cannot
                    # mutate the provider's state through the response.
                    item.update(copy.deepcopy(dict(metadata)))
                data.append(item)
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": data,
                },
            )
            return
        if path in {
            "/__control",
            "/__control/state",
            "/__control/requests",
            "/control",
            "/control/state",
        }:
            if not self._require_control_auth():
                return
            self._send_json(HTTPStatus.OK, self.provider.state)
            return
        self.provider._state.record_unexpected(  # noqa: SLF001
            {"method": "GET", "path": path, "error": "unexpected endpoint"}
        )
        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/__control/arm", "/control/arm"}:
            if not self._require_control_auth():
                return
            self._control_arm()
            return
        if path in {"/__control/reset", "/control/reset"}:
            if not self._require_control_auth():
                return
            self._control_reset()
            return
        if path in {"/__control/release", "/control/release"}:
            if not self._require_control_auth():
                return
            self._control_release()
            return
        if path == "/api/show":
            if not self._require_inference_auth():
                return
            self._ignore_capability_probe()
            return
        if path == "/v1/chat/completions":
            if not self._require_inference_auth():
                return
            self._chat_completion()
            return
        self.provider._state.record_unexpected(  # noqa: SLF001
            {"method": "POST", "path": path, "error": "unexpected endpoint"}
        )
        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def _ignore_capability_probe(self) -> None:
        """Return a probe-down 404 without treating the route as unexpected."""

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def _read_body(self) -> tuple[str, Any | None, str | None]:
        header = self.headers.get("Content-Length")
        try:
            length = int(header) if header is not None else 0
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            return "", None, "request body is too large or has invalid Content-Length"
        raw = self.rfile.read(length) if length else b""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "", None, "request body is not UTF-8"
        if not text:
            return text, None, "request body is empty"
        try:
            return (
                text,
                json.loads(text, parse_constant=_reject_nonfinite_json_constant),
                None,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            return text, None, f"request body is not valid JSON: {detail}"

    def _chat_completion(self) -> None:
        raw, decoded, parse_error = self._read_body()
        headers = _safe_headers(dict(self.headers.items()))
        if parse_error is not None:
            result = self.provider._state.consume(  # noqa: SLF001
                method="POST",
                path="/v1/chat/completions",
                body=raw,
                decoded=decoded,
                headers=headers,
            )
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                parse_error,
                request_id=result.request.request_id,
                generation=result.generation if result.script is not None else None,
            )
            return

        result = self.provider._state.consume(  # noqa: SLF001
            method="POST",
            path="/v1/chat/completions",
            body=raw,
            decoded=decoded,
            headers=headers,
        )
        if result.error_status is not None:
            expected = self.provider.state.get("step_index", 0)
            self._send_error_json(
                result.error_status,
                result.error_message or "unexpected request",
                request_id=result.request.request_id,
                details={"step_index": expected},
                generation=result.generation if result.script is not None else None,
            )
            return
        if not self.provider._state.is_current(result.generation):  # noqa: SLF001
            self._close_connection()
            return
        assert result.step is not None
        assert isinstance(decoded, Mapping)
        step = result.step
        stream = (
            bool(decoded.get("stream", False))
            if isinstance(decoded, Mapping)
            else False
        )
        if step.kind == "http_error":
            self._send_http_error_step(
                step,
                request_id=result.request.request_id,
                generation=result.generation,
            )
            return
        if step.kind == "connection_close":
            self._send_connection_close(
                step,
                stream=stream,
                script=result.script,
                request_id=result.request.request_id,
                generation=result.generation,
            )
            return
        if step.kind == "hold":
            assert result.hold is not None
            hold_result = self.provider._state.wait_hold(  # noqa: SLF001
                result.hold, step.hold_timeout_seconds
            )
            if hold_result == "cancelled" or not self.provider._state.is_current(  # noqa: SLF001
                result.generation
            ):
                self._close_connection()
                return
            if hold_result == "timed_out":
                self._send_error_json(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    "held response timed out before release",
                    request_id=result.request.request_id,
                    generation=result.generation,
                )
                return
            if step.hold_response_kind == "http_error":
                self._send_http_error_step(
                    step,
                    request_id=result.request.request_id,
                    generation=result.generation,
                )
                return
            step = ResponseStep(
                kind=step.hold_response_kind,
                request=step.request,
                text=step.text,
                tool_calls=step.tool_calls,
                chunks=step.chunks,
                usage=step.usage,
            )
        if stream:
            self._send_stream(
                step,
                decoded,
                result.request.request_id,
                script=result.script,
                generation=result.generation,
            )
        else:
            self._send_non_stream(
                step,
                decoded,
                result.request.request_id,
                script=result.script,
                generation=result.generation,
            )

    def _send_stream(
        self,
        step: ResponseStep,
        request: Mapping[str, Any],
        request_id: str,
        *,
        script: Script | None,
        generation: int,
    ) -> None:
        model = str(request.get("model") or (script.model if script else DEFAULT_MODEL))
        created = script.created if script else FIXED_CREATED
        completion_id = _completion_id(request_id)
        usage = _usage(request, step)
        try:
            # A finite scripted stream has no useful keep-alive semantics.
            # Marking the connection close lets simple urllib/httpx clients
            # observe EOF immediately after the [DONE] event.
            self.close_connection = True

            def write_headers() -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

            if not self.provider._state.run_if_current(  # noqa: SLF001
                generation, write_headers
            ):
                self._close_connection()
                return
            chunks = 0

            def emit(
                delta: Mapping[str, Any],
                finish_reason: str | None = None,
                include_usage: bool = False,
            ) -> None:
                nonlocal chunks
                if not self.provider._state.is_current(generation):  # noqa: SLF001
                    raise _StaleGeneration
                payload: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": dict(delta),
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                if include_usage:
                    payload["usage"] = usage

                def write_chunk() -> None:
                    self.wfile.write(
                        f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)}\n\n".encode()
                    )
                    self.wfile.flush()

                if not self.provider._state.run_if_current(  # noqa: SLF001
                    generation, write_chunk
                ):
                    raise _StaleGeneration
                chunks += 1

            emit({"role": "assistant"})
            for text_chunk in _response_text_chunks(step):
                emit({"content": text_chunk})
            if step.tool_calls:
                for index, call in enumerate(step.tool_calls):
                    emit({
                        "tool_calls": [_tool_call_payload(call, completion_id, index)]
                    })
            emit({}, "tool_calls" if step.tool_calls else "stop", include_usage=True)

            def write_done() -> None:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            if not self.provider._state.run_if_current(generation, write_done):  # noqa: SLF001
                raise _StaleGeneration
        except _StaleGeneration:
            self._close_connection()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_non_stream(
        self,
        step: ResponseStep,
        request: Mapping[str, Any],
        request_id: str,
        *,
        script: Script | None,
        generation: int,
    ) -> None:
        if not self.provider._state.is_current(generation):  # noqa: SLF001
            self._close_connection()
            return
        model = str(request.get("model") or (script.model if script else DEFAULT_MODEL))
        created = script.created if script else FIXED_CREATED
        completion_id = _completion_id(request_id)
        # Keep the legacy ``null`` content for an unchunked empty text
        # response, while preserving an explicitly scripted empty chunk as
        # the exact concatenated string.
        content = step.text if step.chunks is not None else (step.text or None)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if step.tool_calls:
            message["tool_calls"] = [
                _tool_call_payload(call, completion_id, index, include_index=False)
                for index, call in enumerate(step.tool_calls)
            ]
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if step.tool_calls else "stop",
                }
            ],
            "usage": _usage(request, step),
        }
        self._send_json(HTTPStatus.OK, response, generation=generation)

    def _send_http_error_step(
        self, step: ResponseStep, *, request_id: str, generation: int
    ) -> None:
        if not self.provider._state.is_current(generation):  # noqa: SLF001
            self._close_connection()
            return
        details = dict(step.error or {})
        if "message" not in details:
            details["message"] = "scripted HTTP error"
        if "type" not in details:
            details["type"] = "scripted_error"
        self._send_error_json(
            step.status,
            details["message"],
            request_id=request_id,
            details=details,
            generation=generation,
        )

    def _send_connection_close(
        self,
        step: ResponseStep,
        *,
        stream: bool,
        script: Script | None,
        request_id: str,
        generation: int,
    ) -> None:
        if not self.provider._state.is_current(generation):  # noqa: SLF001
            self._close_connection()
            return
        self.close_connection = True
        if step.close_before_headers:
            self._close_connection()
            return
        try:

            def write_partial() -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    "text/event-stream" if stream else "application/json",
                )
                self.send_header("Connection", "close")
                self.end_headers()
                if stream:
                    completion_id = _completion_id(request_id)
                    partials = _text_chunks(step.text) if step.text else []
                    # ``close_after_chunks`` counts emitted SSE chunks.  A zero
                    # value deliberately closes after headers, while a positive
                    # value makes it possible to test mid-stream truncation.
                    for text_chunk in partials[: step.close_after_chunks]:
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": script.created if script else FIXED_CREATED,
                            "model": script.model if script else DEFAULT_MODEL,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text_chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        self.wfile.write(
                            f"data: {json.dumps(payload, separators=(',', ':'), allow_nan=False)}\n\n".encode()
                        )
                else:
                    completion_id = _completion_id(request_id)
                    self.wfile.write(f'{{"id":"{completion_id}","choices":['.encode())
                self.wfile.flush()

            if not self.provider._state.run_if_current(  # noqa: SLF001
                generation, write_partial
            ):
                self._close_connection()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.connection.close()

    def _close_connection(self) -> None:
        self.close_connection = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass
        self.connection.close()

    def _control_arm(self) -> None:
        raw, decoded, error = self._read_body()
        if error is not None or not isinstance(decoded, Mapping):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, error or "control body must be an object"
            )
            return
        script_value: Any = decoded.get("script", decoded)
        try:
            script = parse_script(script_value)
        except ScriptValidationError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self.provider._state.arm(script)  # noqa: SLF001
        self._send_json(HTTPStatus.OK, self.provider.state)

    def _control_reset(self) -> None:
        decoded = self._read_control_object("reset")
        if decoded is None:
            return
        disarm = decoded.get("disarm", False)
        if not isinstance(disarm, bool):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "reset.disarm must be a boolean"
            )
            return
        self.provider._state.reset(disarm=disarm)  # noqa: SLF001
        self._send_json(HTTPStatus.OK, self.provider.state)

    def _control_release(self) -> None:
        decoded = self._read_control_object("release")
        if decoded is None:
            return
        hold_id = decoded.get("id", decoded.get("hold_id"))
        if hold_id is not None and not isinstance(hold_id, str):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "release id must be a string")
            return
        released = self.provider._state.release(hold_id)  # noqa: SLF001
        if hold_id is not None and not released:
            self._send_error_json(HTTPStatus.NOT_FOUND, "held response not found")
            return
        self._send_json(
            HTTPStatus.OK, {"released": released, "state": self.provider.state}
        )

    def _read_control_object(self, action: str) -> Mapping[str, Any] | None:
        _raw, decoded, error = self._read_body()
        if error is not None:
            self._send_error_json(HTTPStatus.BAD_REQUEST, error)
            return None
        if not isinstance(decoded, Mapping):
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, f"{action} body must be an object"
            )
            return None
        return decoded

    def _require_control_auth(self) -> bool:
        client = self.client_address[0] if self.client_address else ""
        try:
            is_loopback = ipaddress.ip_address(client).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            self._send_error_json(HTTPStatus.FORBIDDEN, "control API is loopback-only")
            return False
        auth = self.headers.get("Authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not supplied:
            supplied = self.headers.get("X-Control-Token", "")
        if not supplied or not hmac.compare_digest(
            supplied, self.provider.control_token
        ):
            self._send_error_json(
                HTTPStatus.UNAUTHORIZED, "control authentication required"
            )
            return False
        return True

    def _require_inference_auth(self) -> bool:
        """Authenticate public inference routes when an API key is configured."""

        expected = self.provider.api_key
        if expected is None:
            return True
        authorization = self.headers.get("Authorization", "")
        scheme, separator, credential = authorization.partition(" ")
        bearer = credential.strip() if separator and scheme.lower() == "bearer" else ""
        x_api_key = self.headers.get("X-API-Key", "").strip()
        # Evaluate both candidates so either standard OpenAI bearer auth or the
        # conventional X-API-Key form works without ever logging credentials.
        bearer_matches = hmac.compare_digest(bearer, expected)
        x_api_key_matches = hmac.compare_digest(x_api_key, expected)
        if bearer_matches or x_api_key_matches:
            return True
        self._send_error_json(
            HTTPStatus.UNAUTHORIZED, "inference authentication required"
        )
        return False

    def _send_json(
        self,
        status: int | HTTPStatus,
        payload: Mapping[str, Any],
        *,
        generation: int | None = None,
    ) -> None:
        data = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        def write_json() -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        if generation is None:
            write_json()
        elif not self.provider._state.run_if_current(  # noqa: SLF001
            generation, write_json
        ):
            self._close_connection()

    def _send_error_json(
        self,
        status: int | HTTPStatus,
        message: str,
        *,
        request_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        generation: int | None = None,
    ) -> None:
        error: dict[str, Any] = {"message": message, "type": "scripted_provider_error"}
        if request_id is not None:
            error["request_id"] = request_id
        if details:
            error.update(details)
        self._send_json(
            status,
            {"error": error, "schema_version": SCRIPT_SCHEMA_VERSION},
            generation=generation,
        )


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    sensitive = {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "cookie",
        "set-cookie",
    }
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() not in sensitive
        and key.lower() in {"content-type", "accept", "user-agent"}
    }


def _is_loopback_host(host: str) -> bool:
    candidate = host.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # ``localhost`` is the only hostname accepted without DNS
        # resolution; unknown names are conservatively treated as public.
        return candidate.lower() in {"localhost", "ip6-localhost"}


def _text_chunks(text: str) -> list[str]:
    # Preserve spaces exactly while avoiding arbitrary timing or character
    # slicing that can split a Unicode code point.
    chunks: list[str] = []
    current = ""
    for character in text:
        current += character
        if character.isspace() or len(current) >= 16:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _response_text_chunks(step: ResponseStep) -> list[str]:
    """Return exact scripted chunks or the legacy deterministic splitter."""

    if step.chunks is not None:
        return list(step.chunks)
    return _text_chunks(step.text) if step.text else []


def _completion_id(request_id: str) -> str:
    suffix = request_id.rsplit("-s", 1)[-1]
    try:
        sequence = int(suffix)
    except ValueError:
        sequence = 0
    return f"scripted-completion-{sequence:06d}"


def _tool_call_payload(
    call: ToolCall, completion_id: str, index: int, *, include_index: bool = True
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": call.id
        or f"call-{completion_id.removeprefix('scripted-completion-')}-{index:02d}",
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }
    if include_index:
        payload["index"] = index
    return payload


def _usage(request: Mapping[str, Any], step: ResponseStep) -> dict[str, int]:
    if step.usage is not None:
        # ``ResponseStep`` freezes overrides so a caller cannot mutate an
        # armed script while a request is in flight.  Materialise a regular
        # dict at the wire boundary because stdlib ``json`` intentionally does
        # not encode MappingProxyType.
        return dict(step.usage)
    messages = request.get("messages", [])
    prompt_serialized = json.dumps(
        messages,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    prompt_tokens = max(1, (len(prompt_serialized) + 3) // 4)
    completion_serialized = step.text + "".join(
        call.name + call.arguments for call in step.tool_calls
    )
    completion_tokens = max(1, (len(completion_serialized) + 3) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
