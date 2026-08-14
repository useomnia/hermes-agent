"""Schema and validation helpers for the scripted inference provider.

The provider deliberately keeps its wire format small.  A script is JSON (or
the equivalent Python mapping) with a version and an ordered list of steps:

.. code-block:: json

   {
     "schema_version": 1,
     "model": "scripted-model",
     "steps": [
       {"response": {"type": "text", "text": "hello"}},
       {"response": {"type": "tool_calls", "tool_calls": [
         {"name": "lookup", "arguments": {"id": 42}}
       ]}}
     ]
   }

``request`` on a step is optional.  When present it is a recursive subset
match against the decoded chat-completions request.  Keeping matching in the
schema layer makes the HTTP server useful from both Python tests and a Sprite
process without introducing an SDK dependency.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping

SCRIPT_SCHEMA_VERSION = 1
DEFAULT_MODEL = "scripted-model"
FIXED_CREATED = 1_700_000_000
_RESPONSE_KINDS = frozenset({
    "text",
    "tool_calls",
    "http_error",
    "connection_close",
    "hold",
})
_HOLD_RESPONSE_KINDS = frozenset({"text", "tool_calls", "http_error"})
_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
_MODEL_METADATA_FIELDS = frozenset({"pricing"})
_PRICING_FIELDS = frozenset({
    "prompt",
    "completion",
    "request",
    "cache_read",
    "cache_write",
})
# Pricing is deliberately represented as a JSON string rather than a JSON
# number.  That keeps tiny per-token values exact across Python, JavaScript,
# and the OpenAI-compatible wire.  Scientific notation is accepted because
# it is still an exact Decimal representation; whitespace and signed values
# are rejected to prevent implicit coercion and non-canonical JSON numbers.
_DECIMAL_STRING = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class ScriptValidationError(ValueError):
    """Raised when a script does not conform to the supported schema."""


def _freeze_json(value: Any, *, where: str = "value") -> Any:
    """Deep-copy JSON-shaped values into immutable containers.

    Scripts are commonly assembled in a fixture and then mutated by the test
    that owns them.  Keeping a caller's dict/list by reference makes the
    response sequence change underneath an already-running server.  Freeze
    every nested container at parse time and reject non-JSON values early.
    """

    if isinstance(value, Mapping):
        frozen: dict[Any, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScriptValidationError(f"{where} object keys must be strings")
            frozen[key] = _freeze_json(item, where=f"{where}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, where=f"{where}[]") for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScriptValidationError(f"{where} must be finite")
        return copy.deepcopy(value)
    if value is None or isinstance(value, (str, int, bool)):
        return copy.deepcopy(value)
    raise ScriptValidationError(f"{where} must be JSON serialisable")


def _thaw_json(value: Any) -> Any:
    """Return a detached mutable JSON-shaped copy for wire/state output."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _as_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScriptValidationError(f"{where} must be an object")
    return value


def _validate_decimal_string(value: Any, *, where: str) -> str:
    """Validate one exact, finite, non-negative per-token decimal string."""

    if not isinstance(value, str) or not _DECIMAL_STRING.fullmatch(value):
        raise ScriptValidationError(
            f"{where} must be a finite non-negative decimal string"
        )
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ScriptValidationError(
            f"{where} must be a finite non-negative decimal string"
        ) from exc
    if not decimal.is_finite() or decimal < 0:
        raise ScriptValidationError(
            f"{where} must be a finite non-negative decimal string"
        )
    return value


def _validate_model_metadata(
    value: Any,
    *,
    model_ids: tuple[str, ...],
    where: str = "model_metadata",
) -> dict[str, dict[str, Any]]:
    """Validate the optional per-model metadata/pricing contract.

    The contract intentionally starts with one field (``pricing``), whose
    values are exact per-token decimal strings consumed by Hermes' generic
    OpenAI-compatible models parser.  Keeping the outer map keyed by model id
    allows a script to advertise more than one model without coupling the
    testkit to any particular Omnio model name.
    """

    if not isinstance(value, Mapping):
        raise ScriptValidationError(f"{where} must be an object")
    known_models = set(model_ids)
    result: dict[str, dict[str, Any]] = {}
    for model_id, metadata in value.items():
        if not isinstance(model_id, str) or not model_id:
            raise ScriptValidationError(f"{where} keys must be non-empty strings")
        if model_id not in known_models:
            raise ScriptValidationError(
                f"{where}.{model_id} must name a model listed in script.models"
            )
        if not isinstance(metadata, Mapping):
            raise ScriptValidationError(f"{where}.{model_id} must be an object")
        unknown = set(metadata) - _MODEL_METADATA_FIELDS
        if unknown:
            fields = ", ".join(sorted(str(field) for field in unknown))
            raise ScriptValidationError(
                f"{where}.{model_id} has unsupported field(s): {fields}"
            )
        pricing = metadata.get("pricing")
        if not isinstance(pricing, Mapping) or not pricing:
            raise ScriptValidationError(
                f"{where}.{model_id}.pricing must be a non-empty object"
            )
        unknown_pricing = set(pricing) - _PRICING_FIELDS
        if unknown_pricing:
            fields = ", ".join(sorted(str(field) for field in unknown_pricing))
            raise ScriptValidationError(
                f"{where}.{model_id}.pricing has unsupported field(s): {fields}"
            )
        validated_pricing: dict[str, str] = {}
        for field_name, amount in pricing.items():
            if not isinstance(field_name, str):
                raise ScriptValidationError(
                    f"{where}.{model_id}.pricing keys must be strings"
                )
            validated_pricing[field_name] = _validate_decimal_string(
                amount,
                where=f"{where}.{model_id}.pricing.{field_name}",
            )
        result[model_id] = {"pricing": validated_pricing}
    return result


def _validate_status(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 400 <= value <= 599:
        raise ScriptValidationError(f"{where} must be an HTTP 4xx/5xx integer")
    return value


def _validate_usage(value: Any, *, where: str = "response.usage") -> dict[str, int]:
    """Validate an explicit OpenAI completion-usage override.

    The scripted provider normally derives usage from the request and response
    text.  A conformance fixture can instead provide the exact token counts it
    wants Hermes to observe.  Keep this contract deliberately narrower than a
    general OpenAI usage object: all three fields are required, values are
    JSON integers (not booleans/floats), and the total is an arithmetic
    invariant rather than another independently chosen value.
    """

    if not isinstance(value, Mapping):
        raise ScriptValidationError(f"{where} must be an object")
    keys = set(value)
    expected = set(_USAGE_FIELDS)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        fields = ", ".join(sorted(str(field) for field in unknown))
        raise ScriptValidationError(f"{where} has unsupported field(s): {fields}")
    if missing:
        fields = ", ".join(field for field in _USAGE_FIELDS if field in missing)
        raise ScriptValidationError(f"{where} is missing field(s): {fields}")
    result: dict[str, int] = {}
    for field_name in _USAGE_FIELDS:
        amount = value[field_name]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ScriptValidationError(
                f"{where}.{field_name} must be a non-negative integer"
            )
        result[field_name] = amount
    if result["total_tokens"] != (
        result["prompt_tokens"] + result["completion_tokens"]
    ):
        raise ScriptValidationError(
            f"{where}.total_tokens must equal prompt_tokens + completion_tokens"
        )
    return result


def _validate_timeout(value: Any, *, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScriptValidationError(f"{where} must be positive")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ScriptValidationError(f"{where} must be positive and finite")
    return timeout


def _load_json_string(value: str, *, where: str) -> Any:
    def reject_nonfinite(constant: str) -> Any:
        raise ScriptValidationError(f"{where} must not contain {constant}")

    try:
        return json.loads(value, parse_constant=reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise ScriptValidationError(f"{where} must be valid JSON") from exc


def _normalise_tool_call(value: Any, *, where: str) -> "ToolCall":
    item = _as_mapping(value, where=where)
    name = item.get("name")
    if "name" in item:
        if not isinstance(name, str) or not name:
            raise ScriptValidationError(f"{where}.name must be a non-empty string")
        arguments = item.get("arguments", item.get("args", {}))
    else:
        # Accept the OpenAI-shaped function wrapper as a convenience when a
        # script is copied from a captured request/response.
        function = item.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments", "{}")
        else:
            raise ScriptValidationError(f"{where}.name must be a non-empty string")
    if isinstance(arguments, str):
        _load_json_string(arguments, where=f"{where}.arguments")
        arguments_json = arguments
    else:
        # JSON encoding is done by the server with stable key ordering. Freeze
        # first so non-string keys and non-finite numbers fail at arm time.
        arguments_json = _json_dumps(
            _thaw_json(_freeze_json(arguments, where=f"{where}.arguments")),
            where=f"{where}.arguments",
        )

    call_id = item.get("id")
    if not isinstance(name, str) or not name:
        raise ScriptValidationError(f"{where}.name must be a non-empty string")
    if call_id is not None and (not isinstance(call_id, str) or not call_id):
        raise ScriptValidationError(
            f"{where}.id must be a non-empty string when present"
        )
    return ToolCall(name=name, arguments=arguments_json, id=call_id)


def _json_dumps(value: Any, *, where: str) -> str:
    import json

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ScriptValidationError(f"{where} must be JSON serialisable") from exc


@dataclass(frozen=True)
class ToolCall:
    """A deterministic function/tool call emitted by a response step."""

    name: str
    arguments: str = "{}"
    id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ScriptValidationError("tool call name must be a non-empty string")
        if not isinstance(self.arguments, str):
            raise ScriptValidationError("tool call arguments must be a JSON string")
        _load_json_string(self.arguments, where="tool call arguments")
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise ScriptValidationError("tool call id must be a non-empty string")


@dataclass(frozen=True)
class ResponseStep:
    """One expected chat request and the response it should receive."""

    kind: str
    request: Mapping[str, Any] | None = None
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    status: int = 500
    error: Mapping[str, Any] | None = None
    hold_id: str | None = None
    hold_timeout_seconds: float | None = None
    hold_response_kind: str = "text"
    close_after_chunks: int = 0
    close_before_headers: bool = True
    # ``None`` means use the provider's backwards-compatible text splitter.
    # An explicit tuple (including an empty tuple) prescribes the exact SSE
    # text-delta boundaries for a text response.
    chunks: tuple[str, ...] | None = None
    # ``None`` keeps the backwards-compatible derived usage calculation. An
    # explicit mapping is emitted verbatim (after strict validation) on the
    # final streamed/non-streamed response.
    usage: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _RESPONSE_KINDS:
            raise ScriptValidationError(
                f"response.type {self.kind!r} is unsupported; expected text, "
                "tool_calls, http_error, connection_close, or hold"
            )
        if not isinstance(self.text, str):
            raise ScriptValidationError("response.text must be a string")
        if self.request is not None:
            if not isinstance(self.request, Mapping):
                raise ScriptValidationError("request must be an object")
            object.__setattr__(
                self, "request", _freeze_json(self.request, where="request")
            )
        if self.error is not None:
            if not isinstance(self.error, Mapping):
                raise ScriptValidationError("response.error must be an object")
            object.__setattr__(self, "error", _freeze_json(self.error, where="error"))
        if not isinstance(self.tool_calls, (list, tuple)):
            raise ScriptValidationError("response.tool_calls must be an array")
        tool_calls = tuple(self.tool_calls)
        if any(not isinstance(call, ToolCall) for call in tool_calls):
            raise ScriptValidationError(
                "response.tool_calls must contain ToolCall objects"
            )
        object.__setattr__(self, "tool_calls", tool_calls)
        _validate_status(self.status, where="response.status")
        if (
            isinstance(self.close_after_chunks, bool)
            or not isinstance(self.close_after_chunks, int)
            or self.close_after_chunks < 0
        ):
            raise ScriptValidationError(
                "response.after_chunks must be a non-negative integer"
            )
        if not isinstance(self.close_before_headers, bool):
            raise ScriptValidationError("response.before_headers must be boolean")
        if not isinstance(self.hold_response_kind, str):
            raise ScriptValidationError("response.response.type must be a string")
        if self.usage is not None:
            validated_usage = _validate_usage(self.usage)
            object.__setattr__(self, "usage", MappingProxyType(validated_usage))
        timeout = _validate_timeout(
            self.hold_timeout_seconds, where="response.timeout_seconds"
        )
        object.__setattr__(self, "hold_timeout_seconds", timeout)
        if self.hold_id is not None and (
            not isinstance(self.hold_id, str) or not self.hold_id
        ):
            raise ScriptValidationError(
                "response.id must be a non-empty string when present"
            )
        if self.chunks is not None:
            if not isinstance(self.chunks, (list, tuple)):
                raise ScriptValidationError("response.chunks must be an array")
            chunks = tuple(self.chunks)
            if any(not isinstance(chunk, str) for chunk in chunks):
                raise ScriptValidationError("response.chunks must contain only strings")
            object.__setattr__(self, "chunks", chunks)
            if self.kind not in {"text", "hold"} or (
                self.kind == "hold" and self.hold_response_kind != "text"
            ):
                raise ScriptValidationError(
                    "response.chunks is only supported for text responses"
                )
            concatenated = "".join(chunks)
            if self.text and self.text != concatenated:
                raise ScriptValidationError(
                    "response.text must equal the concatenation of response.chunks"
                )
            if not self.text:
                object.__setattr__(self, "text", concatenated)

        if self.usage is not None and self.kind in {
            "http_error",
            "connection_close",
        }:
            raise ScriptValidationError(
                f"response.usage cannot be emitted for {self.kind} responses"
            )

        if self.kind == "hold":
            if self.close_after_chunks or not self.close_before_headers:
                raise ScriptValidationError(
                    "response.before_headers and response.after_chunks require a "
                    "connection_close response"
                )
            if self.hold_response_kind not in _HOLD_RESPONSE_KINDS:
                raise ScriptValidationError(
                    "response.response.type must be text, tool_calls, or http_error"
                )
            if self.hold_response_kind == "tool_calls":
                if not self.tool_calls:
                    raise ScriptValidationError(
                        "response.response.tool_calls must not be empty"
                    )
            elif self.tool_calls:
                raise ScriptValidationError(
                    "response.response.tool_calls are only valid for tool_calls"
                )
            if self.hold_response_kind == "http_error":
                if self.usage is not None:
                    raise ScriptValidationError(
                        "response.usage cannot be emitted for held http_error responses"
                    )
                if self.chunks is not None:
                    raise ScriptValidationError(
                        "response.chunks is only supported for held text responses"
                    )
                if self.text:
                    raise ScriptValidationError(
                        "response.text is not valid for held http_error responses"
                    )
            elif self.error is not None:
                raise ScriptValidationError(
                    "response.error is only valid for http_error responses"
                )
            if self.hold_response_kind != "http_error" and self.status != 500:
                raise ScriptValidationError(
                    "response.status is only valid for http_error responses"
                )
            if self.hold_response_kind != "text" and self.chunks is not None:
                raise ScriptValidationError(
                    "response.chunks is only supported for held text responses"
                )
            return

        if self.hold_id is not None or self.hold_timeout_seconds is not None:
            raise ScriptValidationError(
                "response.id and response.timeout_seconds require a hold response"
            )
        if self.hold_response_kind != "text":
            raise ScriptValidationError(
                "response.response.type requires a hold response"
            )
        if self.kind == "tool_calls":
            if not self.tool_calls:
                raise ScriptValidationError("response.tool_calls must not be empty")
        elif self.tool_calls:
            raise ScriptValidationError(
                "response.tool_calls are only valid for tool_calls responses"
            )
        if self.kind != "connection_close" and (
            self.close_after_chunks or not self.close_before_headers
        ):
            raise ScriptValidationError(
                "response.before_headers and response.after_chunks require a "
                "connection_close response"
            )
        if self.kind == "http_error":
            if self.text:
                raise ScriptValidationError(
                    "response.text is not valid for http_error responses"
                )
            return
        if self.error is not None:
            raise ScriptValidationError(
                "response.error is only valid for http_error responses"
            )
        if self.status != 500:
            raise ScriptValidationError(
                "response.status is only valid for http_error responses"
            )


@dataclass(frozen=True)
class Script:
    """Validated immutable script consumed by :class:`ScriptedProvider`."""

    steps: tuple[ResponseStep, ...]
    schema_version: int = SCRIPT_SCHEMA_VERSION
    model: str = DEFAULT_MODEL
    models: tuple[str, ...] = ()
    created: int = FIXED_CREATED
    metadata: Mapping[str, Any] = field(default_factory=dict)
    model_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def model_ids(self) -> tuple[str, ...]:
        if self.models:
            return self.models
        return (self.model,)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SCRIPT_SCHEMA_VERSION
        ):
            raise ScriptValidationError(
                f"script schema_version {self.schema_version!r} is unsupported; "
                f"expected {SCRIPT_SCHEMA_VERSION}"
            )
        if not isinstance(self.model, str) or not self.model:
            raise ScriptValidationError("script.model must be a non-empty string")
        if not isinstance(self.steps, (list, tuple)):
            raise ScriptValidationError("script.steps must be an array")
        steps = tuple(self.steps)
        if any(not isinstance(step, ResponseStep) for step in steps):
            raise ScriptValidationError(
                "script.steps must contain ResponseStep objects"
            )
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.models, (list, tuple)):
            raise ScriptValidationError("script.models must be an array of strings")
        models = tuple(self.models)
        if any(not isinstance(model, str) or not model for model in models):
            raise ScriptValidationError("script.models must be an array of strings")
        object.__setattr__(self, "models", models)
        if (
            isinstance(self.created, bool)
            or not isinstance(self.created, int)
            or self.created < 0
        ):
            raise ScriptValidationError("script.created must be a non-negative integer")
        if not isinstance(self.metadata, Mapping):
            raise ScriptValidationError("script.metadata must be an object")
        object.__setattr__(
            self, "metadata", _freeze_json(self.metadata, where="metadata")
        )
        validated_model_metadata = _validate_model_metadata(
            self.model_metadata,
            model_ids=self.model_ids,
        )
        object.__setattr__(
            self,
            "model_metadata",
            _freeze_json(validated_model_metadata, where="model_metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for state/debug endpoints."""

        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model": self.model,
            "models": list(self.model_ids),
            "created": self.created,
            "steps": [],
        }
        if self.metadata:
            result["metadata"] = _thaw_json(self.metadata)
        if self.model_metadata:
            result["model_metadata"] = _thaw_json(self.model_metadata)
        for step in self.steps:
            response: dict[str, Any] = {"type": step.kind}
            if step.kind == "text":
                response["text"] = step.text
                if step.chunks is not None:
                    response["chunks"] = list(step.chunks)
                if step.usage is not None:
                    response["usage"] = dict(step.usage)
            elif step.kind == "tool_calls":
                if step.text:
                    response["text"] = step.text
                response["tool_calls"] = [
                    _tool_call_as_dict(call) for call in step.tool_calls
                ]
                if step.usage is not None:
                    response["usage"] = dict(step.usage)
            elif step.kind == "http_error":
                response["status"] = step.status
                if step.error is not None:
                    response["error"] = _thaw_json(step.error)
            elif step.kind == "connection_close":
                response["before_headers"] = step.close_before_headers
                response["after_chunks"] = step.close_after_chunks
                response["text"] = step.text
            elif step.kind == "hold":
                inner: dict[str, Any] = {"type": step.hold_response_kind}
                if step.hold_response_kind in {"text", "tool_calls"}:
                    inner["text"] = step.text
                if step.hold_response_kind == "text" and step.chunks is not None:
                    inner["chunks"] = list(step.chunks)
                if step.usage is not None:
                    inner["usage"] = dict(step.usage)
                if step.hold_response_kind == "tool_calls":
                    inner["tool_calls"] = [
                        _tool_call_as_dict(call) for call in step.tool_calls
                    ]
                if step.hold_response_kind == "http_error":
                    inner["status"] = step.status
                    if step.error is not None:
                        inner["error"] = _thaw_json(step.error)
                response["response"] = inner
                if step.hold_id:
                    response["id"] = step.hold_id
                if step.hold_timeout_seconds is not None:
                    response["timeout_seconds"] = step.hold_timeout_seconds
            item: dict[str, Any] = {"response": response}
            if step.request is not None:
                item["request"] = _thaw_json(step.request)
            result["steps"].append(item)
        return result


def _tool_call_as_dict(call: ToolCall) -> dict[str, Any]:
    result: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
    if call.id is not None:
        result["id"] = call.id
    return result


_KIND_ALIASES = {
    "message": "text",
    "content": "text",
    "tool_call": "tool_calls",
    "tools": "tool_calls",
    "error": "http_error",
    "http-error": "http_error",
    "http_error": "http_error",
    "close": "connection_close",
    "connection-close": "connection_close",
    "connection_close": "connection_close",
    "held": "hold",
    "held_response": "hold",
    "hold": "hold",
}


def _response_mapping(step: Mapping[str, Any], *, where: str) -> Mapping[str, Any]:
    value = step.get("response", step)
    response = _as_mapping(value, where=f"{where}.response")
    return response


def _parse_text_content(
    response: Mapping[str, Any], *, where: str
) -> tuple[str, tuple[str, ...] | None]:
    """Parse text and optional deterministic SSE chunks.

    ``text`` remains the canonical response content for existing scripts.  A
    script may instead provide ``chunks`` (or provide both with matching
    content) to prescribe exact ordered SSE boundaries.  Empty strings are
    valid chunks so fixtures can exercise an explicitly empty delta.
    """

    text_key = next(
        (key for key in ("text", "content", "message") if key in response), None
    )
    raw_text = response.get(text_key, "") if text_key is not None else ""
    if not isinstance(raw_text, str):
        raise ScriptValidationError(f"{where}.response.text must be a string")

    if "chunks" not in response:
        return raw_text, None
    raw_chunks = response["chunks"]
    if not isinstance(raw_chunks, list):
        raise ScriptValidationError(f"{where}.response.chunks must be an array")
    if any(not isinstance(chunk, str) for chunk in raw_chunks):
        raise ScriptValidationError(
            f"{where}.response.chunks must contain only strings"
        )
    chunks = tuple(raw_chunks)
    concatenated = "".join(chunks)
    if text_key is not None and raw_text != concatenated:
        raise ScriptValidationError(
            f"{where}.response.text must equal the concatenation of response.chunks"
        )
    return concatenated, chunks


def _parse_response_usage(
    response: Mapping[str, Any], *, where: str
) -> dict[str, int] | None:
    """Parse a response usage override without conflating omission and null."""

    if "usage" not in response:
        return None
    return _validate_usage(response["usage"], where=f"{where}.response.usage")


def _parse_step(value: Any, index: int) -> ResponseStep:
    where = f"steps[{index}]"
    item = _as_mapping(value, where=where)
    request = item.get(
        "request",
        item.get("when", item.get("expect", item.get("expected_request"))),
    )
    if request is not None:
        request = _as_mapping(request, where=f"{where}.request")

    response = _response_mapping(item, where=where)
    raw_kind = response.get("type", response.get("kind"))
    if raw_kind is None:
        # A compact text step ({"text": "..."}) is useful in tiny scripts.
        raw_kind = "tool_calls" if "tool_calls" in response else "text"
    if not isinstance(raw_kind, str):
        raise ScriptValidationError(f"{where}.response.type must be a string")
    kind = _KIND_ALIASES.get(raw_kind.lower(), raw_kind.lower())

    if kind == "text":
        text, chunks = _parse_text_content(response, where=where)
        return ResponseStep(
            kind=kind,
            request=request,
            text=text,
            chunks=chunks,
            usage=_parse_response_usage(response, where=where),
        )

    if kind == "tool_calls":
        if "chunks" in response:
            raise ScriptValidationError(
                f"{where}.response.chunks is only supported for text responses"
            )
        calls = response.get("tool_calls", response.get("calls", []))
        if not isinstance(calls, list):
            raise ScriptValidationError(f"{where}.response.tool_calls must be an array")
        parsed = tuple(
            _normalise_tool_call(
                call, where=f"{where}.response.tool_calls[{call_index}]"
            )
            for call_index, call in enumerate(calls)
        )
        if not parsed:
            raise ScriptValidationError(
                f"{where}.response.tool_calls must not be empty"
            )
        text = response.get("text", response.get("content", ""))
        if not isinstance(text, str):
            raise ScriptValidationError(f"{where}.response.text must be a string")
        return ResponseStep(
            kind=kind,
            request=request,
            text=text,
            tool_calls=parsed,
            usage=_parse_response_usage(response, where=where),
        )

    if kind == "http_error":
        if "chunks" in response:
            raise ScriptValidationError(
                f"{where}.response.chunks is only supported for text responses"
            )
        status = response.get(
            "status", response.get("status_code", response.get("code", 500))
        )
        status = _validate_status(status, where=f"{where}.response.status")
        error = response.get("error", response.get("body"))
        if error is not None and not isinstance(error, Mapping):
            raise ScriptValidationError(f"{where}.response.error must be an object")
        return ResponseStep(
            kind=kind,
            request=request,
            status=status,
            error=error,
            usage=_parse_response_usage(response, where=where),
        )

    if kind == "connection_close":
        before_headers = response.get(
            "before_headers", response.get("before_response", True)
        )
        if not isinstance(before_headers, bool):
            raise ScriptValidationError(
                f"{where}.response.before_headers must be boolean"
            )
        chunks = response.get("after_chunks", response.get("chunks", 0))
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 0:
            raise ScriptValidationError(
                f"{where}.response.after_chunks must be a non-negative integer"
            )
        text = response.get("text", "")
        if not isinstance(text, str):
            raise ScriptValidationError(f"{where}.response.text must be a string")
        return ResponseStep(
            kind=kind,
            request=request,
            text=text,
            close_before_headers=before_headers,
            close_after_chunks=chunks,
            usage=_parse_response_usage(response, where=where),
        )

    if kind == "hold":
        nested = "response" in response or "then" in response
        if nested:
            if "chunks" in response:
                raise ScriptValidationError(
                    f"{where}.response.chunks must be inside the held text response"
                )
            inner = response.get("response", response.get("then"))
        else:
            # Keep the compact held-text form backwards compatible while
            # allowing its chunks to use the same text contract.
            inner = {"type": "text"}
            if "text" in response:
                inner["text"] = response["text"]
            if "chunks" in response:
                inner["chunks"] = response["chunks"]
        if "usage" in response:
            if isinstance(inner, Mapping) and "usage" in inner:
                raise ScriptValidationError(
                    f"{where}.response.usage must be specified once"
                )
            if isinstance(inner, Mapping):
                inner = dict(inner)
                inner["usage"] = response["usage"]
        inner_item = {"response": inner}
        inner_step = _parse_step(inner_item, index)
        if inner_step.kind in {"hold", "connection_close"}:
            raise ScriptValidationError(
                f"{where}.response.response cannot be {inner_step.kind}"
            )
        hold_id = response.get("id", response.get("hold_id"))
        if hold_id is not None and (not isinstance(hold_id, str) or not hold_id):
            raise ScriptValidationError(
                f"{where}.response.id must be a non-empty string when present"
            )
        timeout = response.get("timeout_seconds", response.get("timeout"))
        timeout = _validate_timeout(timeout, where=f"{where}.response.timeout_seconds")
        return ResponseStep(
            kind=kind,
            request=request,
            text=inner_step.text,
            tool_calls=inner_step.tool_calls,
            status=inner_step.status,
            error=inner_step.error,
            hold_id=hold_id,
            hold_timeout_seconds=timeout,
            hold_response_kind=inner_step.kind,
            chunks=inner_step.chunks,
            usage=inner_step.usage,
        )

    raise ScriptValidationError(
        f"{where}.response.type {raw_kind!r} is unsupported; expected text, tool_calls, "
        "http_error, connection_close, or hold"
    )


def parse_script(value: Any, *, allow_default_version: bool = True) -> Script:
    """Validate and convert a mapping into an immutable :class:`Script`.

    ``allow_default_version`` is intentionally true for arm requests from
    older test harnesses that supplied only ``steps``.  Every resulting script
    still carries ``schema_version: 1`` in state and response metadata.
    """

    if isinstance(value, Script):
        try:
            # Re-parse the canonical public representation so callers cannot
            # bypass mapping validation by constructing a dataclass directly.
            return parse_script(value.as_dict(), allow_default_version=False)
        except ScriptValidationError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScriptValidationError(f"script is invalid: {exc}") from exc
    if isinstance(value, (str, bytes, bytearray)):
        import json

        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ScriptValidationError(f"script is not valid JSON: {exc.msg}") from exc
    root = _as_mapping(value, where="script")
    raw_version = root.get("schema_version", root.get("version"))
    if raw_version is None and allow_default_version:
        raw_version = SCRIPT_SCHEMA_VERSION
    if (
        isinstance(raw_version, bool)
        or not isinstance(raw_version, int)
        or raw_version != SCRIPT_SCHEMA_VERSION
    ):
        raise ScriptValidationError(
            f"script schema_version {raw_version!r} is unsupported; expected {SCRIPT_SCHEMA_VERSION}"
        )

    raw_steps = root.get("steps", root.get("responses"))
    if not isinstance(raw_steps, list):
        raise ScriptValidationError("script.steps must be an array")
    steps = tuple(_parse_step(step, index) for index, step in enumerate(raw_steps))

    model = root.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model:
        raise ScriptValidationError("script.model must be a non-empty string")
    raw_models = root.get("models", [model])
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    if (
        not isinstance(raw_models, list)
        or not raw_models
        or any(not isinstance(item, str) or not item for item in raw_models)
    ):
        raise ScriptValidationError(
            "script.models must be a non-empty array of strings"
        )

    created = root.get("created", FIXED_CREATED)
    if isinstance(created, bool) or not isinstance(created, int) or created < 0:
        raise ScriptValidationError("script.created must be a non-negative integer")
    metadata = root.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ScriptValidationError("script.metadata must be an object")
    model_metadata = root.get("model_metadata", {})
    _validate_model_metadata(model_metadata, model_ids=tuple(raw_models))
    return Script(
        steps=steps,
        schema_version=SCRIPT_SCHEMA_VERSION,
        model=model,
        models=tuple(raw_models),
        created=created,
        metadata=dict(metadata),
        model_metadata=dict(model_metadata),
    )


def matches_request(
    expected: Mapping[str, Any] | None, actual: Mapping[str, Any]
) -> bool:
    """Return whether ``actual`` contains the recursively expected subset."""

    if expected is None:
        return True
    # The natural script form is ``request: {messages: [...]}`` and should
    # match the decoded chat body directly.  Explicit transport keys opt into
    # matching the complete envelope, e.g. ``{method: POST, json: {...}}``.
    transport_keys = {"method", "path", "headers", "json", "body"}
    explicit_transport_keys = {"method", "path", "json", "body"}
    if not transport_keys.intersection(expected):
        body = actual.get("json", {})
        return _subset_matches(expected, body)

    # Header assertions can accompany direct body fields without forcing a
    # fixture to duplicate the ``json`` wrapper.
    if "headers" in expected and not explicit_transport_keys.intersection(expected):
        body_expected = {
            key: value for key, value in expected.items() if key != "headers"
        }
        return _subset_matches(
            expected["headers"], actual.get("headers", {})
        ) and _subset_matches(body_expected, actual.get("json", {}))

    normalized = dict(expected)
    # ``body`` is a convenient alias for the actual envelope's ``json`` key,
    # including when it appears next to an explicit method/path assertion.
    if "body" in normalized and "json" not in normalized:
        normalized["json"] = normalized.pop("body")
    return _subset_matches(normalized, actual)


def _subset_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _subset_matches(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, list) or len(expected) > len(actual):
            return False
        return all(
            _subset_matches(item, actual[index]) for index, item in enumerate(expected)
        )
    # Python treats bool as a subclass of int, but JSON has distinct boolean
    # and number types. Keep numeric equality (1 == 1.0) while refusing the
    # bool/number coercion in either direction, including nested values.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is bool and type(actual) is bool and expected == actual
    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        return (
            isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and expected == actual
        )
    return type(expected) is type(actual) and expected == actual
