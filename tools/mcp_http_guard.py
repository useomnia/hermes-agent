"""Bound pending tool calls when a Streamable HTTP session fails."""

import asyncio
import json
from typing import Any, Callable, Coroutine, Optional, TypeVar

import httpx


_Result = TypeVar("_Result")


class MCPRateLimitedError(RuntimeError):
    """The server rejected an HTTP tool request without a replay."""


class MCPTransportFailedError(RuntimeError):
    """The session's transport ended before the call completed; it was not replayed."""


def describe_transport_failure(failure: Optional[BaseException]) -> str:
    """Name the HTTP status or transport error that ended a session."""
    if failure is None:
        return "the session closed"
    pending: list[Optional[BaseException]] = [failure]
    seen: set[int] = set()
    first_leaf: Optional[BaseException] = None
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            response = current.response
            return f"HTTP {response.status_code} {response.reason_phrase}".rstrip()
        if isinstance(current, httpx.TransportError):
            return f"{type(current).__name__}: {current}".rstrip(": ")
        children = getattr(current, "exceptions", ())
        if not children and first_leaf is None:
            first_leaf = current
        pending.extend(children)
        pending.extend((current.__cause__, current.__context__))
    return type(first_leaf or failure).__name__


# A gateway between Hermes and the MCP server failed; the server may never
# have seen the request. Only these statuses make a read-only call retryable.
RETRYABLE_GATEWAY_STATUSES = frozenset({502, 503, 504})
_JSONRPC_INTERNAL_ERROR = -32603
_JSONRPC_TRANSPORT_ERROR = -32000


def _tool_call_request_id(request: httpx.Request) -> Any:
    """Return the JSON-RPC id of a ``tools/call`` POST, else None."""
    if request.method != "POST":
        return None
    try:
        message = json.loads(request.content)
    except (ValueError, httpx.RequestNotRead):
        return None
    if not isinstance(message, dict) or message.get("method") != "tools/call":
        return None
    return message.get("id")


def _tool_call_error(request: httpx.Request, request_id: Any, code: int, message: str, data: dict) -> httpx.Response:
    return httpx.Response(200, request=request, json={
        "jsonrpc": "2.0", "id": request_id,
        "error": {"code": code, "message": message, "data": data},
    })


def _isolated_failure_data(error: BaseException) -> Optional[dict]:
    data = getattr(getattr(error, "error", None), "data", None)
    if isinstance(data, dict) and ("httpStatus" in data or "transportError" in data):
        return data
    return None


def is_isolated_request_failure(error: BaseException) -> bool:
    """Return True if this tool call's own HTTP request failed (the session is intact)."""
    return _isolated_failure_data(error) is not None


def gateway_status_of(error: BaseException) -> Optional[int]:
    """Return the retryable gateway status an isolated tool call failed with."""
    status = (_isolated_failure_data(error) or {}).get("httpStatus")
    return status if status in RETRYABLE_GATEWAY_STATUSES else None


class RequestScopedFailureClient(httpx.AsyncClient):
    """Fail one tool call's HTTP request on its own, keeping the session open.

    MCP SDK 1.x tears the whole Streamable HTTP session down when one request
    gets a non-2xx status or a transport error, failing every call in flight
    beside it. As SDK 2.x does, a failed ``tools/call`` POST is instead answered
    with that call's own JSON-RPC error. An HTTP 404 (session expiry) and 429
    (HTTPToolCallGuard) keep their existing handling, as do lifecycle requests.
    """

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        request_id = _tool_call_request_id(request)
        if request_id is None:
            return await super().send(request, **kwargs)
        try:
            response = await super().send(request, **kwargs)
        except httpx.TransportError as exc:
            return _tool_call_error(
                request, request_id, _JSONRPC_TRANSPORT_ERROR,
                f"MCP HTTP request failed ({type(exc).__name__}: {exc})".replace(": )", ")"),
                {"transportError": type(exc).__name__},
            )
        if response.status_code < 400 or response.status_code in (404, 429):
            return response
        await response.aclose()
        return _tool_call_error(
            request, request_id, _JSONRPC_INTERNAL_ERROR,
            f"MCP server answered HTTP {response.status_code} {response.reason_phrase}".rstrip(),
            {"httpStatus": response.status_code},
        )


class HTTPToolCallGuard:
    """Wake callers before a failed transport strands them on its shared session.

    The guard belongs to one session. MCP SDK 1.x tears the whole session down
    when one HTTP request fails and never answers the other requests still in
    flight, so their callers would wait for the full tool timeout. An HTTP 429
    fails them as soon as it is observed; any other failure fails the calls in
    flight when the transport ends. A completed result wins a concurrent
    failure. No operation is resubmitted.
    """

    def __init__(self) -> None:
        self._failure: asyncio.Future[RuntimeError] = asyncio.get_running_loop().create_future()
        self._pending = 0

    async def observe_response(self, response: httpx.Response) -> None:
        if response.status_code != 429 or response.request.method != "POST":
            return
        try:
            request = json.loads(response.request.content)
        except (ValueError, httpx.RequestNotRead):
            return
        if not isinstance(request, dict) or request.get("method") != "tools/call":
            return
        if not self._failure.done():
            self._failure.set_result(MCPRateLimitedError(
                "MCP HTTP 429 Too Many Requests; the request was not replayed."
            ))

    def close(self, failure: Optional[BaseException] = None) -> int:
        """Fail the calls still waiting on this session; return how many there were."""
        if not self._failure.done():
            self._failure.set_result(MCPTransportFailedError(
                f"MCP HTTP transport failed before the call completed "
                f"({describe_transport_failure(failure)}); the request was not replayed."
            ))
        return self._pending

    async def call(self, factory: Callable[[], Coroutine[Any, Any, _Result]]) -> _Result:
        failure = self._failure.result() if self._failure.done() else None
        if isinstance(failure, MCPRateLimitedError):
            raise failure
        pending = asyncio.create_task(factory())
        # A call that starts after the transport ended was never sent; the
        # SDK's closed-session error keeps its existing reconnect and retry.
        waiters = {pending} if failure is not None else {pending, self._failure}
        self._pending += 1
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                return pending.result()
            raise self._failure.result()
        finally:
            self._pending -= 1
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
