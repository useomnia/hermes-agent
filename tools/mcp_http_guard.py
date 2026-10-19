"""Bound pending tool calls when a Streamable HTTP session fails."""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Optional, TypeVar

from tools.mcp_tool import sdk_httpx

# The client below is handed to the SDK's transport, so it must come from the
# httpx flavour the installed SDK imports (httpx2 on mcp >= 2.0).
httpx = sdk_httpx()
if httpx is None:  # pragma: no cover - no SDK, so no HTTP MCP session either
    import httpx


logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")

# JSON-RPC code the SDK resolves a waiting request with once its connection
# closes (mcp.types.CONNECTION_CLOSED), and the messages it uses when the
# response stream of a sent request breaks.
_JSONRPC_CONNECTION_CLOSED = -32000
_SDK_STREAM_ABANDONED_PREFIXES = ("SSE stream failed: ", "SSE stream ended ")


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


_JSONRPC_INTERNAL_ERROR = -32603
_JSONRPC_TRANSPORT_ERROR = -32000


def _jsonrpc_message(request: httpx.Request) -> Optional[dict]:
    if request.method != "POST":
        return None
    try:
        message = json.loads(request.content)
    except (ValueError, httpx.RequestNotRead):
        return None
    return message if isinstance(message, dict) else None


def _jsonrpc_method(request: httpx.Request) -> Optional[str]:
    method = (_jsonrpc_message(request) or {}).get("method")
    return method if isinstance(method, str) else None


def _tool_call_request_id(request: httpx.Request) -> Any:
    """Return the JSON-RPC id of a ``tools/call`` POST, else None."""
    message = _jsonrpc_message(request)
    if message is None or message.get("method") != "tools/call":
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


# Transport errors raised before the request left Hermes: the outcome is known.
_NOT_SENT_TRANSPORT_ERRORS = frozenset({"ConnectError", "ConnectTimeout", "PoolTimeout"})


def outcome_may_be_unknown(error: BaseException) -> bool:
    """Return True if the call failed after its request may have reached the server."""
    if isinstance(error, MCPTransportFailedError):
        return True
    name = (_isolated_failure_data(error) or {}).get("transportError")
    return name is not None and name not in _NOT_SENT_TRANSPORT_ERRORS


def is_isolated_request_failure(error: BaseException) -> bool:
    """Return True if this tool call's own HTTP request failed (the session is intact)."""
    return _isolated_failure_data(error) is not None


class RequestScopedFailureClient(httpx.AsyncClient):
    """Fail one tool call's HTTP request on its own and name its failure.

    MCP SDK 2.x fails a request that gets an error status on its own, but as a
    generic "Server returned an error response"; a transport error (connection
    reset, broken pipe) on one POST still ends the whole session and fails
    every call in flight beside it. A failed ``tools/call`` POST is instead
    answered with that call's own JSON-RPC error naming its status or transport
    error, so the call fails alone and is never replayed. An HTTP 404 (session
    expiry) and lifecycle requests keep the SDK's own handling.
    """

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        request_id = _tool_call_request_id(request)
        if request_id is None:
            response = await super().send(request, **kwargs)
            if request.method == "POST" and response.status_code >= 400:
                # The SDK reports this only as "Server returned an error
                # response"; name the status so a failing endpoint is visible.
                logger.warning(
                    "MCP HTTP %s %s answered HTTP %s %s",
                    _jsonrpc_method(request) or "request", request.url,
                    response.status_code, response.reason_phrase,
                )
            return response
        try:
            response = await super().send(request, **kwargs)
        except httpx.TransportError as exc:
            return _tool_call_error(
                request, request_id, _JSONRPC_TRANSPORT_ERROR,
                f"MCP HTTP request failed ({type(exc).__name__}: {exc})".replace(": )", ")"),
                {"transportError": type(exc).__name__},
            )
        if response.status_code < 400 or response.status_code == 404:
            return response
        await response.aclose()
        return _tool_call_error(
            request, request_id, _JSONRPC_INTERNAL_ERROR,
            f"MCP server answered HTTP {response.status_code} {response.reason_phrase}".rstrip(),
            {"httpStatus": response.status_code},
        )


def _closed_with_its_connection(error: BaseException) -> bool:
    """Return True if the SDK, not the server, resolved this call because its connection closed.

    Servers also answer with -32000, so the SDK's own closure is recognised by
    its fixed messages and absent data; a server's error keeps its handling.
    """
    data = getattr(error, "error", None)
    if getattr(data, "code", None) != _JSONRPC_CONNECTION_CLOSED or getattr(data, "data", None) is not None:
        return False
    message = getattr(data, "message", "")
    return message == "Connection closed" or message.startswith(_SDK_STREAM_ABANDONED_PREFIXES)


class HTTPToolCallGuard:
    """Fail calls in flight on a session whose transport ended, without a replay.

    The guard belongs to one session. When its transport ends, every call
    still waiting on it fails at once, and a call the SDK resolved as
    "Connection closed" is reported the same way: either may already have
    reached the server, so neither is resubmitted. A completed result wins a
    concurrent failure.
    """

    def __init__(self) -> None:
        self._failure: asyncio.Future[RuntimeError] = asyncio.get_running_loop().create_future()
        self._pending = 0

    def close(self, failure: Optional[BaseException] = None) -> int:
        """Fail the calls still waiting on this session; return how many there were."""
        if not self._failure.done():
            self._failure.set_result(MCPTransportFailedError(
                f"MCP HTTP transport failed before the call completed "
                f"({describe_transport_failure(failure)}); the request was not replayed."
            ))
        return self._pending

    async def call(self, factory: Callable[[], Coroutine[Any, Any, _Result]]) -> _Result:
        # A call that starts after the transport ended was never sent; the
        # SDK's closed-session error keeps its existing reconnect and retry.
        started_live = not self._failure.done()
        pending = asyncio.create_task(factory())
        waiters = {pending, self._failure} if started_live else {pending}
        self._pending += 1
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if pending not in done:
                raise self._failure.result()
            error = pending.exception()
            if error is not None and started_live and _closed_with_its_connection(error):
                raise MCPTransportFailedError(
                    "MCP HTTP transport closed before the call completed; "
                    "the request was not replayed."
                ) from error
            return pending.result()
        finally:
            self._pending -= 1
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
