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
