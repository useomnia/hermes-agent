"""Bound pending tool calls when a Streamable HTTP session is throttled."""

import asyncio
import json
from typing import Any, Callable, Coroutine, TypeVar

import httpx


_Result = TypeVar("_Result")


class MCPRateLimitedError(RuntimeError):
    """The server rejected an HTTP tool request without a replay."""


class HTTPToolCallGuard:
    """Wake callers before HTTP 429 tears down their shared SDK session.

    The guard belongs to one session. A completed result wins a concurrent
    failure; other callers fail because the transport is closing underneath
    them. No operation is resubmitted.
    """

    def __init__(self) -> None:
        self._failure: asyncio.Future[MCPRateLimitedError] = asyncio.get_running_loop().create_future()

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

    async def call(self, factory: Callable[[], Coroutine[Any, Any, _Result]]) -> _Result:
        if self._failure.done():
            raise self._failure.result()
        pending = asyncio.create_task(factory())
        try:
            done, _ = await asyncio.wait(
                (pending, self._failure), return_when=asyncio.FIRST_COMPLETED,
            )
            if pending in done:
                return pending.result()
            raise self._failure.result()
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
