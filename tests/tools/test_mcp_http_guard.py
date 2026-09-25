import asyncio
import json

import httpx
import pytest

from tools.mcp_http_guard import HTTPToolCallGuard, MCPRateLimitedError


@pytest.mark.asyncio
async def test_http_429_wakes_all_pending_calls_without_replay():
    guard = HTTPToolCallGuard()
    entered = []
    stopped = []
    requests = []

    async def waiting(index):
        entered.append(index)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(index)

    async def respond(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "1"})

    calls = [asyncio.create_task(guard.call(lambda i=i: waiting(i))) for i in range(2)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks={"response": [guard.observe_response]},
    ) as client:
        await client.post("https://mcp.example/mcp", json={"method": "tools/call", "id": 1})
    results = await asyncio.wait_for(asyncio.gather(*calls, return_exceptions=True), 1)
    assert sorted(entered) == sorted(stopped) == [0, 1]
    assert all(isinstance(result, MCPRateLimitedError) for result in results)
    assert len(requests) == 1
    with pytest.raises(MCPRateLimitedError, match="429"):
        await guard.call(lambda: waiting(2))
    assert entered == [0, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,method,payload", [
    (200, "POST", {"method": "tools/call"}),
    (401, "POST", {"method": "tools/call"}),
    (403, "POST", {"method": "tools/call"}),
    (500, "POST", {"method": "tools/call"}),
    (429, "GET", {"method": "tools/call"}),
    (429, "POST", {"method": "ping"}),
    (429, "POST", []),
])
async def test_other_responses_preserve_their_existing_handling(status, method, payload):
    guard = HTTPToolCallGuard()
    response = httpx.Response(status, request=httpx.Request(
        method, "https://mcp.example/mcp", content=json.dumps(payload),
    ))
    await guard.observe_response(response)
    assert await guard.call(lambda: asyncio.sleep(0, result="unchanged")) == "unchanged"


@pytest.mark.asyncio
async def test_malformed_or_unread_request_cannot_trip_the_guard():
    guard = HTTPToolCallGuard()
    for request in [
        httpx.Request("POST", "https://mcp.example/mcp", content=b"invalid"),
        httpx.Request("POST", "https://mcp.example/mcp", stream=httpx.ByteStream(b"{}")),
    ]:
        await guard.observe_response(httpx.Response(429, request=request))
    assert await guard.call(lambda: asyncio.sleep(0, result=True))


@pytest.mark.asyncio
async def test_one_cancelled_caller_does_not_cancel_the_shared_guard():
    guard = HTTPToolCallGuard()
    call = asyncio.create_task(guard.call(lambda: asyncio.Event().wait()))
    await asyncio.sleep(0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert await guard.call(lambda: asyncio.sleep(0, result="healthy")) == "healthy"


@pytest.mark.asyncio
async def test_new_session_does_not_inherit_the_failed_session():
    old = HTTPToolCallGuard()
    request = httpx.Request("POST", "https://mcp.example/mcp", json={"method": "tools/call"})
    await old.observe_response(httpx.Response(429, request=request))
    new = HTTPToolCallGuard()
    assert await new.call(lambda: asyncio.sleep(0, result="recovered")) == "recovered"


@pytest.mark.asyncio
async def test_completed_result_wins_a_simultaneous_transport_failure():
    guard = HTTPToolCallGuard()

    async def completed():
        request = httpx.Request("POST", "https://mcp.example/mcp", json={"method": "tools/call"})
        await guard.observe_response(httpx.Response(429, request=request))
        return "already completed"

    assert await guard.call(completed) == "already completed"
