import asyncio
import json

import httpx
import pytest
from anyio import ClosedResourceError

from tools.mcp_http_guard import (
    HTTPToolCallGuard,
    MCPRateLimitedError,
    MCPTransportFailedError,
    RequestScopedFailureClient,
    describe_transport_failure,
    gateway_status_of,
)


def _http_error(status):
    request = httpx.Request("POST", "https://mcp.example/mcp", json={"method": "tools/call"})
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


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


@pytest.mark.asyncio
async def test_transport_failure_wakes_pending_calls_with_its_cause_without_replay():
    guard = HTTPToolCallGuard()
    started = []
    stopped = []

    async def waiting(index):
        started.append(index)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(index)

    calls = [asyncio.create_task(guard.call(lambda i=i: waiting(i))) for i in range(3)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    failure = ExceptionGroup("unhandled errors in a TaskGroup", [_http_error(502)])
    assert guard.close(failure) == 3
    results = await asyncio.wait_for(asyncio.gather(*calls, return_exceptions=True), 1)
    assert sorted(started) == sorted(stopped) == [0, 1, 2]
    assert all(isinstance(result, MCPTransportFailedError) for result in results)
    assert "HTTP 502 Bad Gateway" in str(results[0])
    assert "not replayed" in str(results[0])
    assert started == [0, 1, 2]


@pytest.mark.asyncio
async def test_call_started_after_the_transport_ended_reaches_the_closed_session():
    guard = HTTPToolCallGuard()
    guard.close(ExceptionGroup("unhandled errors in a TaskGroup", [_http_error(502)]))

    async def closed_session():
        raise ClosedResourceError

    with pytest.raises(ClosedResourceError):
        await guard.call(closed_session)


@pytest.mark.asyncio
async def test_closing_an_idle_session_reports_no_stranded_calls():
    guard = HTTPToolCallGuard()
    assert await guard.call(lambda: asyncio.sleep(0, result="done")) == "done"
    assert guard.close() == 0


@pytest.mark.asyncio
async def test_rate_limit_keeps_its_error_when_the_transport_then_ends():
    guard = HTTPToolCallGuard()
    call = asyncio.create_task(guard.call(lambda: asyncio.Event().wait()))
    await asyncio.sleep(0)
    request = httpx.Request("POST", "https://mcp.example/mcp", json={"method": "tools/call"})
    await guard.observe_response(httpx.Response(429, request=request))
    guard.close(ExceptionGroup("unhandled errors in a TaskGroup", [_http_error(429)]))
    with pytest.raises(MCPRateLimitedError):
        await asyncio.wait_for(call, 1)


@pytest.mark.parametrize("failure,description", [
    (None, "the session closed"),
    (ExceptionGroup("group", [_http_error(503)]), "HTTP 503 Service Unavailable"),
    (ExceptionGroup("group", [ExceptionGroup("inner", [httpx.ReadError("reset")])]), "ReadError: reset"),
    (RuntimeError("wrapped"), "RuntimeError"),
    (ExceptionGroup("group", [ValueError("leaf")]), "ValueError"),
])
def test_transport_failure_names_the_http_status_or_transport_error(failure, description):
    assert describe_transport_failure(failure) == description


def test_transport_failure_found_through_exception_chaining():
    try:
        try:
            raise httpx.ConnectError("refused")
        except httpx.ConnectError as exc:
            raise RuntimeError("wrapped") from exc
    except RuntimeError as exc:
        assert describe_transport_failure(exc) == "ConnectError: refused"


def _isolating_client(status):
    async def respond(request):
        return httpx.Response(status, text="upstream says no")

    return RequestScopedFailureClient(transport=httpx.MockTransport(respond))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503, 504, 401])
async def test_failed_tool_call_is_answered_with_its_own_jsonrpc_error(status):
    async with _isolating_client(status) as client:
        response = await client.post(
            "https://mcp.example/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "tools/call"},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["id"] == 7
    assert f"HTTP {status}" in body["error"]["message"]
    assert body["error"]["data"] == {"httpStatus": status}


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload", [
    (404, {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}),
    (429, {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}),
    (502, {"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    (502, {"jsonrpc": "2.0", "method": "notifications/initialized"}),
])
async def test_other_requests_keep_their_existing_handling(status, payload):
    async with _isolating_client(status) as client:
        response = await client.post("https://mcp.example/mcp", json=payload)
    assert response.status_code == status


@pytest.mark.asyncio
async def test_transport_error_on_a_tool_call_is_answered_without_raising():
    async def reset(request):
        raise httpx.ReadError("connection reset by peer")

    async with RequestScopedFailureClient(transport=httpx.MockTransport(reset)) as client:
        response = await client.post(
            "https://mcp.example/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call"},
        )
    error = response.json()["error"]
    assert "ReadError: connection reset by peer" in error["message"]
    assert gateway_status_of(_McpLikeError(error["data"])) is None


class _McpLikeError(Exception):
    def __init__(self, data):
        super().__init__("error")
        self.error = type("ErrorData", (), {"data": data})()


@pytest.mark.parametrize("data,status", [
    ({"httpStatus": 502}, 502),
    ({"httpStatus": 503}, 503),
    ({"httpStatus": 504}, 504),
    ({"httpStatus": 500}, None),
    ({"httpStatus": 401}, None),
    (None, None),
])
def test_only_gateway_statuses_are_retryable(data, status):
    assert gateway_status_of(_McpLikeError(data)) == status
