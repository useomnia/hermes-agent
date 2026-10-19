import asyncio
import json

import pytest
from anyio import ClosedResourceError

from tools.mcp_http_guard import (
    HTTPToolCallGuard,
    MCPTransportFailedError,
    RequestScopedFailureClient,
    describe_transport_failure,
    httpx,
)


def _http_error(status):
    request = httpx.Request("POST", "https://mcp.example/mcp", json={"method": "tools/call"})
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


@pytest.mark.asyncio
async def test_completed_result_wins_a_transport_failure_that_follows():
    guard = HTTPToolCallGuard()

    async def completed():
        guard.close(ExceptionGroup("unhandled errors in a TaskGroup", [_http_error(502)]))
        return "already completed"

    assert await guard.call(completed) == "already completed"


def _connection_closed():
    from mcp.shared.exceptions import MCPError

    return MCPError(code=-32000, message="Connection closed")


@pytest.mark.asyncio
async def test_call_the_sdk_resolved_as_connection_closed_is_not_replayed():
    """SDK 2.x wakes in-flight calls with "Connection closed" when the
    transport ends; the call may already have reached the server."""
    from tools.mcp_tool import _is_session_expired_error

    guard = HTTPToolCallGuard()

    async def closed_in_flight():
        raise _connection_closed()

    with pytest.raises(MCPTransportFailedError, match="not replayed") as raised:
        await guard.call(closed_in_flight)
    assert _is_session_expired_error(raised.value) is False


@pytest.mark.asyncio
async def test_call_started_after_the_transport_ended_keeps_the_sdk_error():
    guard = HTTPToolCallGuard()
    guard.close()

    async def closed_session():
        raise _connection_closed()

    with pytest.raises(Exception, match="Connection closed") as raised:
        await guard.call(closed_session)
    assert not isinstance(raised.value, MCPTransportFailedError)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "SSE stream failed: ReadError",
    "SSE stream ended without a response",
    "SSE stream ended and reconnection attempts were exhausted",
])
async def test_call_whose_response_stream_broke_is_not_replayed(message):
    from mcp.shared.exceptions import MCPError

    guard = HTTPToolCallGuard()

    async def abandoned():
        raise MCPError(code=-32000, message=message)

    with pytest.raises(MCPTransportFailedError, match="not replayed"):
        await guard.call(abandoned)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "Quota exhausted: upgrade your plan",
    "Invalid or expired session",
])
async def test_server_error_with_the_same_code_keeps_its_own_error(message):
    from mcp.shared.exceptions import MCPError

    guard = HTTPToolCallGuard()

    async def server_error():
        raise MCPError(code=-32000, message=message)

    with pytest.raises(MCPError, match=message) as raised:
        await guard.call(server_error)
    assert not isinstance(raised.value, MCPTransportFailedError)


@pytest.mark.asyncio
async def test_server_connection_closed_error_with_data_keeps_its_own_error():
    from mcp.shared.exceptions import MCPError

    guard = HTTPToolCallGuard()

    async def server_error():
        raise MCPError(code=-32000, message="Connection closed", data={"detail": "upstream"})

    with pytest.raises(MCPError) as raised:
        await guard.call(server_error)
    assert not isinstance(raised.value, MCPTransportFailedError)


@pytest.mark.asyncio
async def test_isolated_transport_error_keeps_its_own_error():
    from mcp.shared.exceptions import MCPError

    guard = HTTPToolCallGuard()

    async def isolated():
        raise MCPError(code=-32000, message="MCP HTTP request failed (ReadError)",
                       data={"transportError": "ReadError"})

    with pytest.raises(MCPError, match="ReadError"):
        await guard.call(isolated)


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
@pytest.mark.parametrize("status", [500, 502, 503, 504, 401, 429])
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
    assert error["data"] == {"transportError": "ReadError"}


@pytest.mark.asyncio
async def test_failed_lifecycle_request_names_its_status_in_the_log(caplog):
    async with _isolating_client(503) as client:
        with caplog.at_level("WARNING", logger="tools.mcp_http_guard"):
            response = await client.post(
                "https://mcp.example/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
    assert response.status_code == 503
    assert "MCP HTTP tools/list https://mcp.example/mcp answered HTTP 503" in caplog.text


class _IsolatedFailure(Exception):
    def __init__(self, data):
        super().__init__("isolated")
        self.error = type("ErrorData", (), {"data": data})()


@pytest.mark.parametrize("failure,unknown", [
    (MCPTransportFailedError("ended"), True),
    (_IsolatedFailure({"transportError": "ReadError"}), True),
    (_IsolatedFailure({"transportError": "WriteError"}), True),
    (_IsolatedFailure({"transportError": "ReadTimeout"}), True),
    (_IsolatedFailure({"transportError": "ConnectError"}), False),
    (_IsolatedFailure({"transportError": "ConnectTimeout"}), False),
    (_IsolatedFailure({"httpStatus": 502}), False),
    (RuntimeError("tool raised"), False),
])
def test_outcome_is_unknown_only_when_the_request_may_have_reached_the_server(failure, unknown):
    from tools.mcp_http_guard import outcome_may_be_unknown

    assert outcome_may_be_unknown(failure) is unknown
