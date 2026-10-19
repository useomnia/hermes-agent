"""A failed Streamable HTTP request must not take the session's other calls with it.

MCP SDK 1.x tore the whole session down when one HTTP request failed and
never answered the requests still in flight, so every concurrent parallel read
waited for the full tool timeout. SDK 2.x isolates a failed status itself but
drops it, and resolves calls in flight on a closed session as "Connection
closed", which must not be replayed. A failed tool call now fails on its own, a
known read-only call is retried once after a gateway error, and a session that
does end fails its in-flight calls at once instead of stranding them.
"""

import asyncio
import concurrent.futures
import json
import threading
import time

import pytest

pytest.importorskip("mcp.client.streamable_http")

from tools.mcp_http_guard import httpx  # the SDK's httpx flavour (httpx2 on mcp 2.x)

TOOL_TIMEOUT = 30


class _MockMCPServer:
    """Stateless JSON Streamable HTTP server whose tools fail on request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.initialized = 0
        self.attempts = {}
        self.hanging = 0

    def _attempt(self, brand):
        with self.lock:
            self.attempts[brand] = self.attempts.get(brand, 0) + 1
            return self.attempts[brand]

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == "initialize":
            with self.lock:
                self.initialized += 1
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock", "version": "1"},
            }
        elif method == "tools/list":
            schema = {"type": "object", "properties": {"brand": {"type": "string"}}}
            result = {"tools": [
                {"name": "get-brand", "inputSchema": schema,
                 "annotations": {"readOnlyHint": True, "destructiveHint": False}},
                {"name": "update-brand", "inputSchema": schema,
                 "annotations": {"readOnlyHint": False, "destructiveHint": True}},
            ]}
        elif method == "tools/call":
            brand = message["params"]["arguments"]["brand"]
            attempt = self._attempt(brand)
            if brand == "hang":
                with self.lock:
                    self.hanging += 1
                await asyncio.Event().wait()
            if brand == "slow":
                await asyncio.sleep(0.5)
            if brand == "bad" or (brand == "flaky" and attempt == 1):
                return httpx.Response(502, text="Bad Gateway")
            if brand == "unavailable":
                return httpx.Response(503, text="Service Unavailable")
            if brand == "reset":
                raise httpx.ReadError("connection reset by peer")
            if brand == "pipe":
                raise httpx.WriteError("[Errno 32] Broken pipe")
            result = {"content": [{"type": "text", "text": brand}], "isError": False}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})


def _wait_for(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.02)


@pytest.fixture
def mock_server(monkeypatch, tmp_path):
    from tools import mcp_http_guard, mcp_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    server = _MockMCPServer()
    transport = httpx.MockTransport(server.handle)

    class _MockedClient(mcp_http_guard.RequestScopedFailureClient):
        def __init__(self, **kwargs):
            super().__init__(transport=transport, **kwargs)

    monkeypatch.setattr(mcp_http_guard, "RequestScopedFailureClient", _MockedClient)
    config = {"mockhttp": {
        "url": "https://mcp.example/mcp",
        "timeout": TOOL_TIMEOUT,
        "skip_preflight": True,
        "supports_parallel_read_tool_calls": True,
        "parallel_read_tool_call_limit": 8,
        "sampling": {"enabled": False},
        "elicitation": {"enabled": False},
    }}
    try:
        assert mcp_tool.register_mcp_servers(config)
        yield server
    finally:
        mcp_tool.shutdown_mcp_servers()


def _handler(tool="get-brand"):
    from tools import mcp_tool

    return mcp_tool._make_tool_handler("mockhttp", tool, TOOL_TIMEOUT)


def test_gateway_error_fails_only_its_own_read_and_retries_it_once(mock_server):
    read = _handler()
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        started = time.monotonic()
        beside = [pool.submit(read, {"brand": "slow"}) for _ in range(3)]
        failed = pool.submit(read, {"brand": "bad"})
        flaky = pool.submit(read, {"brand": "flaky"})
        results = [future.result(timeout=TOOL_TIMEOUT) for future in [*beside, failed, flaky]]
        elapsed = time.monotonic() - started

    assert elapsed < 10, f"reads waited {elapsed:.1f}s"
    for result in results[:3]:
        assert json.loads(result) == {"result": "slow"}
    assert "HTTP 502 Bad Gateway" in json.loads(results[3])["error"]
    assert json.loads(results[4]) == {"result": "flaky"}
    assert mock_server.attempts == {"slow": 3, "bad": 2, "flaky": 2}
    assert mock_server.initialized == 1


def test_write_is_never_retried_after_a_gateway_error(mock_server):
    result = json.loads(_handler("update-brand")({"brand": "unavailable"}))
    assert "HTTP 503 Service Unavailable" in result["error"]
    assert mock_server.attempts == {"unavailable": 1}
    assert mock_server.initialized == 1


def test_transport_error_fails_only_its_own_call_without_a_retry(mock_server):
    read = _handler()
    result = json.loads(read({"brand": "reset"}))
    assert "ReadError" in result["error"]
    assert mock_server.attempts == {"reset": 1}
    assert json.loads(read({"brand": "ok"})) == {"result": "ok"}
    assert mock_server.initialized == 1


def test_session_that_ends_fails_its_in_flight_calls_promptly(mock_server):
    from tools import mcp_tool

    read = _handler()
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        in_flight = [pool.submit(read, {"brand": "hang"}) for _ in range(3)]
        _wait_for(lambda: mock_server.hanging == 3)
        started = time.monotonic()
        assert mcp_tool._signal_reconnect(mcp_tool._servers["mockhttp"])
        stranded = [future.result(timeout=TOOL_TIMEOUT) for future in in_flight]
        elapsed = time.monotonic() - started

    assert elapsed < 10, f"calls waited {elapsed:.1f}s for a dead session"
    for result in stranded:
        assert "not replayed" in json.loads(result)["error"]
    assert mock_server.attempts == {"hang": 3}
    _wait_for(lambda: mock_server.initialized == 2)
    _wait_for(lambda: "error" not in json.loads(read({"brand": "ok"})))


def test_isolated_failure_that_sounds_like_session_loss_is_not_replayed(mock_server):
    result = json.loads(_handler("update-brand")({"brand": "pipe"}))
    assert "Broken pipe" in result["error"]
    assert mock_server.attempts == {"pipe": 1}
    assert mock_server.initialized == 1
