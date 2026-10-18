"""A failed Streamable HTTP request must not strand the session's other calls.

MCP SDK 1.x tears the whole session down when one HTTP request fails and
never answers the requests still in flight. Before the guard failed them,
every concurrent parallel read waited for the full tool timeout, and its read
slot held off the reconnect's handshake for just as long.
"""

import asyncio
import concurrent.futures
import json
import threading
import time

import httpx
import pytest

pytest.importorskip("mcp.client.streamable_http")

TOOL_TIMEOUT = 30


class _MockMCPServer:
    """Stateless JSON Streamable HTTP server whose tool fails on request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.initialized = 0
        self.hanging = 0
        self.gateway_errors = 0

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
            result = {"tools": [{
                "name": "get-brand",
                "inputSchema": {"type": "object", "properties": {"brand": {"type": "string"}}},
                "annotations": {"readOnlyHint": True, "destructiveHint": False},
            }]}
        elif method == "tools/call":
            brand = message["params"]["arguments"]["brand"]
            if brand == "hang":
                with self.lock:
                    self.hanging += 1
                await asyncio.Event().wait()
            if brand == "bad":
                with self.lock:
                    self.gateway_errors += 1
                return httpx.Response(502, text="Bad Gateway")
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


def test_gateway_error_fails_concurrent_reads_promptly_and_reconnects(monkeypatch, tmp_path):
    from tools import mcp_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    server = _MockMCPServer()
    transport = httpx.MockTransport(server.handle)
    real_client = httpx.AsyncClient

    class _MockedClient(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _MockedClient)
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
        handler = mcp_tool._make_tool_handler("mockhttp", "get-brand", TOOL_TIMEOUT)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            in_flight = [pool.submit(handler, {"brand": "hang"}) for _ in range(3)]
            _wait_for(lambda: server.hanging == 3)
            started = time.monotonic()
            failed = handler({"brand": "bad"})
            stranded = [future.result(timeout=TOOL_TIMEOUT) for future in in_flight]
            elapsed = time.monotonic() - started

        assert elapsed < 10, f"calls waited {elapsed:.1f}s for a dead session"
        assert server.gateway_errors == 1
        for result in [failed, *stranded]:
            error = json.loads(result)["error"]
            assert "HTTP 502 Bad Gateway" in error
            assert "not replayed" in error
        _wait_for(lambda: server.initialized == 2)
        _wait_for(lambda: "error" not in json.loads(handler({"brand": "ok"})))
        assert server.gateway_errors == 1
    finally:
        mcp_tool.shutdown_mcp_servers()
