"""Behavioral tests for the bounded, read-only MCP parallel path."""

import asyncio
import concurrent.futures
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _call_result(text="ok"):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        isError=False,
    )


def _prepare_server(name, config, tool_hints):
    """Build one fake MCP task and record its concurrent session calls."""
    from tools.mcp_tool import (
        MCPServerTask,
        _lock,
        _mcp_tool_read_only_hints,
        _mcp_tool_server_names,
        _servers,
        mcp_prefixed_tool_name,
        sanitize_mcp_name_component,
    )

    active = {"current": 0, "max": 0, "lock_seen": []}
    server = MCPServerTask(name)

    async def call_tool(tool_name, *, arguments):
        active["current"] += 1
        active["max"] = max(active["max"], active["current"])
        active["lock_seen"].append(server._rpc_lock.locked())
        try:
            await asyncio.sleep(0.03)
            return _call_result(tool_name)
        finally:
            active["current"] -= 1

    server.session = SimpleNamespace(call_tool=call_tool)
    with _lock:
        _servers[name] = server
        safe_name = sanitize_mcp_name_component(name)
        for tool_name, hint in tool_hints.items():
            prefixed = mcp_prefixed_tool_name(name, tool_name)
            _mcp_tool_server_names[prefixed] = safe_name
            if hint is not None:
                _mcp_tool_read_only_hints[prefixed] = hint
    return server, active


def _cleanup_server(server, tool_hints):
    from tools.mcp_tool import (
        _lock,
        _mcp_tool_read_only_hints,
        _mcp_tool_server_names,
        _parallel_read_safe_servers,
        _servers,
        mcp_prefixed_tool_name,
        sanitize_mcp_name_component,
    )

    with _lock:
        _servers.pop(server.name, None)
        _parallel_read_safe_servers.discard(
            sanitize_mcp_name_component(server.name)
        )
        for tool_name in tool_hints:
            prefixed = mcp_prefixed_tool_name(server.name, tool_name)
            _mcp_tool_server_names.pop(prefixed, None)
            _mcp_tool_read_only_hints.pop(prefixed, None)


def _run_handlers_on_one_mcp_loop(server, handlers, config):
    """Invoke sync handlers concurrently while sharing one MCP event loop."""
    import tools.mcp_tool as mcp_tool

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def loop_thread():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=loop_thread, daemon=True)
    thread.start()
    assert started.wait(timeout=2)

    configured = concurrent.futures.Future()

    def configure():
        try:
            server._config = dict(config)
            server._configure_parallel_reads(config)
            configured.set_result(None)
        except BaseException as exc:
            configured.set_exception(exc)

    loop.call_soon_threadsafe(configure)
    configured.result(timeout=2)

    def run(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    try:
        with patch.object(mcp_tool, "_run_on_mcp_loop", side_effect=run):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(handlers)
            ) as executor:
                futures = [executor.submit(handler, {}) for handler in handlers]
                return [future.result(timeout=5) for future in futures]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


class TestMcpParallelReadCalls:
    def test_http_read_only_calls_overlap_without_rpc_lock(self):
        from tools.mcp_tool import _make_tool_handler

        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": 4,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        hints = {"read_a": True, "read_b": True}
        server, active = _prepare_server("http_reads", config, hints)
        try:
            handlers = [
                _make_tool_handler("http_reads", name, 30)
                for name in hints
            ]
            results = _run_handlers_on_one_mcp_loop(server, handlers, config)
            assert all(json.loads(result)["result"] for result in results)
            assert active["max"] == 2
            assert active["lock_seen"] == [False, False]
            assert server._parallel_read_enabled is True
            assert server._parallel_read_limit == 4
        finally:
            _cleanup_server(server, hints)

    def test_http_read_cap_applies(self):
        from tools.mcp_tool import _make_tool_handler

        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": 2,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        hints = {f"read_{i}": True for i in range(5)}
        server, active = _prepare_server("http_cap", config, hints)
        try:
            handlers = [
                _make_tool_handler("http_cap", name, 30)
                for name in hints
            ]
            results = _run_handlers_on_one_mcp_loop(server, handlers, config)
            assert all(json.loads(result)["result"] for result in results)
            assert active["max"] == 2
            assert server._parallel_read_semaphore is not None
        finally:
            _cleanup_server(server, hints)

    @pytest.mark.parametrize(
        "tool_hints",
        [
            {"write": False, "read": True},
            {"missing": None, "read": True},
        ],
    )
    def test_http_writes_and_missing_hints_serialize_with_reads(self, tool_hints):
        from tools.mcp_tool import _make_tool_handler

        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": 4,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        server, active = _prepare_server("http_mixed", config, tool_hints)
        try:
            handlers = [
                _make_tool_handler("http_mixed", name, 30)
                for name in tool_hints
            ]
            results = _run_handlers_on_one_mcp_loop(server, handlers, config)
            assert all(json.loads(result)["result"] for result in results)
            assert active["max"] == 1
            # The read bypasses _rpc_lock but the write/missing-hint call
            # takes it; the read/write gate still prevents overlap.
            assert active["lock_seen"].count(True) == 1
        finally:
            _cleanup_server(server, tool_hints)

    def test_stdio_read_opt_in_fails_closed_and_serializes(self):
        from tools.mcp_tool import _make_tool_handler

        config = {
            "command": "example-mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": 4,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        hints = {"read_a": True, "read_b": True}
        server, active = _prepare_server("stdio_reads", config, hints)
        try:
            handlers = [
                _make_tool_handler("stdio_reads", name, 30)
                for name in hints
            ]
            with patch("tools.mcp_tool.logger") as logger:
                results = _run_handlers_on_one_mcp_loop(server, handlers, config)
            assert all(json.loads(result)["result"] for result in results)
            assert active["max"] == 1
            assert all(active["lock_seen"])
            assert server._parallel_read_enabled is False
            logger.warning.assert_called()
        finally:
            _cleanup_server(server, hints)

    def test_keepalive_excludes_active_parallel_read(self):
        from tools.mcp_tool import MCPServerTask

        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": 4,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        server = MCPServerTask("keepalive_gate")
        server._config = dict(config)
        server._configure_parallel_reads(config)
        ping_started = asyncio.Event()

        async def send_ping():
            ping_started.set()
            assert server._rpc_lock.locked()

        server.session = SimpleNamespace(send_ping=send_ping)

        async def check():
            async with server._rpc_context(parallel_read=True):
                probe_task = asyncio.create_task(server._keepalive_probe())
                await asyncio.sleep(0)
                assert not ping_started.is_set()
            await asyncio.wait_for(probe_task, timeout=0.5)
            assert ping_started.is_set()

        try:
            asyncio.run(check())
        finally:
            _cleanup_server(server, {})

    @pytest.mark.parametrize(
        "config, enabled",
        [
            ({}, False),
            (
                {
                    "url": "https://example.test/mcp",
                    "supports_parallel_read_tool_calls": True,
                    "parallel_read_tool_call_limit": 2,
                    "sampling": {"enabled": False},
                    "elicitation": {"enabled": False},
                },
                True,
            ),
            (
                {
                    "url": "https://example.test/mcp",
                    "supports_parallel_read_tool_calls": True,
                    "parallel_read_tool_call_limit": 2,
                    "sampling": {"enabled": True},
                    "elicitation": {"enabled": False},
                },
                False,
            ),
        ],
    )
    def test_read_opt_in_is_default_off_and_requires_callbacks_disabled(
        self, config, enabled
    ):
        from tools.mcp_tool import MCPServerTask

        async def check():
            server = MCPServerTask("config_reads")
            server._config = dict(config)
            with patch("tools.mcp_tool.logger") as logger:
                server._configure_parallel_reads(config)
            assert server._parallel_read_enabled is enabled
            if enabled:
                assert server._parallel_read_limit == 2
                assert server._parallel_read_gate.max_readers == 2
            elif config.get("supports_parallel_read_tool_calls"):
                logger.warning.assert_called()

        asyncio.run(check())


class TestMcpReadWriteGate:
    def test_waiting_writer_has_priority_over_late_reader(self):
        from tools.mcp_tool import _MCPReadWriteGate

        async def check():
            gate = _MCPReadWriteGate(2)
            order = []
            writer_acquired = asyncio.Event()

            await gate.acquire_read()
            await gate.acquire_read()

            async def writer():
                await gate.acquire_write()
                try:
                    order.append("writer")
                    writer_acquired.set()
                    await asyncio.sleep(0)
                finally:
                    await gate.release_write()

            async def late_reader():
                await gate.acquire_read()
                try:
                    order.append("reader")
                finally:
                    await gate.release_read()

            writer_task = asyncio.create_task(writer())
            await asyncio.sleep(0)
            reader_task = asyncio.create_task(late_reader())
            await asyncio.sleep(0)
            assert not reader_task.done()

            await gate.release_read()
            await gate.release_read()
            await asyncio.wait_for(writer_acquired.wait(), timeout=0.5)
            await asyncio.wait_for(
                asyncio.gather(writer_task, reader_task), timeout=0.5
            )
            assert order == ["writer", "reader"]

        asyncio.run(check())

    def test_cancelled_reader_returns_reserved_slot(self):
        from tools.mcp_tool import _MCPReadWriteGate

        async def check():
            gate = _MCPReadWriteGate(1)
            await gate.acquire_write()

            reader_task = asyncio.create_task(gate.acquire_read())
            await asyncio.sleep(0)
            reader_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader_task

            await gate.release_write()
            await asyncio.wait_for(gate.acquire_read(), timeout=0.5)
            await gate.release_read()

        asyncio.run(check())

    def test_cancelled_writer_clears_writer_priority(self):
        from tools.mcp_tool import _MCPReadWriteGate

        async def check():
            gate = _MCPReadWriteGate(1)
            await gate.acquire_read()

            writer_task = asyncio.create_task(gate.acquire_write())
            await asyncio.sleep(0)
            writer_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await writer_task

            late_reader = asyncio.create_task(gate.acquire_read())
            await gate.release_read()
            await asyncio.wait_for(late_reader, timeout=0.5)
            await gate.release_read()

        asyncio.run(check())


class TestMcpParallelReadConfig:
    def test_opt_in_uses_default_limit_when_omitted(self):
        from tools.mcp_tool import (
            MCPServerTask,
            _DEFAULT_PARALLEL_READ_LIMIT,
        )

        server = MCPServerTask("default_limit")
        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        try:
            server._configure_parallel_reads(config)
            assert server._parallel_read_enabled is True
            assert server._parallel_read_limit == _DEFAULT_PARALLEL_READ_LIMIT == 4
        finally:
            _cleanup_server(server, {})

    @pytest.mark.parametrize("invalid_limit", [True, None, "nope", 0, -1, 2.5])
    def test_invalid_limit_fails_closed(self, invalid_limit):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("invalid_limit")
        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": invalid_limit,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        try:
            with patch("tools.mcp_tool.logger") as logger:
                server._configure_parallel_reads(config)
            assert server._parallel_read_enabled is False
            assert server._parallel_read_gate is None
            logger.warning.assert_called()
        finally:
            _cleanup_server(server, {})

    @pytest.mark.parametrize("callback_name", ["sampling", "elicitation"])
    def test_quoted_false_callback_setting_fails_closed(self, callback_name):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("quoted_callback")
        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": 4,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        config[callback_name]["enabled"] = "false"
        try:
            with patch("tools.mcp_tool.logger") as logger:
                server._configure_parallel_reads(config)
            assert server._parallel_read_enabled is False
            assert server._parallel_read_gate is None
            logger.warning.assert_called()
        finally:
            _cleanup_server(server, {})

    def test_limit_above_hard_ceiling_is_clamped(self):
        from tools.mcp_tool import (
            MCPServerTask,
            _MAX_PARALLEL_READ_LIMIT,
        )

        server = MCPServerTask("clamped_limit")
        config = {
            "url": "https://example.test/mcp",
            "supports_parallel_read_tool_calls": True,
            "parallel_read_tool_call_limit": _MAX_PARALLEL_READ_LIMIT + 100,
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
        }
        try:
            with patch("tools.mcp_tool.logger") as logger:
                server._configure_parallel_reads(config)
            assert server._parallel_read_enabled is True
            assert server._parallel_read_limit == _MAX_PARALLEL_READ_LIMIT
            assert server._parallel_read_gate.max_readers == _MAX_PARALLEL_READ_LIMIT
            logger.warning.assert_called_once()
        finally:
            _cleanup_server(server, {})
