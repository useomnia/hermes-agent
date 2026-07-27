"""Background MCP discovery on the socket-gateway boot path.

``start_gateway`` starts discovery instead of awaiting it, and the first agent
build joins it. Two properties have to hold for that to be safe:

* the join must not lose tools — no turn may snapshot a partial registry;
* the join must stay off the event loop, or a slow MCP server freezes platform
  heartbeats again (#16856), which is the whole reason discovery was moved off
  the import path in the first place.
"""

import types

import pytest

from hermes_cli import mcp_startup


@pytest.fixture
def runner():
    """Minimal stand-in: the method only reads the join bound off ``self``.

    Constructing a real ``GatewayRunner`` pulls adapters, config and a session
    store — none of which this behaviour depends on.
    """
    from gateway.run import GatewayRunner

    stub = types.SimpleNamespace(
        _MCP_DISCOVERY_JOIN_SECONDS=GatewayRunner._MCP_DISCOVERY_JOIN_SECONDS
    )
    stub._join_mcp_discovery = GatewayRunner._join_mcp_discovery.__get__(stub)
    return stub


class TestDiscoveryWasStarted:
    def test_reports_false_before_any_thread_exists(self, monkeypatch):
        monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

        assert mcp_startup.mcp_discovery_was_started() is False

    def test_reports_true_once_a_thread_exists(self, monkeypatch):
        # Truthiness of a finished thread must not matter: the caller needs
        # "was it ever started", not "is it still running".
        monkeypatch.setattr(
            mcp_startup, "_mcp_discovery_thread", types.SimpleNamespace(is_alive=lambda: False)
        )

        assert mcp_startup.mcp_discovery_was_started() is True


class TestJoinMcpDiscovery:
    def test_joins_the_background_thread_with_a_bound_above_the_internal_wait(
        self, runner, monkeypatch
    ):
        seen = {}

        def fake_wait(timeout=None):
            seen["timeout"] = timeout

        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", fake_wait)

        runner._join_mcp_discovery()

        # Not the 1.5s ``mcp_discovery_timeout`` the CLI uses: this gateway used
        # to wait for discovery in full, and a short bound would silently drop
        # slow servers' tools from turn 1.
        assert seen["timeout"] > 120

    def test_discovers_inline_when_no_thread_was_ever_started(self, runner, monkeypatch):
        # ``start_background_mcp_discovery`` skips on a raw-config probe that can
        # disagree with the merged config discovery itself reads. Falling back
        # here is what stops this change from losing a server the old blocking
        # call would have connected.
        calls = []
        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: False)
        monkeypatch.setattr(
            mcp_startup, "wait_for_mcp_discovery", lambda timeout=None: calls.append("waited")
        )
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: calls.append("discovered"))

        runner._join_mcp_discovery()

        assert calls == ["discovered"]

    def test_both_agent_building_paths_join_before_snapshotting_tools(self):
        """Every tool snapshot is covered, and none of them blocks the loop.

        The turn agent and the background-task agent are the two agents this
        gateway builds with the full toolset, and either can be the first thing
        that runs after boot. Both build inside a closure that
        ``_run_in_executor_with_context`` dispatches to a worker thread, so the
        blocking join is safe there — while an ``await`` of it in the async body
        would freeze platform heartbeats for the whole connect time (#16856).
        """
        import inspect

        from gateway.run import GatewayRunner

        for method in (GatewayRunner._run_agent_inner, GatewayRunner._run_background_task):
            source = inspect.getsource(method)
            assert "self._join_mcp_discovery()" in source, method.__name__
            assert "await self._join_mcp_discovery" not in source, method.__name__

    def test_a_failing_join_never_breaks_the_turn(self, runner, monkeypatch):
        def explode(timeout=None):
            raise RuntimeError("discovery thread wedged")

        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", explode)

        # Degrades to late-binding rather than failing the message.
        runner._join_mcp_discovery()


class TestStartGatewayNoLongerAwaitsDiscovery:
    def test_boot_path_starts_discovery_in_the_background(self):
        # Guards the actual regression: someone re-introducing a blocking
        # ``await ... discover_mcp_tools`` before ``runner.start()`` would put
        # the whole connect time back in front of the gateway serving anything.
        import inspect

        import gateway.run

        source = inspect.getsource(gateway.run.start_gateway)

        assert "start_background_mcp_discovery" in source
        assert "run_in_executor(None, discover_mcp_tools)" not in source
