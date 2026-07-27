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


class TestEnsureMcpDiscoveryComplete:
    def test_joins_the_background_thread_with_a_bound_above_the_internal_wait(self, monkeypatch):
        seen = {}

        def fake_wait(timeout=None):
            seen["timeout"] = timeout

        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", fake_wait)

        mcp_startup.ensure_mcp_discovery_complete()

        # Not the 1.5s ``mcp_discovery_timeout`` the CLI uses: this gateway used
        # to wait for discovery in full, and a short bound would silently drop
        # slow servers' tools from turn 1.
        assert seen["timeout"] > 120

    def test_discovers_inline_when_no_thread_was_ever_started(self, monkeypatch):
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

        mcp_startup.ensure_mcp_discovery_complete()

        assert calls == ["discovered"]

    def test_every_full_toolset_agent_build_joins_discovery(self):
        """Inventory every ``AIAgent(...)`` in the gateway process, not a hand-list.

        Enumerating the call sites by hand is what let the api_server adapter's
        ``_create_agent`` — the build behind every OpenAI-compatible request, and
        so the busiest one — ship without a join: the first turn snapshotted the
        registry before discovery finished and ran without any MCP tool. Walking
        the AST means a new build site fails this test instead of silently losing
        tools at runtime.

        Memory-only agents are exempt: with ``enabled_toolsets=["memory"]`` they
        cannot reach an MCP tool, so waiting would be dead latency.
        """
        import ast
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        joins = ("ensure_mcp_discovery_complete", "_join_mcp_discovery")
        unguarded = []

        for rel in ("gateway/run.py", "gateway/platforms/api_server.py", "gateway/slash_commands.py"):
            tree = ast.parse((repo / rel).read_text(encoding="utf-8"))
            # Map each AIAgent(...) call to the innermost function containing it.
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = ast.unparse(func)
                for call in ast.walk(func):
                    if not (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "AIAgent"
                    ):
                        continue
                    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
                    if kwargs.get("enabled_toolsets") == "['memory']":
                        continue
                    if not any(j in body for j in joins):
                        unguarded.append(f"{rel}:{call.lineno} in {func.name}")

        assert not unguarded, (
            "AIAgent built without joining MCP discovery — its turn would run "
            f"with a partial tool registry: {sorted(set(unguarded))}"
        )

    def test_the_join_is_never_awaited_on_the_loop_thread(self):
        """#16856: the join blocks, so it must only run in a worker thread.

        Every call site sits inside a closure dispatched through
        ``run_in_executor``; an ``await`` of it in an async body would freeze
        platform heartbeats for the whole MCP connect time.
        """
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        for rel in ("gateway/run.py", "gateway/platforms/api_server.py"):
            source = (repo / rel).read_text(encoding="utf-8")
            assert "await self._join_mcp_discovery" not in source, rel
            assert "await ensure_mcp_discovery_complete" not in source, rel

    def test_a_failing_join_never_breaks_the_turn(self, monkeypatch):
        def explode(timeout=None):
            raise RuntimeError("discovery thread wedged")

        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", explode)

        # Degrades to late-binding rather than failing the message.
        mcp_startup.ensure_mcp_discovery_complete()


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
