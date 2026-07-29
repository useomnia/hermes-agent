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


@pytest.fixture(autouse=True)
def _reset_join_abandonment(monkeypatch):
    """``_join_abandoned`` is process-wide state; keep it from leaking between tests."""
    monkeypatch.setattr(mcp_startup, "_join_abandoned", False)


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

    def test_a_wedged_discovery_is_only_waited_out_once_per_process(self, monkeypatch):
        """The 130s bound is a backstop, not a per-request toll.

        A wedged discovery thread never recovers, and the api_server builds a
        fresh agent for every request — so re-waiting would make one stuck MCP
        server cost every later request the full bound AND still hand it a partial
        registry. The first exhaustion has to retire the join for good.
        """
        waits = []
        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(
            mcp_startup, "wait_for_mcp_discovery", lambda timeout=None: waits.append(timeout)
        )
        # Still running after the join returned == the bound was exhausted.
        monkeypatch.setattr(
            mcp_startup, "_mcp_discovery_thread", types.SimpleNamespace(is_alive=lambda: True)
        )

        mcp_startup.ensure_mcp_discovery_complete()
        mcp_startup.ensure_mcp_discovery_complete()
        mcp_startup.ensure_mcp_discovery_complete()

        assert len(waits) == 1, f"re-waited on a thread already known to be wedged: {waits}"

    def test_a_thread_that_finishes_in_time_keeps_the_join_armed(self, monkeypatch):
        # The counterpart to the test above: a join that SUCCEEDS must not retire
        # itself, or a later caller (a reload, a cron job) would skip a wait it
        # genuinely needs.
        waits = []
        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(
            mcp_startup, "wait_for_mcp_discovery", lambda timeout=None: waits.append(timeout)
        )
        monkeypatch.setattr(
            mcp_startup, "_mcp_discovery_thread", types.SimpleNamespace(is_alive=lambda: False)
        )

        mcp_startup.ensure_mcp_discovery_complete()
        mcp_startup.ensure_mcp_discovery_complete()

        assert len(waits) == 2
        assert mcp_startup._join_abandoned is False

    def test_an_explicit_short_timeout_does_not_retire_the_join(self, monkeypatch):
        # An explicit bound is a caller saying "wait this long", not evidence the
        # thread is wedged — so it must not poison the default-bound callers.
        monkeypatch.setattr(mcp_startup, "mcp_discovery_was_started", lambda: True)
        monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", lambda timeout=None: None)
        monkeypatch.setattr(
            mcp_startup, "_mcp_discovery_thread", types.SimpleNamespace(is_alive=lambda: True)
        )

        mcp_startup.ensure_mcp_discovery_complete(timeout=0.01)

        assert mcp_startup._join_abandoned is False

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

        for rel in (
            "gateway/run.py",
            "gateway/platforms/api_server.py",
            "gateway/slash_commands.py",
            # In the inventory because the scheduler runs INSIDE the gateway
            # process: a job due on the first ticker iteration builds its agent
            # while startup discovery is still connecting.
            "cron/scheduler.py",
        ):
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

    def test_no_agent_is_built_directly_on_the_event_loop(self):
        """The join blocks, so an agent build in an async body freezes the loop.

        The sibling test above only proves a join EXISTS in the enclosing
        function — it cannot tell a sync closure dispatched through
        ``run_in_executor`` from a coroutine running on the loop. That blind spot
        is real: ``/v1/chat/completions`` builds its agent inside an
        executor-dispatched ``_run``, but ``/v1/runs`` built its agent inline in
        ``_run_and_close``, an ``asyncio.create_task`` body — so a slow MCP server
        would have stalled every health check and platform heartbeat for the whole
        join, which is #16856 reintroduced through the back door.

        A build nested inside a lambda or a def is fine: that is the shape that
        gets handed to ``run_in_executor``.
        """
        import ast
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        rel = "gateway/platforms/api_server.py"
        tree = ast.parse((repo / rel).read_text(encoding="utf-8"))
        on_loop = []

        def _walk_own_body(node):
            """Yield nodes lexically inside ``node``, NOT descending into nested callables."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                yield child
                yield from _walk_own_body(child)

        for func in ast.walk(tree):
            if not isinstance(func, ast.AsyncFunctionDef):
                continue
            for node in _walk_own_body(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_create_agent"
                ):
                    on_loop.append(f"{rel}:{node.lineno} in async {func.name}")

        assert not on_loop, (
            "_create_agent joins MCP discovery and blocks; called straight from a "
            f"coroutine it stalls the event loop. Wrap it in run_in_executor: {sorted(set(on_loop))}"
        )

    def test_mcp_reload_waits_for_startup_discovery_before_tearing_servers_down(self):
        """A reload arriving before startup discovery finishes must not race it.

        The gateway now serves while discovery is still connecting, so
        ``shutdown_mcp_servers()`` can land underneath the startup thread — and
        ``register_mcp_servers`` dedupes only against connected servers, never
        against in-flight connects. The reload has to join first.
        """
        import ast
        import inspect
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        source = (repo / "gateway/platforms/api_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        handler = next(
            f
            for f in ast.walk(tree)
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "shutdown_mcp_servers" in ast.unparse(f)
            and "_mcp_reload_lock" in ast.unparse(f)
        )
        body = ast.unparse(handler)
        assert "wait_for_startup_mcp_discovery" in body, (
            "MCP reload tears down servers without joining startup discovery first"
        )
        # It must be offloaded, not awaited on the loop.
        assert "run_in_executor(None, wait_for_startup_mcp_discovery)" in body
        assert "await wait_for_startup_mcp_discovery" not in inspect.cleandoc(body)
        # And it must be the JOIN-ONLY helper. The agent-build helper discovers
        # inline when no startup thread exists, which here would connect servers
        # before the handler snapshots the previous set — reporting nothing as
        # `added` and then tearing down what it had just connected. CI caught
        # exactly that: tests/gateway/test_api_server_mcp_reload.py.
        assert "run_in_executor(None, ensure_mcp_discovery_complete)" not in body

    def test_cron_joins_startup_discovery_instead_of_starting_its_own(self):
        """Two concurrent discoveries double-connect the same server.

        ``register_mcp_servers`` filters on ``k not in _servers`` only, so a
        server still in ``_server_connecting`` looks new to a second caller —
        spawning a duplicate stdio child whose connection overwrites the first.
        Before discovery moved to the background this could not happen: the
        gateway finished discovery before the scheduler ever ticked.
        """
        import ast
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        tree = ast.parse((repo / "cron/scheduler.py").read_text(encoding="utf-8"))
        run_job = next(
            f
            for f in ast.walk(tree)
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "run_job"
        )
        body = ast.unparse(run_job)

        assert "ensure_mcp_discovery_complete" in body, (
            "cron builds its agent without joining the gateway's startup discovery"
        )
        # The join must come BEFORE the discovery call it is protecting.
        assert body.index("ensure_mcp_discovery_complete()") < body.index("discover_mcp_tools()")

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
