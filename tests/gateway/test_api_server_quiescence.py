"""Live work must prevent retirement after the handoff protocol is removed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_server as api_server_module
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.quiescence import collect_writer_work_snapshot
from tools import async_delegation
from tools.process_registry import ProcessRegistry
from tools.process_registry import ProcessSession




def _runner():
    return SimpleNamespace(
        _running_agent_count=lambda: 1,
        _active_cron_job_count=lambda: 2,
    )


def _app(adapter):
    app = web.Application()
    app.router.add_get(
        "/v1/omnio/quiescence", adapter._handle_omnio_quiescence_status
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("known,total,status", [(True, 0, 200), (True, 1, 409), (False, 1, 503)])
async def test_retirement_census_keeps_authenticated_read_only_contract(known, total, status):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    try:
        with patch.object(adapter, "_collect_quiescence_snapshot", return_value={
            "known": known, "total": total, "counts": {"api_runs": total}, "errors": [],
        }):
            async with TestClient(TestServer(_app(adapter))) as client:
                unauthorized = await client.get("/v1/omnio/quiescence")
                assert unauthorized.status == 401
                response = await client.get("/v1/omnio/quiescence", headers={"Authorization": "Bearer secret"})
                assert response.status == status
                assert (await response.json())["known"] is known
    finally:
        await adapter.disconnect()


def test_retired_force_marker_cannot_block_a_new_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway_quiescence.json").write_text("malformed retired marker")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._draining_response() is None
    adapter._run_idempotency.close()
    adapter._response_store.close()
    adapter._run_reservation_executor.shutdown()


def _insert_durable_row(delegation_id, state, delivery_state, claim=None):
    with async_delegation._DB_LOCK, async_delegation._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, state, dispatched_at,
                updated_at, delivery_state, delivery_claim)
               VALUES (?, 'session', ?, 1, 1, ?, ?)""",
            (delegation_id, state, delivery_state, claim),
        )


def test_async_count_combines_lifecycle_and_delivery_states():
    _insert_durable_row("running", "running", "pending")
    _insert_durable_row("finalizing", "finalizing", "delivered")
    _insert_durable_row("pending", "completed", "pending")
    _insert_durable_row("claimed", "completed", "pending", "consumer-1")
    _insert_durable_row("claimed-state", "completed", "claimed")
    _insert_durable_row("delivered", "completed", "delivered")

    assert async_delegation.quiescence_work_count() == 5


def test_snapshot_includes_all_writer_categories(monkeypatch):
    from tools.process_registry import process_registry

    monkeypatch.setattr(
        async_delegation, "quiescence_work_count", lambda: 3
    )
    monkeypatch.setattr(
        process_registry,
        "quiescence_work_snapshot",
        lambda: {
            "processes": 4,
            "process_watchers": 5,
            "active_watchers": 2,
            "pending_watchers": 3,
            "completion_queue": 6,
        },
    )
    snapshot = collect_writer_work_snapshot(
        adapter=SimpleNamespace(active_agent_work_count=lambda: 7),
        runner=_runner(),
    )

    assert snapshot["known"] is True
    assert snapshot["counts"]["api_runs"] == 7
    assert snapshot["counts"]["gateway_agents"] == 1
    assert snapshot["counts"]["cron_jobs"] == 2
    assert snapshot["counts"]["background_agent_tasks"] == 3
    assert snapshot["counts"]["async_delegations"] == 3
    assert snapshot["counts"]["processes"] == 4
    assert snapshot["counts"]["process_watchers"] == 5
    assert snapshot["counts"]["completion_queue"] == 6
    assert snapshot["total"] == 28


def test_snapshot_fails_closed_when_api_accounting_breaks():
    def broken_counter():
        raise RuntimeError("counter unavailable")

    snapshot = collect_writer_work_snapshot(
        adapter=SimpleNamespace(quiescence_agent_work_count=broken_counter),
        runner=_runner(),
    )
    assert snapshot["known"] is False
    assert snapshot["counts"]["api_runs"] == 1
    assert snapshot["total"] >= 1
    assert "api_runs" in snapshot["errors"]


def test_reserved_one_shot_task_blocks_zero(monkeypatch):
    class LiveTask:
        def done(self):
            return False

    from tools.process_registry import process_registry

    monkeypatch.setattr(async_delegation, "quiescence_work_count", lambda: 0)
    monkeypatch.setattr(
        process_registry,
        "quiescence_work_snapshot",
        lambda: {"processes": 0, "process_watchers": 0, "completion_queue": 0},
    )
    runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
        _deferred_agent_cleanup_tasks={LiveTask()},
    )
    snapshot = collect_writer_work_snapshot(
        adapter=SimpleNamespace(active_agent_work_count=lambda: 0), runner=runner
    )
    assert snapshot["known"] is True
    assert snapshot["counts"]["background_agent_tasks"] == 1
    assert snapshot["total"] == 1


@pytest.mark.asyncio
async def test_api_detached_one_shot_task_is_counted_before_start(monkeypatch):
    """A shielded cache task cannot disappear between reservation and start."""
    from tools.process_registry import process_registry

    monkeypatch.setattr(async_delegation, "quiescence_work_count", lambda: 0)
    monkeypatch.setattr(
        process_registry,
        "quiescence_work_snapshot",
        lambda: {"processes": 0, "process_watchers": 0, "completion_queue": 0},
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    adapter.gateway_runner = runner
    cache = api_server_module._IdempotencyCache()
    started = asyncio.Event()

    async def compute():
        started.set()
        await asyncio.Future()

    task = asyncio.create_task(
        cache.get_or_set(
            "one-shot", "fingerprint", compute,
            task_registry=adapter._background_agent_tasks,
        )
    )
    try:
        # Let get_or_set publish the child task, but inspect before waiting
        # for its first meaningful await.
        await asyncio.sleep(0)
        snapshot = collect_writer_work_snapshot(adapter=adapter, runner=runner)
        assert snapshot["known"] is True
        assert snapshot["counts"]["api_runs"] == 1
        assert snapshot["total"] == 1
        await started.wait()
        assert adapter.quiescence_agent_work_count() == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_process_registry_counts_active_and_pending_watchers():
    registry = ProcessRegistry()
    registry.enqueue_pending_watcher({"session_id": "pending"})
    registry.register_watcher("active")
    try:
        assert registry.watcher_work_count() == 2
        counts = registry.quiescence_work_snapshot()
        assert counts["process_watchers"] == 2
        assert counts["pending_watchers"] == 1
        assert counts["active_watchers"] == 1
    finally:
        registry.release_watcher("active")


def test_claimed_watcher_never_has_zero_transfer_window():
    registry = ProcessRegistry()
    watcher = {"session_id": "transfer"}
    registry.enqueue_pending_watcher(watcher)
    claimed = registry.claim_pending_watchers()
    assert registry.watcher_work_count() == 1
    assert registry.quiescence_work_snapshot()["process_watchers"] == 1
    registry.release_watcher(claimed[0][1])
    assert registry.watcher_work_count() == 0














def test_process_completion_is_queued_before_running_ownership_disappears(monkeypatch):
    registry = ProcessRegistry()

    class ObservingQueue:
        def __init__(self):
            self.running_counts = []
            self.events = []

        def put(self, event):
            self.running_counts.append(registry.count_running())
            self.events.append(event)

    queue = ObservingQueue()
    registry.completion_queue = queue
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    session = ProcessSession(
        id="proc_atomic",
        command="sleep",
        notify_on_complete=True,
        started_at=1,
    )
    with registry._lock:
        registry._running[session.id] = session
    registry._move_to_finished(session)
    assert queue.running_counts == [1]
    assert registry.count_running() == 0
    assert len(queue.events) == 1


@pytest.mark.asyncio
async def test_watcher_spawn_failure_requeues_and_remains_busy(monkeypatch):
    from tools.process_registry import process_registry

    process_registry.drain_pending_watchers()
    for watcher_id in list(process_registry._active_watchers):
        process_registry.release_watcher(watcher_id)
    watcher = {"session_id": "spawn-failure"}
    process_registry.enqueue_pending_watcher(watcher)
    runner = SimpleNamespace(_running=True)

    # Bind the real dispatcher method without constructing a full gateway.
    from gateway.run import GatewayRunner

    runner._process_watcher_dispatcher = GatewayRunner._process_watcher_dispatcher.__get__(
        runner, type(runner)
    )
    runner._spawn_supervised = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("spawn failed")
    )

    async def stop_after_dispatch(_interval):
        runner._running = False

    monkeypatch.setattr("asyncio.sleep", stop_after_dispatch)
    await runner._process_watcher_dispatcher(interval=0)
    assert process_registry.watcher_work_count() == 1
    assert process_registry.pending_watchers == [watcher]
    process_registry.drain_pending_watchers()
