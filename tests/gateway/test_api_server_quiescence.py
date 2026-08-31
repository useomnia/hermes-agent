"""Handover invariants for the Hermes/Omnio quiescence contract."""

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
from gateway import quiescence
from tools import async_delegation
from tools.process_registry import ProcessRegistry
from tools.process_registry import ProcessSession


@pytest.fixture(autouse=True)
def isolated_quiescence_home(tmp_path, monkeypatch):
    """Keep force-retirement marker tests out of the developer profile."""
    monkeypatch.setattr(quiescence, "get_hermes_home", lambda: tmp_path)


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
    app.router.add_post("/v1/omnio/quiescence", adapter._handle_omnio_quiescence)
    app.router.add_post(
        "/v1/omnio/quiescence/release", adapter._handle_omnio_quiescence_release
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


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


def test_startup_marker_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        quiescence, "write_offline_quiescence_snapshot", lambda *args, **kwargs: False
    )
    assert quiescence.mark_offline_quiescence_unknown() is False


def test_startup_marker_readback_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        quiescence, "write_offline_quiescence_snapshot", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(quiescence, "read_offline_quiescence_snapshot", lambda: {})
    assert quiescence.mark_offline_quiescence_unknown() is False


def test_api_adapter_rejects_malformed_persisted_marker(tmp_path):
    marker = tmp_path / "gateway_quiescence.json"
    marker.write_text("{malformed", encoding="utf-8")
    with patch.object(quiescence, "get_hermes_home", return_value=tmp_path):
        with pytest.raises(RuntimeError, match="marker is unreadable"):
            APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))


def test_startup_does_not_repair_malformed_force_marker(tmp_path):
    marker = tmp_path / "gateway_quiescence.json"
    marker.write_text(
        '{"state":"busy","force_latched":"yes","generation":4}',
        encoding="utf-8",
    )
    with patch.object(quiescence, "get_hermes_home", return_value=tmp_path):
        assert quiescence.mark_offline_quiescence_unknown() is False
        assert marker.read_text(encoding="utf-8") == (
            '{"state":"busy","force_latched":"yes","generation":4}'
        )


def test_offline_marker_fsyncs_parent_after_atomic_replace(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        quiescence,
        "_fsync_parent_directory",
        lambda path: calls.append(path),
    )
    with patch.object(quiescence, "get_hermes_home", return_value=tmp_path):
        assert quiescence.write_offline_quiescence_snapshot(
            {"known": False, "total": 1, "counts": {"x": 1}, "errors": ["x"]},
            lifecycle="starting",
        )
    assert calls == [tmp_path / "gateway_quiescence.json"]


def test_offline_marker_parent_fsync_failure_is_fail_closed(tmp_path, monkeypatch):
    def fail_fsync(_path):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(quiescence, "_fsync_parent_directory", fail_fsync)
    with patch.object(quiescence, "get_hermes_home", return_value=tmp_path):
        with pytest.raises(OSError, match="directory fsync failed"):
            quiescence.write_offline_quiescence_snapshot(
                {"known": False, "total": 1, "counts": {"x": 1}, "errors": ["x"]},
                lifecycle="starting",
            )


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


@pytest.mark.asyncio
async def test_graceful_prepare_is_authenticated_idempotent_and_not_latched():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        unauthorized = await client.post(
            "/v1/omnio/quiescence", json={"mode": "graceful"}
        )
        assert unauthorized.status == 401

        first = await client.post(
            "/v1/omnio/quiescence",
            headers={"Authorization": "Bearer secret"},
            json={"mode": "graceful", "request_id": "same"},
        )
        second = await client.post(
            "/v1/omnio/quiescence",
            headers={"Authorization": "Bearer secret"},
            json={"mode": "graceful", "request_id": "same"},
        )
        assert first.status == second.status == 200
        assert (await first.json())["state"] == "quiescent"
        assert (await second.json())["state"] == "quiescent"
        assert adapter._quiescence_force_latched is False


@pytest.mark.asyncio
async def test_force_timeout_latches_and_release_reopens():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    busy = {
        "counts": {"api_runs": 1},
        "total": 1,
        "known": True,
        "errors": [],
    }
    with patch("gateway.quiescence.collect_writer_work_snapshot", return_value=busy), patch(
        "gateway.quiescence.interrupt_writer_work",
        return_value={"actions": {}, "errors": []},
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force", "timeout_seconds": 0},
            )
            assert response.status == 409
            payload = await response.json()
            assert payload["state"] == "busy"
            assert payload["latched"] is True

            repeated = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force", "timeout_seconds": 0},
            )
            repeated_payload = await repeated.json()
            assert repeated.status == 409
            assert repeated_payload["generation"] == payload["generation"]
            assert repeated_payload["boot_id"] == payload["boot_id"]

            status = await client.get(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
            )
            assert status.status == 409
            assert (await status.json())["state"] == "busy"

            blocked = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={},
            )
            assert blocked.status == 503
            assert (await blocked.json())["error"]["code"] == "gateway_quiescing"

            released = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={
                    "generation": payload["generation"],
                    "boot_id": payload["boot_id"],
                },
            )
            assert released.status == 200
            released_payload = await released.json()
            assert released_payload["state"] == "released"
            assert adapter._quiescence_force_latched is False

            retry = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={
                    "generation": payload["generation"],
                    "boot_id": payload["boot_id"],
                },
            )
            assert retry.status == 200
            assert (await retry.json())["generation"] == released_payload["generation"]


@pytest.mark.asyncio
async def test_force_reports_quiescent_only_after_zero_snapshot():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    snapshots = iter(
        [
            {"counts": {"async_delegations": 1}, "total": 1, "known": True, "errors": []},
            {"counts": {}, "total": 0, "known": True, "errors": []},
        ]
    )
    with patch(
        "gateway.quiescence.collect_writer_work_snapshot",
        side_effect=lambda **_kwargs: next(snapshots),
    ), patch(
        "gateway.quiescence.interrupt_writer_work",
        return_value={"actions": {"async_delegations": 1}, "errors": []},
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force", "timeout_seconds": 1},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["state"] == "quiescent"
            assert payload["latched"] is True


@pytest.mark.asyncio
async def test_force_release_rejects_stale_generation():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    zero = {"counts": {}, "total": 0, "known": True, "errors": []}
    with patch("gateway.quiescence.collect_writer_work_snapshot", return_value=zero), patch(
        "gateway.quiescence.interrupt_writer_work",
        return_value={"actions": {}, "errors": []},
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            prepared = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force"},
            )
            proof = await prepared.json()
            stale = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={
                    "generation": proof["generation"] - 1,
                    "boot_id": proof["boot_id"],
                },
            )
            assert stale.status == 409
            assert (await stale.json())["errors"] == ["stale_generation"]
            assert adapter._quiescence_force_latched is True


@pytest.mark.asyncio
async def test_force_release_requires_exact_request_id_when_supplied():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    zero = {"counts": {}, "total": 0, "known": True, "errors": []}
    with patch("gateway.quiescence.collect_writer_work_snapshot", return_value=zero), patch(
        "gateway.quiescence.interrupt_writer_work",
        return_value={"actions": {}, "errors": []},
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            prepared = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force", "request_id": "barrier-1"},
            )
            proof = await prepared.json()
            proof_identity = {
                "generation": proof["generation"],
                "boot_id": proof["boot_id"],
            }

            stale = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={**proof_identity, "request_id": "barrier-2"},
            )
            assert stale.status == 409
            assert (await stale.json())["errors"] == ["stale_barrier"]
            assert adapter._quiescence_force_latched is True

            missing = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json=proof_identity,
            )
            assert missing.status == 409
            assert (await missing.json())["errors"] == ["stale_barrier"]

            released = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={**proof_identity, "request_id": "barrier-1"},
            )
            assert released.status == 200
            assert adapter._quiescence_force_latched is False


@pytest.mark.asyncio
async def test_old_release_cannot_reopen_new_force_epoch():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    zero = {"counts": {}, "total": 0, "known": True, "errors": []}
    with patch("gateway.quiescence.collect_writer_work_snapshot", return_value=zero), patch(
        "gateway.quiescence.interrupt_writer_work",
        return_value={"actions": {}, "errors": []},
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            first = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force"},
            )
            first_proof = await first.json()
            released = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={
                    "generation": first_proof["generation"],
                    "boot_id": first_proof["boot_id"],
                },
            )
            assert released.status == 200

            second = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force"},
            )
            assert second.status == 200
            stale = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={
                    "generation": first_proof["generation"],
                    "boot_id": first_proof["boot_id"],
                },
            )
            assert stale.status == 409
            assert adapter._quiescence_force_latched is True


def test_successful_release_generation_survives_restart(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter._quiescence_force_latched = True
    adapter._quiescence_generation = 12
    adapter._quiescence_force_boot_id = adapter._quiescence_boot_id
    with patch.object(quiescence, "get_hermes_home", return_value=tmp_path):
        quiescence.write_offline_quiescence_snapshot(
            {"known": True, "total": 0, "counts": {}, "errors": []},
            lifecycle="force_latched",
            force_latched=True,
            generation=12,
            force_boot_id=adapter._quiescence_force_boot_id,
        )
        assert adapter._persist_force_marker(
            {"known": True, "total": 0, "counts": {}, "errors": []},
            latched=False,
            generation=13,
            force_boot_id=adapter._quiescence_force_boot_id,
        )
        original_boot = quiescence._OFFLINE_BOOT_ID
        try:
            quiescence._OFFLINE_BOOT_ID = "replacement-boot"
            quiescence.mark_offline_quiescence_unknown()
            replacement = APIServerAdapter(
                PlatformConfig(enabled=True, extra={"key": "secret"})
            )
            assert replacement._quiescence_force_latched is False
            assert replacement._quiescence_generation == 13
        finally:
            quiescence._OFFLINE_BOOT_ID = original_boot


@pytest.mark.asyncio
async def test_force_marker_write_failure_stays_busy():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    zero = {"counts": {}, "total": 0, "known": True, "errors": []}
    with patch("gateway.quiescence.collect_writer_work_snapshot", return_value=zero), patch(
        "gateway.quiescence.interrupt_writer_work",
        return_value={"actions": {}, "errors": []},
    ) as interrupt, patch.object(adapter, "_persist_force_marker", return_value=False):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/omnio/quiescence",
                headers={"Authorization": "Bearer secret"},
                json={"mode": "force", "timeout_seconds": 1},
            )
            assert response.status == 503
            payload = await response.json()
            assert payload["state"] == "busy"
            assert payload["errors"] == ["force_marker_persistence"]
            assert adapter._quiescence_force_latched is True
            interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_force_release_marker_failure_keeps_latch():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _running_agent_count=lambda: 0,
        _active_cron_job_count=lambda: 0,
    )
    adapter._quiescence_force_latched = True
    adapter._quiescence_generation = 4
    adapter._quiescence_force_boot_id = adapter._quiescence_boot_id
    zero = {"counts": {}, "total": 0, "known": True, "errors": []}
    with patch("gateway.quiescence.collect_writer_work_snapshot", return_value=zero), patch.object(
        adapter, "_persist_force_marker", return_value=False
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/omnio/quiescence/release",
                headers={"Authorization": "Bearer secret"},
                json={
                    "generation": adapter._quiescence_generation,
                    "boot_id": adapter._quiescence_force_boot_id,
                },
            )
            assert response.status == 503
            payload = await response.json()
            assert payload["errors"] == ["force_marker_clear_failed"]
            assert adapter._quiescence_force_latched is True


def test_offline_marker_rejects_stale_shutdown_and_unknown_zero(tmp_path):
    with patch.object(quiescence, "get_hermes_home") as get_home:
        profile_home = tmp_path
        get_home.return_value = profile_home
        marker = profile_home / "gateway_quiescence.json"
        lock = profile_home / "gateway_quiescence.lock"
        original_boot = quiescence._OFFLINE_BOOT_ID
        try:
            quiescence.mark_offline_quiescence_unknown()
            current_boot = quiescence.quiescence_boot_id()
            quiescence.write_offline_quiescence_snapshot(
                {"known": True, "total": 0, "counts": {}, "errors": []},
                lifecycle="stopped",
            )
            assert quiescence.collect_offline_durable_snapshot()["state"] == "quiescent"

            quiescence._OFFLINE_BOOT_ID = "replacement-boot"
            quiescence.mark_offline_quiescence_unknown()
            quiescence._OFFLINE_BOOT_ID = original_boot
            quiescence.write_offline_quiescence_snapshot(
                {"known": True, "total": 0, "counts": {}, "errors": []},
                lifecycle="stopped",
            )
            payload = quiescence.read_offline_quiescence_snapshot()
            assert payload["boot_id"] == "replacement-boot"
            assert payload["state"] == "unknown"
            assert current_boot != "replacement-boot"
        finally:
            quiescence._OFFLINE_BOOT_ID = original_boot
            marker.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)


def test_force_latch_marker_is_rehydrated_on_replacement_boot(tmp_path):
    with patch.object(quiescence, "get_hermes_home", return_value=tmp_path):
        original_boot = quiescence._OFFLINE_BOOT_ID
        try:
            quiescence.write_offline_quiescence_snapshot(
                {"known": True, "total": 0, "counts": {}, "errors": []},
                lifecycle="force_latched",
                force_latched=True,
                generation=9,
                force_boot_id="retired-proof-boot",
            )
            offline = quiescence.collect_offline_durable_snapshot()
            assert offline["state"] == "busy"
            assert offline["errors"] == ["force_latched"]
            quiescence._OFFLINE_BOOT_ID = "replacement-boot"
            quiescence.mark_offline_quiescence_unknown()
            adapter = APIServerAdapter(
                PlatformConfig(enabled=True, extra={"key": "secret"})
            )
            assert adapter._quiescence_force_latched is True
            assert adapter._quiescence_generation == 9
            assert adapter._quiescence_force_boot_id == "retired-proof-boot"
        finally:
            quiescence._OFFLINE_BOOT_ID = original_boot
