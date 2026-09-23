"""Behavior contract for Omnio-managed ``/v1/runs`` identities."""

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _parse_managed_run_identity,
)
from gateway.run_idempotency import ManagedRunIdentity, RunIdempotencyStore


SUBMISSION_ID = "12345678-1234-4234-8234-123456789abc"
FINGERPRINT = "a" * 64
IDENTITY = ManagedRunIdentity(SUBMISSION_ID, FINGERPRINT)


def _managed_body(*, session_id: str = "session") -> dict:
    return {
        "turn_id": "turn-managed",
        "session_id": session_id,
        "omnio_managed": {
            "version": 1,
            "submission_id": SUBMISSION_ID,
            "execution_fingerprint": FINGERPRINT,
        },
    }


def _launch_body(*, session_id: str = "session") -> dict:
    return {"input": "hello", **_managed_body(session_id=session_id)}


def _make_adapter(tmp_path, *, api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    adapter._run_idempotency = RunIdempotencyStore(tmp_path / "state.db")
    return adapter


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post(
        "/v1/runs/managed/reconcile", adapter._handle_reconcile_managed_run
    )
    app.router.add_post(
        "/v1/runs/managed/cancel", adapter._handle_cancel_managed_run
    )
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


def _fast_agent() -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


def _sse_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.parametrize(
    "identity",
    [
        {
            "version": True,
            "submission_id": SUBMISSION_ID,
            "execution_fingerprint": FINGERPRINT,
        },
        {
            "version": 1,
            "submission_id": SUBMISSION_ID.upper(),
            "execution_fingerprint": FINGERPRINT,
        },
        {
            "version": 1,
            "submission_id": SUBMISSION_ID,
            "execution_fingerprint": FINGERPRINT.upper(),
        },
        {
            "version": 1,
            "submission_id": SUBMISSION_ID,
            "execution_fingerprint": FINGERPRINT,
            "extra": "rejected",
        },
    ],
)
def test_managed_identity_parser_is_strict(identity) -> None:
    with pytest.raises(ValueError):
        _parse_managed_run_identity(identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/v1/runs/managed/reconcile", "/v1/runs/managed/cancel"],
)
async def test_managed_controls_require_gateway_auth(tmp_path, path) -> None:
    adapter = _make_adapter(tmp_path, api_key="secret")
    async with TestClient(TestServer(_create_app(adapter))) as client:
        unauthorized = await client.post(path, json=_managed_body())
        authorized = await client.post(
            path,
            json=_managed_body(),
            headers={"Authorization": "Bearer secret"},
        )

    assert unauthorized.status == 401
    assert authorized.status in {200, 404}


@pytest.mark.asyncio
async def test_reconcile_absent_is_non_creating(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/runs/managed/reconcile", json=_managed_body()
        )
        data = await response.json()

    assert response.status == 404
    assert data["error"]["code"] == "managed_run_not_found"
    assert adapter._run_idempotency.get("turn-managed") is None


@pytest.mark.asyncio
async def test_cancel_before_launch_replays_tombstone_even_at_concurrency_cap(
    tmp_path,
) -> None:
    adapter = _make_adapter(tmp_path)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        cancelled = await client.post(
            "/v1/runs/managed/cancel", json=_managed_body()
        )
        cancelled_data = await cancelled.json()

        adapter._max_concurrent_runs = 1
        adapter._inflight_agent_runs = 1
        with patch.object(adapter, "_create_agent") as create_agent:
            replay = await client.post("/v1/runs", json=_launch_body())
            replay_data = await replay.json()

        reconciled = await client.post(
            "/v1/runs/managed/reconcile", json=_managed_body()
        )
        reconciled_data = await reconciled.json()

    assert cancelled.status == 200
    assert cancelled_data["status"] == "cancelled"
    assert replay.status == 202
    assert replay_data == {
        "run_id": cancelled_data["run_id"],
        "status": "cancelled",
        "idempotent": True,
    }
    assert reconciled.status == 200
    assert reconciled_data["run_id"] == cancelled_data["run_id"]
    assert reconciled_data["cancel_requested"] is True
    assert cancelled_data["run_id"] not in adapter._stopping_run_ids
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_before_launch_replays_one_terminal_event_after_restart(
    tmp_path,
) -> None:
    first_adapter = _make_adapter(tmp_path)
    with patch.object(first_adapter, "_create_agent") as create_agent:
        async with TestClient(TestServer(_create_app(first_adapter))) as client:
            cancelled = await client.post(
                "/v1/runs/managed/cancel", json=_managed_body()
            )
            cancelled_data = await cancelled.json()
            launched = await client.post("/v1/runs", json=_launch_body())
            reconciled = await client.post(
                "/v1/runs/managed/reconcile", json=_managed_body()
            )
            first_events = await client.get(
                f"/v1/runs/{cancelled_data['run_id']}/events?after=0"
            )
            first_boundary = first_events.headers["X-Omnio-Replay-Through"]
            first_body = await first_events.text()
            replayed_events = await client.get(
                f"/v1/runs/{cancelled_data['run_id']}/events?after=0"
            )
            replayed_body = await replayed_events.text()
            caught_up = await client.get(
                f"/v1/runs/{cancelled_data['run_id']}/events?after=1"
            )
            caught_up_body = await caught_up.text()

    first = _sse_events(first_body)
    assert cancelled.status == 200
    assert launched.status == 202
    assert reconciled.status == 200
    assert first_boundary == "1"
    assert first == _sse_events(replayed_body)
    assert len(first) == 1
    assert first[0]["sequence_number"] == 1
    assert first[0]["type"] == "response.incomplete"
    assert first[0]["response"]["status"] == "incomplete"
    assert first[0]["response"]["incomplete_details"] == {
        "reason": "cancelled"
    }
    assert _sse_events(caught_up_body) == []
    create_agent.assert_not_called()
    persisted = first_adapter._run_idempotency.get("turn-managed")
    assert persisted is not None
    assert persisted.is_cancel_tombstone
    assert persisted.failure_reason == "cancelled_before_start"

    restarted = _make_adapter(tmp_path)
    with patch.object(restarted, "_create_agent") as restarted_create_agent:
        async with TestClient(TestServer(_create_app(restarted))) as client:
            restored = await client.post(
                "/v1/runs/managed/reconcile", json=_managed_body()
            )
            restored_data = await restored.json()
            restored_events = await client.get(
                f"/v1/runs/{cancelled_data['run_id']}/events?after=0"
            )
            restored_boundary = restored_events.headers[
                "X-Omnio-Replay-Through"
            ]
            restored_body = await restored_events.text()
            repeated = await client.post(
                "/v1/runs/managed/reconcile", json=_managed_body()
            )
            repeated_events = await client.get(
                f"/v1/runs/{cancelled_data['run_id']}/events?after=0"
            )
            repeated_body = await repeated_events.text()

    assert restored.status == repeated.status == 200
    assert restored_data["run_id"] == cancelled_data["run_id"]
    assert restored_boundary == "1"
    restored_event_list = _sse_events(restored_body)
    assert len(restored_event_list) == 1
    restored_event = restored_event_list[0]
    assert restored_event["sequence_number"] == 1
    assert restored_event["type"] == "response.incomplete"
    assert restored_event["response"]["id"] == first[0]["response"]["id"]
    assert restored_event["response"]["status"] == "incomplete"
    assert restored_event["response"]["incomplete_details"] == {
        "reason": "cancelled"
    }
    assert _sse_events(repeated_body) == restored_event_list
    restarted_create_agent.assert_not_called()
    restored_record = restarted._run_idempotency.get("turn-managed")
    assert restored_record is not None
    assert restored_record.is_cancel_tombstone
    assert restored_record.failure_reason == "cancelled_before_start"


@pytest.mark.asyncio
async def test_controls_reject_identity_mismatch(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        assert (
            await client.post("/v1/runs/managed/cancel", json=_managed_body())
        ).status == 200
        mismatch = await client.post(
            "/v1/runs/managed/reconcile",
            json=_managed_body(session_id="different-session"),
        )
        mismatch_data = await mismatch.json()

    assert mismatch.status == 409
    assert mismatch_data["error"]["code"] == "managed_run_identity_conflict"


@pytest.mark.asyncio
async def test_concurrent_identical_managed_launches_create_one_agent(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(
            adapter, "_create_agent", return_value=_fast_agent()
        ) as create_agent:
            first, second = await asyncio.gather(
                client.post("/v1/runs", json=_launch_body()),
                client.post("/v1/runs", json=_launch_body()),
            )
            first_data, second_data = await asyncio.gather(
                first.json(), second.json()
            )
            await asyncio.sleep(0.05)

    assert first.status == second.status == 202
    assert first_data["run_id"] == second_data["run_id"]
    assert {first_data.get("idempotent"), second_data.get("idempotent")} == {
        None,
        True,
    }
    create_agent.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_change",
    [
        {"execution_fingerprint": "b" * 64},
        {"submission_id": "87654321-4321-4321-8321-cba987654321"},
    ],
)
async def test_changed_managed_identity_conflicts_without_second_agent(
    tmp_path, identity_change
) -> None:
    adapter = _make_adapter(tmp_path)
    changed = _launch_body()
    changed["omnio_managed"] = {
        **changed["omnio_managed"],
        **identity_change,
    }
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(
            adapter, "_create_agent", return_value=_fast_agent()
        ) as create_agent:
            first = await client.post("/v1/runs", json=_launch_body())
            conflict = await client.post("/v1/runs", json=changed)
            conflict_data = await conflict.json()
            await asyncio.sleep(0.05)

    assert first.status == 202
    assert conflict.status == 409
    assert conflict_data["error"]["code"] == "managed_run_identity_conflict"
    create_agent.assert_called_once()


@pytest.mark.asyncio
async def test_restart_reconcile_returns_original_managed_mapping(tmp_path) -> None:
    path = tmp_path / "state.db"
    first_adapter = _make_adapter(tmp_path)
    async with TestClient(TestServer(_create_app(first_adapter))) as client:
        with patch.object(
            first_adapter, "_create_agent", return_value=_fast_agent()
        ):
            launched = await client.post("/v1/runs", json=_launch_body())
            launched_data = await launched.json()
            for _ in range(100):
                record = first_adapter._run_idempotency.get("turn-managed")
                if record is not None and record.status == "completed":
                    break
                await asyncio.sleep(0.01)

    restarted = _make_adapter(tmp_path)
    restarted._run_idempotency = RunIdempotencyStore(path)
    async with TestClient(TestServer(_create_app(restarted))) as client:
        reconciled = await client.post(
            "/v1/runs/managed/reconcile", json=_managed_body()
        )
        reconciled_data = await reconciled.json()

    assert reconciled.status == 200
    assert reconciled_data["run_id"] == launched_data["run_id"]
    assert reconciled_data["status"] == "completed"


@pytest.mark.asyncio
async def test_managed_mapping_is_isolated_between_profiles(
    tmp_path, monkeypatch
) -> None:
    profile_homes = {name: tmp_path / name for name in ("foo", "bar")}
    for home in profile_homes.values():
        home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex=True: list(profile_homes.items()),
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: profile_homes[name],
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter.gateway_runner = MagicMock(
        config=GatewayConfig(multiplex_profiles=True)
    )
    app = web.Application(
        middlewares=[adapter._make_profile_prefix_middleware()]
    )
    app.router.add_post(
        "/p/{profile}/v1/runs/managed/reconcile",
        adapter._handle_reconcile_managed_run,
    )
    app.router.add_post(
        "/p/{profile}/v1/runs/managed/cancel",
        adapter._handle_cancel_managed_run,
    )

    async with TestClient(TestServer(app)) as client:
        cancelled = await client.post(
            "/p/foo/v1/runs/managed/cancel", json=_managed_body()
        )
        absent = await client.post(
            "/p/bar/v1/runs/managed/reconcile", json=_managed_body()
        )
        reconciled = await client.post(
            "/p/foo/v1/runs/managed/reconcile", json=_managed_body()
        )

    assert cancelled.status == 200
    assert absent.status == 404
    assert reconciled.status == 200
    assert (profile_homes["foo"] / "state.db").exists()
    assert RunIdempotencyStore(
        profile_homes["bar"] / "state.db"
    ).get("turn-managed") is None


@pytest.mark.asyncio
async def test_cancel_overrides_stale_in_memory_queued_status(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    record, _ = adapter._run_idempotency.reserve(
        turn_id="turn-managed",
        run_id="run-reserved",
        request_fingerprint="launch",
        session_id="session",
        owner_profile=None,
        managed_identity=IDENTITY,
    )
    adapter._turn_event_logs.create_run(record.run_id, record.session_id)
    adapter._set_run_status(
        record.run_id,
        "queued",
        turn_id=record.turn_id,
        session_id=record.session_id,
        owner_profile=None,
    )
    adapter._run_lifecycles[record.run_id] = {
        "accepting": True,
        "agent": None,
        "pending": [],
        "lock": asyncio.Lock(),
    }

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/v1/runs/managed/cancel", json=_managed_body()
        )
        data = await response.json()

    assert response.status == 200
    assert data["status"] == "cancelled"
    assert adapter._run_statuses[record.run_id]["status"] == "cancelled"
    persisted = adapter._run_idempotency.get(record.turn_id)
    assert persisted is not None
    assert persisted.status == "cancelled"
    assert record.run_id in adapter._stopping_run_ids
    adapter._stopping_run_ids.discard(record.run_id)


@pytest.mark.asyncio
async def test_final_persisted_cancel_check_prevents_task_publication(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    real_get = adapter._run_idempotency.get
    cancellation_injected = False

    def cancel_before_publication(turn_id: str):
        nonlocal cancellation_injected
        if not cancellation_injected:
            cancellation_injected = True
            return adapter._run_idempotency.cancel_managed(
                turn_id=turn_id,
                run_id="ignored",
                session_id="session",
                owner_profile=None,
                identity=IDENTITY,
            )[0]
        return real_get(turn_id)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with (
            patch.object(
                adapter._run_idempotency,
                "get",
                side_effect=cancel_before_publication,
            ),
            patch.object(adapter, "_create_agent") as create_agent,
        ):
            response = await client.post("/v1/runs", json=_launch_body())
            data = await response.json()
            await asyncio.sleep(0)

    assert response.status == 202
    assert data["status"] == "cancelled"
    assert cancellation_injected is True
    assert adapter._active_run_tasks == {}
    assert data["run_id"] not in adapter._stopping_run_ids
    create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_after_task_publication_interrupts_the_reserved_run(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    ready = threading.Event()
    interrupted = threading.Event()
    agent = MagicMock()

    def run_conversation(**_kwargs):
        ready.set()
        interrupted.wait(timeout=3)
        return {"final_response": "interrupted", "interrupted": True}

    agent.run_conversation.side_effect = run_conversation
    agent.interrupt.side_effect = lambda _message: interrupted.set()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            launched = await client.post("/v1/runs", json=_launch_body())
            launched_data = await launched.json()
            assert await asyncio.to_thread(ready.wait, 3)
            cancelled = await client.post(
                "/v1/runs/managed/cancel", json=_managed_body()
            )
            cancelled_data = await cancelled.json()
            for _ in range(100):
                record = adapter._run_idempotency.get("turn-managed")
                if record is not None and record.status == "cancelled":
                    break
                await asyncio.sleep(0.01)

    assert launched.status == 202
    assert cancelled.status == 200
    assert cancelled_data["run_id"] == launched_data["run_id"]
    assert cancelled_data["cancel_requested"] is True
    assert interrupted.is_set()
    assert record is not None
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_capability_advertises_managed_contract_and_static_routes(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    route_rows = {(method, path) for method, path, _ in adapter._http_route_table()}
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.get("/v1/capabilities")
        data = await response.json()

    assert data["features"]["managed_run_identity"] == {
        "apiVersion": 1,
        "nonCreatingReconcile": True,
        "durableCancelFence": True,
    }
    assert ("POST", "/v1/runs/managed/reconcile") in route_rows
    assert ("POST", "/v1/runs/managed/cancel") in route_rows
