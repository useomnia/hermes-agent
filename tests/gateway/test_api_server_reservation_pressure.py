"""HTTP admission under real SQLite contention; inference is isolated."""

import asyncio
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hermes_state import SessionDB
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


def _agent():
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


@pytest.mark.asyncio
async def test_session_writer_still_allows_http_admission_and_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session = SessionDB(tmp_path / "state.db")
    session.create_session("conversation", source="api_server")
    adapter = _make_adapter()
    adapter._session_db = session
    blocker = sqlite3.connect(session.db_path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with patch.object(adapter, "_create_agent", return_value=_agent()):
            async with TestClient(TestServer(_create_runs_app(adapter))) as client:
                response = await client.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "turn_id": "session-pressure",
                        "session_id": "conversation",
                    },
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                status = await client.get(f"/v1/runs/{run_id}")
                assert status.status == 200
                blocker.rollback()
                await asyncio.gather(*adapter._active_run_tasks.values())
    finally:
        blocker.rollback()
        blocker.close()
        await adapter.disconnect()
        session.close()


@pytest.mark.asyncio
async def test_reservation_lock_refuses_without_starting_then_replays_once():
    adapter = _make_adapter()
    blocker = sqlite3.connect(adapter._run_idempotency.db_path)
    app = _create_runs_app(adapter)
    app.router.add_get("/v1/runs", adapter._handle_recoverable_runs)
    body = {"input": "hello", "turn_id": "pressure"}
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with patch.object(adapter, "_create_agent", return_value=_agent()) as create:
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/v1/runs", json=body)
                assert response.status == 503
                assert (await response.json())["error"][
                    "code"
                ] == "run_reservation_busy"
                assert response.headers["Retry-After"] == "1"
                assert adapter._run_idempotency.get("pressure") is None
                create.assert_not_called()
                blocker.rollback()
                accepted = await client.post("/v1/runs", json=body)
                replay = await client.post("/v1/runs", json=body)
                assert accepted.status == replay.status == 202
                assert (await accepted.json())["run_id"] == (await replay.json())[
                    "run_id"
                ]
                await asyncio.gather(*adapter._active_run_tasks.values())
                create.assert_called_once()
    finally:
        blocker.rollback()
        blocker.close()
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_inventory_stays_incomplete_during_off_loop_reservation():
    adapter = _make_adapter()
    blocker = sqlite3.connect(adapter._run_idempotency.db_path)
    app = _create_runs_app(adapter)
    app.router.add_get("/v1/runs", adapter._handle_recoverable_runs)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        async with TestClient(TestServer(app)) as client:
            pending = asyncio.create_task(
                client.post("/v1/runs", json={"input": "hello", "turn_id": "pending"})
            )
            async with asyncio.timeout(5):
                while not adapter._run_admission_tasks:
                    await asyncio.sleep(0)
                inventory = await client.get("/v1/runs?recoverable=1")
                assert (await inventory.json())["inventory"]["complete"] is False
                assert not pending.done()
                health = await client.get("/v1/capabilities")
                assert health.status == 200
            assert (await pending).status == 503
            inventory = await client.get("/v1/runs?recoverable=1")
            assert (await inventory.json())["inventory"]["complete"] is True
    finally:
        blocker.rollback()
        blocker.close()
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnected_admission_publishes_one_run_for_same_id_recovery():
    adapter = _make_adapter()
    blocker = sqlite3.connect(adapter._run_idempotency.db_path)
    request = MagicMock()
    request.headers = {}
    from unittest.mock import AsyncMock

    request.json = AsyncMock(return_value={"input": "hello", "turn_id": "disconnect"})
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with patch.object(adapter, "_create_agent", return_value=_agent()) as create:
            pending = asyncio.create_task(adapter._handle_runs(request))
            async with asyncio.timeout(5):
                while not adapter._run_admission_tasks:
                    await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert adapter.quiescence_agent_work_count() > 0
                blocker.rollback()
                await asyncio.gather(*adapter._run_admission_tasks)
                response = await adapter._handle_runs(request)
                assert response.status == 202
                await asyncio.gather(*adapter._active_run_tasks.values())
                create.assert_called_once()
                record = await adapter._run_reservation_io(None, "get", "disconnect")
                assert record.run_id in adapter._run_statuses
    finally:
        blocker.rollback()
        blocker.close()
        await adapter.disconnect()
