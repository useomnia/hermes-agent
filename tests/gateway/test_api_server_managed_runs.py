"""Retired handoff envelopes must never silently become ordinary executions."""

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app


@pytest.mark.asyncio
async def test_retired_managed_envelope_is_rejected():
    adapter = _make_adapter()
    try:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={
                    "input": "hello",
                    "turn_id": "retired",
                    "session_id": "s",
                    "omnio_managed": {"version": 1},
                },
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "unsupported_managed_run"
            assert not adapter._active_run_tasks
    finally:
        await adapter.disconnect()
