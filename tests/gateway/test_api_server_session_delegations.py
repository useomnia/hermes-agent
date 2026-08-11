"""Contract tests for ``GET /api/sessions/{session_id}/delegations``.

The endpoint filters the process-wide async-delegation registry by
``origin_session_id`` so an external UI can rebuild a session's outstanding
background work from the process that owns the children. These tests pin the
contract the Omnio proxy consumes: a ``data`` list, session-scoped filtering,
live-status fields passed through, and auth.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(*, api_key: str = "") -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/delegations",
        adapter._handle_session_delegations,
    )
    return app


async def _get(client: TestClient, session_id: str, headers=None):
    return await client.get(
        f"/api/sessions/{session_id}/delegations", headers=headers or {}
    )


_RECORDS = [
    {
        "delegation_id": "d_1",
        "origin_session_id": "sess-a",
        "origin_turn_id": "turn-1",
        "goal": "research pricing",
        "status": "running",
        "children_activity": [
            {
                "api_calls": 3,
                "current_tool": "web_read",
                "seconds_since_activity": 1.2,
                "subagent_id": "sub_1",
                "finished": False,
            }
        ],
    },
    {
        "delegation_id": "d_2",
        "origin_session_id": "sess-b",
        "origin_turn_id": "turn-9",
        "goal": "unrelated",
        "status": "running",
    },
    {
        "delegation_id": "d_3",
        # Records predating the origin_session_id column surface it as "".
        "origin_session_id": "",
        "origin_turn_id": "",
        "goal": "orphaned",
        "status": "completed",
    },
]


@pytest.mark.asyncio
async def test_filters_to_requested_session():
    adapter = _make_adapter()
    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch(
            "tools.async_delegation.list_async_delegations", return_value=_RECORDS
        ):
            resp = await _get(client, "sess-a")
            assert resp.status == 200
            body = await resp.json()
    assert [d["delegation_id"] for d in body["data"]] == ["d_1"]
    # Live-status fields ride through untouched — the proxy reads them as-is.
    child = body["data"][0]["children_activity"][0]
    assert child["subagent_id"] == "sub_1"
    assert child["finished"] is False
    assert body["data"][0]["origin_turn_id"] == "turn-1"


@pytest.mark.asyncio
async def test_unknown_session_returns_empty_data():
    adapter = _make_adapter()
    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch(
            "tools.async_delegation.list_async_delegations", return_value=_RECORDS
        ):
            resp = await _get(client, "sess-nope")
            assert resp.status == 200
            assert (await resp.json())["data"] == []


@pytest.mark.asyncio
async def test_sessionless_records_never_match():
    """An empty origin_session_id must not leak into any session's listing."""
    adapter = _make_adapter()
    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch(
            "tools.async_delegation.list_async_delegations", return_value=_RECORDS
        ):
            resp = await _get(client, "")
            # aiohttp may 404 an empty path segment; what matters is that a
            # sessionless record (origin_session_id="") can never 200 as data.
            assert resp.status in (200, 404)
            if resp.status == 200:
                assert (await resp.json())["data"] == []


@pytest.mark.asyncio
async def test_registry_failure_returns_500():
    adapter = _make_adapter()
    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch(
            "tools.async_delegation.list_async_delegations",
            side_effect=RuntimeError("boom"),
        ):
            resp = await _get(client, "sess-a")
    assert resp.status == 500


@pytest.mark.asyncio
async def test_auth_required_when_key_configured():
    adapter = _make_adapter(api_key="omnio-test-key")
    async with TestClient(TestServer(_make_app(adapter))) as client:
        with patch(
            "tools.async_delegation.list_async_delegations", return_value=_RECORDS
        ):
            denied = await _get(client, "sess-a")
            allowed = await _get(
                client,
                "sess-a",
                headers={"Authorization": "Bearer omnio-test-key"},
            )
    assert denied.status == 401
    assert allowed.status == 200
