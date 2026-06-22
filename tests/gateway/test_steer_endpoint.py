"""Tests for POST /v1/omnio/steer — mid-turn steer injection into the live
chat agent registered by session id."""

from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_steer_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/omnio/steer", adapter._handle_omnio_steer)
    return app


@pytest.mark.asyncio
async def test_steer_injects_into_the_live_agent_for_the_session(adapter):
    agent = MagicMock()
    agent.steer.return_value = True
    adapter._chat_agents["sess-1"] = [agent]
    app = _create_steer_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/steer",
            json={"text": "actually, keep it formal"},
            headers={"X-Hermes-Session-Id": "sess-1"},
        )
        assert resp.status == 200
        assert (await resp.json())["accepted"] is True
    agent.steer.assert_called_once_with("actually, keep it formal")


@pytest.mark.asyncio
async def test_steer_not_accepted_when_no_live_turn(adapter):
    app = _create_steer_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/steer", json={"text": "hi"}, headers={"X-Hermes-Session-Id": "absent"}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["accepted"] is False
        assert body["reason"] == "no_active_turn"


@pytest.mark.asyncio
async def test_steer_not_accepted_before_the_agent_is_constructed(adapter):
    # The turn registered its ref but the agent isn't built yet (agent_ref[0] None).
    adapter._chat_agents["sess-1"] = [None]
    app = _create_steer_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/steer", json={"text": "hi"}, headers={"X-Hermes-Session-Id": "sess-1"}
        )
        assert resp.status == 200
        assert (await resp.json())["accepted"] is False


@pytest.mark.asyncio
async def test_steer_rejects_missing_text(adapter):
    agent = MagicMock()
    adapter._chat_agents["sess-1"] = [agent]
    app = _create_steer_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/omnio/steer", json={"text": "   "}, headers={"X-Hermes-Session-Id": "sess-1"}
        )
        assert resp.status == 400
    agent.steer.assert_not_called()


@pytest.mark.asyncio
async def test_steer_rejects_missing_session_header(adapter):
    app = _create_steer_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/omnio/steer", json={"text": "hi"})
        assert resp.status == 400
