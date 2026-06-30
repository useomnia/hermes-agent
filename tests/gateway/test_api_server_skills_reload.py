"""Tests for POST /v1/skills/reload — mid-session skills rescan.

Covers auth, that the endpoint runs the same rescan the /reload-skills slash
command uses (agent.skill_commands.reload_skills) and returns its diff, that a
rescan failure surfaces as a 500, and that concurrent reloads are serialized so
the shared slash-command registry can't be left half-populated.
"""

import asyncio
import threading
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.skill_commands as skill_commands
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/skills/reload", adapter._handle_skills_reload)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.mark.asyncio
async def test_reload_requires_auth(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/reload")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_reload_returns_the_rescan_diff(adapter, monkeypatch):
    diff = {
        "added": [{"name": "alpha", "description": "a"}],
        "removed": [],
        "unchanged": ["beta"],
        "total": 2,
        "commands": 2,
    }
    monkeypatch.setattr(skill_commands, "reload_skills", lambda: diff)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/reload")
        assert resp.status == 200
        body = await resp.json()

    assert body["object"] == "hermes.skills.reload"
    assert body["added"] == [{"name": "alpha", "description": "a"}]
    assert body["total"] == 2
    assert body["commands"] == 2


@pytest.mark.asyncio
async def test_reload_reports_a_rescan_failure_as_500(adapter, monkeypatch):
    def _boom():
        raise RuntimeError("scan failed")

    monkeypatch.setattr(skill_commands, "reload_skills", _boom)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/reload")
    assert resp.status == 500


@pytest.mark.asyncio
async def test_concurrent_reloads_are_serialized(adapter, monkeypatch):
    """Two reloads firing at once must not interleave.

    The handler's asyncio.Lock serializes the rescan so a second reload waits for
    the first; scan_skill_commands() resets and repopulates the shared registry,
    so overlapping rescans could otherwise leave it half-populated.
    """
    state = {"active": 0, "max": 0, "starts": 0}
    guard = threading.Lock()

    def _fake_reload():
        with guard:
            state["active"] += 1
            state["starts"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.05)
        with guard:
            state["active"] -= 1
        return {"added": [], "removed": [], "unchanged": [], "total": 0, "commands": 0}

    monkeypatch.setattr(skill_commands, "reload_skills", _fake_reload)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        r1, r2 = await asyncio.gather(
            cli.post("/v1/skills/reload"),
            cli.post("/v1/skills/reload"),
        )
        assert r1.status == 200
        assert r2.status == 200

    # Both reloads ran to completion (neither was dropped) ...
    assert state["starts"] == 2
    # ... but never at the same time.
    assert state["max"] == 1
