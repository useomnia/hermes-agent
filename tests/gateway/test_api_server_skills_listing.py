"""Tests for GET /v1/skills — the deterministic skills listing Omnia's palette reads.

The listing is not just a passthrough of the skills hub's metadata: every entry
carries a ``command`` slug (validated against the live command registry) and the
response ends with the synthetic ``/learn`` built-in. Omnia's "/" palette drops
any skill whose ``command`` is null, so losing that field renders the palette
EMPTY rather than degraded — which is exactly what the 0.19 upstream sync (#42)
did by rewriting this handler without the derive-and-validate block. These tests
pin the payload contract so a future rewrite fails loudly here instead of in prod.
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
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
    app.router.add_get("/v1/skills", adapter._handle_skills)
    return app


def _stub_skills(monkeypatch, skills, registry_slugs):
    """Serve `skills` from the hub and `registry_slugs` as the live command registry."""
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **_: list(skills))
    monkeypatch.setattr(skills_tool, "_sort_skills", lambda found: list(found))
    monkeypatch.setattr(
        skill_commands,
        "get_skill_commands",
        lambda: {f"/{slug}": {} for slug in registry_slugs},
    )


async def _get_skills(adapter) -> list[dict]:
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.get("/v1/skills")
        assert resp.status == 200
        body = await resp.json()
    assert body["object"] == "list"
    return body["data"]


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.mark.asyncio
async def test_listing_requires_auth(auth_adapter):
    async with TestClient(TestServer(_create_app(auth_adapter))) as cli:
        resp = await cli.get("/v1/skills")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_every_registered_skill_carries_its_command_slug(adapter, monkeypatch):
    """The regression guard: a listed skill MUST expose a non-null command slug.

    Omnia's palette filters on `command !== null`, so an entry without it is
    invisible client-side even though the skill is installed and invocable.
    """
    _stub_skills(
        monkeypatch,
        [
            {"name": "Profile the Brand", "description": "d", "category": "omnio"},
            {"name": "web", "description": "d", "category": "toolkits"},
        ],
        ["profile-the-brand", "web"],
    )

    data = await _get_skills(adapter)

    commands = {entry["name"]: entry["command"] for entry in data}
    assert commands["Profile the Brand"] == "profile-the-brand"
    assert commands["web"] == "web"


@pytest.mark.asyncio
async def test_a_slug_absent_from_the_registry_is_reported_as_null(adapter, monkeypatch):
    """A derived slug that doesn't round-trip must be null, not a broken command.

    A non-null command is a promise that `/<command>` resolves on the chat path;
    a skill missing from the registry (platform-disabled, or a display-truncated
    name) would otherwise list but fail on send.
    """
    _stub_skills(
        monkeypatch,
        [
            {"name": "Registered Skill", "description": "d", "category": "omnio"},
            {"name": "Unregistered Skill", "description": "d", "category": "omnio"},
            {"name": "+++", "description": "d", "category": "omnio"},
        ],
        ["registered-skill"],
    )

    data = await _get_skills(adapter)

    commands = {entry["name"]: entry["command"] for entry in data}
    assert commands["Registered Skill"] == "registered-skill"
    assert commands["Unregistered Skill"] is None
    # A name that slugifies to "" can never resolve.
    assert commands["+++"] is None


@pytest.mark.asyncio
async def test_listing_surfaces_the_learn_builtin_command(adapter, monkeypatch):
    """`/learn` is served from this endpoint, not hardcoded client-side."""
    _stub_skills(monkeypatch, [], [])

    data = await _get_skills(adapter)

    assert len(data) == 1
    learn = data[0]
    assert learn["name"] == "learn"
    assert learn["command"] == "learn"
    # category "command" is what buckets it into the palette's "Actions" section.
    assert learn["category"] == "command"
    assert learn["description"]


@pytest.mark.asyncio
async def test_listing_preserves_hub_metadata_alongside_command(adapter, monkeypatch):
    """Deriving `command` must not drop fields the hub supplies (e.g. `hidden`).

    The palette relies on `hidden` to keep internal skills out of the picker, so
    the command derivation spreads the original entry rather than rebuilding it.
    """
    _stub_skills(
        monkeypatch,
        [
            {
                "name": "Manage Credentials",
                "description": "internal",
                "category": "omnio",
                "hidden": True,
            }
        ],
        ["manage-credentials"],
    )

    data = await _get_skills(adapter)

    skill = data[0]
    assert skill["command"] == "manage-credentials"
    assert skill["hidden"] is True
    assert skill["description"] == "internal"
    assert skill["category"] == "omnio"


@pytest.mark.asyncio
async def test_every_listed_command_resolves_on_the_chat_expansion_path(
    adapter, monkeypatch
):
    """The cross-path invariant: what the palette LISTS, the chat path EXPANDS.

    A listed `command` is a promise that sending `/<command>` expands server-side
    into the skill payload instead of reaching the model as literal text. Both
    sides settle it with the same membership test against the same
    `get_skill_commands()` registry — the listing checks `f"/{slug}" in registry`,
    and `resolve_skill_command_key` checks `f"/{cmd}" in registry` — so they
    cannot disagree. This pins that: deriving the slug and resolving it must stay
    a single source of truth, not two parallel reimplementations that can drift.
    """
    from agent.skill_commands import resolve_skill_command_key

    _stub_skills(
        monkeypatch,
        [
            {"name": "Site Audit", "description": "d", "category": "omnio"},
            {"name": "Profile the Brand", "description": "d", "category": "omnio"},
            {"name": "Orphan Skill", "description": "d", "category": "omnio"},
        ],
        ["site-audit", "profile-the-brand"],
    )

    data = await _get_skills(adapter)

    listed = [entry for entry in data if entry["command"] and entry["category"] != "command"]
    assert listed, "expected at least one invocable skill in the listing"
    for entry in listed:
        # The exact resolver the chat path calls before building the payload.
        assert resolve_skill_command_key(entry["command"]) is not None, (
            f"listing advertised /{entry['command']} but the chat path would not "
            "expand it — the slug would reach the model as literal text"
        )
    # The unregistered skill is withheld rather than advertised as invocable.
    assert {entry["name"] for entry in listed} == {"Site Audit", "Profile the Brand"}


@pytest.mark.asyncio
async def test_enumeration_failure_is_reported_as_500(adapter, monkeypatch):
    def _boom(**_):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(skills_tool, "_find_all_skills", _boom)

    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.get("/v1/skills")
    assert resp.status == 500
