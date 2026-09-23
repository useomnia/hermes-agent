"""Run submission loads real skill files before the agent receives its message."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run_idempotency import RunIdempotencyStore
from hermes_constants import get_config_path


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    for name, content in [
        ("review-report", "REPORT PROCEDURE"),
        ("review-report-custom", "CUSTOM PROCEDURE"),
    ]:
        directory = skills_dir / "reporting" / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test procedure.\n"
            "metadata:\n  hidden: true\n---\n\n"
            f"{content}\nSession: ${{HERMES_SESSION_ID}}\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._run_idempotency = RunIdempotencyStore(tmp_path / "state.db")
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.run_conversation.side_effect = lambda **kwargs: {
        "final_response": json.dumps(kwargs)
    }
    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: agent)

    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app, skills_dir


async def _submit(client, input_value, **options):
    response = await client.post("/v1/runs", json={"input": input_value, **options})
    assert response.status == 202, await response.text()
    accepted = await response.json()
    events = await client.get(f"/v1/runs/{accepted['run_id']}/events")
    await asyncio.wait_for(events.text(), timeout=10)
    status = await client.get(f"/v1/runs/{accepted['run_id']}")
    result = await status.json()
    assert result["status"] == "completed", result
    return accepted, json.loads(result["output"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "/review-report Report ID: report-123",
        "Load /review-report Report ID: report-123",
        [{"role": "user", "content": "/review-report Report ID: report-123"}],
    ],
)
async def test_run_loads_exact_registered_skill_before_agent(runtime, message):
    app, _ = runtime
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        _, received = await _submit(client, message, session_id="skill-session")

    assert "REPORT PROCEDURE" in received["user_message"]
    assert "CUSTOM PROCEDURE" not in received["user_message"]
    assert "Report ID: report-123" in received["user_message"]
    assert "Session: skill-session" in received["user_message"]
    assert received["task_id"] == "skill-session"


@pytest.mark.asyncio
async def test_run_expansion_uses_generated_session_identity(runtime):
    app, _ = runtime
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        accepted, received = await _submit(client, "/review-report")

    assert f"Session: {accepted['run_id']}" in received["user_message"]
    assert received["task_id"] == accepted["run_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    ["ordinary request", "/unknown-skill do work", "read /brand/report.md", "/"],
)
async def test_unrecognized_run_input_is_preserved(runtime, message):
    app, _ = runtime
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        _, received = await _submit(client, message)

    assert received["user_message"] == message


@pytest.mark.asyncio
async def test_run_preserves_multimodal_content(runtime):
    app, _ = runtime
    content = [
        {"type": "text", "text": "/review-report"},
        {"type": "image_url", "image_url": {"url": "https://example.com/report.png"}},
    ]
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        _, received = await _submit(client, [{"role": "user", "content": content}])

    assert received["user_message"] == content


@pytest.mark.asyncio
async def test_run_expands_only_current_message(runtime):
    app, _ = runtime
    history = [
        {"role": "user", "content": "/review-report-custom"},
        {"role": "assistant", "content": "Previous report."},
    ]
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        _, received = await _submit(
            client, "/review-report", conversation_history=history
        )

    assert "REPORT PROCEDURE" in received["user_message"]
    assert received["conversation_history"] == history


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "skills_config",
    [
        {"disabled": ["review-report"]},
        {"platform_disabled": {"api_server": ["review-report"]}},
    ],
)
async def test_run_respects_disabled_skills(runtime, skills_config):
    app, _ = runtime
    get_config_path().write_text(
        json.dumps({"skills": skills_config}), encoding="utf-8"
    )
    message = "/review-report Report ID: report-123"
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        _, received = await _submit(client, message)

    assert received["user_message"] == message


@pytest.mark.asyncio
async def test_failed_skill_load_preserves_original_run_input(runtime, monkeypatch):
    app, _ = runtime
    monkeypatch.setattr(
        skill_commands, "_load_skill_payload", lambda *args, **kwargs: None
    )
    message = "/review-report Report ID: report-123"
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        _, received = await _submit(client, message)

    assert received["user_message"] == message


@pytest.mark.asyncio
async def test_run_retry_keeps_original_expansion_after_skill_changes(runtime):
    app, skills_dir = runtime
    message = "/review-report Report ID: report-123"
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        first, first_received = await _submit(client, message, turn_id="skill-turn")
        (skills_dir / "reporting/review-report/SKILL.md").write_text(
            "---\nname: review-report\n---\nCHANGED PROCEDURE", encoding="utf-8"
        )
        second, second_received = await _submit(client, message, turn_id="skill-turn")

    assert second["run_id"] == first["run_id"]
    assert second["idempotent"] is True
    assert second_received == first_received
    assert "REPORT PROCEDURE" in second_received["user_message"]


@pytest.mark.asyncio
async def test_run_rejects_unauthenticated_skill_request(runtime):
    app, _ = runtime
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/runs", json={"input": "/review-report"})

    assert response.status == 401


@pytest.mark.asyncio
async def test_run_advertises_slash_command_support(runtime):
    app, _ = runtime
    async with TestClient(
        TestServer(app), headers={"Authorization": "Bearer test-key"}
    ) as client:
        response = await client.get("/v1/capabilities")
        capabilities = await response.json()

    assert capabilities["features"]["run_slash_commands"] is True
