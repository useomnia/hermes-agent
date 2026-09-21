"""Exercise real YAML loading, agent construction, and provider-bound prompts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
from hermes_cli.config import get_config_path
from hermes_cli.dump import _config_overrides
from run_agent import AIAgent


def _write_config(**agent_options):
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "agent": {"environment_probe": False, "tool_use_enforcement": False,
                  **agent_options},
        "compression": {"enabled": False},
    }), encoding="utf-8")


@pytest.fixture
def make_agent():
    tools = [{"type": "function", "function": {
        "name": "terminal", "description": "Execute a command",
        "parameters": {"type": "object", "properties": {}},
    }}]
    with (
        patch("run_agent.get_tool_definitions", return_value=tools),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        def build():
            agent = AIAgent(
                model="openai/gpt-5.6-luna@preset/internal",
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            agent._use_prompt_caching = False
            agent.save_trajectories = False
            return agent

        yield build


def _request_system_prompt(agent, history=None):
    message = SimpleNamespace(content="Acknowledged.", tool_calls=None,
                              reasoning=None, reasoning_content=None,
                              reasoning_details=None)
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="openai/gpt-5.6-luna", usage=None,
    )
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Reply with Acknowledged.", conversation_history=history)
    assert result["completed"] is True
    request = agent.client.chat.completions.create.call_args.kwargs
    assert request["model"] == "openai/gpt-5.6-luna@preset/internal"
    # Hermes maps the system prompt to OpenAI's developer role for GPT models.
    assert request["messages"][0]["role"] == "developer"
    return request["messages"][0]["content"], result["messages"]


@pytest.mark.parametrize("options,enabled", [
    ({}, True), ({"execution_guidance": False}, False),
    ({"execution_guidance": True}, True),
    ({"execution_guidance": ["LUNA"]}, True),
    ({"execution_guidance": ["deepseek"]}, False),
])
def test_yaml_config_controls_luna_request_prompt(make_agent, options, enabled):
    _write_config(**options)
    prompt, _ = _request_system_prompt(make_agent())
    assert prompt.count(OPENAI_MODEL_EXECUTION_GUIDANCE) == int(enabled)


def test_config_edit_applies_to_new_agents_without_changing_existing_prompt(make_agent):
    _write_config()
    agent = make_agent()
    first, history = _request_system_prompt(agent)
    assert OPENAI_MODEL_EXECUTION_GUIDANCE in first
    first_stable = agent._build_system_prompt_parts()["stable"]

    _write_config(execution_guidance=False)
    second, _ = _request_system_prompt(agent, history)
    assert second == first
    assert agent._build_system_prompt_parts()["stable"] == first_stable

    new_prompt, _ = _request_system_prompt(make_agent())
    assert OPENAI_MODEL_EXECUTION_GUIDANCE not in new_prompt


def test_diagnostic_dump_reports_execution_override():
    overrides = _config_overrides({"agent": {"execution_guidance": False}})
    assert overrides["agent.execution_guidance"] == "False"
