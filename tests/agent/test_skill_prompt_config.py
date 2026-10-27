"""Exercise real YAML loading and agent construction for the skill prompt switches."""

from unittest.mock import patch

import pytest
import yaml

from agent.prompt_builder import (
    MEMORY_GUIDANCE,
    MEMORY_GUIDANCE_WITHOUT_SKILL_CAPTURE,
    SKILLS_GUIDANCE,
    SKILLS_LEARNING_GUIDANCE,
    SKILLS_SAFETY_GUIDANCE,
    clear_skills_system_prompt_cache,
)
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home
from run_agent import AIAgent

SKILL_CAPTURE = "save it as a skill with the skill tool"
NEUTRAL_NAMES_ONLY_NOTE = "(Categories marked [names only] omit their descriptions"


def _tool(name):
    return {"type": "function", "function": {
        "name": name, "description": f"{name} tool",
        "parameters": {"type": "object", "properties": {}},
    }}


def _write_config(agent=None, skills=None):
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "agent": {"environment_probe": False, "tool_use_enforcement": False,
                  "execution_guidance": False, **(agent or {})},
        "skills": skills or {},
        "compression": {"enabled": False},
    }), encoding="utf-8")


def _write_skill(category, name, description):
    skill_dir = get_hermes_home() / "skills" / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n", encoding="utf-8"
    )


@pytest.fixture
def make_agent():
    tools = [_tool(name) for name in ("terminal", "memory", "skill_view", "skill_manage")]
    with (
        patch("run_agent.get_tool_definitions", return_value=tools),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        def build():
            clear_skills_system_prompt_cache(clear_snapshot=True)
            return AIAgent(
                model="openai/gpt-5.6-luna@preset/internal",
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        yield build


def _stable_prompt(agent):
    return agent._build_system_prompt_parts()["stable"]


class TestSkillLearningGuidance:
    def test_default_keeps_the_full_skill_and_memory_guidance(self, make_agent):
        _write_config()

        prompt = _stable_prompt(make_agent())

        assert SKILLS_GUIDANCE in prompt
        assert MEMORY_GUIDANCE in prompt

    def test_false_drops_skill_capture_and_keeps_safety_rules(self, make_agent):
        _write_config(agent={"skill_learning_guidance": False})

        prompt = _stable_prompt(make_agent())

        assert SKILLS_SAFETY_GUIDANCE in prompt
        assert SKILLS_LEARNING_GUIDANCE not in prompt
        assert MEMORY_GUIDANCE_WITHOUT_SKILL_CAPTURE in prompt
        assert SKILL_CAPTURE not in prompt
        assert "skill_manage(action='patch')" not in prompt

    def test_true_matches_the_default(self, make_agent):
        _write_config()
        default = _stable_prompt(make_agent())
        _write_config(agent={"skill_learning_guidance": True})

        assert _stable_prompt(make_agent()) == default

    def test_default_guidance_is_the_learning_and_safety_text_joined(self):
        assert SKILLS_GUIDANCE == f"{SKILLS_LEARNING_GUIDANCE}\n\n{SKILLS_SAFETY_GUIDANCE}"
        assert SKILL_CAPTURE in MEMORY_GUIDANCE
        assert SKILL_CAPTURE not in MEMORY_GUIDANCE_WITHOUT_SKILL_CAPTURE
        assert MEMORY_GUIDANCE_WITHOUT_SKILL_CAPTURE.endswith(
            "Procedures and workflows belong in skills, not memory."
        )


class TestCompactCategories:
    def test_configured_category_lists_names_only(self, make_agent):
        _write_skill("connectors", "hubspot", "Use HubSpot CRM well")
        _write_skill("connectors", "gmail", "Use Gmail well")
        _write_skill("research", "market-map", "Map the market")
        _write_config(skills={"compact_categories": ["connectors"]})

        prompt = _stable_prompt(make_agent())

        assert "  connectors [names only]: gmail, hubspot" in prompt
        assert "Use HubSpot CRM well" not in prompt
        assert "    - market-map: Map the market" in prompt
        assert NEUTRAL_NAMES_ONLY_NOTE in prompt

    def test_nested_categories_follow_their_top_level_segment(self, make_agent):
        _write_skill("connectors/crm", "hubspot", "Use HubSpot CRM well")
        _write_config(skills={"compact_categories": ["connectors"]})

        prompt = _stable_prompt(make_agent())

        assert "connectors/crm [names only]: hubspot" in prompt
        assert "Use HubSpot CRM well" not in prompt

    @pytest.mark.parametrize("skills", [{}, {"compact_categories": []},
                                        {"compact_categories": "connectors"},
                                        {"compact_categories": [" ", ""]}])
    def test_missing_empty_or_malformed_values_leave_the_index_untouched(
        self, make_agent, skills
    ):
        _write_skill("connectors", "hubspot", "Use HubSpot CRM well")
        _write_config(skills=skills)

        prompt = _stable_prompt(make_agent())

        assert "    - hubspot: Use HubSpot CRM well" in prompt
        assert "[names only]" not in prompt
        assert NEUTRAL_NAMES_ONLY_NOTE not in prompt
