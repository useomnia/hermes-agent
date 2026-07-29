"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    HERMES_AGENT_HELP_GUIDANCE,
    STEER_CHANNEL_NOTE,
)
from agent.system_prompt import build_system_prompt_parts
from hermes_cli.default_soul import DEFAULT_SOUL_MD
from run_agent import AIAgent


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False, context_length=None):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _prompt_parts_for_disclosure_tests(agent, *, soul=""):
    environment_hints = MagicMock(
        side_effect=lambda *, terminal_backend_hint=True: (
            "Terminal backend: sprites" if terminal_backend_hint else ""
        )
    )
    profile_resolver = MagicMock(return_value="tenant-a")
    with (
        patch("run_agent.load_soul_md", return_value=soul),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", environment_hints),
        patch("run_agent.build_context_files_prompt", return_value=""),
        patch(
            "agent.file_safety._resolve_active_profile_name",
            profile_resolver,
        ),
        patch("hermes_time.now", return_value=datetime(2026, 7, 29)),
    ):
        parts = build_system_prompt_parts(agent)
    return parts, environment_hints, profile_resolver


def _real_identity_slot(monkeypatch, hermes_home, *, fallback_identity=""):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    agent = _make_agent(
        _agent_help_guidance=False,
        _fallback_identity=fallback_identity,
        _profile_hint=False,
        _terminal_backend_hint=False,
        _model_info_hint=False,
    )
    with (
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        stable = build_system_prompt_parts(agent)["stable"]
    return stable.split("\n\n", 1)[0]


class TestFallbackIdentityRealSoulPath:
    def test_unwritable_home_without_fallback_uses_default_identity(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "not-a-directory"
        hermes_home.write_text("blocks directory creation", encoding="utf-8")

        identity = _real_identity_slot(monkeypatch, hermes_home)

        assert identity == DEFAULT_AGENT_IDENTITY

    def test_unwritable_home_with_fallback_uses_configured_identity(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "not-a-directory"
        hermes_home.write_text("blocks directory creation", encoding="utf-8")
        fallback_identity = "You are Acme, a test assistant."

        identity = _real_identity_slot(
            monkeypatch,
            hermes_home,
            fallback_identity=fallback_identity,
        )

        assert identity == fallback_identity

    def test_auto_seeded_soul_without_fallback_preserves_default_identity(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "hermes-home"

        identity = _real_identity_slot(monkeypatch, hermes_home)

        assert (hermes_home / "SOUL.md").read_text(encoding="utf-8").strip() == (
            DEFAULT_SOUL_MD.strip()
        )
        assert identity == DEFAULT_AGENT_IDENTITY

    def test_auto_seeded_soul_with_fallback_uses_configured_identity(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "hermes-home"
        fallback_identity = "You are Acme, a test assistant."

        identity = _real_identity_slot(
            monkeypatch,
            hermes_home,
            fallback_identity=fallback_identity,
        )

        assert (hermes_home / "SOUL.md").read_text(encoding="utf-8").strip() == (
            DEFAULT_SOUL_MD.strip()
        )
        assert identity == fallback_identity

    def test_custom_soul_without_fallback_uses_custom_identity(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        custom_identity = "You are the identity authored in SOUL.md."
        (hermes_home / "SOUL.md").write_text(custom_identity, encoding="utf-8")

        identity = _real_identity_slot(monkeypatch, hermes_home)

        assert identity == custom_identity

    def test_custom_soul_with_fallback_keeps_custom_identity(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        custom_identity = "You are the identity authored in SOUL.md."
        (hermes_home / "SOUL.md").write_text(custom_identity, encoding="utf-8")

        identity = _real_identity_slot(
            monkeypatch,
            hermes_home,
            fallback_identity="You are Acme, a test assistant.",
        )

        assert identity == custom_identity


class TestPromptDisclosureHints:
    def test_defaults_preserve_prompt_byte_for_byte(self):
        common = dict(
            model="vendor/model",
            provider="vendor",
            pass_session_id=True,
            session_id="session-123",
        )
        implicit_parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(**common)
        )
        explicit_parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                **common,
                _profile_hint=True,
                _terminal_backend_hint=True,
                _model_info_hint=True,
                _agent_help_guidance=True,
                _fallback_identity="",
            )
        )

        assert implicit_parts == explicit_parts
        assert implicit_parts["stable"].startswith(DEFAULT_AGENT_IDENTITY)
        assert HERMES_AGENT_HELP_GUIDANCE in implicit_parts["stable"]
        assert "Active Hermes profile: tenant-a" in implicit_parts["stable"]
        assert "Terminal backend: sprites" in implicit_parts["stable"]
        assert implicit_parts["volatile"].endswith(
            "Conversation started: Wednesday, July 29, 2026\n"
            "Session ID: session-123\n"
            "Model: vendor/model\n"
            "Provider: vendor"
        )

    def test_profile_hint_can_be_suppressed_without_resolving_profile(self):
        parts, environment_hints, profile_resolver = (
            _prompt_parts_for_disclosure_tests(
                _make_agent(
                    _profile_hint=False,
                    model="vendor/model",
                    provider="vendor",
                )
            )
        )

        prompt = "\n\n".join(parts.values())
        assert "Active Hermes profile" not in prompt
        profile_resolver.assert_not_called()
        environment_hints.assert_called_once_with(terminal_backend_hint=True)
        assert "Terminal backend: sprites" in parts["stable"]
        assert "Model: vendor/model" in parts["volatile"]

    def test_terminal_backend_hint_can_be_suppressed_independently(self):
        parts, environment_hints, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                _terminal_backend_hint=False,
                model="vendor/model",
                provider="vendor",
            )
        )

        prompt = "\n\n".join(parts.values())
        assert "Terminal backend:" not in prompt
        environment_hints.assert_called_once_with(terminal_backend_hint=False)
        assert "Active Hermes profile: tenant-a" in parts["stable"]
        assert "Model: vendor/model" in parts["volatile"]

    def test_model_info_hint_preserves_timestamp_and_session_id(self):
        parts, environment_hints, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                _model_info_hint=False,
                model="vendor/model",
                provider="vendor",
                pass_session_id=True,
                session_id="session-123",
            )
        )

        assert parts["volatile"].endswith(
            "Conversation started: Wednesday, July 29, 2026\n"
            "Session ID: session-123"
        )
        assert "\nModel:" not in parts["volatile"]
        assert "\nProvider:" not in parts["volatile"]
        assert "Active Hermes profile: tenant-a" in parts["stable"]
        environment_hints.assert_called_once_with(terminal_backend_hint=True)

    def test_model_info_hint_suppresses_alibaba_model_identity(self):
        parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                _model_info_hint=False,
                model="alibaba/qwen3-coder-plus",
                provider="alibaba",
            )
        )

        prompt = "\n\n".join(parts.values())
        assert "qwen3-coder-plus" not in prompt
        assert "You are powered by the model named" not in prompt

    def test_alibaba_model_identity_present_when_model_info_hint_defaults_true(self):
        parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                model="alibaba/qwen3-coder-plus",
                provider="alibaba",
            )
        )

        assert "You are powered by the model named qwen3-coder-plus." in parts["stable"]
        assert "The exact model ID is alibaba/qwen3-coder-plus." in parts["stable"]

    def test_agent_help_guidance_can_be_suppressed_independently(self):
        parts, environment_hints, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                _agent_help_guidance=False,
                model="vendor/model",
                provider="vendor",
            ),
            soul="You are a neutral assistant.",
        )

        stable = parts["stable"]
        assert HERMES_AGENT_HELP_GUIDANCE not in stable
        for vendor_text in (
            "Hermes Agent",
            "Nous Research",
            "nousresearch.com",
            "hermes-agent",
        ):
            assert vendor_text not in stable
        assert "Active Hermes profile: tenant-a" in stable
        assert "Terminal backend: sprites" in stable
        assert "Model: vendor/model" in parts["volatile"]
        environment_hints.assert_called_once_with(terminal_backend_hint=True)

    def test_agent_help_guidance_is_forwarded_to_skills_prompt(self):
        skills_prompt = MagicMock(return_value="Neutral skills prompt")
        with (
            patch("run_agent.build_skills_system_prompt", skills_prompt),
            patch("run_agent.get_toolset_for_tool", return_value="skills"),
            patch(
                "agent.coding_context.coding_compact_skill_categories",
                return_value=frozenset(),
            ),
        ):
            parts, _, _ = _prompt_parts_for_disclosure_tests(
                _make_agent(
                    valid_tool_names=["skill_view"],
                    _agent_help_guidance=False,
                )
            )

        assert "Neutral skills prompt" in parts["stable"]
        skills_prompt.assert_called_once_with(
            available_tools=["skill_view"],
            available_toolsets={"skills"},
            compact_categories=None,
            agent_help_guidance=False,
        )

    def test_custom_fallback_identity_replaces_the_default_verbatim(self):
        fallback_identity = "You are Acme, a test assistant."
        parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                _fallback_identity=fallback_identity,
                _agent_help_guidance=False,
                _profile_hint=False,
            )
        )

        assert parts["stable"].split("\n\n", 1)[0] == fallback_identity
        assert "Hermes Agent" not in parts["stable"]
        assert "Nous Research" not in parts["stable"]

    def test_soul_identity_takes_precedence_over_custom_fallback(self):
        fallback_identity = "You are Acme, a test assistant."
        soul_identity = "You are the identity loaded from SOUL.md."
        parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(
                _fallback_identity=fallback_identity,
                _agent_help_guidance=False,
                _profile_hint=False,
            ),
            soul=soul_identity,
        )

        assert parts["stable"].split("\n\n", 1)[0] == soul_identity
        assert fallback_identity not in parts["stable"]

    def test_steering_note_remains_for_agents_with_tools(self):
        parts, _, _ = _prompt_parts_for_disclosure_tests(
            _make_agent(valid_tool_names=["read_file"])
        )

        assert STEER_CHANNEL_NOTE in parts["stable"]
        assert "Trust ONLY this exact marker" in parts["stable"]
        assert "Hermes" not in STEER_CHANNEL_NOTE


class TestPromptDisclosureConfig:
    def test_flags_default_true_and_read_false_from_agent_config(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["agent"]["profile_hint"] is True
        assert DEFAULT_CONFIG["agent"]["terminal_backend_hint"] is True
        assert DEFAULT_CONFIG["agent"]["model_info_hint"] is True
        assert DEFAULT_CONFIG["agent"]["agent_help_guidance"] is True
        assert DEFAULT_CONFIG["agent"]["fallback_identity"] == ""

        def make_initialized_agent(agent_config):
            with (
                patch("run_agent.get_tool_definitions", return_value=[]),
                patch("run_agent.check_toolset_requirements", return_value={}),
                patch("run_agent.OpenAI"),
                patch("hermes_logging.setup_logging"),
                patch(
                    "hermes_cli.config.load_config",
                    return_value={"agent": agent_config},
                ),
            ):
                return AIAgent(
                    model="anthropic/claude-opus-4.8",
                    api_key="test-key-1234567890",
                    base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True,
                    skip_context_files=True,
                    skip_memory=True,
                    enabled_toolsets=[],
                )

        defaults = make_initialized_agent({})
        assert defaults._profile_hint is True
        assert defaults._terminal_backend_hint is True
        assert defaults._model_info_hint is True
        assert defaults._agent_help_guidance is True
        assert defaults._fallback_identity == ""

        fallback_identity = "You are Acme, a test assistant."
        disabled = make_initialized_agent(
            {
                "profile_hint": False,
                "terminal_backend_hint": False,
                "model_info_hint": False,
                "agent_help_guidance": False,
                "fallback_identity": fallback_identity,
            }
        )
        assert disabled._profile_hint is False
        assert disabled._terminal_backend_hint is False
        assert disabled._model_info_hint is False
        assert disabled._agent_help_guidance is False
        assert disabled._fallback_identity == fallback_identity


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)
