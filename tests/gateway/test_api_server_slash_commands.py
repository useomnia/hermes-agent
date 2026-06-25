"""Unit tests for ``/skill-name`` slash-command expansion on /v1/chat/completions.

The OpenAI chat path honors slash commands like every other Hermes surface: a
recognized ``/<command> [instruction]`` is expanded into its skill-invocation
payload before the agent runs, while anything else (a path, a question about
``/etc``, an unknown command) passes through untouched.  These tests exercise
``APIServerAdapter._maybe_expand_slash_command`` directly — it carries no
instance state, so it is constructed via ``__new__`` and the skill-resolution
functions it imports lazily are patched at their source modules.
"""

from unittest.mock import patch

from gateway.platforms.api_server import APIServerAdapter

SESSION_ID = "api_123_abcd"


def _adapter() -> APIServerAdapter:
    # _maybe_expand_slash_command touches no instance attributes; skip __init__.
    return APIServerAdapter.__new__(APIServerAdapter)


def _patch_skills(*, skill_key=None, skill_msg=None, bundle_key=None, bundle_result=None):
    """Patch the bundle + skill resolvers/builders the method imports lazily."""
    return (
        patch("agent.skill_bundles.resolve_bundle_command_key", return_value=bundle_key),
        patch("agent.skill_bundles.build_bundle_invocation_message", return_value=bundle_result),
        patch("agent.skill_commands.resolve_skill_command_key", return_value=skill_key),
        patch("agent.skill_commands.build_skill_invocation_message", return_value=skill_msg),
    )


class TestNoExpansion:
    def test_non_string_content_passes_through(self):
        # Multimodal content is a list — never a slash command.
        content = [{"type": "text", "text": "/site-audit"}]
        assert _adapter()._maybe_expand_slash_command(content, SESSION_ID) is None

    def test_message_without_leading_slash_passes_through(self):
        assert _adapter()._maybe_expand_slash_command("hello there", SESSION_ID) is None

    def test_lone_slash_passes_through(self):
        assert _adapter()._maybe_expand_slash_command("/", SESSION_ID) is None
        assert _adapter()._maybe_expand_slash_command("/   ", SESSION_ID) is None

    def test_unrecognized_command_passes_through(self):
        # A "/"-leading message that matches no bundle and no skill (e.g. a path
        # like "/Users/x") must be forwarded untouched, not rejected.
        patches = _patch_skills(skill_key=None, bundle_key=None)
        with patches[0], patches[1], patches[2], patches[3]:
            assert _adapter()._maybe_expand_slash_command("/Users/pablo/notes", SESSION_ID) is None

    def test_skill_resolves_but_build_returns_none_passes_through_and_logs(self, caplog):
        # A command that resolves to a real skill but builds no payload is a
        # failure, not a non-match: it still passes through, but loudly.
        patches = _patch_skills(skill_key="/site-audit", skill_msg=None, bundle_key=None)
        with caplog.at_level("ERROR"), patches[0], patches[1], patches[2], patches[3]:
            assert _adapter()._maybe_expand_slash_command("/site-audit", SESSION_ID) is None
        assert any("resolved but built no payload" in record.message for record in caplog.records)

    def test_bundle_resolves_but_build_returns_none_passes_through_and_logs(self, caplog):
        patches = _patch_skills(skill_key=None, bundle_key="/research", bundle_result=None)
        with caplog.at_level("ERROR"), patches[0], patches[1], patches[2], patches[3]:
            assert _adapter()._maybe_expand_slash_command("/research", SESSION_ID) is None
        assert any("resolved but built no payload" in record.message for record in caplog.records)

    def test_resolver_exception_passes_through(self):
        with patch("agent.skill_bundles.resolve_bundle_command_key", return_value=None), \
             patch("agent.skill_commands.resolve_skill_command_key", side_effect=RuntimeError("boom")):
            assert _adapter()._maybe_expand_slash_command("/site-audit", SESSION_ID) is None


class TestSkillExpansion:
    def test_known_skill_is_expanded(self):
        patches = _patch_skills(skill_key="/site-audit", skill_msg="EXPANDED PAYLOAD", bundle_key=None)
        with patches[0], patches[1], patches[2], patches[3] as build:
            out = _adapter()._maybe_expand_slash_command("/site-audit run on example.com", SESSION_ID)
        assert out == "EXPANDED PAYLOAD"
        # command stripped of slash; trailing instruction + session id threaded through.
        build.assert_called_once_with("/site-audit", "run on example.com", task_id=SESSION_ID)

    def test_skill_without_instruction_uses_empty_string(self):
        patches = _patch_skills(skill_key="/site-audit", skill_msg="X", bundle_key=None)
        with patches[0], patches[1], patches[2], patches[3] as build:
            out = _adapter()._maybe_expand_slash_command("/site-audit", SESSION_ID)
        assert out == "X"
        build.assert_called_once_with("/site-audit", "", task_id=SESSION_ID)

    def test_leading_whitespace_before_slash_is_tolerated(self):
        patches = _patch_skills(skill_key="/site-audit", skill_msg="X", bundle_key=None)
        with patches[0], patches[1], patches[2], patches[3] as build:
            out = _adapter()._maybe_expand_slash_command("   /site-audit go", SESSION_ID)
        assert out == "X"
        build.assert_called_once_with("/site-audit", "go", task_id=SESSION_ID)

    def test_multiline_instruction_is_preserved(self):
        patches = _patch_skills(skill_key="/site-audit", skill_msg="X", bundle_key=None)
        with patches[0], patches[1], patches[2], patches[3] as build:
            _adapter()._maybe_expand_slash_command("/site-audit\nline one\nline two", SESSION_ID)
        build.assert_called_once_with("/site-audit", "line one\nline two", task_id=SESSION_ID)

    def test_resolver_receives_command_resolution(self):
        patches = _patch_skills(skill_key="/site-audit", skill_msg="X", bundle_key=None)
        with patches[0], patches[1], patches[2] as resolve, patches[3]:
            _adapter()._maybe_expand_slash_command("/site_audit now", SESSION_ID)
        # The raw command token (underscores and all) is handed to the resolver,
        # which owns underscore->hyphen normalization.
        resolve.assert_called_once_with("site_audit")


class TestBundlePrecedence:
    def test_bundle_wins_over_skill(self):
        patches = _patch_skills(
            skill_key="/research",
            skill_msg="SKILL PAYLOAD",
            bundle_key="/research",
            bundle_result=("BUNDLE PAYLOAD", ["a", "b"], []),
        )
        with patches[0], patches[1] as build_bundle, patches[2], patches[3] as build_skill:
            out = _adapter()._maybe_expand_slash_command("/research deep dive", SESSION_ID)
        assert out == "BUNDLE PAYLOAD"
        build_bundle.assert_called_once_with("/research", "deep dive", task_id=SESSION_ID)
        build_skill.assert_not_called()

    def test_bundle_build_failure_falls_back_to_skill(self):
        # Bundle resolves but produces no message -> fall through to skill.
        patches = _patch_skills(
            skill_key="/research",
            skill_msg="SKILL PAYLOAD",
            bundle_key="/research",
            bundle_result=None,
        )
        with patches[0], patches[1], patches[2], patches[3]:
            out = _adapter()._maybe_expand_slash_command("/research", SESSION_ID)
        assert out == "SKILL PAYLOAD"


class TestLearnCommand:
    def test_learn_with_instruction_is_rewritten_to_the_learn_prompt(self):
        with patch("agent.learn_prompt.build_learn_prompt", return_value="LEARN PROMPT") as build:
            out = _adapter()._maybe_expand_slash_command("/learn from this repo dir", SESSION_ID)
        assert out == "LEARN PROMPT"
        build.assert_called_once_with("from this repo dir")

    def test_learn_without_instruction_passes_empty_string(self):
        with patch("agent.learn_prompt.build_learn_prompt", return_value="LEARN PROMPT") as build:
            out = _adapter()._maybe_expand_slash_command("/learn", SESSION_ID)
        assert out == "LEARN PROMPT"
        build.assert_called_once_with("")

    def test_learn_is_dispatched_before_skill_and_bundle_resolution(self):
        with (
            patch("agent.learn_prompt.build_learn_prompt", return_value="LEARN PROMPT"),
            patch("agent.skill_commands.resolve_skill_command_key") as resolve_skill,
            patch("agent.skill_bundles.resolve_bundle_command_key") as resolve_bundle,
        ):
            out = _adapter()._maybe_expand_slash_command("/learn x", SESSION_ID)
        assert out == "LEARN PROMPT"
        resolve_skill.assert_not_called()
        resolve_bundle.assert_not_called()

    def test_learn_prompt_build_failure_passes_through(self):
        with patch("agent.learn_prompt.build_learn_prompt", side_effect=RuntimeError("boom")):
            assert _adapter()._maybe_expand_slash_command("/learn x", SESSION_ID) is None
