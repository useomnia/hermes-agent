"""OpenRouter preset context resolution and turn-start refresh tests."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import agent.model_metadata as model_metadata
from agent.model_metadata import (
    DEFAULT_FALLBACK_CONTEXT,
    OPENROUTER_PRESETS_URL,
    OpenRouterPresetContextResolution,
    get_model_context_length,
    resolve_openrouter_preset_context,
)
from agent.conversation_loop import _stored_prompt_matches_runtime
from agent.turn_context import _refresh_openrouter_preset_context
from tests.agent.test_turn_context import _FakeAgent, _build


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _preset_response(config, version="version-1"):
    response = MagicMock()
    response.json.return_value = {
        "data": {
            "designated_version_id": version,
            "designated_version": {"id": version, "config": config},
        }
    }
    return response


@pytest.fixture(autouse=True)
def _clear_preset_cache():
    model_metadata._openrouter_preset_context_cache.clear()
    yield
    model_metadata._openrouter_preset_context_cache.clear()


def test_single_model_preset_uses_authenticated_metadata_minimum():
    response = _preset_response({"model": "openai/gpt-5"})
    with (
        patch.object(model_metadata.requests, "get", return_value=response) as request,
        patch.object(
            model_metadata,
            "fetch_model_metadata",
            return_value={"openai/gpt-5": {"context_length": 400_000}},
        ),
    ):
        assert (
            get_model_context_length(
                "@preset/team",
                base_url=OPENROUTER_BASE_URL,
                provider="openrouter",
                api_key="openrouter-secret",
            )
            == 400_000
        )

    request.assert_called_once_with(
        f"{OPENROUTER_PRESETS_URL}/team",
        headers={"Authorization": "Bearer openrouter-secret"},
        timeout=(5, 10),
        verify=True,
    )


def test_models_list_uses_conservative_minimum_and_one_forced_refresh():
    response = _preset_response({"models": ["openai/large", "~openai/small"]})
    metadata = [
        {"openai/large": {"context_length": 400_000}},
        {
            "openai/large": {"context_length": 400_000},
            "openai/small": {"context_length": 128_000},
        },
    ]
    with (
        patch.object(model_metadata.requests, "get", return_value=response),
        patch.object(
            model_metadata, "fetch_model_metadata", side_effect=metadata
        ) as fetch,
    ):
        result = resolve_openrouter_preset_context(
            "@preset/team",
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            api_key="key",
        )

    assert result is not None
    assert result.context_length == 128_000
    assert result.candidate_count == 2
    assert fetch.call_args_list[1].kwargs == {"force_refresh": True}


@pytest.mark.parametrize(
    "api_key, request_side_effect",
    [
        ("", None),
        ("key", requests.RequestException("provider unavailable")),
    ],
)
def test_missing_auth_or_preset_api_failure_uses_exact_fallback(
    api_key, request_side_effect
):
    with patch.object(
        model_metadata.requests, "get", side_effect=request_side_effect
    ) as request:
        result = get_model_context_length(
            "@preset/team",
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            api_key=api_key,
        )

    assert result == DEFAULT_FALLBACK_CONTEXT == 1_000_000
    if not api_key:
        request.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [{}, {"model": ""}, {"models": []}, {"models": ["openai/gpt-5", 4]}],
)
def test_missing_or_invalid_candidates_use_fallback(config):
    response = _preset_response(config)
    with (
        patch.object(model_metadata.requests, "get", return_value=response),
        patch.object(model_metadata, "fetch_model_metadata") as fetch,
    ):
        assert (
            get_model_context_length(
                "@preset/team",
                base_url=OPENROUTER_BASE_URL,
                provider="openrouter",
                api_key="key",
            )
            == DEFAULT_FALLBACK_CONTEXT
        )
    fetch.assert_not_called()


def test_unknown_candidate_after_forced_refresh_uses_fallback():
    response = _preset_response({"model": "unknown/model"})
    with (
        patch.object(model_metadata.requests, "get", return_value=response),
        patch.object(
            model_metadata,
            "fetch_model_metadata",
            side_effect=[{}, {}],
        ) as fetch,
    ):
        assert (
            resolve_openrouter_preset_context(
                "@preset/team",
                base_url=OPENROUTER_BASE_URL,
                provider="openrouter",
                api_key="key",
            )
            is None
        )
    assert fetch.call_count == 2


def test_preset_cache_is_authenticated_short_lived_and_keeps_version():
    responses = [
        _preset_response({"model": "openai/model"}, version="version-1"),
        _preset_response({"model": "openai/model"}, version="version-2"),
    ]
    now = [100.0]
    metadata = [
        {"openai/model": {"context_length": 128_000}},
        {"openai/model": {"context_length": 256_000}},
    ]
    with (
        patch.object(model_metadata.time, "time", side_effect=lambda: now[0]),
        patch.object(model_metadata.requests, "get", side_effect=responses) as request,
        patch.object(model_metadata, "fetch_model_metadata", side_effect=metadata),
    ):
        first = resolve_openrouter_preset_context(
            "@preset/team",
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            api_key="secret-1",
        )
        now[0] = 101.0
        cached = resolve_openrouter_preset_context(
            "@preset/team",
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            api_key="secret-1",
        )
        now[0] = 161.0
        refreshed = resolve_openrouter_preset_context(
            "@preset/team",
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            api_key="secret-1",
        )

    assert first is not None and first.designated_version_id == "version-1"
    assert cached is not None
    assert cached.source == "cache" and cached.designated_version_id == "version-1"
    assert refreshed is not None
    assert refreshed.context_length == 256_000
    assert refreshed.designated_version_id == "version-2"
    assert request.call_count == 2
    cache_keys = list(model_metadata._openrouter_preset_context_cache)
    assert cache_keys == [
        ("team", model_metadata._openrouter_api_key_fingerprint("secret-1"))
    ]


def test_only_pure_openrouter_alias_is_resolved():
    with patch.object(model_metadata.requests, "get") as request:
        assert (
            resolve_openrouter_preset_context(
                "concrete-model@preset/team",
                base_url=OPENROUTER_BASE_URL,
                provider="openrouter",
                api_key="key",
            )
            is None
        )
        assert (
            resolve_openrouter_preset_context(
                "@preset/team",
                base_url="https://example.test/v1",
                provider="openrouter",
                api_key="key",
            )
            is None
        )
        request.assert_not_called()
        assert (
            resolve_openrouter_preset_context(
                "@preset/team",
                base_url=OPENROUTER_BASE_URL,
                provider="openrouter",
                api_key="key",
            )
            is None
        )
    request.assert_called_once()


def test_concrete_model_with_preset_suffix_uses_concrete_metadata_without_fetching_preset():
    with (
        patch.object(model_metadata.requests, "get") as request,
        patch.object(
            model_metadata,
            "fetch_model_metadata",
            return_value={"openai/example": {"context_length": 777_777}},
        ),
    ):
        context = get_model_context_length(
            "openai/example@preset/omnio",
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            api_key="key",
        )

    assert context == 777_777
    request.assert_not_called()


def test_generic_unknown_model_uses_1m_fallback():
    with patch.object(model_metadata, "fetch_model_metadata", return_value={}):
        assert get_model_context_length("unknown/model") == 1_000_000


def test_turn_start_context_change_updates_engine_and_prompt_cache():
    agent = _FakeAgent()
    agent.model = "@preset/team"
    agent.context_compressor.context_length = 256_000
    agent.context_compressor.update_model = MagicMock()
    agent._cached_system_prompt_static = "old static"

    def invalidate():
        agent._cached_system_prompt = None
        agent._cached_system_prompt_static = None

    agent._invalidate_system_prompt = MagicMock(side_effect=invalidate)
    resolution = OpenRouterPresetContextResolution(
        slug="team",
        context_length=512_000,
        designated_version_id="version-2",
        candidate_count=1,
    )
    with patch(
        "agent.turn_context.resolve_openrouter_preset_context",
        return_value=resolution,
    ):
        _refresh_openrouter_preset_context(agent)

    agent.context_compressor.update_model.assert_called_once_with(
        model="@preset/team",
        context_length=512_000,
        base_url=OPENROUTER_BASE_URL,
        api_key="sk-x",
        provider="openrouter",
        api_mode="chat_completions",
    )
    agent._invalidate_system_prompt.assert_called_once_with()
    assert agent._cached_system_prompt is None


def test_persisted_prompt_context_marker_detects_preset_change_across_agent_rebuild():
    agent = _FakeAgent()
    agent.model = "@preset/team"
    agent.context_compressor.context_length = 1_050_000
    prompt = "Model: @preset/team\nProvider: openrouter\nContext window: 256000"

    assert _stored_prompt_matches_runtime(agent, prompt) is False


def test_turn_start_unchanged_context_does_not_rebuild_prompt():
    agent = _FakeAgent()
    agent.model = "@preset/team"
    agent.context_compressor.context_length = 256_000
    agent.context_compressor.update_model = MagicMock()
    agent._invalidate_system_prompt = MagicMock()
    agent._openrouter_preset_context_length = 256_000
    resolution = OpenRouterPresetContextResolution(
        slug="team",
        context_length=256_000,
        designated_version_id="version-1",
        candidate_count=1,
    )
    with patch(
        "agent.turn_context.resolve_openrouter_preset_context",
        return_value=resolution,
    ):
        _refresh_openrouter_preset_context(agent)

    agent.context_compressor.update_model.assert_not_called()
    agent._invalidate_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == "SYSTEM"


def test_turn_start_resolution_failure_drops_previous_context_to_fallback():
    agent = _FakeAgent()
    agent.model = "@preset/team"
    agent.context_compressor.update_model = MagicMock()
    agent._invalidate_system_prompt = MagicMock()
    agent._openrouter_preset_context_length = 512_000
    agent.context_compressor.context_length = 512_000
    with patch(
        "agent.turn_context.resolve_openrouter_preset_context", return_value=None
    ):
        _refresh_openrouter_preset_context(agent)

    agent.context_compressor.update_model.assert_called_once()
    assert (
        agent.context_compressor.update_model.call_args.kwargs["context_length"]
        == 1_000_000
    )
    assert agent._openrouter_preset_context_length == 1_000_000


def test_turn_start_refresh_runs_before_prompt_assembly():
    agent = _FakeAgent()
    order = []
    agent._restore_primary_runtime = lambda: order.append("restore")
    agent._cached_system_prompt = None

    def restore_or_build(*_args):
        order.append("prompt")
        agent._cached_system_prompt = "new prompt"

    with patch(
        "agent.turn_context._refresh_openrouter_preset_context",
        side_effect=lambda _agent: order.append("refresh"),
    ):
        _build(agent, restore_or_build_system_prompt=restore_or_build)

    assert order.index("refresh") < order.index("prompt")
