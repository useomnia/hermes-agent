"""Fork-owned coverage for api_server provider-routing propagation."""

import logging

import pytest

from agent.chat_completion_helpers import _provider_preferences_for_agent
from agent.transports.chat_completions import ChatCompletionsTransport
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _FakeAgent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _create_adapter(
    monkeypatch,
    *,
    provider_routing=None,
    service_tier=None,
    model="test/model",
    use_real_config=False,
):
    """Build an adapter with deterministic _create_agent dependencies."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-test",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: model)
    if not use_real_config:
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
        monkeypatch.setattr(
            "gateway.run._load_gateway_runtime_config",
            lambda: {"provider_routing": provider_routing or {}},
        )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 10)
    monkeypatch.setattr(
        GatewayRunner,
        "_load_reasoning_config",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_load_provider_routing",
        staticmethod(
            lambda: (_ for _ in ()).throw(
                AssertionError("api_server must use profile-scoped runtime config")
            )
        ),
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_load_service_tier",
        staticmethod(lambda: service_tier),
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_complete",
        lambda: None,
    )

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    return adapter


def _emitted_provider_body(agent):
    preferences = _provider_preferences_for_agent(agent)
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=agent.model,
        messages=[{"role": "user", "content": "hello"}],
        is_openrouter=True,
        provider_preferences=preferences,
    )
    return kwargs.get("extra_body", {})


def _built_chat_completion_body(agent):
    return ChatCompletionsTransport().build_kwargs(
        model=agent.model,
        messages=[{"role": "user", "content": "hello"}],
        request_overrides=agent.request_overrides,
    )


def test_config_provider_routing_reaches_agent_and_openrouter_body(monkeypatch):
    routing = {
        "only": ["anthropic", "google"],
        "ignore": ["deepinfra"],
        "order": ["deepseek", "anthropic"],
        "sort": "price",
        "require_parameters": True,
        "data_collection": "deny",
    }
    adapter = _create_adapter(
        monkeypatch,
        provider_routing=routing,
        service_tier="priority",
    )

    agent = adapter._create_agent(session_id="config-routing")

    assert agent.providers_allowed == routing["only"]
    assert agent.providers_ignored == routing["ignore"]
    assert agent.providers_order == routing["order"]
    assert agent.provider_sort == "price"
    assert agent.provider_require_parameters is True
    assert agent.provider_data_collection == "deny"
    assert agent.service_tier == "priority"
    assert _emitted_provider_body(agent)["provider"] == routing


def test_unset_config_does_not_emit_provider_routing(monkeypatch):
    adapter = _create_adapter(monkeypatch)

    agent = adapter._create_agent(session_id="no-routing")

    assert agent.providers_allowed is None
    assert agent.providers_ignored is None
    assert agent.providers_order is None
    assert agent.provider_sort is None
    assert agent.provider_require_parameters is False
    assert agent.provider_data_collection is None
    assert agent.service_tier is None
    assert "provider" not in _emitted_provider_body(agent)


@pytest.mark.parametrize("malformed_config", ["enabled", ["deepseek"]])
def test_non_object_config_routing_warns_and_falls_back_to_empty(
    monkeypatch,
    caplog,
    malformed_config,
):
    adapter = _create_adapter(monkeypatch, provider_routing=malformed_config)

    with caplog.at_level(logging.WARNING):
        agent = adapter._create_agent(session_id="malformed-config-routing")

    assert agent.providers_allowed is None
    assert agent.providers_order is None
    assert agent.provider_sort is None
    assert "provider" not in _emitted_provider_body(agent)
    assert "Ignoring invalid provider_routing config: expected an object" in caplog.text


def test_config_service_tier_reaches_supported_provider_request(monkeypatch):
    adapter = _create_adapter(
        monkeypatch,
        service_tier="priority",
        model="gpt-5.4",
    )

    agent = adapter._create_agent(session_id="config-fast-mode")

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority"}
    assert _built_chat_completion_body(agent)["service_tier"] == "priority"


@pytest.mark.parametrize(
    "model_options",
    [{"service_tier": "priority"}, {"fast": True}],
    ids=["service-tier", "fast-flag"],
)
def test_request_service_tier_reaches_supported_provider_request(
    monkeypatch,
    model_options,
):
    adapter = _create_adapter(monkeypatch, model="gpt-5.4")

    agent = adapter._create_agent(
        session_id="request-fast-mode",
        model_options=model_options,
    )

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority"}
    assert _built_chat_completion_body(agent)["service_tier"] == "priority"


def test_request_non_priority_service_tier_does_not_enable_fast_mode(monkeypatch):
    adapter = _create_adapter(monkeypatch, model="gpt-5.4")

    agent = adapter._create_agent(
        session_id="request-flex-tier",
        model_options={"service_tier": "flex"},
    )

    assert agent.service_tier == "flex"
    assert agent.request_overrides == {}
    assert "service_tier" not in _built_chat_completion_body(agent)


def test_config_service_tier_is_inert_for_unsupported_model(monkeypatch):
    adapter = _create_adapter(
        monkeypatch,
        service_tier="priority",
        model="deepseek/deepseek-chat",
    )

    agent = adapter._create_agent(session_id="unsupported-fast-mode")

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {}
    assert "service_tier" not in _built_chat_completion_body(agent)


def test_request_provider_routing_is_ignored_and_config_wins(monkeypatch):
    """A caller cannot steer provider routing — config.yaml is the only source.

    Routing is a per-model pin (Omnio pins DeepSeek first-party because cheap
    third-party OpenRouter hosts mangle V4's DSML tool-call format), so honoring
    a request-supplied block would let a caller silently unpin it.
    """
    config_routing = {
        "order": ["config-provider"],
        "sort": "price",
        "require_parameters": True,
        "data_collection": "deny",
    }
    adapter = _create_adapter(monkeypatch, provider_routing=config_routing)

    agent = adapter._create_agent(
        session_id="request-routing",
        model_options={
            "provider_routing": {
                "only": ["request-provider"],
                "order": ["request-provider"],
                "sort": "latency",
            }
        },
    )

    assert agent.providers_allowed is None
    assert agent.providers_order == ["config-provider"]
    assert agent.provider_sort == "price"
    assert agent.provider_data_collection == "deny"
    assert _emitted_provider_body(agent)["provider"] == config_routing
    assert config_routing["order"] == ["config-provider"]


def test_config_routing_uses_profile_runtime_config_with_env_expansion(
    monkeypatch,
    tmp_path,
):
    import yaml

    import gateway.run as gateway_run

    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "secondary"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"provider_routing": {"order": ["default-provider"], "sort": "price"}}
        ),
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "provider_routing": {
                    "order": ["${PROFILE_ROUTING_PROVIDER}"],
                    "sort": "${PROFILE_ROUTING_SORT}",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROFILE_ROUTING_PROVIDER", "profile-provider")
    monkeypatch.setenv("PROFILE_ROUTING_SORT", "latency")
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    adapter = _create_adapter(monkeypatch, use_real_config=True)

    with gateway_run._profile_runtime_scope(profile_home):
        agent = adapter._create_agent(session_id="secondary-profile-routing")

    assert agent.providers_order == ["profile-provider"]
    assert agent.provider_sort == "latency"


def test_config_routing_survives_malformed_request_block(monkeypatch):
    """A malformed request block is inert, not a way to clear the config pin."""
    config_routing = {
        "order": ["config-provider"],
        "sort": "price",
        "require_parameters": True,
    }
    adapter = _create_adapter(monkeypatch, provider_routing=config_routing)

    agent = adapter._create_agent(
        session_id="malformed-routing",
        model_options={"provider_routing": "deepseek"},
    )

    assert _emitted_provider_body(agent)["provider"] == config_routing
