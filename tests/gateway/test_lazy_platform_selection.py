"""Behavioral coverage for manifest-guided deferred platform selection."""

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platform_registry import PlatformEntry, platform_registry


@pytest.fixture
def isolated_registry():
    """Keep the process singleton isolated while exercising config loading."""
    with platform_registry._lock:
        original_entries = dict(platform_registry._entries)
        original_deferred = dict(platform_registry._deferred)
        original_deferred_specs = dict(platform_registry._deferred_specs)
        original_deferred_env = dict(platform_registry._deferred_env)
        original_deferred_activation_env = dict(
            platform_registry._deferred_activation_env
        )
        original_resolving = dict(platform_registry._resolving)
        platform_registry._entries.clear()
        platform_registry._deferred.clear()
        platform_registry._deferred_specs.clear()
        platform_registry._deferred_env.clear()
        platform_registry._deferred_activation_env.clear()
        platform_registry._resolving.clear()
    try:
        with patch("hermes_cli.plugins.discover_plugins", lambda *a, **k: None):
            yield platform_registry
    finally:
        with platform_registry._lock:
            platform_registry._entries.clear()
            platform_registry._entries.update(original_entries)
            platform_registry._deferred.clear()
            platform_registry._deferred.update(original_deferred)
            platform_registry._deferred_specs.clear()
            platform_registry._deferred_specs.update(original_deferred_specs)
            platform_registry._deferred_env.clear()
            platform_registry._deferred_env.update(original_deferred_env)
            platform_registry._deferred_activation_env.clear()
            platform_registry._deferred_activation_env.update(
                original_deferred_activation_env
            )
            platform_registry._resolving.clear()
            platform_registry._resolving.update(original_resolving)


def test_empty_gateway_config_does_not_import_deferred_platform(
    tmp_path, monkeypatch, isolated_registry
):
    loader = MagicMock()
    isolated_registry.register_deferred(
        "lazy-platform",
        loader,
        required_env=("LAZY_PLATFORM_TOKEN",),
    )
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("LAZY_PLATFORM_TOKEN", raising=False)

    load_gateway_config()

    loader.assert_not_called()


def test_explicit_yaml_platform_loads_and_keeps_yaml_hook_semantics(
    tmp_path, monkeypatch, isolated_registry
):
    loader_calls = []

    def loader():
        loader_calls.append("loaded")
        isolated_registry.register(
            PlatformEntry(
                name="lazy-platform",
                label="Lazy Platform",
                adapter_factory=lambda cfg: object(),
                check_fn=lambda: True,
                apply_yaml_config_fn=lambda yaml_cfg, platform_cfg: {
                    "hook_seen": platform_cfg.get("enabled") is True,
                },
                source="plugin",
            )
        )

    isolated_registry.register_deferred(
        "lazy-platform",
        loader,
        required_env=("LAZY_PLATFORM_TOKEN",),
    )
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n"
        "  lazy-platform:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("LAZY_PLATFORM_TOKEN", raising=False)

    config = load_gateway_config()

    assert loader_calls == ["loaded"]
    lazy = config.platforms[Platform("lazy-platform")]
    assert lazy.enabled is True
    assert lazy.extra["hook_seen"] is True


def test_activation_env_loads_only_matching_deferred_platform(
    isolated_registry, monkeypatch
):
    loader_calls = []

    def loader():
        loader_calls.append("loaded")
        isolated_registry.register(
            PlatformEntry(
                name="env-platform",
                label="Env Platform",
                adapter_factory=lambda cfg: object(),
                check_fn=lambda: True,
                is_connected=lambda cfg: True,
                source="plugin",
            )
        )

    isolated_registry.register_deferred(
        "env-platform",
        loader,
        required_env=("ENV_PLATFORM_TOKEN",),
    )
    monkeypatch.setenv("ENV_PLATFORM_TOKEN", "configured")

    config = GatewayConfig()
    # The normal env override path is covered here through the public helper
    # so the existing is_connected/check_fn enablement contract stays intact.
    from gateway.config import _apply_env_overrides

    _apply_env_overrides(config)

    assert loader_calls == ["loaded"]
    assert config.platforms[Platform("env-platform")].enabled is True
