"""Regression tests: ``_apply_env_overrides`` must not lazy-install platform
SDKs for platforms the user has not configured.

For adapter plugins, ``PlatformEntry.check_fn`` doubles as the lazy-installer
(it pip-installs the platform SDK as a side effect — see e.g.
``plugins/platforms/discord/adapter.py::check_discord_requirements``).  The
enablement sweep in ``_apply_env_overrides`` used to call ``check_fn`` for
*every* registered plugin platform unconditionally, so a single
``load_gateway_config()`` — which the desktop/dashboard readiness probe
(``GET /api/status``) awaits synchronously — pip-installed Discord, Telegram,
Slack, Feishu and Dingtalk even with ``platforms: none``.  That blocked
startup until every install finished and made the desktop app time out and
boot-loop (stuck at 94%).

The fix consults the cheap ``is_connected`` credential check FIRST and only
runs the install-triggering ``check_fn`` for platforms that are already
enabled or actually configured.  These tests pin that contract.
"""

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platform_registry import PlatformEntry, platform_registry


@pytest.fixture
def isolated_registry():
    """Run with a registry containing only the entries the test registers."""
    original = dict(platform_registry._entries)
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
        # ``_apply_env_overrides`` calls ``discover_plugins()`` (idempotent),
        # which would re-register the real bundled platforms and clobber the
        # fakes below.  Neutralize it so the test controls the registry.
        with patch("hermes_cli.plugins.discover_plugins", lambda *a, **k: None):
            yield platform_registry
    finally:
        platform_registry._entries.clear()
        platform_registry._entries.update(original)
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


def _register_fake_platform(name, *, check_fn, is_connected):
    platform_registry.register(
        PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=check_fn,
            is_connected=is_connected,
            source="plugin",
        )
    )


def test_unconfigured_platform_is_not_probed_for_install(isolated_registry):
    # is_connected reports "no credentials" → the platform must be skipped
    # without ever calling check_fn (which would lazy-install the SDK).
    check_fn = MagicMock(return_value=True)
    _register_fake_platform(
        "discord", check_fn=check_fn, is_connected=lambda cfg: False
    )

    config = GatewayConfig()
    _apply_env_overrides(config)

    check_fn.assert_not_called()
    assert not config.platforms.get(Platform.DISCORD, PlatformConfig()).enabled


def test_configured_platform_is_still_installed_and_enabled(isolated_registry):
    # is_connected reports "credentials present" → check_fn must run (so the
    # SDK is verified/installed) and the platform is auto-enabled, exactly as
    # before the fix.
    check_fn = MagicMock(return_value=True)
    _register_fake_platform(
        "discord", check_fn=check_fn, is_connected=lambda cfg: True
    )

    config = GatewayConfig()
    _apply_env_overrides(config)

    check_fn.assert_called_once()
    assert config.platforms[Platform.DISCORD].enabled is True


def test_failed_install_does_not_enable_configured_platform(isolated_registry):
    # Credentials present but the SDK genuinely cannot be installed/imported
    # (check_fn returns False) → platform must not be enabled.
    check_fn = MagicMock(return_value=False)
    _register_fake_platform(
        "discord", check_fn=check_fn, is_connected=lambda cfg: True
    )

    config = GatewayConfig()
    _apply_env_overrides(config)

    check_fn.assert_called_once()
    assert not config.platforms.get(Platform.DISCORD, PlatformConfig()).enabled


def test_unconfigured_deferred_platform_is_not_imported(isolated_registry):
    """Manifest candidates stay lazy when neither config nor activation env exists."""
    loader = MagicMock()
    platform_registry.register_deferred(
        "lazy-platform",
        loader,
        required_env=("LAZY_PLATFORM_TOKEN",),
    )

    _apply_env_overrides(GatewayConfig())

    loader.assert_not_called()


def test_env_candidate_is_loaded_for_existing_enablement_hooks(isolated_registry, monkeypatch):
    """An activation env selects only that deferred platform for normal hooks."""
    loader_calls = []
    check_fn = MagicMock(return_value=True)

    def loader():
        loader_calls.append("loaded")
        platform_registry.register(
            PlatformEntry(
                name="env-platform",
                label="Env Platform",
                adapter_factory=lambda cfg: MagicMock(),
                check_fn=check_fn,
                is_connected=lambda cfg: True,
                source="plugin",
            )
        )

    platform_registry.register_deferred(
        "env-platform",
        loader,
        required_env=("ENV_PLATFORM_TOKEN",),
    )
    monkeypatch.setenv("ENV_PLATFORM_TOKEN", "configured")

    config = GatewayConfig()
    _apply_env_overrides(config)

    assert loader_calls == ["loaded"]
    assert check_fn.call_count == 1
    assert config.platforms[Platform("env-platform")].enabled is True
