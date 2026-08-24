"""Tests for the platform adapter registry and dynamic Platform enum."""

import os
import threading
import time
import pytest
from unittest.mock import MagicMock

import gateway.platform_registry as platform_registry_module
from gateway.platform_registry import PlatformRegistry, PlatformEntry
from gateway.config import Platform, GatewayConfig


# ── Platform enum dynamic members ─────────────────────────────────────────


class TestPlatformEnumDynamic:
    """Test that Platform enum accepts unknown values for plugin platforms."""

    def test_builtin_members_still_work(self):
        assert Platform.TELEGRAM.value == "telegram"
        assert Platform("telegram") is Platform.TELEGRAM

    def test_dynamic_member_created(self):
        p = Platform("irc")
        assert p.value == "irc"
        assert p.name == "IRC"

    def test_dynamic_member_identity_stable(self):
        """Same value returns same object (cached)."""
        a = Platform("irc")
        b = Platform("irc")
        assert a is b

    def test_dynamic_member_case_normalised(self):
        """Mixed case normalised to lowercase."""
        a = Platform("IRC")
        b = Platform("irc")
        assert a is b
        assert a.value == "irc"

    def test_dynamic_member_with_hyphens(self):
        """Registered plugin platforms with hyphens work once registered."""
        from gateway.platform_registry import platform_registry as _reg

        entry = PlatformEntry(
            name="my-platform",
            label="My Platform",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="plugin",
        )
        _reg.register(entry)
        try:
            p = Platform("my-platform")
            assert p.value == "my-platform"
            assert p.name == "MY_PLATFORM"
        finally:
            _reg.unregister("my-platform")

    def test_dynamic_member_rejects_unregistered(self):
        """Arbitrary strings are rejected to prevent enum pollution."""
        with pytest.raises(ValueError):
            Platform("totally-fake-platform")

    def test_dynamic_member_rejects_non_string(self):
        with pytest.raises(ValueError):
            Platform(123)

    def test_dynamic_member_rejects_empty(self):
        with pytest.raises(ValueError):
            Platform("")

    def test_dynamic_member_rejects_whitespace_only(self):
        with pytest.raises(ValueError):
            Platform("   ")


# ── PlatformRegistry ──────────────────────────────────────────────────────


class TestPlatformRegistry:
    """Test the PlatformRegistry itself."""

    def _make_entry(self, name="test", check_ok=True, validate_ok=True, factory_ok=True):
        adapter_mock = MagicMock()
        return PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg, _m=adapter_mock: _m if factory_ok else (_ for _ in ()).throw(RuntimeError("factory error")),
            check_fn=lambda: check_ok,
            validate_config=lambda cfg: validate_ok,
            required_env=[],
            source="plugin",
        ), adapter_mock

    def test_register_and_get(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("alpha")
        reg.register(entry)
        assert reg.get("alpha") is entry
        assert reg.is_registered("alpha")

    def test_get_unknown_returns_none(self):
        reg = PlatformRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("beta")
        reg.register(entry)
        assert reg.unregister("beta") is True
        assert reg.get("beta") is None
        assert reg.unregister("beta") is False  # already gone

    def test_create_adapter_success(self):
        reg = PlatformRegistry()
        entry, mock_adapter = self._make_entry("gamma")
        reg.register(entry)
        result = reg.create_adapter("gamma", MagicMock())
        assert result is mock_adapter

    def test_create_adapter_unknown_name(self):
        reg = PlatformRegistry()
        assert reg.create_adapter("unknown", MagicMock()) is None

    def test_create_adapter_check_fails(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("delta", check_ok=False)
        reg.register(entry)
        assert reg.create_adapter("delta", MagicMock()) is None

    def test_create_adapter_validate_fails(self):
        reg = PlatformRegistry()
        entry, _ = self._make_entry("epsilon", validate_ok=False)
        reg.register(entry)
        assert reg.create_adapter("epsilon", MagicMock()) is None

    def test_create_adapter_factory_exception(self):
        reg = PlatformRegistry()
        entry = PlatformEntry(
            name="broken",
            label="Broken",
            adapter_factory=lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")),
            check_fn=lambda: True,
            validate_config=None,
            source="plugin",
        )
        reg.register(entry)
        # factory raises → create_adapter returns None instead of propagating
        assert reg.create_adapter("broken", MagicMock()) is None

    def test_create_adapter_no_validate(self):
        """When validate_config is None, skip validation."""
        reg = PlatformRegistry()
        mock_adapter = MagicMock()
        entry = PlatformEntry(
            name="novalidate",
            label="NoValidate",
            adapter_factory=lambda cfg: mock_adapter,
            check_fn=lambda: True,
            validate_config=None,
            source="plugin",
        )
        reg.register(entry)
        assert reg.create_adapter("novalidate", MagicMock()) is mock_adapter

    def test_all_entries(self):
        reg = PlatformRegistry()
        e1, _ = self._make_entry("one")
        e2, _ = self._make_entry("two")
        reg.register(e1)
        reg.register(e2)
        names = {e.name for e in reg.all_entries()}
        assert names == {"one", "two"}

    def test_plugin_entries(self):
        reg = PlatformRegistry()
        plugin_entry, _ = self._make_entry("plugged")
        builtin_entry = PlatformEntry(
            name="core",
            label="Core",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="builtin",
        )
        reg.register(plugin_entry)
        reg.register(builtin_entry)
        plugin_names = {e.name for e in reg.plugin_entries()}
        assert plugin_names == {"plugged"}

    def test_re_register_replaces(self):
        reg = PlatformRegistry()
        entry1, mock1 = self._make_entry("dup")
        entry2 = PlatformEntry(
            name="dup",
            label="Dup v2",
            adapter_factory=lambda cfg: "v2",
            check_fn=lambda: True,
            source="plugin",
        )
        reg.register(entry1)
        reg.register(entry2)
        assert reg.get("dup").label == "Dup v2"

    def test_deferred_platform_resolution_is_single_flight(self):
        """Concurrent first-use callers all observe the loaded platform."""
        reg = PlatformRegistry()
        loader_started = threading.Event()
        release_loader = threading.Event()
        start_gate = threading.Barrier(5)
        loader_calls = []
        resolved = []
        errors = []

        entry = PlatformEntry(
            name="lazy",
            label="Lazy",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="plugin",
        )

        def loader():
            loader_calls.append(1)
            loader_started.set()
            assert release_loader.wait(timeout=2.0)
            reg.register(entry)

        reg.register_deferred("lazy", loader)

        def worker():
            try:
                start_gate.wait(timeout=2.0)
                resolved.append(reg.get("lazy") is entry)
            except BaseException as exc:  # pragma: no cover - failure detail
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        start_gate.wait(timeout=2.0)
        assert loader_started.wait(timeout=2.0)
        assert reg.is_registered("lazy")
        release_loader.set()
        for thread in threads:
            thread.join(timeout=2.0)

        assert not errors
        assert loader_calls == [1]
        assert resolved == [True] * 4

    def test_deferred_resolution_timeout_fails_closed_without_partial_entry(
        self, monkeypatch
    ):
        """A wedged loader cannot deadlock callers or expose half-registration."""
        reg = PlatformRegistry()
        loader_started = threading.Event()
        release_loader = threading.Event()
        owner_done = threading.Event()
        entry = PlatformEntry(
            name="stuck",
            label="Stuck",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="plugin",
        )

        def loader():
            loader_started.set()
            release_loader.wait(timeout=2.0)
            reg.register(entry)

        reg.register_deferred("stuck", loader)

        def resolve_as_owner():
            reg.get("stuck")
            owner_done.set()

        owner = threading.Thread(target=resolve_as_owner)
        owner.start()
        assert loader_started.wait(timeout=2.0)

        monkeypatch.setattr(
            platform_registry_module,
            "DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS",
            0.01,
        )
        started = time.monotonic()
        assert reg.get("stuck") is None
        assert time.monotonic() - started < 0.5
        assert "stuck" not in reg._entries
        assert reg.is_registered("stuck")

        release_loader.set()
        owner.join(timeout=2.0)
        assert owner_done.is_set()
        assert reg.get("stuck") is entry

    def test_candidate_resolution_uses_manifest_env_or_explicit_config(self):
        """Startup selection stays lazy while explicit/configured use loads."""
        reg = PlatformRegistry()
        loaded = []

        def make_loader(name):
            def loader():
                loaded.append(name)
                reg.register(
                    PlatformEntry(
                        name=name,
                        label=name.title(),
                        adapter_factory=lambda cfg: MagicMock(),
                        check_fn=lambda: True,
                        source="plugin",
                    )
                )

            return loader

        reg.register_deferred(
            "env-platform",
            make_loader("env-platform"),
            required_env=("ENV_PLATFORM_TOKEN",),
            optional_env=("ENV_PLATFORM_ALLOWED_USERS",),
        )
        reg.register_deferred(
            "configured-platform",
            make_loader("configured-platform"),
            required_env=("CONFIGURED_PLATFORM_TOKEN",),
        )

        assert reg.plugin_entries(resolve_deferred=False) == []
        assert reg.resolve_candidates(environ={"ENV_PLATFORM_TOKEN": "yes"}) == (
            "env-platform",
        )
        assert loaded == ["env-platform"]
        assert reg.plugin_entries(resolve_deferred=False)[0].name == "env-platform"

        assert reg.resolve_candidates({"configured-platform"}) == (
            "configured-platform",
        )
        assert loaded == ["env-platform", "configured-platform"]

    def test_activation_alternatives_extend_required_env(self):
        reg = PlatformRegistry()
        calls = []

        def loader():
            calls.append("loaded")
            reg.register(
                PlatformEntry(
                    name="multi-mode",
                    label="Multi Mode",
                    adapter_factory=lambda cfg: object(),
                    check_fn=lambda: True,
                    source="plugin",
                )
            )

        reg.register_deferred(
            "multi-mode",
            loader,
            required_env=("NORMAL_MODE_TOKEN",),
            activation_env=("CALLBACK_MODE_TOKEN",),
        )

        assert reg.resolve_candidates(environ={"NORMAL_MODE_TOKEN": "yes"}) == (
            "multi-mode",
        )
        assert calls == ["loaded"]

    def test_deferred_manifest_allowlist_names_are_available_without_import(self):
        reg = PlatformRegistry()
        reg.register_deferred(
            "lazy-auth",
            lambda: None,
            required_env=("LAZY_AUTH_TOKEN", "LAZY_AUTH_ALLOWED_USERS"),
            optional_env=("LAZY_AUTH_ALLOW_ALL_USERS",),
        )

        assert reg.plugin_entries(resolve_deferred=False) == []
        assert reg.plugin_authorization_env_vars() == (
            ("LAZY_AUTH_ALLOWED_USERS",),
            ("LAZY_AUTH_ALLOW_ALL_USERS",),
        )

    def test_full_plugin_enumeration_still_materializes_deferred_entries(self):
        reg = PlatformRegistry()
        entry = PlatformEntry(
            name="enumerated",
            label="Enumerated",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            source="plugin",
        )
        reg.register_deferred("enumerated", lambda: reg.register(entry))

        assert [loaded.name for loaded in reg.plugin_entries()] == ["enumerated"]

    def test_secondary_aliases_share_one_loader_and_publish_atomically(self):
        reg = PlatformRegistry()
        calls = []
        primary = PlatformEntry(
            name="primary",
            label="Primary",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )
        secondary = PlatformEntry(
            name="secondary",
            label="Secondary",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )

        def loader():
            calls.append("load")
            reg.register(primary)
            reg.register(secondary)

        reg.register_deferred("primary", loader, aliases=("secondary",))

        assert reg.get("secondary") is secondary
        assert reg.get("primary") is primary
        assert calls == ["load"]

    def test_failed_deferred_load_retries_without_leaking_partial_entries(self):
        reg = PlatformRegistry()
        attempts = []
        partial = PlatformEntry(
            name="retryable",
            label="Partial",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )
        complete = PlatformEntry(
            name="retryable",
            label="Complete",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )

        def loader():
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                reg.register(partial)
                raise RuntimeError("first import failed")
            reg.register(complete)

        reg.register_deferred("retryable", loader)

        assert reg.get("retryable") is None
        assert "retryable" not in reg._entries
        assert reg.get("retryable") is complete
        assert attempts == [1, 2]

    def test_unregister_invalidates_inflight_loader(self):
        reg = PlatformRegistry()
        loader_started = threading.Event()
        release_loader = threading.Event()
        entry = PlatformEntry(
            name="unregistered",
            label="Unregistered",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )

        def loader():
            loader_started.set()
            release_loader.wait(timeout=2.0)
            reg.register(entry)

        reg.register_deferred("unregistered", loader)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(reg.get("unregistered"))
        )
        worker.start()
        assert loader_started.wait(timeout=2.0)
        assert reg.unregister("unregistered") is False
        release_loader.set()
        worker.join(timeout=2.0)

        assert result == [None]
        assert reg.get("unregistered") is None
        assert not reg.is_registered("unregistered")

    def test_new_generation_wins_over_inflight_loader(self):
        reg = PlatformRegistry()
        old_started = threading.Event()
        release_old = threading.Event()
        old_finished = threading.Event()
        stale_registration = []
        old_entry = PlatformEntry(
            name="generation",
            label="Old",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )
        new_entry = PlatformEntry(
            name="generation",
            label="New",
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: True,
            source="plugin",
        )

        def old_loader():
            old_started.set()
            release_old.wait(timeout=2.0)
            # The registry must reject this stale publication.
            stale_registration.append(reg.register(old_entry))
            old_finished.set()

        reg.register_deferred("generation", old_loader)
        worker = threading.Thread(target=lambda: reg.get("generation"))
        worker.start()
        assert old_started.wait(timeout=2.0)

        reg.register_deferred("generation", lambda: reg.register(new_entry))
        release_old.set()
        assert old_finished.wait(timeout=2.0)
        worker.join(timeout=2.0)

        assert stale_registration == [False]
        assert reg.get("generation") is new_entry
        assert reg.get("generation").label == "New"

    def test_full_enumeration_starts_independent_loaders_before_waiting(self):
        reg = PlatformRegistry()
        started = {name: threading.Event() for name in ("one", "two")}
        release = threading.Event()

        def make_loader(name):
            def loader():
                started[name].set()
                release.wait(timeout=2.0)
                reg.register(
                    PlatformEntry(
                        name=name,
                        label=name,
                        adapter_factory=lambda cfg: object(),
                        check_fn=lambda: True,
                        source="plugin",
                    )
                )

            return loader

        reg.register_deferred("one", make_loader("one"))
        reg.register_deferred("two", make_loader("two"))

        result = []
        enumerator = threading.Thread(
            target=lambda: result.extend(entry.name for entry in reg.plugin_entries())
        )
        enumerator.start()
        assert started["one"].wait(timeout=2.0)
        assert started["two"].wait(timeout=2.0)
        release.set()
        enumerator.join(timeout=2.0)

        assert set(result) == {"one", "two"}


# ── GatewayConfig integration ────────────────────────────────────────────


class TestGatewayConfigPluginPlatform:
    """Test that GatewayConfig parses and validates plugin platforms."""

    def test_from_dict_accepts_plugin_platform(self):
        data = {
            "platforms": {
                "telegram": {"enabled": True, "token": "test-token"},
                "irc": {"enabled": True, "extra": {"server": "irc.libera.chat"}},
            }
        }
        cfg = GatewayConfig.from_dict(data)
        platform_values = {p.value for p in cfg.platforms}
        assert "telegram" in platform_values
        assert "irc" in platform_values

    def test_get_connected_platforms_includes_registered_plugin(self):
        """Plugin platform with registry entry passes get_connected_platforms."""
        # Register a fake plugin platform
        from gateway.platform_registry import platform_registry as _reg

        test_entry = PlatformEntry(
            name="testplat",
            label="TestPlat",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            validate_config=lambda cfg: bool(cfg.extra.get("token")),
            source="plugin",
        )
        _reg.register(test_entry)
        try:
            data = {
                "platforms": {
                    "testplat": {"enabled": True, "extra": {"token": "abc"}},
                }
            }
            cfg = GatewayConfig.from_dict(data)
            connected = cfg.get_connected_platforms()
            connected_values = {p.value for p in connected}
            assert "testplat" in connected_values
        finally:
            _reg.unregister("testplat")

    def test_get_connected_platforms_excludes_unregistered_plugin(self):
        """Plugin platform without registry entry is excluded."""
        data = {
            "platforms": {
                "unknown_plugin": {"enabled": True, "extra": {"token": "abc"}},
            }
        }
        cfg = GatewayConfig.from_dict(data)
        connected = cfg.get_connected_platforms()
        connected_values = {p.value for p in connected}
        assert "unknown_plugin" not in connected_values

    def test_get_connected_platforms_excludes_invalid_config(self):
        """Plugin platform with failing validate_config is excluded."""
        from gateway.platform_registry import platform_registry as _reg

        test_entry = PlatformEntry(
            name="badconfig",
            label="BadConfig",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=lambda: True,
            validate_config=lambda cfg: False,  # always fails
            source="plugin",
        )
        _reg.register(test_entry)
        try:
            data = {
                "platforms": {
                    "badconfig": {"enabled": True, "extra": {}},
                }
            }
            cfg = GatewayConfig.from_dict(data)
            connected = cfg.get_connected_platforms()
            connected_values = {p.value for p in connected}
            assert "badconfig" not in connected_values
        finally:
            _reg.unregister("badconfig")


# ── Extended PlatformEntry fields ─────────────────────────────────────


class TestPlatformEntryExtendedFields:
    """Test the auth, message length, and display fields on PlatformEntry."""

    def test_default_field_values(self):
        entry = PlatformEntry(
            name="test",
            label="Test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
        assert entry.allowed_users_env == ""
        assert entry.allow_all_env == ""
        assert entry.max_message_length == 0
        assert entry.pii_safe is False
        assert entry.emoji == "🔌"
        assert entry.allow_update_command is True

    def test_custom_auth_fields(self):
        entry = PlatformEntry(
            name="irc",
            label="IRC",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            allowed_users_env="IRC_ALLOWED_USERS",
            allow_all_env="IRC_ALLOW_ALL_USERS",
            max_message_length=450,
            pii_safe=False,
            emoji="💬",
        )
        assert entry.allowed_users_env == "IRC_ALLOWED_USERS"
        assert entry.allow_all_env == "IRC_ALLOW_ALL_USERS"
        assert entry.max_message_length == 450
        assert entry.emoji == "💬"


# ── Cron platform resolution ─────────────────────────────────────────


class TestCronPlatformResolution:
    """Test that cron delivery accepts plugin platform names."""

    def test_builtin_platform_resolves(self):
        """Built-in platform names resolve via Platform() call."""
        p = Platform("telegram")
        assert p is Platform.TELEGRAM

    def test_plugin_platform_resolves(self):
        """Plugin platform names create dynamic enum members."""
        p = Platform("irc")
        assert p.value == "irc"

    def test_invalid_platform_type_rejected(self):
        """Non-string values are still rejected."""
        with pytest.raises(ValueError):
            Platform(None)


# ── platforms.py integration ──────────────────────────────────────────


class TestPlatformsMerge:
    """Test get_all_platforms() merges with registry."""

    def test_get_all_platforms_includes_builtins(self):
        from hermes_cli.platforms import get_all_platforms, PLATFORMS
        merged = get_all_platforms()
        for key in PLATFORMS:
            assert key in merged

    def test_get_all_platforms_includes_plugin(self):
        from hermes_cli.platforms import get_all_platforms
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="testmerge",
            label="TestMerge",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            emoji="🧪",
        ))
        try:
            merged = get_all_platforms()
            assert "testmerge" in merged
            assert "TestMerge" in merged["testmerge"].label
        finally:
            _reg.unregister("testmerge")

    def test_platform_label_plugin_fallback(self):
        from hermes_cli.platforms import platform_label
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="labeltest",
            label="LabelTest",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            emoji="🏷️",
        ))
        try:
            label = platform_label("labeltest")
            assert "LabelTest" in label
        finally:
            _reg.unregister("labeltest")


# ── apply_yaml_config_fn (PlatformEntry field + load_gateway_config dispatch) ──


class TestApplyYamlConfigFnField:
    """The hook field itself — defaults, custom values, signature."""

    def test_default_is_none(self):
        entry = PlatformEntry(
            name="test",
            label="Test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
        assert entry.apply_yaml_config_fn is None

    def test_accepts_callable(self):
        def _hook(yaml_cfg, platform_cfg):
            return None

        entry = PlatformEntry(
            name="test",
            label="Test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            apply_yaml_config_fn=_hook,
        )
        assert entry.apply_yaml_config_fn is _hook
        # Sanity-check the signature contract.
        assert entry.apply_yaml_config_fn({"x": 1}, {"y": 2}) is None


class TestApplyYamlConfigFnDispatch:
    """End-to-end dispatch through load_gateway_config().

    Each test registers a temporary PlatformEntry, writes a config.yaml in
    a tmp HERMES_HOME, calls load_gateway_config(), and asserts the hook
    was invoked correctly.  Cleanup unregisters the entry.
    """

    def _write_config(self, tmp_path, content: str):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home

    def _register_hook(self, name, hook_fn):
        from gateway.platform_registry import platform_registry as _reg

        entry = PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=hook_fn,
        )
        _reg.register(entry)
        return _reg

    def test_hook_can_mutate_environ(self, tmp_path, monkeypatch):
        """A hook that mutates os.environ has its env vars set after load."""
        env_var = "MYHOOKPLAT_FLAG"
        monkeypatch.delenv(env_var, raising=False)

        def _hook(yaml_cfg, platform_cfg):
            if "flag" in platform_cfg and not os.getenv(env_var):
                os.environ[env_var] = str(platform_cfg["flag"]).lower()
            return None

        reg = self._register_hook("myhookplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "myhookplat:\n  flag: true\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert os.environ.get(env_var) == "true"
        finally:
            reg.unregister("myhookplat")
            os.environ.pop(env_var, None)

    def test_hook_returned_dict_merges_into_extra(self, tmp_path, monkeypatch):
        """A hook that returns a dict has it merged into PlatformConfig.extra."""

        def _hook(yaml_cfg, platform_cfg):
            return {"seeded_key": "seeded_value", "flag": platform_cfg.get("flag")}

        reg = self._register_hook("myextraplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "myextraplat:\n  flag: yes\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            cfg = load_gateway_config()

            plat = Platform("myextraplat")
            assert plat in cfg.platforms
            extra = cfg.platforms[plat].extra
            assert extra.get("seeded_key") == "seeded_value"
            # flag value carried through from yaml_cfg arg.
            assert extra.get("flag") is True
        finally:
            reg.unregister("myextraplat")

    def test_hook_receives_full_yaml_and_platform_subdict(
        self, tmp_path, monkeypatch
    ):
        """Hook receives both the full yaml_cfg and its own platform sub-dict."""
        captured: dict = {}

        def _hook(yaml_cfg, platform_cfg):
            captured["yaml_cfg"] = yaml_cfg
            captured["platform_cfg"] = platform_cfg
            return None

        reg = self._register_hook("mycaptureplat", _hook)
        try:
            home = self._write_config(
                tmp_path,
                "top_level_key: 1\n"
                "mycaptureplat:\n"
                "  inner_key: deep\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert captured["yaml_cfg"].get("top_level_key") == 1
            assert captured["platform_cfg"] == {"inner_key": "deep"}
        finally:
            reg.unregister("mycaptureplat")

    def test_hook_exception_swallowed(self, tmp_path, monkeypatch):
        """A misbehaving hook never aborts load_gateway_config()."""

        def _bad_hook(yaml_cfg, platform_cfg):
            raise RuntimeError("plugin author bug")

        # Also register a well-behaved hook to ensure dispatch continues
        # iterating after a bad one.
        good_called = {"count": 0}

        def _good_hook(yaml_cfg, platform_cfg):
            good_called["count"] += 1
            return None

        from gateway.platform_registry import platform_registry as _reg
        _reg.register(PlatformEntry(
            name="mybadplat",
            label="MyBad",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=_bad_hook,
        ))
        _reg.register(PlatformEntry(
            name="mygoodplat",
            label="MyGood",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            apply_yaml_config_fn=_good_hook,
        ))
        try:
            home = self._write_config(
                tmp_path,
                "mybadplat:\n  k: v\n"
                "mygoodplat:\n  k: v\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            # Must not raise.
            from gateway.config import load_gateway_config
            load_gateway_config()

            assert good_called["count"] == 1
        finally:
            _reg.unregister("mybadplat")
            _reg.unregister("mygoodplat")

    def test_hook_skipped_when_platform_section_missing(
        self, tmp_path, monkeypatch
    ):
        """Hook is NOT called when the platform's YAML section is absent."""
        called = {"count": 0}

        def _hook(yaml_cfg, platform_cfg):
            called["count"] += 1
            return None

        reg = self._register_hook("myabsentplat", _hook)
        try:
            home = self._write_config(tmp_path, "telegram:\n  k: v\n")
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert called["count"] == 0
        finally:
            reg.unregister("myabsentplat")

    def test_hook_skipped_when_platform_section_not_dict(
        self, tmp_path, monkeypatch
    ):
        """Hook is NOT called when the platform's YAML section isn't a dict."""
        called = {"count": 0}

        def _hook(yaml_cfg, platform_cfg):
            called["count"] += 1
            return None

        reg = self._register_hook("mybadshapeplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "mybadshapeplat: just-a-string\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            assert called["count"] == 0
        finally:
            reg.unregister("mybadshapeplat")

    def test_env_var_takes_precedence_when_hook_uses_getenv_guard(
        self, tmp_path, monkeypatch
    ):
        """The standard `not os.getenv(...)` guard preserves env > YAML."""
        env_var = "MYPRECPLAT_FLAG"
        monkeypatch.setenv(env_var, "preexisting")

        def _hook(yaml_cfg, platform_cfg):
            if "flag" in platform_cfg and not os.getenv(env_var):
                os.environ[env_var] = str(platform_cfg["flag"]).lower()
            return None

        reg = self._register_hook("myprecplat", _hook)
        try:
            home = self._write_config(
                tmp_path, "myprecplat:\n  flag: yaml-value\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config
            load_gateway_config()

            # Pre-existing env var was NOT clobbered by the hook.
            assert os.environ.get(env_var) == "preexisting"
        finally:
            reg.unregister("myprecplat")
            os.environ.pop(env_var, None)


class TestPluginPlatformSharedKeyBridge:
    """Plugin-registered platforms get the same shared-key bridging as built-ins.

    Without this, plugin authors using ``apply_yaml_config_fn`` would have to
    re-implement bridging for every common key (``unauthorized_dm_behavior``,
    ``notice_delivery``, ``reply_prefix``, ``require_mention``, ``dm_policy``,
    ``allow_from``, etc.) — defeating the hook's whole point of letting
    plugins focus on their *platform-specific* keys.
    """

    def _write_config(self, tmp_path, content: str):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home

    def test_shared_keys_bridged_for_plugin_platform(self, tmp_path, monkeypatch):
        """A plugin platform's ``require_mention``/``dm_policy``/etc. flow into
        ``PlatformConfig.extra`` without the plugin needing its own bridge."""
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="mysharedplat",
            label="MySharedPlat",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
        ))
        try:
            home = self._write_config(
                tmp_path,
                "mysharedplat:\n"
                "  require_mention: true\n"
                "  dm_policy: allow\n"
                "  reply_prefix: \"→ \"\n"
                "  allow_from: [\"alice\", \"bob\"]\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("mysharedplat")
            assert plat in cfg.platforms
            extra = cfg.platforms[plat].extra
            assert extra.get("require_mention") is True
            assert extra.get("dm_policy") == "allow"
            assert extra.get("reply_prefix") == "→ "
            assert extra.get("allow_from") == ["alice", "bob"]
        finally:
            _reg.unregister("mysharedplat")


class TestPluginEnablementGate:
    """Plugin platforms must NOT auto-enable on check_fn alone (#31116).

    When a plugin registers ``is_connected`` (the "did the user actually
    configure credentials" probe), ``load_gateway_config`` must consult it
    before flipping ``enabled = True``.  Without this gate, ``check_fn``
    semantics ("the SDK is importable") get conflated with "the user wants
    this platform on", and the gateway tries to connect to e.g. Discord
    with no token — emitting noisy retry-forever errors on every fresh
    install that has the plugin loaded.
    """

    def _write_config(self, tmp_path, content: str = ""):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home

    def test_plugin_with_is_connected_false_is_NOT_enabled(
        self, tmp_path, monkeypatch
    ):
        """check_fn=True + is_connected=False must NOT enable the platform.

        Reproduces #31116: Discord plugin loads, its check_fn lazy-installs
        discord.py and returns True, but the user has no DISCORD_BOT_TOKEN.
        Previously this auto-enabled Discord and the gateway spammed
        ``ERROR ... [Discord] No bot token configured`` on every reconnect.
        """
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="myunconfiguredplat",
            label="MyUnconfigured",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,             # SDK available
            is_connected=lambda cfg: False,    # but user hasn't set credentials
            source="plugin",
        ))
        try:
            home = self._write_config(tmp_path)
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("myunconfiguredplat")
            # Either absent entirely, or present but explicitly disabled.
            if plat in cfg.platforms:
                assert cfg.platforms[plat].enabled is False, (
                    "Plugin with is_connected=False must NOT be auto-enabled"
                )
        finally:
            _reg.unregister("myunconfiguredplat")

    def test_plugin_with_is_connected_true_is_enabled(
        self, tmp_path, monkeypatch
    ):
        """check_fn=True + is_connected=True still enables the platform."""
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="myconfiguredplat",
            label="MyConfigured",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            is_connected=lambda cfg: True,
            source="plugin",
        ))
        try:
            home = self._write_config(tmp_path)
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("myconfiguredplat")
            assert plat in cfg.platforms
            assert cfg.platforms[plat].enabled is True
        finally:
            _reg.unregister("myconfiguredplat")

    def test_plugin_without_is_connected_falls_back_to_check_fn(
        self, tmp_path, monkeypatch
    ):
        """Legacy plugins that don't register is_connected keep working.

        For plugins where ``is_connected is None``, gating on ``check_fn``
        alone remains the contract — that's what callers without a
        credential probe have always done.
        """
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="mylegacyplat",
            label="MyLegacy",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            # is_connected intentionally omitted (None)
            source="plugin",
        ))
        try:
            home = self._write_config(tmp_path)
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("mylegacyplat")
            assert plat in cfg.platforms
            assert cfg.platforms[plat].enabled is True
        finally:
            _reg.unregister("mylegacyplat")

    def test_is_connected_raises_does_not_enable(self, tmp_path, monkeypatch):
        """A buggy is_connected must not silently enable the platform.

        Treat a raising is_connected as "configuration unknown" — refuse to
        enable, log, and move on.  Anything else would re-introduce the
        #31116 bug for plugins whose probe has a transient failure.
        """
        from gateway.platform_registry import platform_registry as _reg

        def _bad_probe(cfg):
            raise RuntimeError("plugin bug")

        _reg.register(PlatformEntry(
            name="mybadprobeplat",
            label="MyBadProbe",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            is_connected=_bad_probe,
            source="plugin",
        ))
        try:
            home = self._write_config(tmp_path)
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("mybadprobeplat")
            if plat in cfg.platforms:
                assert cfg.platforms[plat].enabled is False
        finally:
            _reg.unregister("mybadprobeplat")

    def test_yaml_enabled_true_overrides_is_connected_false(
        self, tmp_path, monkeypatch
    ):
        """Explicit YAML ``enabled: true`` wins over is_connected=False.

        If the user wrote ``platforms.X.enabled: true`` themselves, respect
        that — they may be using a credential mechanism the plugin's
        is_connected probe doesn't know about.  Don't fight them.
        """
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="myexplicitplat",
            label="MyExplicit",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            is_connected=lambda cfg: False,
            source="plugin",
        ))
        try:
            home = self._write_config(
                tmp_path,
                "platforms:\n"
                "  myexplicitplat:\n"
                "    enabled: true\n",
            )
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("myexplicitplat")
            assert plat in cfg.platforms
            assert cfg.platforms[plat].enabled is True, (
                "Explicit YAML enabled: true must win over plugin's "
                "is_connected=False — user has the final say"
            )
        finally:
            _reg.unregister("myexplicitplat")

    def test_is_connected_sees_env_seeded_extras(self, tmp_path, monkeypatch):
        """``env_enablement_fn`` extras must be visible to ``is_connected``.

        Some plugins (e.g. Google Chat) implement ``is_connected`` by
        inspecting ``config.extra`` (where ``env_enablement_fn`` deposits
        env-var-derived state) rather than reading ``os.environ`` directly.
        If the gate runs BEFORE the seeding step, those plugins fail the
        gate even when the user is genuinely configured via env vars.

        Pin the contract: when both hooks are present, ``env_enablement_fn``
        feeds a candidate config to ``is_connected``.
        """
        from gateway.platform_registry import platform_registry as _reg

        seen_extras: dict = {}

        def _is_connected(cfg):
            seen_extras["snapshot"] = dict(getattr(cfg, "extra", {}) or {})
            extra = getattr(cfg, "extra", {}) or {}
            return bool(extra.get("project_id") and extra.get("subscription_name"))

        def _env_enablement():
            return {"project_id": "p", "subscription_name": "s"}

        _reg.register(PlatformEntry(
            name="myextrasplat",
            label="MyExtras",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            is_connected=_is_connected,
            env_enablement_fn=_env_enablement,
            source="plugin",
        ))
        try:
            home = self._write_config(tmp_path)
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("myextrasplat")
            assert plat in cfg.platforms, (
                "is_connected was called with empty extras — "
                "env_enablement_fn must seed the probe BEFORE the gate"
            )
            assert cfg.platforms[plat].enabled is True
            # extras populated on the live config too
            assert cfg.platforms[plat].extra.get("project_id") == "p"
            assert cfg.platforms[plat].extra.get("subscription_name") == "s"
            # and the probe saw them
            assert seen_extras["snapshot"]["project_id"] == "p"
        finally:
            _reg.unregister("myextrasplat")

    def test_is_connected_failed_gate_does_not_leak_extras(
        self, tmp_path, monkeypatch
    ):
        """When the gate rejects, env-seeded extras must NOT leak onto
        ``config.platforms``.  A rejected plugin should be invisible, not
        present-but-partially-populated.
        """
        from gateway.platform_registry import platform_registry as _reg

        _reg.register(PlatformEntry(
            name="myrejectedplat",
            label="MyRejected",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            is_connected=lambda cfg: False,
            env_enablement_fn=lambda: {"some_key": "should-not-leak"},
            source="plugin",
        ))
        try:
            home = self._write_config(tmp_path)
            monkeypatch.setenv("HERMES_HOME", str(home))

            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()

            plat = Platform("myrejectedplat")
            if plat in cfg.platforms:
                assert cfg.platforms[plat].enabled is False
                assert "some_key" not in cfg.platforms[plat].extra, (
                    "Rejected plugin's env-seeded extras leaked onto "
                    "config.platforms"
                )
        finally:
            _reg.unregister("myrejectedplat")
