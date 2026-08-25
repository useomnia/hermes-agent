"""
Platform Adapter Registry

Allows platform adapters (built-in and plugin) to self-register so the gateway
can discover and instantiate them without hardcoded if/elif chains.

Built-in adapters continue to use the existing if/elif in _create_adapter()
for now.  Plugin adapters register here via PluginContext.register_platform()
and are looked up first -- if nothing is found the gateway falls through to
the legacy code path.

Usage (plugin side):

    from gateway.platform_registry import platform_registry, PlatformEntry

    platform_registry.register(PlatformEntry(
        name="irc",
        label="IRC",
        adapter_factory=lambda cfg: IRCAdapter(cfg),
        check_fn=check_requirements,
        validate_config=lambda cfg: bool(cfg.extra.get("server")),
        required_env=["IRC_SERVER"],
        install_hint="pip install irc",
    ))

Usage (gateway side):

    adapter = platform_registry.create_adapter("irc", platform_config)
"""

import logging
import os
import threading
import time
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# A deferred platform loader may import a third-party SDK or run plugin
# registration code.  First-use callers must not be able to deadlock forever
# behind a broken loader, but the bound is deliberately generous enough for a
# cold import on a slow sandbox.  A timed-out waiter fails closed; the owner is
# still allowed to finish and publish the entry, so a later lookup can retry
# without ever observing a partially registered platform.
DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS = 30.0


@dataclass
class PlatformEntry:
    """Metadata and factory for a single platform adapter."""

    # Identifier used in config.yaml (e.g. "irc", "viber").
    name: str

    # Human-readable label (e.g. "IRC", "Viber").
    label: str

    # Factory callable: receives a PlatformConfig, returns an adapter instance.
    # Using a factory instead of a bare class lets plugins do custom init
    # (e.g. passing extra kwargs, wrapping in try/except).
    adapter_factory: Callable[[Any], Any]

    # Returns True when the platform's dependencies are available.
    check_fn: Callable[[], bool]

    # Optional: given a PlatformConfig, is it properly configured?
    # If None, the registry skips config validation and lets the adapter
    # fail at connect() time with a descriptive error.
    validate_config: Optional[Callable[[Any], bool]] = None

    # Optional: given a PlatformConfig, is the platform connected/enabled?
    # Used by ``GatewayConfig.get_connected_platforms()`` and setup UI status.
    # If None, falls back to ``validate_config`` or ``check_fn``.
    is_connected: Optional[Callable[[Any], bool]] = None

    # Env vars this platform needs (for ``hermes setup`` display).
    required_env: list = field(default_factory=list)

    # Hint shown when check_fn returns False.
    install_hint: str = ""

    # Optional setup function for interactive configuration.
    # Signature: () -> None (prompts user, saves env vars).
    # If None, falls back to _setup_standard_platform (needs token_var + vars)
    # or a generic "set these env vars" display.
    setup_fn: Optional[Callable[[], None]] = None

    # "builtin" or "plugin"
    source: str = "plugin"

    # Name of the plugin manifest that registered this entry (empty for
    # built-ins).  Used by ``hermes gateway setup`` to auto-enable the
    # owning plugin when the user configures its platform.
    plugin_name: str = ""

    # ── Auth env var names (for _is_user_authorized integration) ──
    # E.g. "IRC_ALLOWED_USERS" — checked for comma-separated user IDs.
    allowed_users_env: str = ""
    # E.g. "IRC_ALLOW_ALL_USERS" — if truthy, all users authorized.
    allow_all_env: str = ""

    # ── Message limits ──
    # Max message length for smart-chunking.  0 = no limit.
    max_message_length: int = 0

    # ── Privacy ──
    # If True, session descriptions redact PII (phone numbers, etc.)
    pii_safe: bool = False

    # ── Display ──
    # Emoji for CLI/gateway display (e.g. "💬")
    emoji: str = "🔌"

    # Whether this platform should appear in _UPDATE_ALLOWED_PLATFORMS
    # (allows /update command from this platform).
    allow_update_command: bool = True

    # ── LLM guidance ──
    # Platform hint injected into the system prompt (e.g. "You are on IRC.
    # Do not use markdown.").  Empty string = no hint.
    platform_hint: str = ""

    # ── Env-driven auto-configuration ──
    # Optional: read env vars, return a dict of ``PlatformConfig.extra`` fields
    # to seed when the platform is auto-enabled.  Called during
    # ``_apply_env_overrides`` BEFORE the adapter is constructed, so
    # ``gateway status`` etc. can reflect env-only configuration without
    # instantiating the adapter.  Return ``None`` (or an empty dict) to skip.
    # Signature: () -> Optional[dict[str, Any]]
    env_enablement_fn: Optional[Callable[[], Optional[dict]]] = None

    # ── YAML→env config bridge ──
    # Optional: translate this platform's ``config.yaml`` keys into env vars
    # and/or seed ``PlatformConfig.extra`` directly.  Lets a plugin own its
    # YAML config translation instead of forcing core ``gateway/config.py``
    # to know every platform's schema.
    #
    # Signature: (yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]
    # Called from ``load_gateway_config()`` after the generic shared-key loop
    # and before ``_apply_env_overrides``.  Mutating ``os.environ`` is allowed
    # (use ``not os.getenv(...)`` guards to preserve env > YAML precedence);
    # any returned dict is merged into ``PlatformConfig.extra``.  Exceptions
    # are caught and logged at debug level.
    # See website/docs/developer-guide/adding-platform-adapters.md for the
    # full contract and a worked example.
    apply_yaml_config_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None

    # Optional: home-channel env var name for cron/notification delivery
    # (e.g. ``"IRC_HOME_CHANNEL"``).  When set, ``cron.scheduler`` treats this
    # platform as a valid ``deliver=<name>`` target and reads the env var to
    # resolve the default chat/room ID.  Empty = no cron home-channel support.
    cron_deliver_env_var: str = ""

    # ── Standalone (out-of-process) sending ──
    # Optional: async coroutine that delivers a message without a live
    # gateway adapter.  Called by ``tools/send_message_tool._send_via_adapter``
    # when ``cron`` runs in a separate process from the gateway and the
    # in-process adapter weakref is therefore ``None``.
    #
    # Signature:
    #     async (pconfig, chat_id, message, *, thread_id=None,
    #            media_files=None, force_document=False) -> dict
    #
    # Returns ``{"success": True, "message_id": ...}`` on success or
    # ``{"error": str}`` on failure.  Plugin authors typically open an
    # ephemeral connection / acquire a fresh OAuth token, send, and close.
    # Without this hook, plugin platforms cannot serve as cron ``deliver=``
    # targets when the gateway is not co-resident with the cron process.
    standalone_sender_fn: Optional[Callable[..., Awaitable[dict]]] = None


@dataclass
class _DeferredRegistration:
    """One lazy loader registration, possibly covering multiple aliases."""

    loader: Callable[[], None]
    names: tuple[str, ...]
    required_env: tuple[str, ...]
    optional_env: tuple[str, ...]
    activation_env: tuple[str, ...]
    generation: int
    source: str = "plugin"


@dataclass
class _DeferredResolution:
    """State shared by all waiters resolving one deferred registration."""

    registration: _DeferredRegistration
    done: threading.Event = field(default_factory=threading.Event)
    owner_thread_id: int | None = None
    cancelled: bool = False
    success: bool = False
    published_entries: dict[str, PlatformEntry] = field(default_factory=dict)


class PlatformRegistry:
    """Central registry of platform adapters.

    Thread-safe for reads and deferred resolution. A loader is single-flight so
    concurrent first-use callers cannot observe a half-registered platform.
    """

    def __init__(self) -> None:
        self._entries: dict[str, PlatformEntry] = {}
        # Deferred platform loaders: name -> zero-arg callable that imports the
        # owning plugin module (which calls register() and populates _entries).
        #
        # Why this exists: platform adapter modules import heavy, platform-
        # specific SDKs at module level (lark_oapi, microsoft_teams, discord.py,
        # slack_bolt, ...). Eagerly loading all ~20 bundled platform plugins at
        # plugin-discovery time added several seconds to *every* `hermes`
        # invocation -- including plain `hermes chat`, which never touches any
        # gateway platform. Discovery now registers a cheap deferred loader per
        # platform; the real module is imported only when a registry lookup
        # actually asks for that platform (gateway start, cron delivery,
        # `hermes setup`/`gateway status`, send_message).
        self._deferred: dict[str, Callable[[], None]] = {}
        # The callable map remains deliberately simple for compatibility with
        # existing diagnostic callers; the registration records below carry
        # aliases, generations, and source ownership for safe invalidation.
        self._deferred_specs: dict[str, _DeferredRegistration] = {}
        # Manifest-declared environment names kept beside each deferred
        # loader.  Reading this metadata is intentionally cheap: it lets
        # config loading materialize only explicitly configured platforms or
        # platforms whose activation environment is present, while setup and
        # status can still request full enumeration explicitly.
        self._deferred_env: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self._deferred_activation_env: dict[str, tuple[str, ...]] = {}
        # A platform loader imports third-party SDKs and registers the real
        # entry as a side effect. Multiple startup/first-use paths can ask for
        # the same platform concurrently, so enforce per-name single-flight
        # resolution and allow concurrent callers to wait for that loader.
        self._lock = threading.RLock()
        # Every alias in a registration points to the same resolution object.
        # The id is ``None`` during the tiny window between starting a loader
        # thread and its first instruction; this still makes recursive lookups
        # fail closed.
        self._resolving: dict[str, _DeferredResolution] = {}
        self._generation = 0
        self._loader_local = threading.local()

    # -- deferred loading ----------------------------------------------------

    def register_deferred(
        self,
        name: str,
        loader: Callable[[], None],
        *,
        aliases: tuple[str, ...] | list[str] = (),
        required_env: tuple[str, ...] | list[str] = (),
        optional_env: tuple[str, ...] | list[str] = (),
        activation_env: tuple[str, ...] | list[str] = (),
        source: str = "plugin",
    ) -> None:
        """Register a lazy loader for a platform that hasn't been imported yet.

        *loader* is a zero-arg callable that imports the owning plugin module,
        which is expected to call :meth:`register` with the real entries for
        *name* and any *aliases*. The loader runs at most once per generation,
        the first time any covered name is looked up (or when the full entry
        list is materialized). A real entry that is registered directly (e.g. a
        built-in) takes precedence -- the whole deferred registration is then
        dropped.
        """
        names = tuple(dict.fromkeys((str(name), *(str(alias) for alias in aliases))))
        required = tuple(str(value) for value in required_env)
        optional = tuple(str(value) for value in optional_env)
        # Explicit activation alternatives extend the required credential set;
        # they do not replace it. A multi-mode adapter such as WeCom must still
        # load for its normal bot credentials as well as its callback-only
        # credentials.
        activation = tuple(
            dict.fromkeys((*required, *(str(value) for value in activation_env)))
        )
        with self._lock:
            if any(platform_name in self._entries for platform_name in names):
                # Already concretely registered; no need to defer.
                return

            # A new registration supersedes an old deferred generation or an
            # in-flight one. The old worker cannot publish after cancellation.
            for platform_name in names:
                old_spec = self._deferred_specs.get(platform_name)
                if old_spec is not None:
                    self._remove_deferred_spec_locked(old_spec)
                old_resolution = self._resolving.get(platform_name)
                if old_resolution is not None:
                    self._cancel_resolution_locked(old_resolution)

            self._generation += 1
            registration = _DeferredRegistration(
                loader=loader,
                names=names,
                required_env=required,
                optional_env=optional,
                activation_env=activation,
                generation=self._generation,
                source=source,
            )
            for platform_name in names:
                self._deferred[platform_name] = loader
                self._deferred_specs[platform_name] = registration
                self._deferred_env[platform_name] = (required, optional)
                self._deferred_activation_env[platform_name] = activation

    def _remove_deferred_spec_locked(self, registration: _DeferredRegistration) -> None:
        """Remove all aliases belonging to *registration* (lock required)."""
        for platform_name in registration.names:
            if self._deferred_specs.get(platform_name) is registration:
                self._deferred_specs.pop(platform_name, None)
                self._deferred.pop(platform_name, None)
                self._deferred_env.pop(platform_name, None)
                self._deferred_activation_env.pop(platform_name, None)

    def _remove_resolution_locked(self, resolution: _DeferredResolution) -> None:
        """Remove all in-flight aliases for *resolution* (lock required)."""
        for platform_name, current in tuple(self._resolving.items()):
            if current is resolution:
                self._resolving.pop(platform_name, None)

    def _cancel_resolution_locked(self, resolution: _DeferredResolution) -> None:
        """Cancel a resolution and retract entries it published (lock required)."""
        if resolution.cancelled:
            return
        resolution.cancelled = True
        resolution.success = False
        for platform_name, entry in tuple(resolution.published_entries.items()):
            if self._entries.get(platform_name) is entry:
                self._entries.pop(platform_name, None)
        self._remove_resolution_locked(resolution)
        resolution.done.set()

    def clear_plugin_platforms(self) -> None:
        """Invalidate all plugin-owned platform state for force rediscovery."""
        with self._lock:
            resolutions: list[_DeferredResolution] = []
            seen_resolutions: set[int] = set()
            for resolution in self._resolving.values():
                if (
                    resolution.registration.source == "plugin"
                    and id(resolution) not in seen_resolutions
                ):
                    seen_resolutions.add(id(resolution))
                    resolutions.append(resolution)
            for resolution in resolutions:
                self._cancel_resolution_locked(resolution)

            registrations: list[_DeferredRegistration] = []
            seen_registrations: set[int] = set()
            for registration in self._deferred_specs.values():
                if (
                    registration.source == "plugin"
                    and id(registration) not in seen_registrations
                ):
                    seen_registrations.add(id(registration))
                    registrations.append(registration)
            for registration in registrations:
                self._remove_deferred_spec_locked(registration)

            for platform_name, entry in tuple(self._entries.items()):
                if entry.source == "plugin":
                    self._entries.pop(platform_name, None)

    def is_loader_current(self, name: str) -> bool:
        """Return whether the current deferred loader may still publish.

        Plugin loaders use this before entering their normal import/register
        path so a force rediscovery can cancel work that has not started yet.
        """
        resolution = getattr(self._loader_local, "resolution", None)
        if resolution is None:
            return False
        with self._lock:
            return (
                not resolution.cancelled
                and self._resolving.get(name) is resolution
            )

    def resolve_candidates(
        self,
        configured_names: set[str] | list[str] | tuple[str, ...] = (),
        *,
        environ: dict[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Materialize only platforms relevant to the current config.

        A deferred platform is a candidate when its registry name appears in
        *configured_names* or any manifest-declared required/alternative
        activation environment variable is populated. The latter is
        deliberately an ``any`` check:
        the adapter's existing ``is_connected``/``env_enablement_fn`` hooks
        remain the authority for whether an incomplete credential set should
        actually enable the platform.  Unknown names are ignored, so callers
        can pass raw YAML keys without importing plugin code first.

        Full enumeration remains available through ``all_entries()`` and
        ``plugin_entries()`` (their default behavior is unchanged).
        """
        env = os.environ if environ is None else environ
        configured = {str(name) for name in configured_names}
        with self._lock:
            candidates = tuple(self._deferred_activation_env.items())
        selected = tuple(
            name
            for name, activation_env in candidates
            if name in configured or any(env.get(var) for var in activation_env)
        )
        self._resolve_many(selected, deadline=time.monotonic() + DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS)
        return selected

    def plugin_authorization_env_vars(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return plugin allowlist env names without forcing full discovery.

        Loaded entries provide their exact fields. Deferred bundled platform
        manifests provide the same names through ``optional_env``; keeping
        those candidates lazy avoids importing every platform merely to decide
        whether the startup allowlist warning applies.
        """
        allowed: set[str] = set()
        allow_all: set[str] = set()
        with self._lock:
            entries = tuple(
                entry for entry in self._entries.values() if entry.source == "plugin"
            )
            deferred_env = tuple(self._deferred_env.values())
            resolving_specs = tuple(
                resolution.registration
                for resolution in {
                    id(value): value for value in self._resolving.values()
                }.values()
                if resolution.registration.source == "plugin"
            )
        for entry in entries:
            if entry.allowed_users_env:
                allowed.add(entry.allowed_users_env)
            if entry.allow_all_env:
                allow_all.add(entry.allow_all_env)
        for required, optional in deferred_env:
            # A third-party manifest may put an allowlist variable in either
            # block. Keep both for parity with the exact fields exposed by a
            # materialized PlatformEntry.
            for name in (*required, *optional):
                if name.endswith("_ALLOWED_USERS"):
                    allowed.add(name)
                if name.endswith("_ALLOW_ALL_USERS"):
                    allow_all.add(name)
        for registration in resolving_specs:
            for name in (*registration.required_env, *registration.optional_env):
                if name.endswith("_ALLOWED_USERS"):
                    allowed.add(name)
                if name.endswith("_ALLOW_ALL_USERS"):
                    allow_all.add(name)
        return tuple(sorted(allowed)), tuple(sorted(allow_all))

    def _wait_for_resolution(
        self,
        name: str,
        resolution: _DeferredResolution,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Wait for one deferred load, returning False on a bounded timeout."""
        if deadline is None:
            deadline = time.monotonic() + DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS
        remaining = max(0.0, deadline - time.monotonic())
        if resolution.done.wait(timeout=remaining):
            with self._lock:
                return resolution.success and not resolution.cancelled
        logger.warning(
            "Timed out waiting for deferred platform '%s' within %.1fs; "
            "failing closed until its loader finishes",
            name,
            DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS,
        )
        return False

    def _run_deferred_loader(
        self,
        resolution: _DeferredResolution,
        loader_context: Context,
    ) -> None:
        """Run one loader and publish its complete registration atomically."""
        current_thread = threading.get_ident()
        with self._lock:
            if resolution.cancelled:
                return
            resolution.owner_thread_id = current_thread
        self._loader_local.resolution = resolution
        error: BaseException | None = None
        try:
            # Deferred platform loading can happen under a multiplexed profile
            # secret/home context. Carry that context into the worker so lazy
            # imports observe the same profile-scoped state as the caller.
            loader_context.run(resolution.registration.loader)
        except BaseException as e:  # noqa: BLE001 — fail closed and requeue
            error = e
            logger.warning(
                "Deferred load of platform '%s' failed: %s",
                resolution.registration.names[0],
                e,
                exc_info=True,
            )
        finally:
            with self._lock:
                if not resolution.cancelled:
                    complete = all(
                        platform_name in self._entries
                        for platform_name in resolution.registration.names
                    )
                    if error is not None or not complete:
                        # A loader that raised after publishing an entry must
                        # not leak that partial state. Requeue the same
                        # generation so a later first-use call can retry.
                        for platform_name, entry in tuple(
                            resolution.published_entries.items()
                        ):
                            if self._entries.get(platform_name) is entry:
                                self._entries.pop(platform_name, None)
                        for platform_name in resolution.registration.names:
                            if platform_name not in self._entries:
                                self._deferred[platform_name] = (
                                    resolution.registration.loader
                                )
                                self._deferred_specs[platform_name] = (
                                    resolution.registration
                                )
                                self._deferred_env[platform_name] = (
                                    resolution.registration.required_env,
                                    resolution.registration.optional_env,
                                )
                                self._deferred_activation_env[platform_name] = (
                                    resolution.registration.activation_env
                                )
                        resolution.success = False
                    else:
                        resolution.success = True
                    self._remove_resolution_locked(resolution)
                resolution.done.set()
            self._loader_local.resolution = None

    def _begin_resolution(
        self,
        name: str,
    ) -> tuple[_DeferredResolution | None, bool]:
        """Return a resolution and whether the caller is its owner."""
        current_thread = threading.get_ident()
        with self._lock:
            pending = self._resolving.get(name)
            if pending is not None:
                return pending, pending.owner_thread_id == current_thread
            loader = self._deferred.get(name)
            registration = self._deferred_specs.get(name)
            if loader is None or registration is None:
                return None, False

            resolution = _DeferredResolution(registration=registration)
            for platform_name in registration.names:
                if self._deferred_specs.get(platform_name) is registration:
                    self._deferred.pop(platform_name, None)
                    self._deferred_specs.pop(platform_name, None)
                    self._deferred_env.pop(platform_name, None)
                    self._deferred_activation_env.pop(platform_name, None)
                self._resolving[platform_name] = resolution

        try:
            worker = threading.Thread(
                target=self._run_deferred_loader,
                args=(resolution, copy_context()),
                name=f"hermes-platform-{registration.names[0]}",
                daemon=True,
            )
            worker.start()
        except Exception as e:
            logger.warning(
                "Could not start deferred loader for '%s': %s",
                registration.names[0],
                e,
            )
            with self._lock:
                self._cancel_resolution_locked(resolution)
        return resolution, False

    def _resolve(
        self,
        name: str,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Start one deferred loader and wait at most until *deadline*."""
        resolution, owner = self._begin_resolution(name)
        if resolution is None or owner:
            return True
        return self._wait_for_resolution(name, resolution, deadline=deadline)

    def _resolve_many(self, names: tuple[str, ...], *, deadline: float) -> None:
        """Start independent loaders together, then wait under one deadline."""
        resolutions: list[tuple[str, _DeferredResolution]] = []
        seen: set[int] = set()
        for name in names:
            resolution, owner = self._begin_resolution(name)
            if resolution is None or owner or id(resolution) in seen:
                continue
            seen.add(id(resolution))
            resolutions.append((name, resolution))
        for name, resolution in resolutions:
            self._wait_for_resolution(name, resolution, deadline=deadline)

    def _resolve_all(self) -> None:
        """Run every pending deferred loader.

        Used by the iterate-all accessors (``all_entries``/``plugin_entries``),
        which are only called by paths that genuinely need every adapter:
        ``hermes setup``/``gateway status``, platform menus, and explicit
        diagnostics.  Gateway channel-directory startup uses connected
        adapters directly and does not iterate the full set. CLI chat never
        iterates the full set.
        """
        with self._lock:
            if not self._deferred and not self._resolving:
                return
            # Snapshot keys -- loaders mutate _deferred as they resolve. Keep
            # in-flight names too: a full-enumeration caller must wait for a
            # prior candidate resolution instead of returning a transiently
            # incomplete registry.
            names = tuple(dict.fromkeys((*self._deferred, *self._resolving)))
        self._resolve_many(
            names,
            deadline=time.monotonic() + DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS,
        )

    def register(self, entry: PlatformEntry) -> bool:
        """Register a platform adapter entry.

        If an entry with the same name exists, it is replaced (last writer
        wins -- this lets plugins override built-in adapters if desired).

        Returns ``False`` when a stale, cancelled deferred loader attempted to
        publish an entry.  Existing callers ignore the return value; exposing
        it lets ``PluginContext`` avoid recording a force-rediscovery-invalid
        platform in its attribution set.
        """
        active_resolution = getattr(self._loader_local, "resolution", None)
        with self._lock:
            pending = self._resolving.get(entry.name)
            if active_resolution is not None:
                if active_resolution.cancelled:
                    logger.debug(
                        "Discarding stale deferred registration for '%s'",
                        entry.name,
                    )
                    return False
                if pending is not None and pending is not active_resolution:
                    logger.warning(
                        "Discarding conflicting deferred registration for '%s'",
                        entry.name,
                    )
                    return False
                # Hide an entry published by a deferred loader from all
                # readers until the loader completes successfully.
                self._resolving[entry.name] = active_resolution
                active_resolution.published_entries[entry.name] = entry
            elif pending is not None:
                # An explicit direct registration supersedes an in-flight
                # loader; its stale worker is generation-invalidated.
                self._cancel_resolution_locked(pending)

            deferred_spec = self._deferred_specs.get(entry.name)
            if deferred_spec is not None:
                self._remove_deferred_spec_locked(deferred_spec)

            # A concrete registration supersedes any pending deferred loader.
            if entry.name in self._entries:
                prev = self._entries[entry.name]
                logger.info(
                    "Platform '%s' re-registered (was %s, now %s)",
                    entry.name,
                    prev.source,
                    entry.source,
                )
            self._entries[entry.name] = entry
        logger.debug("Registered platform adapter: %s (%s)", entry.name, entry.source)
        return True

    def unregister(self, name: str) -> bool:
        """Remove a platform entry.  Returns True if it existed."""
        with self._lock:
            deferred_spec = self._deferred_specs.get(name)
            if deferred_spec is not None:
                self._remove_deferred_spec_locked(deferred_spec)
            pending = self._resolving.get(name)
            if pending is not None:
                self._cancel_resolution_locked(pending)
            return self._entries.pop(name, None) is not None

    def get(self, name: str) -> Optional[PlatformEntry]:
        """Look up a platform entry by name."""
        with self._lock:
            needs_resolution = name not in self._entries or name in self._resolving
        if needs_resolution:
            if not self._resolve(
                name,
                deadline=time.monotonic() + DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS,
            ):
                return None
        with self._lock:
            if name in self._resolving:
                return None
            return self._entries.get(name)

    def all_entries(self, *, resolve_deferred: bool = True) -> list[PlatformEntry]:
        """Return all registered platform entries.

        ``resolve_deferred=False`` is for config/startup paths that have
        already selected relevant candidates with :meth:`resolve_candidates`.
        Interactive setup/status callers keep the default full enumeration.
        """
        if resolve_deferred:
            self._resolve_all()
        with self._lock:
            return [
                entry for name, entry in self._entries.items()
                if name not in self._resolving
            ]

    def plugin_entries(self, *, resolve_deferred: bool = True) -> list[PlatformEntry]:
        """Return only plugin-registered platform entries.

        ``resolve_deferred=False`` returns the currently materialized subset;
        callers that need every bundled platform should retain the default.
        """
        if resolve_deferred:
            self._resolve_all()
        with self._lock:
            return [
                entry
                for name, entry in self._entries.items()
                if name not in self._resolving and entry.source == "plugin"
            ]

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) or in-flight platform still counts as
        # registered -- the loader will materialize it on first real use. This
        # keeps cheap membership checks (toolset resolution, webhook
        # deliver-target checks) from triggering a heavy import.
        with self._lock:
            return (
                name in self._entries
                or name in self._deferred
                or name in self._resolving
            )

    def create_adapter(self, name: str, config: Any) -> Optional[Any]:
        """Create an adapter instance for the given platform name.

        Returns None if:
        - No entry registered for *name*
        - check_fn() returns False (missing deps)
        - validate_config() returns False (misconfigured)
        - The factory raises an exception
        """
        with self._lock:
            needs_resolution = name not in self._entries or name in self._resolving
        if needs_resolution:
            if not self._resolve(
                name,
                deadline=time.monotonic() + DEFERRED_PLATFORM_LOAD_TIMEOUT_SECONDS,
            ):
                return None
        with self._lock:
            if name in self._resolving:
                return None
            entry = self._entries.get(name)
        if entry is None:
            return None

        if not entry.check_fn():
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning(
                "Platform '%s' requirements not met%s",
                entry.label,
                hint,
            )
            return None

        if entry.validate_config is not None:
            try:
                if not entry.validate_config(config):
                    logger.warning(
                        "Platform '%s' config validation failed",
                        entry.label,
                    )
                    return None
            except Exception as e:
                logger.warning(
                    "Platform '%s' config validation error: %s",
                    entry.label,
                    e,
                )
                return None

        try:
            adapter = entry.adapter_factory(config)
            return adapter
        except Exception as e:
            logger.error(
                "Failed to create adapter for platform '%s': %s",
                entry.label,
                e,
                exc_info=True,
            )
            return None


# Module-level singleton
platform_registry = PlatformRegistry()
