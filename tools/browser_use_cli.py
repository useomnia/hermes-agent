"""Hermes' Browser Use CLI 3.0 browser surface.

The CLI is a managed subprocess boundary and the selected browser surface;
when the exact Hermes-managed installation is missing, ``browser_exec``
returns an actionable setup error rather than downgrading. There is no
user-facing backend switch or floating ``uvx`` fallback. Omnio supplies a
conversation-scoped CDP relay. Hermes keeps a
hashed logical identity for every conversation/session; shared browsers use
it as ``BU_NAME``, while Omnio uses a hashed private IPC directory for every
conversation/session (``default`` only for the unnamed live Toolbox tab).
"""

import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import is_truthy_value

logger = logging.getLogger(__name__)

_DIRECT_PROVIDER_KEY = "browser-use"
BROWSER_USE_PACKAGE = "browser-use==0.13.8"
BROWSER_USE_CLI_VERSION = "0.1.9"

# Shared-browser daemon names become the BU_NAME env var. Omnio's templated
# CDP path uses a private hashed runtime for every logical session and keeps
# ``default`` only for the unnamed live Toolbox tab.
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_DEFAULT_TIMEOUT_S = 300
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 1800
_STDERR_CAP_CHARS = 4000

# Screenshot paths printed by capture_screenshot() in the exec output.
# Two alternatives: POSIX absolute (/tmp/shot.png) and Windows drive-letter
# absolute (C:\Users\...\shot.png or C:/Users/.../shot.png). Browser Use on
# Windows prints native paths — the POSIX-only pattern silently dropped them
# and screenshot_path / the multimodal attach never fired (#83884).
_IMAGE_PATH_RE = re.compile(
    r"((?:[A-Za-z]:[\\/]|/)[^\s\"']+?\.(?:png|jpe?g|webp))", re.IGNORECASE
)

# http(s) URL literals in exec code checked against browser_navigate's policy
_URL_RE = re.compile(r"https?://[^\s'\"\\)]+", re.IGNORECASE)

# Browser Harness already owns the daemon lifecycle.  Hermes tracks only the
# exact named endpoint it started so timeout/turn cleanup can ask the managed
# CLI to stop that daemon without ever killing Toolbox Chrome or another
# conversation's process.
_HARNESS_LOCK = threading.RLock()
_ACTIVE_HARNESSES: Dict[str, Dict[str, Any]] = {}
_HARNESS_CLEANUP_STOP = threading.Event()
_HARNESS_CLEANUP_THREAD: Optional[threading.Thread] = None


def _blocked_url_in_code(code: str) -> Optional[str]:
    """Return an error if a URL literal fails the built-in navigation checks."""
    from tools.browser_tool import evaluate_url_safety

    for url in _URL_RE.findall(code or ""):
        err = evaluate_url_safety(url)
        if err:
            return err.get("error", "Blocked: unsafe URL")
    return None


def _base_subprocess_env() -> dict:
    from tools.browser_tool import _build_browser_env

    env = _build_browser_env()
    # The browser-use CLI runs under its own Python (uv tool / uvx), which
    # may differ from Hermes's venv Python. PYTHONPATH/PYTHONHOME inherited
    # from the agent process point at Hermes's venv site-packages, and a
    # child interpreter honors them ahead of its own site-packages — so the
    # CLI imports compiled C-extensions (e.g. pydantic_core) built for the
    # wrong interpreter and crashes on ABI mismatch (#83427, #84841, #86006,
    # #86104). Strip both — the CLI manages its own environment and never
    # needs Hermes's import path.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # These values belong to Hermes' gateway/session context or to another
    # harness instance. Resolve the task-aware CDP URL below and then export
    # only the concrete BU_* endpoint plus the hashed instance directories.
    for key in (
        "HERMES_SESSION_ID",
        "HERMES_SESSION_KEY",
        "BU_NAME",
        "BH_RUNTIME_DIR",
        "BH_RUNTIME_DIR_SHARED",
        "BH_TMP_DIR",
        "BH_TMP_DIR_SHARED",
        "BH_AGENT_WORKSPACE",
        "BU_AUTOSPAWN",
        "BROWSER_CDP_URL_TEMPLATE",
        "BROWSER_CDP_URL",
        "BU_CDP_WS",
        "BU_CDP_URL",
    ):
        env.pop(key, None)
    # Same class of hazard, PATH flavor: profile-spawned workers (kanban
    # bots, cron jobs) can hand down a PATH of only version-manager dirs,
    # which kills the uv trampoline before the CLI's Python starts. Floor
    # the PATH so coreutils are always reachable (see below).
    env["PATH"] = _floor_subprocess_path(env.get("PATH", ""))
    # Browser Use and Browser Harness both honor this switch.  Force it even
    # when the parent gateway inherited a truthy value: a managed child must
    # not emit third-party usage telemetry on behalf of Hermes.
    env["ANONYMIZED_TELEMETRY"] = "false"
    env["BROWSER_USE_TELEMETRY"] = "0"
    env["BROWSER_HARNESS_TELEMETRY"] = "0"
    env["DO_NOT_TRACK"] = "1"
    return env


def _floor_subprocess_path(path: str) -> str:
    """Guarantee core system dirs survive onto the CLI subprocess PATH.

    Profile workers can inherit a PATH holding only version-manager dirs
    (observed: the nvm node dir repeated 7x, nothing else). That is fatal
    for the uv-installed browser-use binary: its POSIX sh trampoline
    resolves ``dirname``/``realpath`` through PATH, so without /usr/bin it
    dies with ``realpath: not found … exec: /python: not found`` (exit
    127) before its own Python ever starts. Reuses browser_tool's
    ``_merge_browser_path`` floor — same hazard, same sane-dir list — and
    falls back to appending FHS bin dirs if that import is unavailable.
    Windows .cmd shims don't trampoline through PATH, so no-op there.
    """
    if os.name == "nt":
        return path
    try:
        from tools.browser_tool import _merge_browser_path

        return _merge_browser_path(path or "")
    except Exception:
        pass
    parts = [p for p in (path or "").split(os.pathsep) if p]
    existing = set(parts)
    for directory in (
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ):
        if directory not in existing and os.path.isdir(directory):
            parts.append(directory)
    return os.pathsep.join(parts)


def _read_browser_cfg() -> dict:
    """Return the ``browser:`` config section, or {} on any failure."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        cfg = cfg_get(read_raw_config(), "browser", default={})
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.debug("Could not read browser config section: %s", e)
        return {}


def is_browser_use_cli_mode() -> bool:
    """Return whether the clean Browser Use surface owns browser tools.

    Surface selection is deliberately independent of installation state.  A
    failed provisioning step must produce an actionable ``browser_exec``
    install error, not silently restore the legacy browser_* tools with a
    different isolation and CDP contract.  The only exception is Camofox,
    whose HTTP/Firefox backend cannot be driven by Browser Harness.
    """
    try:
        from tools.browser_camofox import is_camofox_mode

        if is_camofox_mode():
            return False
    except Exception as e:  # pragma: no cover - optional plugin
        logger.debug("Camofox activity check failed: %s", e)
    return True


def _managed_bin_dir() -> Optional[str]:
    """Hermes' install-scoped bin dir, shared by every profile.

    Profile gateways replace ``HERMES_HOME`` with
    ``<root>/profiles/<name>``.  Browser Use is installed once by Hermes (and
    by Omnio provisioning) under ``<root>/bin``; profile cloning deliberately
    does not duplicate managed executables.  Resolve the root explicitly so a
    brand gateway sees the same pinned CLI as the default profile.
    """
    try:
        from hermes_constants import get_default_hermes_root

        return str(get_default_hermes_root() / "bin")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("Could not resolve managed bin dir: %s", e)
        return None


def _cli_version_probe_env() -> dict:
    """Minimal, scrubbed environment for ``browser-use --version``."""
    # Reuse the same credential-scrubbed boundary as a real exec. Version
    # probing is still an untrusted subprocess invocation; it must not receive
    # the gateway's full environment merely because it only asks for a version.
    try:
        env = _base_subprocess_env()
    except Exception:
        # Keep discovery usable in a minimal bootstrap/test interpreter while
        # still passing only the process essentials needed by a CLI shim.
        env = {
            key: os.environ[key]
            for key in ("HOME", "USER", "SystemRoot", "WINDIR")
            if os.environ.get(key)
        }
        env["PATH"] = _floor_subprocess_path(os.environ.get("PATH", ""))
        env["ANONYMIZED_TELEMETRY"] = "false"
        env["BROWSER_USE_TELEMETRY"] = "0"
        env["BROWSER_HARNESS_TELEMETRY"] = "0"
        env["DO_NOT_TRACK"] = "1"
    for key in (
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSER_USE_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_BROWSER_TTL",
    ):
        env.pop(key, None)
    return env


def _managed_cli_is_current(path: str) -> bool:
    """Verify the managed executable is the pinned CLI 0.1.9."""
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=_cli_version_probe_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    return bool(
        result.returncode == 0
        and re.search(
            rf"(?<![0-9]){re.escape(BROWSER_USE_CLI_VERSION)}(?![0-9])",
            output,
        )
    )


def _find_cli() -> Optional[List[str]]:
    """Locate only the exact Hermes-managed Browser Use executable.

    A PATH or ``uvx`` hit is deliberately ignored.  Those paths are mutable
    user state and can silently drift away from the harness version that the
    Omnio runtime was provisioned with.
    """
    bin_dir = _managed_bin_dir()
    if not bin_dir:
        return None
    for name in ("browser-use", "browser-use.exe", "browser-use.cmd"):
        candidate = Path(bin_dir) / name
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            path = str(candidate)
            if _managed_cli_is_current(path):
                return [path]
    direct = shutil.which("browser-use", path=bin_dir)
    return [direct] if direct and _managed_cli_is_current(direct) else None


def install_cli(timeout_s: int = 600) -> Tuple[bool, str]:
    """Install the browser-use CLI persistently via ``uv tool install``.

    Resolution order for uv: Hermes' managed uv (bootstrapped on demand via
    ``hermes_cli.managed_uv.ensure_uv``) → uv on PATH. The binary is linked
    into the installation root's ``bin`` directory (``UV_TOOL_BIN_DIR``) so
    ``_find_cli()`` resolves it for every profile without touching the user's
    PATH.

    Returns ``(ok, message)`` — never raises.
    """
    # MANAGED-FIRST: only the managed copy short-circuits the install. A
    # browser-use found on PATH is a user-level side install — it must NOT
    # prevent provisioning the canonical Hermes-managed copy, or resolution
    # stays pinned to a binary we don't control (version drift, no updates
    # through hermes tools).
    bin_dir = _managed_bin_dir()
    if bin_dir:
        managed = _find_cli()
        if managed:
            return True, f"browser-use CLI {BROWSER_USE_CLI_VERSION} already installed ({managed[0]})"

    uv_bin: Optional[str] = None
    try:
        from hermes_cli.managed_uv import ensure_uv

        uv_bin = str(ensure_uv() or "") or None
    except Exception as e:
        logger.debug("Managed uv bootstrap unavailable: %s", e)
    if not uv_bin:
        uv_bin = shutil.which("uv")
    if not uv_bin:
        return False, (
            "uv is not available and could not be bootstrapped. Install uv "
            "(https://docs.astral.sh/uv/) and run `uv tool install "
            f"{BROWSER_USE_PACKAGE}`."
        )

    env = dict(os.environ)
    env["UV_NO_CONFIG"] = "1"
    if bin_dir:
        try:
            Path(bin_dir).mkdir(parents=True, exist_ok=True)
            env["UV_TOOL_BIN_DIR"] = bin_dir
        except OSError as e:
            logger.debug("Could not prepare %s: %s", bin_dir, e)

    try:
        result = subprocess.run(
            [uv_bin, "tool", "install", "--force", BROWSER_USE_PACKAGE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"`uv tool install {BROWSER_USE_PACKAGE}` timed out after {timeout_s}s"
    except Exception as e:
        return False, f"Failed to run `uv tool install {BROWSER_USE_PACKAGE}`: {e}"

    if result.returncode != 0:
        tail = "\n".join(
            (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        )
        return False, f"`uv tool install {BROWSER_USE_PACKAGE}` failed:\n{tail}"

    found = _find_cli()
    if not found or len(found) != 1:
        return False, (
            "install reported success but the browser-use binary is still "
            f"not resolvable — run `uv tool install {BROWSER_USE_PACKAGE}` manually"
        )
    return True, f"browser-use CLI installed ({found[0]})"


def _canonical_conversation_id(task_id: Optional[str]) -> str:
    """Return the stable conversation identity used for harness naming.

    Omnio binds ``HERMES_SESSION_ID`` in the gateway context.  A delegated
    child task may have a different task id, but it must still share the
    parent conversation's browser.  The task id is therefore only a fallback
    for non-gateway callers.  The final process/thread fallback is unique to
    this process and intentionally never the literal ``default``.
    """
    candidates: list[str] = []
    try:
        from gateway.session_context import get_session_env

        for key in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
            value = str(get_session_env(key) or "").strip()
            if value:
                candidates.append(value)
    except Exception:
        pass
    for key in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
        value = os.environ.get(key, "").strip()
        if value:
            candidates.append(value)
    task = str(task_id or "").strip()
    if task:
        candidates.append(f"task:{task}")
    if candidates:
        return candidates[0]
    return f"process:{os.getpid()}:thread:{threading.get_ident()}"


def _derive_bu_name(task_id: Optional[str], session_name: str = "") -> str:
    """Hash conversation + optional name into a valid, non-sensitive BU_NAME."""
    conversation = _canonical_conversation_id(task_id)
    optional = str(session_name or "").strip()
    digest = hashlib.sha256(
        f"hermes-browser-use-v1\0{conversation}\0{optional}".encode("utf-8")
    ).hexdigest()
    # ``bu-`` keeps the value valid for browser_harness' [A-Za-z0-9_-] name
    # grammar while making it obvious in diagnostics that this is a managed
    # Hermes endpoint, never an operator-provided raw session name.
    return f"bu-{digest[:48]}"


def _omnio_template_cdp_configured() -> bool:
    """Whether Omnio's conversation-scoped CDP relay is configured."""
    return bool(os.environ.get("BROWSER_CDP_URL_TEMPLATE", "").strip())


def _owner_marker_path(runtime_dir: Path) -> Path:
    """Return the gateway ownership marker inside one private runtime dir."""
    return runtime_dir / "gateway.owner_pid"


def _read_omnio_owner_pid(runtime_dir: Path) -> Optional[int]:
    """Read a validated owner marker, returning ``None`` when absent/stale."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(_owner_marker_path(runtime_dir)), flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                return None
            owner = getattr(os, "getuid", lambda: metadata.st_uid)()
            if metadata.st_uid != owner or (
                os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                return None
            raw = os.read(fd, 128).decode("ascii", errors="strict").strip()
        finally:
            os.close(fd)
    except (OSError, UnicodeError):
        return None
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _write_omnio_owner_pid(runtime_dir: Path) -> Optional[str]:
    """Record this gateway as the private runtime's owner.

    The marker is deliberately owner-only and opened with ``O_NOFOLLOW`` so a
    hostile or stale symlink cannot redirect writes outside the already
    validated runtime directory. It lets a new gateway process distinguish a
    dead owner (whose exact Browser Harness daemon can be reloaded) from a
    live gateway using the same canonical conversation.
    """
    marker = _owner_marker_path(runtime_dir)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(marker), flags, 0o600)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                return f"Browser Harness owner marker {marker} is not a regular file"
            owner = getattr(os, "getuid", lambda: metadata.st_uid)()
            if metadata.st_uid != owner:
                return f"Browser Harness owner marker {marker} is owned by another user"
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            payload = f"{os.getpid()}\n".encode("ascii")
            written = 0
            while written < len(payload):
                written += os.write(fd, payload[written:])
            os.fsync(fd)
            verified = os.fstat(fd)
            if os.name != "nt" and stat.S_IMODE(verified.st_mode) != 0o600:
                return f"Browser Harness owner marker {marker} is not owner-only"
        finally:
            os.close(fd)
    except OSError as exc:
        logger.debug("Could not write Browser Harness owner marker %s: %s", marker, exc)
        return f"Browser Harness owner marker {marker} is unavailable: {exc}"
    return None


def _owner_pid_is_alive(pid: int) -> bool:
    """Use a cross-platform PID probe without importing it at module load.

    ``gateway.status._pid_exists`` is the canonical Hermes probe and handles
    Windows without the dangerous ``os.kill(pid, 0)`` fallback. The direct
    ``psutil`` fallback is only for scaffold/partial-import environments. A
    probe failure is deliberately treated as alive: uncertainty must never
    let a restarted gateway take over another process's runtime.
    """
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists
    except Exception as exc:
        logger.debug("Could not import Hermes PID probe: %s", exc)
        _pid_exists = None

    if _pid_exists is not None:
        try:
            return bool(_pid_exists(pid))
        except Exception as exc:
            logger.debug("Hermes PID probe failed for %s: %s", pid, exc)

    # ``psutil.pid_exists`` is the repository's dependency-backed,
    # cross-platform fallback. Do not substitute ``os.kill(pid, 0)`` here:
    # on Windows it sends CTRL+C to the target's console process group.
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception as exc:
        logger.debug("Fallback PID probe failed for %s: %s", pid, exc)
        return True


def _configure_omnio_harness_dirs(
    env: dict,
    logical_bu_name: str,
    harness_name: str = "default",
    cmd: Optional[List[str]] = None,
) -> Optional[str]:
    """Isolate one Omnio harness instance while reusing Toolbox's sole tab.

    Browser Harness treats a non-default ``BU_NAME`` as a request for a
    dedicated automation tab. The unnamed Omnio call already has its own
    Toolbox Chrome process/context, and the live screencast is attached to
    that process's original tab, so only that path keeps the harness name
    ``default``. Explicit names retain their hashed non-default name while
    every path gets isolated IPC/temp files and an owner marker.
    """
    digest = hashlib.sha256(
        f"hermes-browser-use-omnio-runtime-v1\0{logical_bu_name}".encode("utf-8")
    ).hexdigest()[:24]
    # AF_UNIX sun_path is only 104 bytes on macOS.  ``tempfile.gettempdir()``
    # commonly expands to a long per-user TMPDIR there, so use the guaranteed
    # short POSIX root for the harness IPC path.
    runtime_root = Path("/tmp") if os.name != "nt" else Path(tempfile.gettempdir())
    runtime_dir = runtime_root / f"hermes-bu-{digest}"
    try:
        from hermes_constants import get_hermes_home

        tmp_dir = Path(get_hermes_home()) / "cache" / "browser-use" / "tmp" / digest
    except Exception:
        tmp_dir = Path(tempfile.gettempdir()) / f"hermes-bu-tmp-{digest}"
    for path in (runtime_dir, tmp_dir):
        try:
            # Do not let Browser Harness inherit a pre-existing 0755/symlink
            # directory: it contains the daemon socket/pid and must be a
            # private endpoint owned by this gateway user.
            try:
                current = path.lstat()
            except FileNotFoundError:
                path.mkdir(parents=True, mode=0o700, exist_ok=False)
                current = path.lstat()
            if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
                return f"Browser Harness directory {path} is not a private directory"
            owner = getattr(os, "getuid", lambda: current.st_uid)()
            if current.st_uid != owner:
                return f"Browser Harness directory {path} is owned by another user"
            path.chmod(0o700)
            verified = path.lstat()
            if (
                stat.S_ISLNK(verified.st_mode)
                or not stat.S_ISDIR(verified.st_mode)
                or verified.st_uid != owner
                or (
                    os.name != "nt" and stat.S_IMODE(verified.st_mode) != 0o700
                )
            ):
                return f"Browser Harness directory {path} could not be made private"
        except OSError as exc:
            logger.debug("Could not prepare Browser Harness directory %s: %s", path, exc)
            return f"Browser Harness directory {path} is unavailable: {exc}"
    if not _SESSION_RE.match(harness_name):
        return "Browser Harness name is invalid"
    env["BU_NAME"] = harness_name
    env["BH_RUNTIME_DIR"] = str(runtime_dir)
    env["BH_TMP_DIR"] = str(tmp_dir)
    env.pop("BH_RUNTIME_DIR_SHARED", None)
    env.pop("BH_TMP_DIR_SHARED", None)

    # A gateway restart loses the in-memory registry while the named harness
    # daemon may still be alive. Reap only when the marker's exact owner PID
    # is dead. A live different owner means another gateway is actively using
    # this canonical conversation; fail closed rather than sharing its daemon
    # or overwriting its marker.
    previous_owner = _read_omnio_owner_pid(runtime_dir)
    if previous_owner and previous_owner != os.getpid():
        if _owner_pid_is_alive(previous_owner):
            return (
                f"Browser Harness runtime {runtime_dir} is owned by a live "
                f"gateway process (PID {previous_owner})"
            )
        if not _stop_harness_daemon(harness_name, env, cmd):
            return (
                f"Browser Harness runtime {runtime_dir} has a dead owner "
                "but its exact daemon could not be reloaded safely; retry "
                "after confirming the previous gateway is stopped"
            )
    marker_error = _write_omnio_owner_pid(runtime_dir)
    if marker_error:
        return marker_error
    return None


def _browser_use_cloud_autospawn_enabled() -> bool:
    """Whether the existing Browser Use cloud credentials opt into spawning.

    This is provider configuration, not a Browser Use-vs-legacy feature flag:
    the only browser surface remains ``browser_exec``.  Keep the old opt-in
    semantics so a direct Browser Use API key does not unexpectedly create a
    billable remote browser when a paired CDP endpoint is absent.
    """
    cfg = _read_browser_cfg()
    if not isinstance(cfg, dict) or is_truthy_value(cfg.get("use_gateway"), default=False):
        return False
    provider = str(cfg.get("cloud_provider") or "").strip().lower()
    if provider not in {"", _DIRECT_PROVIDER_KEY}:
        return False
    try:
        from tools.browser_camofox import is_camofox_mode

        if is_camofox_mode():
            return False
    except Exception:
        pass
    return bool(os.getenv("BROWSER_USE_API_KEY"))


def _workspace_dir(task_id: Optional[str], session_name: str = "") -> Optional[str]:
    """Stable conversation/session scratch dir across delegated browser calls.

    A delegated child task id is an execution detail, not the browser
    conversation identity.  Keep workspace files alongside the daemon's
    canonical conversation/name key so a child call can see helpers and
    accumulated results created by its parent (and vice versa).
    """
    expected_name = _derive_bu_name(task_id, session_name)
    existing = os.environ.get("BH_AGENT_WORKSPACE")
    # Preserve an explicitly provisioned workspace only when it is already
    # keyed by Hermes' digest. Never inherit a generic/raw path that would
    # make delegated tasks in one conversation appear to lose their helpers.
    if existing:
        try:
            if Path(existing).name == expected_name:
                return existing
        except (TypeError, ValueError):
            pass
    try:
        from hermes_constants import get_hermes_home

        # Include the optional session label only through the same digest used
        # for BU_NAME.  The human label and canonical session id must never be
        # written into a path, and delegated task ids must not split one
        # conversation's durable browser workspace into several directories.
        safe = expected_name[:80]
        path = Path(get_hermes_home()) / "cache" / "browser-use" / "workspace" / safe
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except Exception as e:
        logger.debug("browser_exec workspace unavailable: %s", e)
        return None


def _find_screenshot(stdout: str, since: float) -> Optional[str]:
    """Return the last screenshot path printed during this exec, or None.

    Only accepts files that exist and were written after the exec started
    """
    for path in reversed(_IMAGE_PATH_RE.findall(stdout or "")):
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= since - 1:
                return path
        except OSError:
            continue
    return None


def _native_screenshot_result(result: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    """Build a multimodal tool result attaching path for vision models"""
    try:
        from pathlib import Path

        from tools.vision_tools import (
            _EMBED_MAX_DIMENSION,
            _EMBED_TARGET_BYTES,
            _resize_image_for_vision,
            _should_use_native_vision_fast_path,
        )

        if not _should_use_native_vision_fast_path():
            return None
        # History-reuse cap (#92699): this data URL bakes into the tool
        # result and is re-sent on every later turn — same policy as the
        # vision_analyze / browser_vision native embeds (256 KB / 1568 px,
        # JPEG quality ladder instead of PNG dimension-halving).
        data_url = _resize_image_for_vision(
            Path(path),
            mime_type="image/png",
            max_base64_bytes=_EMBED_TARGET_BYTES,
            max_dimension=_EMBED_MAX_DIMENSION,
            force_jpeg=True,
        )
        text = json.dumps(result, ensure_ascii=False)
        return {
            "_multimodal": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        text
                        + "\n\nThe screenshot from this call is attached — "
                        "inspect it with your native vision."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
            "text_summary": text,
            "meta": {"screenshot_path": path, "native_vision": True},
        }
    except Exception as e:
        logger.debug("Native screenshot attach failed (falling back to text): %s", e)
        return None


def _resolve_backend_cdp(
    env: dict,
    task_id: Optional[str],
    session_name: str = "",
    bu_name: Optional[str] = None,
) -> Optional[str]:
    """Point the harness at the configured browser backend's CDP endpoint.

    Resolution order (first hit wins):

    1. When Omnio's ``BROWSER_CDP_URL_TEMPLATE`` is present, the task-aware
       ``_get_task_cdp_override(task_id)`` path. The template is mandatory;
       an unresolved value is an error, never permission to attach elsewhere.
       Process-level ``BU_CDP_*`` values cannot outrank this isolation path.
    2. Outside Omnio, ``BU_CDP_WS`` / ``BU_CDP_URL`` already in the
       environment — explicit user/operator overrides passed through intact.
    3. A configured cloud browser provider (Browserbase, Firecrawl, Nous
       gateway/Browser Use cloud, …): reuse the legacy stack's
       ``_get_session_info()`` so browser_exec shares the SAME provider
       session machinery — per-task session cache, expiry replacement,
       inactivity reaper, and atexit cleanup — instead of duplicating it.
    4. Nothing configured: return None; the harness attaches to local
       Chrome (or Browser Use cloud via BU_AUTOSPAWN for legacy configs).

    ``session_name`` (the tool's ``session`` argument / BU_NAME) keys the
    provider session cache when set, so every distinct name gets its OWN
    cloud browser and the same name reuses one — that is what makes named
    sessions actually concurrent-safe on provider backends instead of all
    names sharing a single per-task browser.

    Returns an error string on provider failure, None on success.
    """
    template = os.environ.get("BROWSER_CDP_URL_TEMPLATE", "").strip()
    if not template:
        # Explicit CDP values are copied from the parent only after the
        # credential-scrubbed environment is built. They remain a supported
        # generic-Hermes override, but never supersede Omnio's conversation
        # template above.
        for key in ("BU_CDP_WS", "BU_CDP_URL"):
            value = str(env.get(key) or os.environ.get(key) or "").strip()
            if value:
                env[key] = value
                return None

    try:
        from tools.browser_tool import (
            _get_task_cdp_override,
            _get_cloud_provider,
            _get_session_info,
        )
    except Exception as e:  # pragma: no cover — stubbed browser_tool in tests
        logger.debug("browser_tool backend resolution unavailable: %s", e)
        if os.environ.get("BROWSER_CDP_URL_TEMPLATE", "").strip():
            return (
                "Omnio browser CDP is configured but its task-aware resolver "
                "is unavailable; refusing to attach to local Chrome."
            )
        return None

    try:
        # This call must stay task-aware: the Omnio relay path includes the
        # canonical conversation id, and a child task id must not accidentally
        # resolve a different conversation's endpoint.
        override = _get_task_cdp_override(str(task_id or ""))
    except Exception as exc:
        logger.warning("Omnio CDP resolution failed: %s", exc)
        override = ""
    if template and not override:
        return (
            "Omnio browser CDP is configured but unresolved for this "
            "conversation; refusing to attach to local Chrome. Verify the "
            "canonical Hermes session id and Toolbox browser relay."
        )
    if override:
        env["BU_CDP_URL" if override.startswith(("http://", "https://")) else "BU_CDP_WS"] = override
        return None

    try:
        provider = _get_cloud_provider()
    except Exception as e:
        logger.debug("Cloud provider lookup failed: %s", e)
        provider = None
    if provider is None:
        return None

    # Browser Use direct-API configs: the CLI talks to Browser Use cloud
    # natively (BU_AUTOSPAWN / auth login) — routing through the legacy
    # provider here would just create a second, redundant session. The
    # Nous-gateway variant (use_gateway: true) DOES resolve through the
    # provider: the gateway provisions the cloud browser server-side and
    # returns its CDP URL, giving subscribers CLI mode with no raw key.
    provider_key = str(getattr(provider, "name", "") or "").strip().lower()
    if provider_key == _DIRECT_PROVIDER_KEY and not is_truthy_value(
        _read_browser_cfg().get("use_gateway"), default=False
    ):
        return None

    try:
        # Named sessions get their OWN provider browser, keyed by name so the
        # same name reuses one browser across calls and tasks, and different
        # names never collide. Unnamed calls keep the per-task key.
        cache_key = f"bu-named-{bu_name or _derive_bu_name(task_id, session_name)}"
        session_info = _get_session_info(cache_key)
    except Exception as e:
        return (
            f"Cloud browser provider {type(provider).__name__} failed to "
            f"provide a session: {e}. Fix the provider configuration or "
            "switch backends via `hermes tools` → Browser Automation."
        )
    cdp = str((session_info or {}).get("cdp_url") or "")
    if not cdp:
        return (
            f"Cloud browser provider {type(provider).__name__} returned no "
            "CDP endpoint, so Browser Use mode cannot drive it. Switch to "
            "a provider configuration that exposes CDP, then retry."
        )
    env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp
    return None


def _stop_harness_daemon(
    bu_name: str,
    env: Optional[dict] = None,
    cmd: Optional[List[str]] = None,
    timeout_s: float = 15.0,
) -> bool:
    """Stop exactly one Browser Harness daemon through its managed CLI.

    ``browser-use --reload`` delegates to Browser Harness' identity-checked
    IPC shutdown.  It scopes the operation by ``BU_NAME`` and never sends a
    signal to Toolbox Chrome or its browser context.  In particular, do not
    replace this with ``pkill``, a raw pid-file read, or CDP ``Browser.close``:
    those can terminate a sibling conversation or the Toolbox-owned browser.
    """
    stop_name = str((env or {}).get("BU_NAME") or bu_name or "")
    if not stop_name or not _SESSION_RE.match(stop_name):
        return False
    command = list(cmd or _find_cli() or [])
    if not command:
        return False
    stop_env = dict(env or _base_subprocess_env())
    stop_env["BU_NAME"] = stop_name
    stop_extra: dict = {}
    if os.name == "nt":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            stop_extra["creationflags"] = windows_hide_flags()
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            stop_extra["startupinfo"] = startup
        except Exception as exc:
            logger.debug("Windows cleanup hide-flags unavailable: %s", exc)
    try:
        result = subprocess.run(
            [command[0], "--reload"],
            input="",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=stop_env,
            **stop_extra,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Browser Harness cleanup failed for %s: %s", bu_name, exc)
        return False


def _harness_inactivity_timeout() -> int:
    """Use the existing browser inactivity policy, with a safe floor."""
    try:
        from tools.browser_tool import _get_session_inactivity_timeout

        return max(30, int(_get_session_inactivity_timeout()))
    except Exception:
        return 120


def _cleanup_inactive_harnesses() -> None:
    now = time.time()
    inactivity_timeout = _harness_inactivity_timeout()
    with _HARNESS_LOCK:
        stale = [
            (name, info, float(info.get("last_activity", now)))
            for name, info in _ACTIVE_HARNESSES.items()
            # A lease is held for the whole browser_exec subprocess.  A long
            # extraction may legitimately exceed the inactivity TTL; never
            # reload its daemon underneath it.
            if not int(info.get("in_flight", 0) or 0)
            and now - float(info.get("last_activity", now)) > inactivity_timeout
        ]
    for name, info, observed_activity in stale:
        # Re-check identity, lease, and activity while holding the same lock
        # used by browser_exec registration/release.  Holding it through the
        # exact-name --reload prevents a new call from becoming active after
        # selection but before the stop.  The stop is scoped by BU_NAME and
        # therefore cannot touch Toolbox Chrome or another daemon.
        with _HARNESS_LOCK:
            current = _ACTIVE_HARNESSES.get(name)
            if (
                current is not info
                or int(current.get("in_flight", 0) or 0)
                or float(current.get("last_activity", now)) != observed_activity
            ):
                continue
            logger.info("Cleaning inactive Browser Harness daemon %s", name)
            _stop_harness_daemon(name, current.get("env"), current.get("cmd"))
            # Do not remove a replacement entry if a future refactor ever
            # releases this lock around the subprocess.  The identity and
            # activity check also makes cleanup idempotent.
            if (
                _ACTIVE_HARNESSES.get(name) is current
                and not int(current.get("in_flight", 0) or 0)
                and float(current.get("last_activity", now)) == observed_activity
            ):
                _ACTIVE_HARNESSES.pop(name, None)


def _harness_cleanup_worker() -> None:
    while not _HARNESS_CLEANUP_STOP.wait(30.0):
        try:
            _cleanup_inactive_harnesses()
        except Exception as exc:  # pragma: no cover - defensive thread guard
            logger.debug("Browser Harness inactivity cleanup failed: %s", exc)


def _touch_harness(bu_name: str, info: Optional[dict] = None) -> None:
    """Record activity and lazily start the bounded inactivity reaper."""
    global _HARNESS_CLEANUP_THREAD
    with _HARNESS_LOCK:
        if info is not None:
            info["last_activity"] = time.time()
        else:
            existing = _ACTIVE_HARNESSES.get(bu_name)
            if existing is not None:
                existing["last_activity"] = time.time()
        if (
            _HARNESS_CLEANUP_THREAD is None
            or not _HARNESS_CLEANUP_THREAD.is_alive()
            or _HARNESS_CLEANUP_STOP.is_set()
        ):
            _HARNESS_CLEANUP_STOP.clear()
            _HARNESS_CLEANUP_THREAD = threading.Thread(
                target=_harness_cleanup_worker,
                name="browser-use-cleanup",
                daemon=True,
            )
            _HARNESS_CLEANUP_THREAD.start()


def _release_harness_lease(bu_name: str) -> bool:
    """Release one browser_exec lease and report whether it was the last.

    ``browser_exec`` calls this from a ``finally`` block, including timeout
    and launch-error paths.  The final activity touch keeps a successfully
    used daemon alive for the normal inactivity window, while the in-flight
    count prevents the reaper from stopping a daemon during a long call.
    """
    with _HARNESS_LOCK:
        info = _ACTIVE_HARNESSES.get(bu_name)
        if info is None:
            return True
        info["in_flight"] = max(0, int(info.get("in_flight", 0) or 0) - 1)
        info["last_activity"] = time.time()
        return info["in_flight"] == 0


def _stop_harness_if_idle(
    bu_name: str,
    entry: dict,
    env: dict,
    cmd: List[str],
) -> bool:
    """Reload and remove *entry* only while it is still the idle endpoint.

    This is used after a timeout.  A sibling call can acquire a lease between
    the timed-out subprocess's ``finally`` block and this helper, so identity
    and lease state are checked under the registry lock before the exact-name
    Browser Harness reload.  The lock stays held through the reload to make
    that check atomic with new registration.
    """
    with _HARNESS_LOCK:
        current = _ACTIVE_HARNESSES.get(bu_name)
        if current is not entry or int(current.get("in_flight", 0) or 0):
            return False
        stopped = _stop_harness_daemon(
            bu_name,
            current.get("env") or env,
            current.get("cmd") or cmd,
        )
        if _ACTIVE_HARNESSES.get(bu_name) is current and not int(
            current.get("in_flight", 0) or 0
        ):
            _ACTIVE_HARNESSES.pop(bu_name, None)
        return stopped


def _drop_harness_if_idle(bu_name: str, entry: dict) -> None:
    """Remove a failed launch's bookkeeping without stopping a sibling."""
    with _HARNESS_LOCK:
        current = _ACTIVE_HARNESSES.get(bu_name)
        if current is entry and not int(current.get("in_flight", 0) or 0):
            _ACTIVE_HARNESSES.pop(bu_name, None)


def cleanup_browser_use(task_id: Optional[str] = None) -> None:
    """Stop Browser Use daemons owned by *task_id*, or all when omitted."""
    with _HARNESS_LOCK:
        entries = list(_ACTIVE_HARNESSES.items())
    if task_id is None:
        selected = entries
    else:
        task = str(task_id)
        conversation = _canonical_conversation_id(task_id)
        selected = [
            (name, info)
            for name, info in entries
            if info.get("task_id") == task
            or info.get("conversation_id") == conversation
        ]
    for name, info in selected:
        _stop_harness_daemon(name, info.get("env"), info.get("cmd"))
        with _HARNESS_LOCK:
            _ACTIVE_HARNESSES.pop(name, None)


def cleanup_all_browser_use() -> None:
    """Stop every exact named Browser Harness daemon tracked by Hermes."""
    _HARNESS_CLEANUP_STOP.set()
    cleanup_browser_use()


atexit.register(cleanup_all_browser_use)


def browser_exec(
    code: str,
    session: str = "",
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    task_id: Optional[str] = None,
):
    """Run Python code through the browser-use CLI, and return its output"""
    from tools.registry import tool_error, tool_result

    if not code or not code.strip():
        return tool_error("No code provided. Pass Python that uses the pre-imported helpers, e.g. new_tab(\"https://example.com\") then print(page_info()).")

    blocked = _blocked_url_in_code(code)
    if blocked:
        return tool_error(blocked)

    omnio_local_cdp = _omnio_template_cdp_configured()
    cmd = _find_cli()
    if not cmd:
        if omnio_local_cdp:
            return tool_error(
                "Omnio's pinned Browser Use CLI is unavailable in the agent "
                "runtime. Reprovision this Omnio sandbox, then retry; installing "
                "browser-use from the Toolbox terminal or PATH cannot repair "
                "the agent-side managed runtime."
            )
        return tool_error(
            "The Hermes-managed Browser Use CLI is not installed. Install the "
            f"pinned package `{BROWSER_USE_PACKAGE}` with `hermes tools`, then "
            "retry. A PATH or floating uvx installation is not used."
        )

    env = _base_subprocess_env()
    if omnio_local_cdp:
        # Omnio runs Browser Use locally against Toolbox-owned Chrome. Never
        # pass a Browser Use cloud credential or permit cloud autospawn on
        # this path: an unavailable conversation CDP relay must fail closed,
        # not create a remote browser as a fallback.
        env.pop("BROWSER_USE_API_KEY", None)
        env["BU_AUTOSPAWN"] = "0"
    if session:
        if not _SESSION_RE.match(session):
            return tool_error(
                f"Invalid session name {session!r}: use 1-64 letters, digits, "
                "dashes, or underscores (e.g. 'r7k2')."
            )
    # Never export the human-readable session. The same explicit name is
    # stable within one conversation, while two conversations produce
    # different logical identities and private state paths.
    # ``bu_name`` is always the hashed logical identity used by Hermes' own
    # registry/workspace bookkeeping. Omnio gets private IPC/temp directories
    # for every logical session. Only the unnamed path uses the harness's
    # ``default`` tab so the live view remains on Toolbox's original tab;
    # explicit sessions keep their hashed non-default dedicated tab.
    bu_name = _derive_bu_name(task_id, session)
    if omnio_local_cdp:
        harness_name = bu_name if session else "default"
        harness_error = _configure_omnio_harness_dirs(
            env, bu_name, harness_name=harness_name, cmd=cmd
        )
        if harness_error:
            return tool_error(harness_error)
    else:
        env["BU_NAME"] = bu_name
    # Route through the configured browser backend (Browserbase, Firecrawl,
    # Nous gateway, CDP override, local Chrome, …). Named sessions compose
    # with the backend: BU_NAME namespaces the harness daemon (its IPC
    # socket, log, and pid), and on provider backends the name additionally
    # keys its own cloud browser — so concurrent sessions stop clobbering
    # each other's daemon (#86894). Browser Use direct-API cloud configs
    # are the one exception: the CLI manages named cloud browsers natively,
    # and _resolve_backend_cdp skips provider resolution for them.
    backend_err = _resolve_backend_cdp(
        env, task_id, session_name=session, bu_name=bu_name
    )
    if backend_err:
        return tool_error(backend_err)

    workspace = _workspace_dir(task_id, session)
    if workspace:
        env["BH_AGENT_WORKSPACE"] = workspace

    # BU_AUTOSPAWN makes the CLI start a Browser Use cloud browser when no
    # local Chrome/CDP endpoint is reachable (their API key authenticates it)
    if (
        "BU_CDP_WS" not in env
        and "BU_CDP_URL" not in env
        and "BU_AUTOSPAWN" not in env
        and _browser_use_cloud_autospawn_enabled()
    ):
        env["BU_AUTOSPAWN"] = "1"

    try:
        timeout = max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_S

    # Windows: hide the console the .cmd shim would flash (as browser_tool does)
    popen_extra: dict = {}
    if os.name == "nt":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            popen_extra["creationflags"] = windows_hide_flags()
            _si = subprocess.STARTUPINFO()
            _si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            popen_extra["startupinfo"] = _si
        except Exception as e:
            logger.debug("Windows hide-flags unavailable: %s", e)

    with _HARNESS_LOCK:
        entry = _ACTIVE_HARNESSES.get(bu_name)
        if entry is None:
            entry = {
                "task_id": str(task_id or ""),
                "conversation_id": _canonical_conversation_id(task_id),
                "cmd": list(cmd),
                "env": dict(env),
                "last_activity": time.time(),
                "in_flight": 0,
            }
            _ACTIVE_HARNESSES[bu_name] = entry
        else:
            # Same conversation/name may issue concurrent calls; keep one
            # daemon record while leasing it once per subprocess.
            entry.update(
                {
                    "task_id": str(task_id or ""),
                    "conversation_id": _canonical_conversation_id(task_id),
                    "cmd": list(cmd),
                    "env": dict(env),
                }
            )
        entry["in_flight"] = int(entry.get("in_flight", 0) or 0) + 1
        _touch_harness(bu_name, entry)

    started = time.time()
    proc = None
    timeout_error: Optional[subprocess.TimeoutExpired] = None
    launch_error: Optional[OSError] = None
    try:
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            **popen_extra,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_error = exc
    except OSError as exc:
        launch_error = exc
    finally:
        # Always release the lease after subprocess completion.  The activity
        # touch here protects a successful long call from the reaper, while
        # the in-flight count prevents reload during that call.
        last_lease = _release_harness_lease(bu_name)

    if timeout_error is not None:
        stopped = (
            _stop_harness_if_idle(bu_name, entry, env, cmd) if last_lease else False
        )
        return tool_error(
            f"browser-use exec timed out after {timeout}s. The daemon may "
            f"have been stopped ({'yes' if stopped else 'no'}); retry with a larger timeout_s (max "
            f"{_MAX_TIMEOUT_S}), or split the work into several calls that "
            "append to workspace files — anything already written to the "
            "workspace is preserved."
        )
    if launch_error is not None:
        _drop_harness_if_idle(bu_name, entry)
        return tool_error(f"Failed to launch browser-use CLI: {launch_error}")

    if proc is None:
        _drop_harness_if_idle(bu_name, entry)
        return tool_error("browser-use CLI returned no process result")

    result = {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": proc.stdout,
    }
    if workspace:
        result["workspace"] = workspace
    if session:
        result["session"] = session
    stderr = (proc.stderr or "").strip()
    if stderr:
        if len(stderr) > _STDERR_CAP_CHARS:
            stderr = stderr[:_STDERR_CAP_CHARS] + "\n… (stderr truncated)"
        result["stderr"] = stderr

    screenshot = _find_screenshot(proc.stdout, started)
    if screenshot:
        result["screenshot_path"] = screenshot
        native = _native_screenshot_result(result, screenshot)
        if native is not None:
            return native
    return tool_result(result)


# The tool description is intentionally static.  Reading a third-party skill
# file at runtime would make the model prompt version-dependent and would
# import uncontrolled text into every conversation.
_HEADER_BASE = (
    "Drive a real web browser via the Browser Use CLI. The `code` argument "
    "is piped verbatim to the `browser-use` CLI on stdin and executed as "
    "full Python (standard library available) with the CLI's pre-imported "
    "browser helpers; stdout comes back in the result. Start `code` with a "
    "one-line comment describing the step for the user in plain, "
    "non-technical language, max 60 chars (e.g. `# Searching Amazon for "
    "paper towels`) — the UI displays it as the step label.\n\n"
    "STATE: the browser session and the workspace persist across calls; "
    "Python variables do NOT (each call is a fresh interpreter). The "
    "workspace is a stable directory — path in $BH_AGENT_WORKSPACE and "
    "returned as `workspace` in every result. For multi-item tasks "
    "('collect all N products / every entry / the full table'), append each "
    "batch to a JSON/CSV file in the workspace as you go, then read it back "
    "to assemble the final answer; define reusable functions in "
    "agent_helpers.py there — the harness auto-imports it into every call. "
    "Do aggregation in code, not in your head: dedupe, count, sort, and "
    "format with Python inside the exec. Before giving a final answer on a "
    "multi-item task, verify the collected count against what was asked "
    "and go back for anything missing.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call "
    "— do not spend a call per action — but for long extractions prefer "
    "several medium calls that append to workspace files over one giant "
    "call, so progress survives timeouts. For an isolated concurrent "
    "browser session (parallel tasks that must not share tabs), pass "
    "session=<name> (never BU_NAME env syntax) and reuse the same name on "
    "every related call."
)

_HEADER_VISION = (
    " Screenshots are attached to your context automatically: when the exec "
    "output contains a capture_screenshot() path, the image arrives with "
    "this tool's result and you inspect it directly with your own vision — "
    "never send browser screenshots to a separate vision tool."
)

_HEADER_TEXT_ONLY = (
    " Your model cannot view images, so work text-first: page_info() for "
    "state, js() for reading/extracting DOM text, fill_input(selector, "
    "text) for inputs, and js(\"document.querySelector('…').click()\") for "
    "clicks — skip the screenshot-driven workflow described below."
)

_DESCRIPTION_HEADER = _HEADER_BASE  # back-compat alias for external imports

# NOTE: browser_exec is additionally gated at tool-definition time — sessions
# whose resolved toolsets do not include ``terminal`` never see it (see
# model_tools._compute_tool_definitions). The check_fn registered below only
# answers "is Browser Use mode configured"; surface policy lives with the
# session, not in the process-wide TTL-cached check_fn.


def _description_header() -> str:
    """Header tailored to whether the active model can see images natively"""
    try:
        from tools.vision_tools import _should_use_native_vision_fast_path

        if _should_use_native_vision_fast_path():
            return _HEADER_BASE + _HEADER_VISION
    except Exception:
        pass
    return _HEADER_BASE + _HEADER_TEXT_ONLY

_skill_text_cache: Optional[str] = None
_skill_text_fetched = False

# Pinned quick-reference for the CLI's pre-imported helpers. Replaces the
# live ``browser-use skill`` fetch: embedding whatever text the installed CLI
# version prints would ship uncontrolled third-party content into every
# session's system-side schema (version drift across machines, supply-chain
# exposure, and a byte-unstable prompt). A/B benchmarked Aug 2026 (108 runs,
# opus-4.8 + kimi-k3, 6 multi-step tasks x 3 reps): header-only schema went
# 36/36 vs 36/36 for the full skill dump at ~equal tokens (-60% vs the
# legacy browser_* toolset either way). The pinned digest below keeps the
# first-call reliability of the helper names without the 7.7KB dump.
_HELPERS_DIGEST = (
    "\n\nHELPERS (pre-imported): new_tab(url) opens/navigates (use for the "
    "FIRST navigation), goto_url(url) navigates the current tab, "
    "wait_for_load() after navigation, page_info() summarizes the current "
    "page state, js(expr) evaluates a JS expression and returns its value "
    "(js('document.title'); wrap function bodies as js('(() => {...})()') — "
    "a bare '() => {...}' returns the function itself, uncalled), "
    "fill_input(selector, text) types into inputs, click_at_xy(x, y) clicks "
    "viewport coordinates, capture_screenshot() saves and prints a "
    "screenshot path, cdp('Domain.method', **kwargs) is raw CDP — "
    "cdp('Accessibility.getFullAXTree')['nodes'] lists every element's "
    "role/name/backendDOMNodeId (filter in Python before printing; it is "
    "thousands of nodes), then cdp('DOM.getBoxModel', backendNodeId=n) gives "
    "click coordinates. ensure_real_tab() recovers from a stale/internal "
    "tab. Login walls: stop and ask the user; never guess credentials."
)


def _cli_skill_text() -> str:
    """Deprecated: always returns "" — the schema uses the pinned header.

    Kept so tests and any external callers keep importing a stable symbol;
    see _HELPERS_DIGEST for the rationale (benchmark-backed removal of the
    live ``browser-use skill`` fetch).
    """
    return _skill_text_cache or ""


def _dynamic_schema_overrides() -> dict:
    return {"description": _description_header() + _HELPERS_DIGEST}


BROWSER_EXEC_SCHEMA = {
    "name": "browser_exec",
    # Static fallback, used only when the managed CLI is unavailable.
    "description": (
        _HEADER_BASE
        + _HELPERS_DIGEST
        + "\n\n(The browser-use CLI is not installed yet. Install it with "
        f"`uv tool install {BROWSER_USE_PACKAGE}`.)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute using the pre-imported browser helpers. Use print(...) for any data you need back.",
            },
            "session": {
                "type": "string",
                "description": "Optional human-readable session label. Hermes hashes the canonical conversation id plus this label into the Browser Harness identity and private state paths, so the same label in two conversations stays isolated. Reuse the same label across calls in one conversation.",
            },
            "timeout_s": {
                "type": "integer",
                "description": f"Max seconds to wait for the code to finish (default {_DEFAULT_TIMEOUT_S}, max {_MAX_TIMEOUT_S}).",
                "default": _DEFAULT_TIMEOUT_S,
            },
        },
        "required": ["code"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="browser_exec",
    toolset="browser-use",
    schema=BROWSER_EXEC_SCHEMA,
    handler=lambda args, **kw: browser_exec(
        code=args.get("code", ""),
        session=args.get("session", "") or "",
        timeout_s=args.get("timeout_s", _DEFAULT_TIMEOUT_S),
        task_id=kw.get("task_id"),
    ),
    check_fn=is_browser_use_cli_mode,
    dynamic_schema_overrides=_dynamic_schema_overrides,
    emoji="🌐",
)
