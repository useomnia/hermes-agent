"""Map host skill directory paths to backend-visible paths.

When the active terminal backend is remote (Docker, SSH, Daytona, Singularity,
Modal), the skills tree lives at a different filesystem location inside the
sandbox than on the host.  Skill content that references the skill directory
(``${HERMES_SKILL_DIR}``, ``[Skill directory: ...]``, supporting-file hints)
must use the backend-visible path, or the agent will try to run bundled
scripts via paths that do not exist in the sandbox (hermes-agent#41541,
#73842).

The authoritative mount layout is computed by
``tools.credential_files.get_skills_directory_layout()``; this module consumes
it with a longest-prefix match and falls back to the host path whenever the
backend is local or unknown, so behavior on local backends is unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

# Container backends whose skills mount root is /root/.hermes inside the
# sandbox.  Mirrors the class names in tools/environments/*.py.
_CONTAINER_ENV_CLASSES = {
    "DockerEnvironment",
    "SingularityEnvironment",
    "ModalEnvironment",
    "ManagedModalEnvironment",
}

# TERMINAL_ENV values that map to the /root/.hermes container layout even
# before a live environment object exists (used at first skill render).
_CONTAINER_BACKEND_NAMES = {"docker", "singularity", "modal"}

# Remote backends whose .hermes root is only known from the live environment
# (SSH/Daytona resolve a remote home at connect time).
_REMOTE_BACKEND_NAMES = {"ssh", "daytona"}


def _active_terminal_env(task_id: str | None) -> Any:
    """Return the live terminal environment for *task_id*, or None."""
    if not task_id:
        return None
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id)
    except Exception:
        logger.debug("Could not resolve active terminal env", exc_info=True)
        return None


def _backend_name() -> str:
    return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"


def _hermes_base_for_env(env: Any, backend_name: str) -> str | None:
    """Resolve the backend-visible ``.hermes`` root, or None if unknown.

    None means "the host path is the backend path" (local/unknown backend) or
    the backend root cannot be determined without a live environment.
    """
    if env is not None:
        remote_home = getattr(env, "_remote_home", None)
        if remote_home:
            return f"{str(remote_home).rstrip('/')}/.hermes"
        if type(env).__name__ in _CONTAINER_ENV_CLASSES:
            return "/root/.hermes"
        return None
    if backend_name in _CONTAINER_BACKEND_NAMES:
        return "/root/.hermes"
    return None


def map_skill_dir_for_backend(
    host_skill_dir: Path | str | None,
    task_id: str | None = None,
) -> str:
    """Translate *host_skill_dir* to the path the agent sees on the backend.

    Longest-prefix-matches the host path against the existing skills mount
    layout (``get_skills_directory_layout``) and returns the corresponding
    backend-visible path (POSIX form, since container/remote paths are
    POSIX).  Falls back to the host path unchanged when:

    - the backend is local or unknown (TERMINAL_ENV unset / "local"),
    - the backend root cannot be determined without a live environment
      (SSH/Daytona before first connect),
    - the directory is not under any known skills mount, or
    - the mount layout cannot be resolved.
    """
    if host_skill_dir is None:
        return ""
    host = str(host_skill_dir)
    if _backend_name() == "sprites":
        return _map_sprites_skill_dir(host)
    base = _hermes_base_for_env(_active_terminal_env(task_id), _backend_name())
    if not base:
        return host
    try:
        from tools.credential_files import get_skills_directory_layout

        mounts = get_skills_directory_layout(container_base=base)
    except Exception:
        logger.debug("Could not resolve skills directory mount layout", exc_info=True)
        return host
    if not mounts:
        return host

    # Longest-prefix match against mount host paths.  Normalize separators so
    # Windows hosts (backslash paths) match against POSIX container prefixes.
    # On Windows, filesystems are case-insensitive, so comparisons are
    # lowercased there; on POSIX hosts the match stays case-sensitive.
    host_norm = host.replace("\\", "/")
    case_fold = os.name == "nt"
    best_prefix: str | None = None
    best_container: str | None = None
    for m in mounts:
        # Match against the canonical source path AND the actual mount
        # source: when symlinks are present the mount host_path is a
        # sanitized copy while agent-visible skill dirs live under the
        # canonical skills tree (source_path).
        for key in ("source_path", "host_path"):
            candidate = m.get(key)
            if not candidate:
                continue
            prefix = str(candidate).rstrip("/").replace("\\", "/")
            match_norm = host_norm if not case_fold else host_norm.lower()
            prefix_norm = prefix if not case_fold else prefix.lower()
            if match_norm == prefix_norm or match_norm.startswith(prefix_norm + "/"):
                if best_prefix is None or len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_container = m["container_path"]
    if best_container is None:
        return host

    rel = host_norm[len(best_prefix):].lstrip("/")
    if not rel:
        return best_container
    return f"{best_container.rstrip('/')}/{rel}"


def _map_sprites_skill_dir(host: str) -> str:
    """Use the same roots as Sprites' secret-free skills sync (no host mounts)."""
    from agent.skill_utils import get_external_skills_dirs
    from tools.credential_files import _resolve_hermes_home

    roots = [(_resolve_hermes_home() / "skills", "/skills")]
    roots.extend((root, f"/skills/external_skills/{index}")
                 for index, root in enumerate(get_external_skills_dirs()))
    path = Path(host).absolute()
    for root, target in sorted(roots, key=lambda item: len(item[0].parts), reverse=True):
        try:
            relative = path.relative_to(root.absolute())
        except ValueError:
            continue
        # The sync skips symlinks below each catalog root. Never advertise
        # a path to bytes that are intentionally excluded from the projection.
        current = root
        if ".." in relative.parts:
            return host
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return host
        return str(PurePosixPath(target) / relative.as_posix())
    return host
