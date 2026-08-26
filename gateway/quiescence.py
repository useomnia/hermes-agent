"""Writer-quiescence accounting shared by the gateway and Omnio handover.

The handover caller needs a stronger signal than ``active_agents == 0``:
detached async children, API requests before agent publication, cron workers,
terminal processes, process watchers, and completion delivery can all outlive
the root turn that started them.  This module keeps the cross-subsystem
accounting in one small, dependency-light seam.  Individual subsystems own
their locks; async delegation additionally exposes one SQLite-transactional
count for its lifecycle and delivery state.

The snapshot is deliberately read-only with respect to admission.  Omnia
closes its external admission in its own durable transaction before asking for
this snapshot, then takes its SQL turn fence after a zero result.  Hermes does
not latch a graceful gate here, so pending completion delivery cannot be
stranded behind a local gate.
"""

from __future__ import annotations

import os
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_OFFLINE_SNAPSHOT_FILENAME = "gateway_quiescence.json"
_OFFLINE_LOCK_FILENAME = "gateway_quiescence.lock"
_OFFLINE_SNAPSHOT_OBJECT = "hermes.gateway.quiescence.offline"
_OFFLINE_BOOT_ID = uuid.uuid4().hex

# These are the categories included in ``total``.  The response may also
# expose aliases/details, but aliases must not double-count the total.
_TOTAL_KEYS = (
    "api_runs",
    "gateway_agents",
    "background_agent_tasks",
    "cron_jobs",
    "processes",
    "process_watchers",
    "completion_queue",
)


def _nonnegative_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _record_failure(
    counts: Dict[str, int], errors: List[str], key: str, exc: BaseException
) -> None:
    """Fail closed for one unavailable subsystem without leaking internals."""
    counts[key] = max(1, counts.get(key, 0))
    errors.append(key)
    logger.warning("Quiescence count unavailable for %s: %s", key, exc)


def _resolve_runner(adapter: Any = None, runner: Any = None) -> Any:
    if runner is not None:
        return runner
    if adapter is not None:
        candidate = getattr(adapter, "gateway_runner", None)
        if candidate is not None:
            return candidate
    try:
        from gateway.run import _gateway_runner_ref

        return _gateway_runner_ref()
    except Exception:
        return None


def _count_task_set(owner: Any, *names: str) -> int:
    """Count explicitly registered writer-capable task sets, if present.

    ``GatewayRunner._background_tasks`` intentionally contains always-on
    watchers and therefore cannot be used: counting it would make a running
    gateway permanently non-quiescent. New background agent owners can opt in
    by exposing ``_background_agent_tasks`` (or the compatibility name
    ``_writer_tasks``) without changing this contract.
    """
    total = 0
    for name in names:
        tasks = getattr(owner, name, None)
        if tasks is None:
            continue
        try:
            total += sum(not task.done() for task in tasks)
        except AttributeError:
            try:
                total += len(tasks)
            except TypeError:
                continue
    return _nonnegative_count(total)


def collect_writer_work_snapshot(
    *, adapter: Any = None, runner: Any = None
) -> Dict[str, Any]:
    """Collect all Hermes writer-capable work visible to a handover caller.

    ``known`` is false when a subsystem cannot be read.  Its category is set
    to one so consumers that only inspect ``total`` still fail closed.  The
    async delegation category comes from one durable SQL count that combines
    running/finalizing lifecycle rows with pending/claimed delivery rows.
    """
    runner = _resolve_runner(adapter, runner)
    counts: Dict[str, int] = {}
    errors: List[str] = []

    # API requests reserve a slot before their first await; active_agent_work_count
    # therefore includes request parsing/agent-publication gaps as well as live
    # /v1/runs tasks.
    api_owner = adapter
    if api_owner is None and runner is not None:
        try:
            from gateway.config import Platform

            api_owner = getattr(runner, "adapters", {}).get(Platform.API_SERVER)
        except Exception:
            api_owner = None
    try:
        helper = getattr(api_owner, "quiescence_agent_work_count", None)
        if not callable(helper):
            helper = getattr(api_owner, "active_agent_work_count", None)
        if callable(helper):
            counts["api_runs"] = _nonnegative_count(helper())
        elif runner is not None:
            helper = getattr(runner, "_active_api_run_count", None)
            counts["api_runs"] = _nonnegative_count(helper()) if callable(helper) else 0
        else:
            counts["api_runs"] = 0
    except Exception as exc:  # pragma: no cover - defensive integration seam
        _record_failure(counts, errors, "api_runs", exc)

    try:
        helper = getattr(runner, "_running_agent_count", None)
        counts["gateway_agents"] = _nonnegative_count(helper()) if callable(helper) else 0
    except Exception as exc:  # pragma: no cover - defensive integration seam
        _record_failure(counts, errors, "gateway_agents", exc)

    try:
        # This optional registry is for one-shot background agent tasks that
        # are not represented by the API reservation, session agent, or async
        # delegation durable row. It is intentionally distinct from the
        # runner's always-on _background_tasks watcher set.
        counts["background_agent_tasks"] = _count_task_set(
            runner,
            "_background_agent_tasks",
            "_writer_tasks",
            # Existing one-shot ownership sets. The broad _background_tasks
            # set is intentionally excluded because it also contains
            # always-on supervisors.
            "_deferred_agent_cleanup_tasks",
            "_startup_restore_tasks",
        ) if runner is not None else 0
    except Exception as exc:  # pragma: no cover - defensive integration seam
        _record_failure(counts, errors, "background_agent_tasks", exc)

    try:
        helper = getattr(runner, "_active_cron_job_count", None)
        if callable(helper):
            counts["cron_jobs"] = _nonnegative_count(helper())
        else:
            from cron.scheduler import get_running_job_ids

            counts["cron_jobs"] = len(get_running_job_ids())
    except Exception as exc:
        # Cron is optional, so a missing module is not an unknown writer. A
        # present but broken scheduler must still fail closed.
        try:
            import cron.scheduler  # noqa: F401
        except ImportError:
            counts["cron_jobs"] = 0
        else:
            _record_failure(counts, errors, "cron_jobs", exc)

    try:
        from tools.async_delegation import quiescence_work_count

        async_count = _nonnegative_count(quiescence_work_count())
        counts["background_agent_tasks"] += async_count
        # Keep the more precise name available to callers without adding it to
        # total a second time. This makes mixed-version Omnia clients easier to
        # diagnose while preserving one canonical aggregate.
        counts["async_delegations"] = async_count
    except Exception as exc:
        _record_failure(counts, errors, "background_agent_tasks", exc)
        counts["async_delegations"] = counts.get("background_agent_tasks", 1)

    try:
        from tools.process_registry import process_registry

        helper = getattr(process_registry, "quiescence_work_snapshot", None)
        if callable(helper):
            process_counts = helper() or {}
            counts["processes"] = _nonnegative_count(process_counts.get("processes", 0))
            counts["process_watchers"] = _nonnegative_count(
                process_counts.get("process_watchers", 0)
            )
            counts["completion_queue"] = _nonnegative_count(
                process_counts.get("completion_queue", 0)
            )
            # Preserve useful detail fields without making consumers infer
            # watcher totals from implementation-specific names.
            counts["active_watchers"] = _nonnegative_count(
                process_counts.get("active_watchers", 0)
            )
            counts["pending_watchers"] = _nonnegative_count(
                process_counts.get("pending_watchers", 0)
            )
        else:
            counts["processes"] = _nonnegative_count(process_registry.count_running())
            counts["process_watchers"] = _nonnegative_count(
                process_registry.watcher_work_count()
            )
            counts["completion_queue"] = _nonnegative_count(
                process_registry.completion_queue.qsize()
            )
    except Exception as exc:
        _record_failure(counts, errors, "processes", exc)
        counts.setdefault("process_watchers", 1)
        counts.setdefault("completion_queue", 1)

    total = sum(counts.get(key, 0) for key in _TOTAL_KEYS)
    known = not errors
    return {
        "counts": counts,
        "total": total,
        "known": known,
        "errors": errors,
    }


# Short alias for callers that prefer the term used by the HTTP contract.
collect_quiescence_snapshot = collect_writer_work_snapshot


def _offline_snapshot_path() -> Path:
    """Return the profile-local snapshot path used by cold-state readers."""
    return get_hermes_home() / _OFFLINE_SNAPSHOT_FILENAME


def offline_quiescence_marker_exists() -> bool:
    """Whether a persisted marker exists (including malformed markers)."""
    try:
        return _offline_snapshot_path().exists()
    except OSError:
        # An inaccessible profile directory is fail-closed by startup callers.
        return True


def quiescence_boot_id() -> str:
    """Return this gateway process's opaque boot identity."""
    return _OFFLINE_BOOT_ID


@contextmanager
def _offline_marker_lock():
    """Serialize startup/shutdown marker ownership across processes."""
    lock_path = get_hermes_home() / _OFFLINE_LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            # Windows has no fcntl; its gateway lock/runner replacement path
            # provides the process exclusion there. Thread exclusion remains
            # available through the atomic marker write itself.
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def write_offline_quiescence_snapshot(
    snapshot: Dict[str, Any], *, lifecycle: str = "unknown",
    force_latched: bool = False, generation: Optional[int] = None,
    force_boot_id: Optional[str] = None
) -> bool:
    """Persist a fail-closed handover snapshot for a stopped gateway.

    The process registry and watcher queue are in-memory, so no cold reader
    can infer their absence from ``state.db``.  Gateway startup first writes an
    ``unknown`` marker; clean teardown replaces it only after the final live
    snapshot has been collected.  A stale ``quiescent`` marker therefore
    cannot survive an unclean restart, and a cold profile with a ``busy`` or
    ``unknown`` marker must wait for a live gateway proof.
    """
    counts = dict(snapshot.get("counts") or {})
    total = int(snapshot.get("total") or 0)
    known = bool(snapshot.get("known"))
    state = "quiescent" if known and total == 0 else "busy" if total else "unknown"
    payload = {
        "object": _OFFLINE_SNAPSHOT_OBJECT,
        "lifecycle": str(lifecycle or "unknown"),
        "state": state,
        "known": known,
        "counts": counts,
        "total": total,
        "errors": list(snapshot.get("errors") or []),
        "pid": os.getpid(),
        "boot_id": _OFFLINE_BOOT_ID,
        "force_latched": bool(force_latched),
        "generation": int(generation) if generation is not None else None,
        "force_boot_id": str(force_boot_id or (_OFFLINE_BOOT_ID if force_latched else "")),
        "observed_at": time.time(),
    }
    path = _offline_snapshot_path()
    with _offline_marker_lock():
        # A previous process may finish shutting down after its replacement
        # has already marked this profile as starting. Never let that stale
        # process overwrite the replacement's marker.
        if lifecycle in {"running", "force_latched", "stopped"}:
            existing = _read_offline_snapshot_unlocked(path)
            if existing and existing.get("boot_id") != _OFFLINE_BOOT_ID:
                logger.warning(
                    "Skipping stale offline quiescence marker from boot %s",
                    _OFFLINE_BOOT_ID,
                )
                return False
        atomic_json_write(path, payload, indent=None, separators=(",", ":"))
        _fsync_parent_directory(path)
    return True


def mark_offline_quiescence_unknown() -> bool:
    """Invalidate the previous clean marker and verify the replacement."""
    previous = read_offline_quiescence_snapshot() or {}
    if offline_quiescence_marker_exists() and (
        not previous or not offline_quiescence_marker_well_formed(previous)
    ):
        # Distinguish a first boot (no file) from a corrupt/unreadable file.
        # Never replace an unknown persisted state with a fresh ordinary boot.
        return False
    written = write_offline_quiescence_snapshot(
        {
            "known": False,
            "total": 0,
            "counts": {"unknown": 1},
            "errors": ["gateway_starting"],
        },
        lifecycle="starting",
        # Preserve a failed/successful force retirement across a process
        # replacement. The new adapter rehydrates this latch before it can
        # admit work; only a release carrying the old proof identity can clear
        # the persisted retirement state.
        force_latched=bool(previous.get("force_latched")),
        generation=previous.get("generation"),
        force_boot_id=(
            previous.get("force_boot_id") or previous.get("boot_id")
            if previous.get("force_latched")
            else None
        ),
    )
    if not written:
        return False
    marker = read_offline_quiescence_snapshot() or {}
    return bool(
        marker.get("boot_id") == _OFFLINE_BOOT_ID
        and marker.get("lifecycle") == "starting"
        and marker.get("known") is False
        and "gateway_starting" in (marker.get("errors") or [])
    )


def read_offline_quiescence_snapshot() -> Optional[Dict[str, Any]]:
    """Read the last clean/unknown marker without importing live gateway state."""
    path = _offline_snapshot_path()
    with _offline_marker_lock():
        return _read_offline_snapshot_unlocked(path)


def _read_offline_snapshot_unlocked(path: Path) -> Optional[Dict[str, Any]]:
    """Read a marker while the caller owns the cross-process marker lock."""
    try:
        import json

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def offline_quiescence_marker_well_formed(payload: Any) -> bool:
    """Validate persisted marker fields before treating them as proof state.

    Older Hermes versions may omit additive fields, so validation is limited
    to fields that are present.  A malformed force/generation identity is
    unsafe to repair in place: startup must remain closed until an operator or
    a matching control-plane release resolves it.
    """
    if not isinstance(payload, dict):
        return False
    if "force_latched" in payload and not isinstance(payload["force_latched"], bool):
        return False
    generation = payload.get("generation")
    if generation is not None and (
        isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
    ):
        return False
    for key in ("boot_id", "force_boot_id"):
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            return False
    if payload.get("force_latched") and not (
        payload.get("force_boot_id") or payload.get("boot_id")
    ):
        return False
    if "state" in payload and payload["state"] not in {"unknown", "busy", "quiescent"}:
        return False
    if "known" in payload and not isinstance(payload["known"], bool):
        return False
    if "total" in payload and (
        isinstance(payload["total"], bool)
        or not isinstance(payload["total"], int)
        or payload["total"] < 0
    ):
        return False
    if "counts" in payload and not isinstance(payload["counts"], dict):
        return False
    return True


def _fsync_parent_directory(path: Path) -> None:
    """Durably publish an atomic rename, not just the file contents."""
    flags = getattr(os, "O_RDONLY", 0)
    directory_fd = os.open(str(path.parent), flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def collect_offline_durable_snapshot() -> Dict[str, Any]:
    """Read durable async work for a cold profile, failing closed on zero.

    This helper is safe for a supervisor that cannot reach the HTTP listener.
    A non-zero async count proves the profile is busy.  A zero async count is
    only ``unknown`` because process completions/watchers are memory-resident;
    callers must then use a clean ``quiescent`` offline marker or wait for a
    live ``/v1/omnio/quiescence`` response.
    """
    try:
        from tools.async_delegation import quiescence_work_count

        async_count = _nonnegative_count(quiescence_work_count())
    except Exception as exc:  # pragma: no cover - defensive/offline caller
        logger.warning("Offline quiescence durable count unavailable: %s", exc)
        return {
            "object": _OFFLINE_SNAPSHOT_OBJECT,
            "state": "unknown",
            "known": False,
            "counts": {"async_delegations": 1},
            "total": 1,
            "errors": ["async_delegations"],
        }
    marker = read_offline_quiescence_snapshot()
    if async_count:
        return {
            "object": _OFFLINE_SNAPSHOT_OBJECT,
            "state": "busy",
            "known": True,
            "counts": {"async_delegations": async_count},
            "total": async_count,
            "errors": [],
        }
    if marker and marker.get("force_latched"):
        return {
            "object": _OFFLINE_SNAPSHOT_OBJECT,
            "state": "busy",
            "known": True,
            "counts": {"force_latched": 1},
            "total": 1,
            "errors": ["force_latched"],
            "boot_id": marker.get("force_boot_id") or marker.get("boot_id", ""),
            "generation": marker.get("generation"),
        }
    if (
        marker
        and marker.get("state") == "quiescent"
        and marker.get("known")
        and marker.get("lifecycle") == "stopped"
        and marker.get("boot_id")
    ):
        return marker
    return {
        "object": _OFFLINE_SNAPSHOT_OBJECT,
        "state": "unknown",
        "known": False,
        "counts": {"async_delegations": 0},
        "total": 0,
        "errors": ["process_registry_not_durable"],
    }


def interrupt_writer_work(
    *, adapter: Any = None, runner: Any = None, reason: str = "quiescence force"
) -> Dict[str, Any]:
    """Best-effort cooperative/force interruption of writer-capable work.

    This function only signals existing primitives.  Callers must collect a
    fresh snapshot and report busy until every signalled worker has actually
    settled; an interrupt request itself is never proof of quiescence.
    """
    runner = _resolve_runner(adapter, runner)
    actions: Dict[str, int] = {}
    errors: List[str] = []

    if runner is not None:
        helper = getattr(runner, "_interrupt_running_agents", None)
        if callable(helper):
            try:
                helper(reason)
                actions["gateway_agents"] = 1
            except Exception as exc:  # pragma: no cover - defensive seam
                errors.append("gateway_agents")
                logger.warning("Could not interrupt gateway agents: %s", exc)

    if adapter is not None:
        # Covers /v1/runs plus chat/responses agents that are still executing
        # off-loop. The adapter owns this helper so the quiescence module does
        # not depend on its private maps.
        helper = getattr(adapter, "interrupt_active_agents", None)
        if callable(helper):
            try:
                actions["api_runs"] = _nonnegative_count(helper(reason))
            except Exception as exc:  # pragma: no cover
                errors.append("api_runs")
                logger.warning("Could not interrupt API agents: %s", exc)

    try:
        from cron.scheduler import mark_running_jobs_interrupted

        interrupted = mark_running_jobs_interrupted(reason)
        actions["cron_jobs"] = len(interrupted or [])
    except ImportError:
        actions["cron_jobs"] = 0
    except Exception as exc:  # pragma: no cover
        errors.append("cron_jobs")
        logger.warning("Could not interrupt cron jobs: %s", exc)

    try:
        from tools.async_delegation import interrupt_all

        actions["async_delegations"] = _nonnegative_count(interrupt_all(reason))
    except Exception as exc:  # pragma: no cover
        errors.append("async_delegations")
        logger.warning("Could not interrupt async delegations: %s", exc)

    try:
        from tools.process_registry import process_registry

        actions["processes"] = _nonnegative_count(process_registry.kill_all())
    except Exception as exc:  # pragma: no cover
        errors.append("processes")
        logger.warning("Could not kill background processes: %s", exc)

    return {"actions": actions, "errors": errors}
