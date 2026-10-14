"""Read-only census of writer-capable work used before runtime retirement.

An idle root Turn is insufficient: child agents, cron, process watchers and
pending completion delivery can still own work. Unknown counts prevent retirement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

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
    """Collect all Hermes writer-capable work visible to a retirement caller.

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
