"""Public task-query/control API for the native persistent WorkItem queue.

CONCEPT:AU-KG.compute.persistent-task-tracking - Persistent Task Tracking

Split out of ``engine_tasks.TaskManagerMixin`` (CX wD10 file-decomposition):
this mixin owns the READ/CONTROL surface a caller uses to inspect or manage a
submitted task by job id (``get_task_status``, ``list_tasks``, ``cancel_task``,
``prioritize_task``, and the two immutable-audit-record rejections
``remove_task``/``clear_tasks``/``clear_completed_tasks``). It is pure
presentation and control-flow over WorkItem — it owns no scheduling, claiming,
or execution state itself; those live in ``TaskManagerMixin`` proper (worker
claim/lease, background task handlers, maintenance ticks, ...), which composes
this mixin in via ``TaskManagerMixin(TaskQueryMixin, GraphEngineProtocol)``.
Every method here calls back onto sibling mixin state through ``self``
(``self._work_item_engine``, ``self._ingest_work_item_index()``,
``self._active_work_item_claim``) exactly as it did before the split — Python
resolves those through the composed class's MRO, so moving the methods here
changes nothing about runtime behavior.
"""

from __future__ import annotations

import time
from typing import Any, Protocol


class _TaskQueryHost(Protocol):
    """Structural contract this mixin needs from whatever it is composed with.

    ``TaskQueryMixin`` calls back onto sibling ``TaskManagerMixin`` state
    (worker claim/lease bookkeeping lives there, not here); a plain
    ``TaskQueryMixin`` checked in isolation has no way for mypy to know those
    attributes exist. This Protocol names exactly the three it uses, typed
    the same way their own definitions and every other caller already type
    them (``_work_item_engine`` is opaque — ``orchestration.work_item``'s own
    functions accept it as ``Any``, e.g. ``get_work_item(engine: Any, ...)``).
    """

    @property
    def _work_item_engine(self) -> Any: ...

    def _ingest_work_item_index(self) -> dict[str, dict[str, Any]]: ...

    def _active_work_item_claim(
        self, job_id: str, *, pop: bool = False
    ) -> dict[str, Any] | None: ...

# Rendered public status vocabulary a job can be bucketed under in
# ``list_tasks``'s response (CONCEPT:AU-KG.compute.persistent-task-tracking).
_TASK_STATUS_BUCKETS = (
    "running",
    "pending",
    "scheduled",
    "blocked",
    "completed",
    "failed",
    "cancelled",
    "dead_letter",
    "unknown",
)

# Result-summary keys copied onto a completed job's public listing entry.
_COMPLETED_SUMMARY_KEYS = (
    "chunks_added",
    "nodes_added",
    "edges_added",
    "diffs_added",
    "chunks_skipped",
    "skip_reason",
)


def _task_status_from_work_item(item: dict[str, Any] | None) -> str:
    """Render the public job-status vocabulary from the sole WorkItem."""
    status = str((item or {}).get("status") or "")
    if (
        status == "ready"
        and float((item or {}).get("next_retry_at") or 0) > time.time()
    ):
        return "scheduled"
    return {
        "submitted": "blocked",
        "ready": "pending",
        "leased": "running",
        "running": "running",
        "succeeded": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "dead_letter": "dead_letter",
    }.get(status, "unknown")


def _coerce_prio_bucket(value: Any, default: int = 2) -> int:
    """Validate a current WorkItem claim bucket in the closed interval 0..3."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError("WorkItem prio_bucket must be an integer")
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        raise ValueError("WorkItem prio_bucket must be between 0 and 3")
    raise TypeError("WorkItem prio_bucket must be an integer")


def _task_list_entry(job_id: str, item: dict[str, Any], status: str) -> dict[str, Any]:
    """The public per-job record rendered into a ``list_tasks`` bucket."""
    meta = item.get("metadata") or {}
    job_info: dict[str, Any] = {
        "job_id": job_id,
        "target": meta.get("target", "unknown"),
    }
    if status in {"failed", "dead_letter"}:
        job_info["error"] = item.get("error_ref") or "Unknown error"
    elif status == "completed":
        # Include result summary for completed jobs
        for key in _COMPLETED_SUMMARY_KEYS:
            if key in meta:
                job_info[key] = meta[key]
    return job_info


def _stamp_task_progress(response: dict[str, Any], total_tasks: int) -> None:
    """Add the progress rollup onto a rendered ``list_tasks`` response."""
    completed_count = len(response["completed"])
    progress = round((completed_count / total_tasks) * 100, 2)
    response["progress_percentage"] = f"{progress}% complete"
    response["progress_stats"] = {
        "total_tasks": total_tasks,
        "completed": completed_count,
        "pending_in_graph": len(response["pending"]),
        "running_in_graph": len(response["running"]),
        "scheduled": len(response["scheduled"]),
        "blocked": len(response["blocked"]),
    }


class TaskQueryMixin:
    """Read/control surface over the persistent WorkItem task queue.

    Composed onto ``TaskManagerMixin``; see the module docstring for the
    state-sharing contract with the sibling mixin(s) it is composed with.
    """

    def get_task_status(self: _TaskQueryHost, job_id: str) -> dict | None:
        """Render one ingestion WorkItem using the public job vocabulary."""
        from agent_utilities.orchestration import work_item as _wi

        item = _wi.get_work_item(
            self._work_item_engine, _wi.ingest_task_work_item_id(job_id)
        )
        if item is None or item.get("kind") != "ingest_task":
            return None
        status = _task_status_from_work_item(item)

        return {
            "job_id": job_id,
            "status": status,
            "metadata": dict(item.get("metadata") or {}),
            "attempt": item.get("attempt"),
            "max_attempts": item.get("max_attempts"),
            "resource_class": item.get("resource_class"),
            "lease_expires_at": item.get("lease_expires_at"),
            "heartbeat_at": item.get("heartbeat_at"),
            "updated_at": item.get("updated_at"),
        }

    def list_tasks(self: _TaskQueryHost) -> dict:
        """Group ingestion WorkItems by their rendered public status."""
        response: dict[str, Any] = {name: [] for name in _TASK_STATUS_BUCKETS}

        for job_id, item in self._ingest_work_item_index().items():
            status = _task_status_from_work_item(item)
            if status not in response:
                continue
            response[status].append(_task_list_entry(job_id, item, status))

        total_tasks = sum(len(items) for items in response.values())
        if total_tasks > 0:
            _stamp_task_progress(response, total_tasks)
        return response

    def remove_task(self, job_id: str) -> bool:
        """WorkItem audit records are immutable and cannot be removed."""
        return False

    def clear_completed_tasks(self: _TaskQueryHost) -> dict:
        """Reject deletion of immutable WorkItem audit records."""
        return {
            "status": "error",
            "error": "WorkItem audit records cannot be cleared",
            "cleared": 0,
            "remaining": len(self._ingest_work_item_index()),
        }

    def cancel_task(self: _TaskQueryHost, job_id: str) -> dict:
        """Cancel a single queued/running task by id (terminal 'cancelled').

        The native engine owns cancellation and preserves the audit record.
        """
        if not job_id:
            return {"status": "error", "error": "job_id required"}
        try:
            from agent_utilities.orchestration import work_item as _wi

            item_id = _wi.ingest_task_work_item_id(job_id)
            prior = _wi.get_work_item(self._work_item_engine, item_id)
            if prior is None:
                return {"status": "error", "error": f"job {job_id} not found"}
            cancelled = _wi.cancel_work_item(
                self._work_item_engine,
                item_id,
                reason="cancel_task",
            )
        except Exception as e:  # noqa: BLE001 — public control API is structured
            return {"status": "error", "error": f"WorkItem cancel failed: {e}"}
        if not cancelled:
            return {
                "status": "error",
                "error": "WorkItem cancellation was rejected by its current lease",
            }
        self._active_work_item_claim(job_id, pop=True)
        return {
            "status": "success",
            "job_id": job_id,
            "prev_status": _task_status_from_work_item(prior),
        }

    def clear_tasks(self: _TaskQueryHost, status: str = "completed") -> dict:
        """Reject deletion of immutable WorkItem audit records."""
        status = (status or "completed").strip().lower()
        valid = {
            "pending",
            "running",
            "scheduled",
            "blocked",
            "completed",
            "failed",
            "cancelled",
            "dead_letter",
            "all",
        }
        if status not in valid:
            return {
                "status": "error",
                "error": f"status must be one of {sorted(valid)}",
            }

        return {
            "status": "error",
            "error": "WorkItem audit records cannot be cleared",
            "cleared": 0,
            "filter": status,
            "remaining": len(self._ingest_work_item_index()),
        }

    def prioritize_task(self: _TaskQueryHost, job_id: str, priority: int = 1) -> dict:
        """Re-prioritize a task by setting its claim bucket (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task).

        The worker claim iterates integer buckets 0..3 in ascending order, so
        a lower bucket runs first. Named priority aliases are not accepted.
        """
        try:
            bucket = _coerce_prio_bucket(priority)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "error": "priority must be an integer bucket from 0 through 3",
            }
        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.ingest_task_work_item_id(job_id)
        if _wi.get_work_item(self._work_item_engine, item_id) is None:
            return {"status": "error", "error": f"job {job_id} not found"}
        if not _wi.set_work_item_priority(self._work_item_engine, item_id, bucket):
            return {
                "status": "error",
                "error": "WorkItem priority update was rejected",
            }
        return {
            "status": "success",
            "job_id": job_id,
            "prio_bucket": bucket,
            "task_status": _task_status_from_work_item(
                _wi.get_work_item(self._work_item_engine, item_id)
            ),
        }
