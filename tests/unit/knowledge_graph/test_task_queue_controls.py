"""Ingestion queue controls preserve WorkItem as immutable authority."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.orchestration import work_item as wi


class Harness:
    cancel_task = TaskManagerMixin.cancel_task
    clear_tasks = TaskManagerMixin.clear_tasks
    clear_completed_tasks = TaskManagerMixin.clear_completed_tasks
    remove_task = TaskManagerMixin.remove_task
    prioritize_task = TaskManagerMixin.prioritize_task
    _active_work_item_claim = TaskManagerMixin._active_work_item_claim

    def __init__(self) -> None:
        self._work_item_engine_cache = object()
        self._active_work_item_claims: dict[str, dict[str, Any]] = {}
        import threading

        self._active_work_item_claims_lock = threading.Lock()

    @property
    def _work_item_engine(self) -> object:
        return self._work_item_engine_cache

    def _ingest_work_item_index(self) -> dict[str, dict[str, Any]]:
        return {"job-1": {"status": "succeeded"}}


def _item(status: str = "ready") -> dict[str, Any]:
    return {"status": status, "kind": "ingest_task", "metadata": {}}


def test_clear_and_remove_reject_immutable_audit_deletion() -> None:
    harness = Harness()
    assert harness.remove_task("job-1") is False
    assert harness.clear_tasks("completed") == {
        "status": "error",
        "error": "WorkItem audit records cannot be cleared",
        "cleared": 0,
        "filter": "completed",
        "remaining": 1,
    }
    assert harness.clear_completed_tasks()["status"] == "error"


def test_cancel_delegates_to_native_work_item() -> None:
    harness = Harness()
    with (
        patch.object(wi, "get_work_item", return_value=_item()),
        patch.object(wi, "cancel_work_item", return_value=True) as cancel,
    ):
        result = harness.cancel_task("job-1")
    assert result["status"] == "success"
    cancel.assert_called_once_with(
        harness._work_item_engine,
        "workitem:ingest_task:job-1",
        reason="cancel_task",
    )


def test_prioritize_updates_work_item_only() -> None:
    harness = Harness()
    with (
        patch.object(wi, "get_work_item", return_value=_item()),
        patch.object(wi, "set_work_item_priority", return_value=True) as prioritize,
    ):
        result = harness.prioritize_task("job-1", 0)
    assert result["status"] == "success"
    assert result["prio_bucket"] == 0
    prioritize.assert_called_once_with(
        harness._work_item_engine,
        "workitem:ingest_task:job-1",
        0,
    )
