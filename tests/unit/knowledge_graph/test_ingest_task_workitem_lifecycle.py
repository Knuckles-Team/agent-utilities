"""TaskManager delegates ingestion lifecycle exclusively to native WorkItems."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.orchestration import work_item as wi


class Harness:
    _claim_next_task = TaskManagerMixin._claim_next_task
    _remember_work_item_claim = TaskManagerMixin._remember_work_item_claim
    _active_work_item_claim = TaskManagerMixin._active_work_item_claim
    _update_task_status = TaskManagerMixin._update_task_status
    _fail_or_retry_task = TaskManagerMixin._fail_or_retry_task

    def __init__(self) -> None:
        self.backend = object()
        self._work_item_engine_cache = object()
        self._active_work_item_claims: dict[str, dict[str, Any]] = {}
        self._active_work_item_claims_lock = threading.Lock()

    @property
    def _work_item_engine(self) -> object:
        return self._work_item_engine_cache

    def _get_host_token(self) -> str:
        return "worker-opaque"

    def _ingest_task_metadata(self, job_id: str) -> dict[str, Any]:
        assert job_id == "job-1"
        return {"target": "workspace:repo", "type": "codebase"}

    def _checkpoint_db(self) -> None:
        pass


def _claim() -> dict[str, Any]:
    return {
        "_native": True,
        "work_item_id": "workitem:ingest_task:job-1",
        "payload_ref": "job-1",
        "tenant": "tenant-a",
        "lease_owner": "worker-opaque",
        "lease_epoch": 4,
        "fence_token": 4,
        "fencing_token": "fence-4",
    }


def test_claim_next_uses_native_claim_and_keeps_fence_only_in_memory() -> None:
    harness = Harness()
    with (
        patch.object(wi, "claim_next", return_value=_claim()) as native_claim,
        patch.object(wi, "mark_running", return_value=True),
    ):
        result = harness._claim_next_task()

    assert result == (
        "job-1",
        {"target": "workspace:repo", "type": "codebase"},
    )
    assert harness._active_work_item_claim("job-1") == _claim()
    native_claim.assert_called_once_with(
        harness._work_item_engine,
        queue="ingest_task",
        token="worker-opaque",
        lease_ttl_s=7200.0,
    )


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("completed", "succeeded"), ("failed", "failed"), ("cancelled", "cancelled")],
)
def test_terminal_status_commits_active_native_claim(
    status: str, outcome: str
) -> None:
    harness = Harness()
    harness._remember_work_item_claim("job-1", _claim())
    with patch.object(wi, "commit_result", return_value="committed") as commit:
        harness._update_task_status("job-1", status, {"ignored": "result body"})
    assert harness._active_work_item_claim("job-1") is None
    assert commit.call_args.kwargs["outcome"] == outcome
    assert commit.call_args.args[:3] == (
        harness._work_item_engine,
        "workitem:ingest_task:job-1",
        _claim(),
    )


def test_retry_uses_native_commit_and_drops_local_claim() -> None:
    harness = Harness()
    harness._remember_work_item_claim("job-1", _claim())
    with patch.object(wi, "commit_result", return_value="retry_scheduled") as commit:
        harness._fail_or_retry_task("job-1", "failure")
    assert commit.call_args.kwargs["retryable"] is True
    assert harness._active_work_item_claim("job-1") is None


def test_missing_active_claim_fails_closed() -> None:
    harness = Harness()
    with pytest.raises(wi.WorkItemBackendUnavailable):
        harness._update_task_status("job-1", "completed", {})


def test_portable_target_resolves_from_runtime_workspace(tmp_path: Path) -> None:
    from agent_utilities.knowledge_graph.core import engine_tasks

    target = tmp_path / "repo"
    target.mkdir()
    with patch(
        "agent_utilities.core.workspace.get_agent_workspace", return_value=tmp_path
    ):
        portable = engine_tasks._portable_task_target(str(target))
    assert portable == "workspace:repo"
