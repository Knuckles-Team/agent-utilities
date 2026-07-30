"""TaskManager delegates ingestion lifecycle exclusively to native WorkItems."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _TASK_WORK_ITEM_LEASE_SEC,
    TaskManagerMixin,
    _retryable_partial_materialization,
)
from agent_utilities.orchestration import work_item as wi


class Harness:
    _claim_next_task = TaskManagerMixin._claim_next_task
    _remember_work_item_claim = TaskManagerMixin._remember_work_item_claim
    _active_work_item_claim = TaskManagerMixin._active_work_item_claim
    _update_task_status = TaskManagerMixin._update_task_status
    _fail_or_retry_task = TaskManagerMixin._fail_or_retry_task
    _defer_task_for_materialization = TaskManagerMixin._defer_task_for_materialization

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

    def _require_live_work_item_lease(self, job_id: str, claim: dict[str, Any]) -> None:
        assert job_id == "job-1"
        assert claim == _claim()


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
        lease_ttl_s=_TASK_WORK_ITEM_LEASE_SEC,
    )


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("completed", "succeeded"), ("failed", "failed"), ("cancelled", "cancelled")],
)
def test_terminal_status_commits_active_native_claim(status: str, outcome: str) -> None:
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


def test_retryable_partial_materialization_defers_without_consuming_attempt() -> None:
    harness = Harness()
    harness._remember_work_item_claim("job-1", _claim())
    with patch.object(wi, "defer_work_item", return_value=True) as defer:
        deferred = harness._defer_task_for_materialization(
            "job-1",
            {"code": "PARTIAL_MATERIALIZATION", "retryable": True},
        )

    assert deferred is True
    assert defer.call_count == 1
    assert defer.call_args.kwargs["reason_ref"] == "partial_materialization"
    assert harness._active_work_item_claim("job-1") is None


def test_fenced_materialization_defer_drops_stale_local_claim() -> None:
    harness = Harness()
    harness._remember_work_item_claim("job-1", _claim())
    with patch.object(wi, "defer_work_item", return_value=False):
        deferred = harness._defer_task_for_materialization(
            "job-1",
            {"code": "PARTIAL_MATERIALIZATION", "retryable": True},
        )

    assert deferred is False
    assert harness._active_work_item_claim("job-1") is None


def test_claimed_metadata_read_partial_is_deferred_before_returning() -> None:
    harness = Harness()
    partial = RuntimeError(
        '{"code":"PARTIAL_MATERIALIZATION","phase":"partial","retryable":true}'
    )
    with (
        patch.object(wi, "claim_next", return_value=_claim()),
        patch.object(wi, "mark_running", return_value=True),
        patch.object(harness, "_ingest_task_metadata", side_effect=partial),
        patch.object(wi, "defer_work_item", return_value=True) as defer,
    ):
        result = harness._claim_next_task()

    assert result is None
    assert defer.call_count == 1
    assert harness._active_work_item_claim("job-1") is None


def test_claim_next_task_drops_local_claim_when_materialization_defer_fails() -> None:
    """An unexpected failure releasing the native lease must not strand the
    in-memory claim or escape as an application error.

    Mirrors ``_task_worker_loop``'s in-body materialization handling
    (``defer_error`` branch): the native WorkItem itself self-heals via its
    own lease TTL + the pre-existing expired-lease reaper, so this must
    behave like a clean defer from the caller's perspective — return
    ``None``, never raise, never leave a stale local claim behind.
    """
    harness = Harness()
    partial = RuntimeError(
        '{"code":"PARTIAL_MATERIALIZATION","phase":"partial","retryable":true}'
    )
    with (
        patch.object(wi, "claim_next", return_value=_claim()),
        patch.object(wi, "mark_running", return_value=True),
        patch.object(harness, "_ingest_task_metadata", side_effect=partial),
        patch.object(
            wi, "defer_work_item", side_effect=RuntimeError("engine unavailable")
        ) as defer,
    ):
        result = harness._claim_next_task()

    assert result is None
    assert defer.call_count == 1
    assert harness._active_work_item_claim("job-1") is None


@pytest.mark.parametrize(
    ("error", "recognized"),
    [
        (
            RuntimeError(
                '{"code":"PARTIAL_MATERIALIZATION","phase":"partial",'
                '"retryable":true,"completeness_cursor":{"node_offset":1}}'
            ),
            True,
        ),
        (
            RuntimeError(
                '{"code":"PARTIAL_MATERIALIZATION","phase":"failed","retryable":false}'
            ),
            False,
        ),
        (RuntimeError("prefix PARTIAL_MATERIALIZATION suffix"), False),
        (RuntimeError('{"code":"OTHER","retryable":true}'), False),
    ],
)
def test_partial_materialization_recognition_is_exact(
    error: RuntimeError, recognized: bool
) -> None:
    assert (_retryable_partial_materialization(error) is not None) is recognized


def test_worker_waits_for_retryable_materialization_before_claiming() -> None:
    class WaitingWorker:
        _task_worker_loop = TaskManagerMixin._task_worker_loop

        def __init__(self) -> None:
            self.failed = False

        def _claim_next_task(self, *, worker_id: str | None = None):
            raise RuntimeError(
                '{"code":"PARTIAL_MATERIALIZATION","phase":"partial",'
                '"retryable":true,"completeness_cursor":{"node_offset":16384}}'
            )

        def _fail_or_retry_task(self, *_args: object) -> None:
            self.failed = True

    worker = WaitingWorker()
    with patch(
        "agent_utilities.knowledge_graph.core.engine_tasks.time.sleep",
        side_effect=StopIteration,
    ):
        with pytest.raises(StopIteration):
            worker._task_worker_loop()

    assert worker.failed is False


def test_worker_never_routes_partial_materialization_to_failure_attempt() -> None:
    class Registry:
        def finish(self, _worker_id: str) -> None:
            pass

    class WaitingWorker:
        _task_worker_loop = TaskManagerMixin._task_worker_loop

        def __init__(self) -> None:
            self.claims = 0
            self.failed = False
            self.deferred = False

        def _claim_next_task(self, *, worker_id: str | None = None):
            self.claims += 1
            if self.claims > 1:
                raise KeyboardInterrupt
            return "job-1", {"target": "workspace:repo", "type": "codebase"}

        def _execute_claimed_task(self, *_args: object) -> None:
            raise RuntimeError(
                '{"code":"PARTIAL_MATERIALIZATION","phase":"partial","retryable":true}'
            )

        def _defer_task_for_materialization(
            self, _job_id: str, _materialization: dict[str, Any]
        ) -> bool:
            self.deferred = True
            return False

        def _fail_or_retry_task(self, *_args: object) -> None:
            self.failed = True

        def _worker_registry(self) -> Registry:
            return Registry()

    worker = WaitingWorker()
    with pytest.raises(KeyboardInterrupt):
        worker._task_worker_loop()

    assert worker.deferred is True
    assert worker.failed is False


def test_worker_generic_error_path_is_unchanged() -> None:
    class Registry:
        def finish(self, _worker_id: str) -> None:
            pass

    class FailingWorker:
        _task_worker_loop = TaskManagerMixin._task_worker_loop

        def __init__(self) -> None:
            self.failed_with: tuple[str, str] | None = None

        def _claim_next_task(self, *, worker_id: str | None = None):
            return "job-1", {"target": "workspace:repo", "type": "codebase"}

        def _execute_claimed_task(self, *_args: object) -> None:
            raise ValueError("ordinary failure")

        def _fail_or_retry_task(self, job_id: str, error: str) -> None:
            self.failed_with = (job_id, error)

        def _worker_registry(self) -> Registry:
            return Registry()

    worker = FailingWorker()
    with patch(
        "agent_utilities.knowledge_graph.core.engine_tasks.time.sleep",
        side_effect=KeyboardInterrupt,
    ):
        with pytest.raises(KeyboardInterrupt):
            worker._task_worker_loop()

    assert worker.failed_with == ("job-1", "ordinary failure")


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
