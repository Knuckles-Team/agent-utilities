"""TaskManager's short, renewable ingestion WorkItem lease contract."""

from __future__ import annotations

import threading
import time

import pytest

from agent_utilities.knowledge_graph.core import engine_tasks
from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _LeaseHarness:
    """Minimal TaskManager surface for the lease heartbeat helpers."""

    _active_work_item_claim = TaskManagerMixin._active_work_item_claim
    _start_work_item_lease_heartbeat = TaskManagerMixin._start_work_item_lease_heartbeat
    _stop_work_item_lease_heartbeat = TaskManagerMixin._stop_work_item_lease_heartbeat
    _require_live_work_item_lease = TaskManagerMixin._require_live_work_item_lease

    def __init__(self) -> None:
        self._active_work_item_claims = {
            "job-1": {
                "work_item_id": "workitem:ingest_task:job-1",
                "tenant": "tenant-a",
                "lease_owner": "worker-a",
                "lease_epoch": 1,
                "fencing_token": "fence-1",
            }
        }
        self._active_work_item_claims_lock = threading.Lock()
        self._work_item_lease_heartbeats = {}
        self._work_item_lease_heartbeats_lock = threading.Lock()
        self._work_item_engine = object()

    def _background_session_for_spawn(self):
        return object()


class _ClaimLeaseHarness:
    """Minimal claim path proving the crash-recovery lease is bounded."""

    _claim_next_task = TaskManagerMixin._claim_next_task
    _work_item_engine = object()

    def _get_host_token(self):
        return "worker-a"

    def _ingest_task_metadata(self, _job_id):
        return {"type": "skill_workflows"}

    def _remember_work_item_claim(self, _job_id, _claim):
        pass


def _thread(_session, target, *, name: str, args=()):
    return threading.Thread(target=target, args=args, daemon=True, name=name)


def test_active_ingestion_task_renews_short_fenced_lease(monkeypatch):
    harness = _LeaseHarness()
    renewed = threading.Event()
    calls: list[dict] = []

    def heartbeat(_engine, item_id, claim, *, lease_ttl_s):
        calls.append({"item_id": item_id, "claim": claim, "lease_ttl_s": lease_ttl_s})
        renewed.set()
        return True

    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(engine_tasks, "_authorized_background_thread", _thread)
    monkeypatch.setattr(engine_tasks, "_TASK_WORK_ITEM_HEARTBEAT_SEC", 0.01)
    monkeypatch.setattr(work_item, "heartbeat", heartbeat)

    harness._start_work_item_lease_heartbeat("job-1")
    assert renewed.wait(timeout=1.0)
    harness._stop_work_item_lease_heartbeat("job-1")
    calls_after_stop = len(calls)
    time.sleep(0.03)

    assert calls[0]["item_id"] == "workitem:ingest_task:job-1"
    assert calls[0]["lease_ttl_s"] == engine_tasks._TASK_WORK_ITEM_LEASE_SEC == 60.0
    assert len(calls) == calls_after_stop


def test_ingestion_claim_uses_short_crash_recovery_lease(monkeypatch):
    """Without subsequent heartbeats, native recovery starts after 60 seconds."""
    observed: dict[str, float] = {}

    def claim_next(_engine, **kwargs):
        observed["lease_ttl_s"] = kwargs["lease_ttl_s"]
        return {"work_item_id": "workitem:ingest_task:job-1", "payload_ref": "job-1"}

    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(work_item, "claim_next", claim_next)
    monkeypatch.setattr(work_item, "mark_running", lambda *_args: True)

    assert _ClaimLeaseHarness()._claim_next_task() == (
        "job-1",
        {"type": "skill_workflows"},
    )
    assert observed["lease_ttl_s"] == engine_tasks._TASK_WORK_ITEM_LEASE_SEC == 60.0


def test_lost_or_stopped_heartbeat_fences_terminal_commit(monkeypatch):
    harness = _LeaseHarness()
    lost = threading.Event()
    called = threading.Event()

    def heartbeat(*_args, **_kwargs):
        called.set()
        return False

    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(engine_tasks, "_authorized_background_thread", _thread)
    monkeypatch.setattr(engine_tasks, "_TASK_WORK_ITEM_HEARTBEAT_SEC", 0.01)
    monkeypatch.setattr(work_item, "heartbeat", heartbeat)

    harness._start_work_item_lease_heartbeat("job-1")
    assert called.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with harness._work_item_lease_heartbeats_lock:
            heartbeat_state = harness._work_item_lease_heartbeats["job-1"]
        if heartbeat_state[1].is_set():
            lost.set()
            break
        time.sleep(0.01)
    assert lost.is_set()
    with pytest.raises(
        work_item.WorkItemBackendUnavailable, match="lost its WorkItem lease"
    ):
        harness._require_live_work_item_lease(
            "job-1", harness._active_work_item_claim("job-1")
        )
    harness._stop_work_item_lease_heartbeat("job-1")
