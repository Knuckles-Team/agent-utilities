"""Wire-First live-path test for ``AdmissionPolicy`` in the worker-claim path
(CONCEPT:AU-ORCH.dispatch.worker-scheduling).

Before this wiring, ``TaskManagerMixin._claim_next_task`` constructed a real
:class:`~agent_utilities.knowledge_graph.core.worker_scheduler.WorkerRegistry`
(``_worker_registry``, used live at ``engine_tasks.py``) and even built the
lane/pending-by-lane inputs :class:`AdmissionPolicy` needs (``_pending_by_lane``)
— but never actually constructed an ``AdmissionPolicy`` or called
``.decide()``/``.admit()`` anywhere outside its own unit tests
(``tests/unit/knowledge_graph/test_worker_scheduler.py``). Every native
``work_item.claim_next`` win was executed unconditionally: the fair-admission
rules (hot spare, heavy-type cap, per-lane min coverage, interactive floor)
were fully implemented and unit-tested dead code on the live claim path.

``_claim_next_task`` now runs a real :class:`AdmissionPolicy` (built once via
the new ``_admission_policy`` cache) against the registry's live state plus a
real ``pending_by_lane`` snapshot before it lets a claimed task proceed; a
denied admission defers the native lease (``_defer_task_for_admission``)
instead of starting the worker. This test drives the REAL
``TaskManagerMixin._claim_next_task`` entry point end to end (not a bare
``AdmissionPolicy(...).decide(...)`` call) and asserts the admission decision
actually gates whether the worker claims/starts the work.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.knowledge_graph.core.worker_scheduler import (
    SchedulerConfig,
    WorkerRegistry,
)


class _AdmissionClaimHarness:
    """Minimal claim-path surface — mirrors ``_ClaimHarness`` in
    ``test_hydration_reserved_worker.py``, plus the real scheduler state
    ``_admission_policy``/``_defer_task_for_admission`` read.
    """

    # Bind the REAL methods under test — this is the live wiring, not a copy.
    _claim_next_task = TaskManagerMixin._claim_next_task
    _admission_policy = TaskManagerMixin._admission_policy
    _defer_task_for_admission = TaskManagerMixin._defer_task_for_admission

    _work_item_engine = object()

    def __init__(
        self,
        job_types: dict[str, str],
        *,
        sched_config: SchedulerConfig,
        registry: WorkerRegistry,
        pending_by_lane: dict[str, int],
    ) -> None:
        self._job_types = job_types
        self._sched_config = sched_config
        self._registry = registry
        self._pending = dict(pending_by_lane)
        self._active_claims: dict[str, dict] = {}

    def _get_host_token(self) -> str:
        return "worker-a"

    def _ingest_task_metadata(self, job_id: str) -> dict[str, str]:
        return {"type": self._job_types[job_id]}

    def _remember_work_item_claim(self, job_id, claim) -> None:
        self._active_claims[job_id] = dict(claim)

    def _active_work_item_claim(self, job_id: str, *, pop: bool = False):
        if pop:
            return self._active_claims.pop(job_id, None)
        return self._active_claims.get(job_id)

    # The real _admission_policy/_pending_by_lane read exactly these two.
    def _worker_registry(self) -> WorkerRegistry:
        return self._registry

    def _pending_by_lane(self) -> dict[str, int]:
        return self._pending


def _patch_claim(monkeypatch, *, job_id: str, work_item_id: str):
    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(
        work_item,
        "claim_next",
        lambda _engine, **_kw: {"work_item_id": work_item_id, "payload_ref": job_id},
    )
    monkeypatch.setattr(work_item, "mark_running", lambda *_a: True)


def test_claim_next_task_admission_denied_defers_claim_live_path(monkeypatch):
    """A candidate whose OWN lane is already covered, while another pending
    lane has zero coverage, is denied admission ("steer to uncovered lane")
    — and the real claim path defers the native lease instead of starting
    the worker."""
    from agent_utilities.orchestration import work_item

    deferred_calls: list[dict] = []

    def fake_defer(_engine, work_item_id, claim, *, next_retry_at, reason_ref=""):
        deferred_calls.append(
            {"work_item_id": work_item_id, "claim": claim, "reason_ref": reason_ref}
        )
        return True

    monkeypatch.setattr(work_item, "defer_work_item", fake_defer)
    _patch_claim(
        monkeypatch,
        job_id="new-connector-sync",
        work_item_id="workitem:ingest_task:new-connector-sync",
    )

    # A worker is already busy covering the "connectors" lane (candidate's own
    # lane) — while "research" has 5 pending items and zero running coverage.
    registry = WorkerRegistry()
    registry.start("existing-worker", "connectors", "connector_sync")
    cfg = SchedulerConfig(worker_count=4, reserved=0, per_lane_min=1)

    harness = _AdmissionClaimHarness(
        {"new-connector-sync": "connector_sync"},
        sched_config=cfg,
        registry=registry,
        pending_by_lane={"connectors": 0, "research": 5},
    )

    result = harness._claim_next_task(worker_id="worker-a", hydration_reserved=False)

    # Denied admission → no claim handed back, the worker never starts it.
    assert result is None
    # The side effect that PROVES the gate ran for real: the native lease was
    # released for retry instead of the task executing.
    assert len(deferred_calls) == 1
    assert deferred_calls[0]["reason_ref"] == "admission_denied"
    assert (
        deferred_calls[0]["work_item_id"] == "workitem:ingest_task:new-connector-sync"
    )
    # The registry still shows only the pre-existing worker — this worker was
    # never recorded as claiming the new task.
    assert registry.running_by_lane() == {"connectors": 1}
    assert registry.busy_count() == 1


def test_claim_next_task_admission_allowed_starts_worker_live_path(monkeypatch):
    """An uncontested candidate (empty pool, only its own lane pending) is
    admitted — the real claim path starts the worker in the live registry."""
    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(
        work_item,
        "defer_work_item",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("defer_work_item must not be called on an admitted claim")
        ),
    )
    _patch_claim(
        monkeypatch,
        job_id="first-connector-sync",
        work_item_id="workitem:ingest_task:first-connector-sync",
    )

    registry = WorkerRegistry()
    cfg = SchedulerConfig(worker_count=4, reserved=0, per_lane_min=1)

    harness = _AdmissionClaimHarness(
        {"first-connector-sync": "connector_sync"},
        sched_config=cfg,
        registry=registry,
        pending_by_lane={"connectors": 1},
    )

    result = harness._claim_next_task(worker_id="worker-a", hydration_reserved=False)

    # _claim_next_task stamps the winning claim's own lease identity
    # (claimed_by/work_item_epoch/work_item_id) onto the returned metadata so
    # a caller/log line can say who holds this task and under which fencing
    # generation — see the "Stamp the winning claim's own lease identity"
    # comment at its call site. _patch_claim's stub claim carries no
    # lease_owner/lease_epoch, so those surface as None here.
    assert result == (
        "first-connector-sync",
        {
            "type": "connector_sync",
            "claimed_by": None,
            "work_item_epoch": None,
            "work_item_id": "workitem:ingest_task:first-connector-sync",
        },
    )
    # The side effect that PROVES admission ran (and allowed) for real: the
    # live registry now shows this worker busy in the connectors lane.
    assert registry.running_by_lane() == {"connectors": 1}
    assert registry.busy_count() == 1


def test_admission_denied_retry_delay_is_jittered_not_fixed_one_second(monkeypatch):
    """U-65/BUG-111 regression: a live pod's admission-denied retry was a
    FIXED ``next_retry_at = now + 1.0`` for every denied worker, synchronizing
    the whole pool into a claim/defer thundering herd (2,278 claims + 344
    defers observed in one 5-minute window). The fenced defer must now offer a
    retry somewhere in a bounded 5-15s jittered window instead."""
    import time as time_mod

    from agent_utilities.orchestration import work_item

    captured: list[float] = []

    def fake_defer(_engine, work_item_id, claim, *, next_retry_at, reason_ref=""):
        captured.append(next_retry_at)
        return True

    monkeypatch.setattr(work_item, "defer_work_item", fake_defer)
    _patch_claim(
        monkeypatch,
        job_id="new-connector-sync",
        work_item_id="workitem:ingest_task:new-connector-sync",
    )

    registry = WorkerRegistry()
    registry.start("existing-worker", "connectors", "connector_sync")
    cfg = SchedulerConfig(worker_count=4, reserved=0, per_lane_min=1)

    harness = _AdmissionClaimHarness(
        {"new-connector-sync": "connector_sync"},
        sched_config=cfg,
        registry=registry,
        pending_by_lane={"connectors": 0, "research": 5},
    )

    before = time_mod.time()
    harness._claim_next_task(worker_id="worker-a", hydration_reserved=False)
    after = time_mod.time()

    assert len(captured) == 1
    delay = captured[0] - before
    # Bounded 5-15s window (with a small tolerance for wall-clock skew around
    # `after`), and NOT the old fixed ~1.0s.
    assert 5.0 <= delay <= 15.0 + (after - before)
    assert delay > 1.5, (
        "admission-denied retry must not collapse back to the old fixed ~1s "
        "delay that synchronized every denied worker onto the same retry tick"
    )
