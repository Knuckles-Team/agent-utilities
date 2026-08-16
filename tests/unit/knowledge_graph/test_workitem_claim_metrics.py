"""U-66/U-92/BUG-171: WorkItem claim/poll/admission-deferral metrics.

Prior incident waves (U-70/U-73/U-90) had to reconstruct claim/defer/poll
rates from ad hoc log-grepping over a fixed observation window — none of it
was a queryable Prometheus series. ``TaskManagerMixin._claim_next_task``
now records every claim-poll outcome (claimed vs. empty) and every
reserved-worker AdmissionPolicy deferral onto the existing no-op-safe
gateway metrics registry (``observability.gateway_metrics``).

Reuses the exact ``_AdmissionClaimHarness``/``_patch_claim`` fixtures
``test_admission_policy_live_path.py`` already built to drive the REAL
``TaskManagerMixin._claim_next_task`` entry point (not a bare unit test of
the metric-recording helpers in isolation) — this proves the metrics fire
on the live claim path, not merely that the helper functions work.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.worker_scheduler import (
    SchedulerConfig,
    WorkerRegistry,
)
from agent_utilities.observability import gateway_metrics as gm
from tests.unit.knowledge_graph.test_admission_policy_live_path import (
    _AdmissionClaimHarness,
    _patch_claim,
)


def _claims_value(outcome: str) -> float:
    return gm.WORKITEM_CLAIMS.labels(queue="ingest_task", outcome=outcome)._value.get()


def _deferrals_value(task_type: str) -> float:
    return gm.WORKITEM_ADMISSION_DEFERRALS.labels(task_type=task_type)._value.get()


def test_empty_poll_increments_empty_claim_counter(monkeypatch):
    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(work_item, "claim_next", lambda _engine, **_kw: None)

    harness = _AdmissionClaimHarness(
        {},
        sched_config=SchedulerConfig(worker_count=4, reserved=0, per_lane_min=1),
        registry=WorkerRegistry(),
        pending_by_lane={},
    )

    before = _claims_value("empty")
    result = harness._claim_next_task(worker_id="worker-a", hydration_reserved=False)
    after = _claims_value("empty")

    assert result is None
    assert after == before + 1


def test_admitted_claim_increments_claimed_counter(monkeypatch):
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
        job_id="metrics-connector-sync",
        work_item_id="workitem:ingest_task:metrics-connector-sync",
    )

    harness = _AdmissionClaimHarness(
        {"metrics-connector-sync": "connector_sync"},
        sched_config=SchedulerConfig(worker_count=4, reserved=0, per_lane_min=1),
        registry=WorkerRegistry(),
        pending_by_lane={"connectors": 1},
    )

    before_claimed = _claims_value("claimed")
    before_empty = _claims_value("empty")
    result = harness._claim_next_task(worker_id="worker-a", hydration_reserved=False)

    assert result is not None
    assert _claims_value("claimed") == before_claimed + 1
    # An admitted claim is never ALSO counted as an empty poll.
    assert _claims_value("empty") == before_empty


def test_admission_denied_increments_deferral_counter_by_task_type(monkeypatch):
    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(work_item, "defer_work_item", lambda *_a, **_k: True)
    _patch_claim(
        monkeypatch,
        job_id="metrics-new-connector-sync",
        work_item_id="workitem:ingest_task:metrics-new-connector-sync",
    )

    registry = WorkerRegistry()
    registry.start("existing-worker", "connectors", "connector_sync")
    harness = _AdmissionClaimHarness(
        {"metrics-new-connector-sync": "connector_sync"},
        sched_config=SchedulerConfig(worker_count=4, reserved=0, per_lane_min=1),
        registry=registry,
        pending_by_lane={"connectors": 0, "research": 5},
    )

    before = _deferrals_value("connector_sync")
    result = harness._claim_next_task(worker_id="worker-a", hydration_reserved=False)
    after = _deferrals_value("connector_sync")

    # Denied admission → no claim handed back (same live-path proof as
    # test_admission_policy_live_path.py), AND now a recorded deferral.
    assert result is None
    assert after == before + 1
