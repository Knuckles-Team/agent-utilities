"""Reserved-capacity worker claiming for priority-1 capability/boot hydration
work (CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness).

Proven root cause: the priority-1 fleet tool-schema boot job
(``task_type="capability_hydration"``) shared the ``connectors`` lane/pool
with however many ordinary per-connector ``connector_sync`` jobs the */20m
fleet sweep had enqueued. Since queue priority cannot preempt work a worker is
ALREADY running, the boot job could sit pending until one of those
already-claimed legacy jobs happened to finish — which, live, took long enough
to blow past the lane's soft timeout with the worker still wedged.

A worker reserved for hydration-priority work (:func:`TaskManagerMixin.
start_task_workers`) ALWAYS tries a hydration-scoped claim first
(:func:`TaskManagerMixin._claim_next_task`), so it is never left waiting on
however many legacy jobs are occupying the rest of the pool. These tests
exercise the pure claim-selection logic (no engine/backend).
"""

from __future__ import annotations

import threading

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _ClaimHarness:
    """Minimal claim-path surface — mirrors ``_ClaimLeaseHarness`` in
    ``test_ingest_workitem_lease_recovery.py``."""

    _claim_next_task = TaskManagerMixin._claim_next_task
    _work_item_engine = object()

    def __init__(self, job_types: dict[str, str]) -> None:
        self._job_types = job_types

    def _get_host_token(self) -> str:
        return "worker-a"

    def _ingest_task_metadata(self, job_id: str) -> dict[str, str]:
        return {"type": self._job_types[job_id]}

    def _remember_work_item_claim(self, _job_id, _claim) -> None:
        pass


def test_hydration_reserved_worker_claims_capability_hydration_over_saturating_legacy_jobs(
    monkeypatch,
):
    """The acquisition pool is saturated with legacy ``connector_sync`` work AND
    the pending priority-1 ``capability_hydration`` boot job is sitting in the
    same lane. A reserved worker's hydration-scoped claim finds the boot job —
    an unfiltered claim (what an ordinary worker would issue) would instead
    have surfaced the legacy job.
    """
    from agent_utilities.orchestration import work_item

    attempts: list[dict] = []

    def claim_next(_engine, **kwargs):
        attempts.append(kwargs)
        if kwargs.get("fairness_group") == "capability_hydration":
            return {
                "work_item_id": "workitem:ingest_task:boot-fleet",
                "payload_ref": "boot-fleet",
            }
        if kwargs.get("fairness_group"):
            return None  # e.g. skill_workflows: nothing pending
        # The plain unfiltered claim — what a NON-reserved worker would issue,
        # and what this reserved worker would ALSO get if it fell through.
        return {
            "work_item_id": "workitem:ingest_task:legacy-connector",
            "payload_ref": "legacy-connector",
        }

    monkeypatch.setattr(work_item, "claim_next", claim_next)
    monkeypatch.setattr(work_item, "mark_running", lambda *_a: True)

    harness = _ClaimHarness(
        {"boot-fleet": "capability_hydration", "legacy-connector": "connector_sync"}
    )

    claimed = harness._claim_next_task(hydration_reserved=True)

    assert claimed == (
        "boot-fleet",
        {
            "type": "capability_hydration",
            "claimed_by": None,
            "work_item_epoch": None,
            "work_item_id": "workitem:ingest_task:boot-fleet",
        },
    )
    # It tried a hydration-scoped claim (never the unfiltered fallback that
    # would have handed it the legacy job) to land the boot job.
    assert all(kw.get("fairness_group") for kw in attempts)


def test_non_reserved_worker_is_unaffected_and_gets_the_unfiltered_claim(monkeypatch):
    """A normal (non-reserved) worker still does the plain unfiltered claim —
    proving the guarantee comes from the reservation, not a change to how
    ordinary claiming behaves."""
    from agent_utilities.orchestration import work_item

    attempts: list[dict] = []

    def claim_next(_engine, **kwargs):
        attempts.append(kwargs)
        return {
            "work_item_id": "workitem:ingest_task:legacy-connector",
            "payload_ref": "legacy-connector",
        }

    monkeypatch.setattr(work_item, "claim_next", claim_next)
    monkeypatch.setattr(work_item, "mark_running", lambda *_a: True)

    harness = _ClaimHarness({"legacy-connector": "connector_sync"})
    claimed = harness._claim_next_task(hydration_reserved=False)

    assert claimed == (
        "legacy-connector",
        {
            "type": "connector_sync",
            "claimed_by": None,
            "work_item_epoch": None,
            "work_item_id": "workitem:ingest_task:legacy-connector",
        },
    )
    # Exactly one attempt, unfiltered — no hydration probing for a plain worker.
    assert len(attempts) == 1
    assert attempts[0]["queue"] == "ingest_task"
    assert attempts[0]["token"] == "worker-a"
    assert "fairness_group" not in attempts[0]
    assert "resource_class" not in attempts[0]


def test_hydration_reserved_worker_falls_back_to_ordinary_work_when_idle(monkeypatch):
    """No hydration work pending → the reserved worker still claims ordinary
    work instead of idling: capacity is never wasted, only ever prioritized
    (dynamic scaling, never starved to zero — mirrors the shared-LLM priority
    gate's same principle, CONCEPT:AU-ORCH.scheduling.also-fold-vllm-scheduler)."""
    from agent_utilities.orchestration import work_item

    def claim_next(_engine, **kwargs):
        if kwargs.get("fairness_group"):
            return None
        return {
            "work_item_id": "workitem:ingest_task:legacy-connector",
            "payload_ref": "legacy-connector",
        }

    monkeypatch.setattr(work_item, "claim_next", claim_next)
    monkeypatch.setattr(work_item, "mark_running", lambda *_a: True)

    harness = _ClaimHarness({"legacy-connector": "connector_sync"})
    claimed = harness._claim_next_task(hydration_reserved=True)

    assert claimed == (
        "legacy-connector",
        {
            "type": "connector_sync",
            "claimed_by": None,
            "work_item_epoch": None,
            "work_item_id": "workitem:ingest_task:legacy-connector",
        },
    )


class _StopLoop(BaseException):
    """Escapes ``_task_worker_loop``'s own ``except Exception`` handler —
    unlike ``StopIteration``/any ``Exception`` subclass, which that handler's
    error-recovery path (a genuine, legitimate behavior: one bad claim must
    not kill the worker thread) would swallow and keep looping past."""


def test_task_worker_loop_alternates_hydration_priority_on_the_degenerate_single_worker_case():
    """D-43: with no second thread to keep pinned to general work,
    ``hydration_reserved`` stays ``False`` at the thread level for a
    single-worker pool, but ``hydration_alternate=True`` makes
    ``_task_worker_loop`` check the hydration lane first on every OTHER poll —
    a bounded floor that never fully starves general work even under a
    sustained hydration-type firehose."""
    from unittest.mock import patch

    seen_hydration_reserved: list[bool] = []

    class AlternatingWorker:
        _task_worker_loop = TaskManagerMixin._task_worker_loop
        # U-65/BUG-111: _task_worker_loop now gates each claim attempt through
        # a process-local nonblocking lock (self._claim_gate) -- a real one is
        # needed here, same as production TaskManagerMixin.__init__ sets.
        _claim_gate = threading.Lock()

        def _claim_next_task(
            self, *, worker_id: str | None = None, hydration_reserved: bool = False
        ):
            seen_hydration_reserved.append(hydration_reserved)
            if len(seen_hydration_reserved) >= 4:
                raise _StopLoop
            return None  # idle each time -> falls to the time.sleep backoff

    worker = AlternatingWorker()
    with (
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks.time.sleep",
            return_value=None,
        ),
        patch("agent_utilities.core.background_throttle.get_throttle") as mock_throttle,
    ):
        mock_throttle.return_value.should_yield_background = False
        with pytest.raises(_StopLoop):
            worker._task_worker_loop(hydration_reserved=False, hydration_alternate=True)

    # True, False, True, False, ... — every other poll is hydration-first.
    assert seen_hydration_reserved == [True, False, True, False]


def test_task_worker_loop_no_alternation_when_disabled():
    """A thread-level-reserved worker (``hydration_reserved=True``) or an
    ordinary one (both ``False``) never alternates — D-43's alternation is
    strictly additive for the degenerate single-worker case."""
    from unittest.mock import patch

    seen_hydration_reserved: list[bool] = []

    class PlainWorker:
        _task_worker_loop = TaskManagerMixin._task_worker_loop
        # U-65/BUG-111: see AlternatingWorker above.
        _claim_gate = threading.Lock()

        def _claim_next_task(
            self, *, worker_id: str | None = None, hydration_reserved: bool = False
        ):
            seen_hydration_reserved.append(hydration_reserved)
            if len(seen_hydration_reserved) >= 3:
                raise _StopLoop
            return None

    worker = PlainWorker()
    with (
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks.time.sleep",
            return_value=None,
        ),
        patch("agent_utilities.core.background_throttle.get_throttle") as mock_throttle,
    ):
        mock_throttle.return_value.should_yield_background = False
        with pytest.raises(_StopLoop):
            worker._task_worker_loop(
                hydration_reserved=False, hydration_alternate=False
            )

    assert seen_hydration_reserved == [False, False, False]


def test_hydration_task_types_land_in_the_connectors_or_ingestion_lane_as_expected():
    """Sanity: the two HYDRATION_TASK_TYPES resolve to the lanes the reserved
    worker's per-type ``resource_class`` scoping relies on."""
    from agent_utilities.core.resource_priority import HYDRATION_TASK_TYPES
    from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type

    assert HYDRATION_TASK_TYPES == frozenset(
        {"skill_workflows", "capability_hydration"}
    )
    assert lane_for_task_type("capability_hydration") == "connectors"
    assert lane_for_task_type("skill_workflows") == "ingestion"
