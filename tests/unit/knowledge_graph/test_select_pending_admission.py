"""Admission gate composes with the lane/type rotation in the claim path
(CONCEPT:AU-ORCH.dispatch.worker-scheduling).

These bind the REAL ``_select_pending_task`` over a fake ``query_cypher`` that
serves controlled pending rows, plus the REAL ``WorkerRegistry`` + an
``AdmissionPolicy``-derived ``admit`` predicate. They prove: (1) a denied (lane,
type) is skipped by the rotation; (2) the spare-holding case returns None instead
of grabbing a lane-less legacy task; (3) the headline regression — a pending
content_url in its own uncovered lane is selected even while codebase saturates
the ingestion lane.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type
from agent_utilities.knowledge_graph.core.worker_scheduler import (
    AdmissionPolicy,
    SchedulerConfig,
    WorkerRegistry,
)

INGEST_LANE = lane_for_task_type("codebase")
CONN_LANE = lane_for_task_type("connector_sync")


class _SelectHarness:
    """Serves pending rows keyed by (tkind, prio_bucket) via a fake query_cypher."""

    def __init__(self, rows_by_type: dict[str, str]):
        # rows_by_type: task_type -> job_id available at bucket 2 (normal).
        self._rows = rows_by_type

    def query_cypher(self, query: str, params: dict | None = None):
        params = params or {}
        # Only the typed bucket-2 query returns rows in these tests; everything
        # else (lane-stamped, lane-less, legacy) returns empty so we isolate the
        # typed rotation + admission interaction.
        if "tkind" in query and params.get("b") == 2:
            tk = params.get("tk")
            if tk in self._rows:
                return [{"id": self._rows[tk], "meta": None}]
        return []

    # :Task selection now reads the isolated control plane via _control_cypher
    # (CONCEPT:AU-KG.backend.schedule-on-control-graph); in these tests the control plane is the same fake query
    # source, so delegate to query_cypher.
    def _control_cypher(self, query: str, params: dict | None = None):
        return self.query_cypher(query, params)

    _select_pending_task = TaskManagerMixin._select_pending_task


def _admit_fn(cfg: SchedulerConfig, reg: WorkerRegistry, pending: dict[str, int]):
    policy = AdmissionPolicy(cfg, reg)

    def _admit(lane: str, task_type: str) -> bool:
        return policy.admit(lane, task_type, pending)

    return _admit


def test_denied_type_is_skipped_admitted_type_is_selected():
    """codebase is capped (denied); a content_url in the same lane is admitted."""
    h = _SelectHarness({"codebase": "cb-1", "content_url": "url-1"})
    cfg = SchedulerConfig(worker_count=4, reserved=1, per_lane_min=1, codebase_cap=1)
    reg = WorkerRegistry()
    reg.start("w0", INGEST_LANE, "codebase")  # 1 codebase running == cap
    pending = {INGEST_LANE: 10}
    admit = _admit_fn(cfg, reg, pending)

    # Rotation may offer codebase first, but admission denies it (cap); the
    # content_url candidate IS admitted and selected.
    row = h._select_pending_task(admit=admit)
    assert row is not None
    assert row["id"] == "url-1"


def test_spare_held_returns_none_not_legacy_task():
    """When admission denies everything to hold the spare, no legacy task is
    grabbed — the selector returns None so the worker stays free."""

    class _LegacyHarness(_SelectHarness):
        def query_cypher(self, query, params=None):
            params = params or {}
            # A typed codebase row exists (would be claimed without admission)…
            if (
                "tkind" in query
                and params.get("b") == 2
                and params.get("tk") == "codebase"
            ):
                return [{"id": "cb-legacy", "meta": None}]
            # …AND a lane-less legacy pending row exists.
            if "tkind" not in query and "lane" not in query and "priority" not in query:
                return [{"id": "legacy-1", "meta": None}]
            return []

    h = _LegacyHarness({})
    cfg = SchedulerConfig(worker_count=4, reserved=1, per_lane_min=1, codebase_cap=1)
    reg = WorkerRegistry()
    reg.start("w0", INGEST_LANE, "codebase")  # cap reached → codebase denied
    pending = {INGEST_LANE: 10}
    admit = _admit_fn(cfg, reg, pending)

    # codebase denied (cap) and it's the only typed work → spare held → the
    # lane-less legacy sweep must NOT fire. Returns None.
    assert h._select_pending_task(admit=admit) is None


def test_content_url_own_lane_selected_while_codebase_saturated():
    """The headline regression at the claim-selection level: codebase saturates
    the ingestion lane, but a pending connector_sync (its own uncovered lane) is
    selected immediately."""
    h = _SelectHarness({"codebase": "cb-1", "connector_sync": "conn-1"})
    cfg = SchedulerConfig(worker_count=4, reserved=1, per_lane_min=1)
    reg = WorkerRegistry()
    for i in range(3):
        reg.start(f"w{i}", INGEST_LANE, "codebase")  # ingestion saturated, 1 free
    pending = {INGEST_LANE: 50, CONN_LANE: 1}
    admit = _admit_fn(cfg, reg, pending)

    row = h._select_pending_task(admit=admit)
    assert row is not None
    assert row["id"] == "conn-1"  # the bursty connector jumped the codebase backlog


def test_no_admit_preserves_legacy_rotation():
    """admit=None is the pre-ORCH-1.81 behaviour: typed rotation selects normally."""
    h = _SelectHarness({"codebase": "cb-1"})
    row = h._select_pending_task(admit=None)
    assert row is not None and row["id"] == "cb-1"
