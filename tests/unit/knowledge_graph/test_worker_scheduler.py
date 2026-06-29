"""Reserved-worker fair admission scheduler (CONCEPT:ORCH-1.81).

The KG worker pool drains one lane-partitioned queue. The admission policy keeps a
hot spare, caps the heavy ``codebase`` type, and guarantees per-lane minimum
coverage — composing with the existing rotation + engine CAS. These tests exercise
the pure policy (no engine/backend) over a live :class:`WorkerRegistry`.
"""

from __future__ import annotations

import os
from unittest import mock

from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type
from agent_utilities.knowledge_graph.core.worker_scheduler import (
    HEAVY_TYPE,
    AdmissionPolicy,
    SchedulerConfig,
    WorkerRegistry,
    scheduler_config_from_env,
)

INGEST_LANE = lane_for_task_type(HEAVY_TYPE)  # "ingestion"
QUERIES_LANE = lane_for_task_type("conversation")  # "queries"
MAINT_LANE = lane_for_task_type("scheduled_job")  # "maint" (best-effort)


def _policy(worker_count=4, reserved=1, per_lane_min=1, codebase_cap=None):
    cfg = SchedulerConfig(
        worker_count=worker_count,
        reserved=reserved,
        per_lane_min=per_lane_min,
        codebase_cap=codebase_cap,
    )
    reg = WorkerRegistry()
    return AdmissionPolicy(cfg, reg), reg


# --- WorkerRegistry --------------------------------------------------------
def test_registry_tracks_busy_and_free():
    reg = WorkerRegistry()
    assert reg.free_count(4) == 4
    reg.start("w0", INGEST_LANE, "codebase")
    reg.start("w1", QUERIES_LANE, "conversation")
    assert reg.busy_count() == 2
    assert reg.free_count(4) == 2
    assert reg.running_by_lane() == {INGEST_LANE: 1, QUERIES_LANE: 1}
    assert reg.running_by_type() == {"codebase": 1, "conversation": 1}
    reg.finish("w0")
    assert reg.free_count(4) == 3
    assert reg.running_by_type() == {"conversation": 1}
    reg.finish("w0")  # idempotent
    assert reg.free_count(4) == 3


# --- Hot spare reservation -------------------------------------------------
def test_reservation_keeps_at_least_one_free():
    """The last free worker is refused a routine claim so a spare remains."""
    policy, reg = _policy(worker_count=4, reserved=1)
    reg.start("w0", INGEST_LANE, "document")
    reg.start("w1", INGEST_LANE, "document")
    # 2 busy, 2 free → claiming leaves 1 free == reserved → allowed.
    assert policy.admit(INGEST_LANE, "document", {INGEST_LANE: 5}) is True
    reg.start("w2", INGEST_LANE, "document")
    # 3 busy, 1 free → claiming would leave 0 free < reserved=1 → refused,
    # because no OTHER pending lane is uncovered (only ingestion pending, covered).
    assert policy.admit(INGEST_LANE, "document", {INGEST_LANE: 5}) is False


def test_reservation_relaxed_to_cover_uncovered_pending_lane():
    """The hot spare is spent to cover an uncovered pending lane rather than starve
    it — but only down to the interactive floor (CONCEPT:KG-2.289), never into the
    reserved interactive worker (that boundary is exercised in
    test_ingest_tail_optimization)."""
    policy, reg = _policy(worker_count=5, reserved=1)
    # 3 ingestion workers busy, 2 free. Covering the uncovered connectors lane
    # leaves 1 free == the interactive floor, so the relaxation still applies.
    reg.start("w0", INGEST_LANE, "codebase")
    reg.start("w1", INGEST_LANE, "codebase")
    reg.start("w2", INGEST_LANE, "document")
    pending = {INGEST_LANE: 5, "connectors": 1}
    # Claiming the connectors lane covers an uncovered pending lane → admitted
    # (degrade toward the floor, never starve), without touching the reserved slot.
    assert policy.admit("connectors", "connector_sync", pending) is True


# --- Coverage steering -----------------------------------------------------
def test_steer_away_from_covered_lane_toward_uncovered():
    """A free worker is steered off an already-covered lane while an uncovered
    pending lane exists, so the rotation can offer the uncovered lane."""
    policy, reg = _policy(worker_count=6, reserved=1)
    reg.start("w0", INGEST_LANE, "codebase")  # ingestion covered
    pending = {INGEST_LANE: 10, "research": 3}  # research pending, uncovered
    # Offered ingestion again → denied (steer to the uncovered research lane).
    assert policy.admit(INGEST_LANE, "document", pending) is False
    # Offered the uncovered research lane → admitted.
    assert policy.admit("research", "research_paper_fetch", pending) is True


# --- Heavy-type (codebase) cap --------------------------------------------
def test_explicit_codebase_cap_respected():
    policy, reg = _policy(worker_count=6, reserved=1, codebase_cap=2)
    reg.start("w0", INGEST_LANE, "codebase")
    reg.start("w1", INGEST_LANE, "codebase")
    # 2 codebase running == cap → further codebase refused even with free workers.
    assert policy.admit(INGEST_LANE, "codebase", {INGEST_LANE: 10}) is False
    # A non-heavy ingestion type is still admissible.
    assert policy.admit(INGEST_LANE, "document", {INGEST_LANE: 10}) is True


def test_derived_codebase_cap_leaves_room_for_other_lanes():
    """Derived cap = workers - reserved - Σ(other pending lane minimums).

    Pins the durable shard width to 1 so the KG-2.279 shard floor is inert and
    this test isolates the pure room-for-other-lanes derivation.
    """
    policy, _reg = _policy(worker_count=6, reserved=1, per_lane_min=1)
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=1,
    ):
        # Two OTHER lanes pending (queries, research) → cap = 6-1-2 = 3.
        pending = {INGEST_LANE: 100, QUERIES_LANE: 2, "research": 2}
        assert policy.codebase_cap(pending) == 3
        # No other pending lane → cap = 6-1-0 = 5.
        assert policy.codebase_cap({INGEST_LANE: 100}) == 5


def test_shard_floor_lifts_collapsed_codebase_cap():
    """CONCEPT:KG-2.279 — the derived cap collapses to ~1 on a busy box (many
    pending lanes), which would idle K-1 of the engine's K durable shard writers.
    The shard floor lifts it back to ``min(K, workers - reserved)`` so K concurrent
    codebase ingests (→ K distinct per-repo graphs → K distinct redb shards) keep
    every writer busy — without starving the spare or another lane's minimum."""
    policy, _reg = _policy(worker_count=6, reserved=1, per_lane_min=1)
    # Five OTHER lanes pending → derived = 6-1-5 = 0 → would floor to 1 (one hot
    # shard writer, the profiled bottleneck).
    busy = {
        INGEST_LANE: 100,
        QUERIES_LANE: 1,
        "research": 1,
        "connectors": 1,
        "worldview": 1,
        "extraction": 1,
    }
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=4,
    ):
        # Floored to min(K=4, workers-reserved=5) = 4 → all 4 shard writers usable.
        assert policy.codebase_cap(busy) == 4
    # The floor never exceeds the pool minus the hot spare.
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=8,
    ):
        assert policy.codebase_cap(busy) == 5  # min(8, 6-1)


def test_shard_floor_respects_explicit_redb_shards_env(monkeypatch):
    """An explicit ``EPISTEMIC_GRAPH_REDB_SHARDS`` sets the floor (no engine call)."""
    from agent_utilities.knowledge_graph.core import worker_scheduler

    worker_scheduler.set_engine_shard_writers(None)  # ensure no cached engine K
    monkeypatch.setenv("EPISTEMIC_GRAPH_REDB_SHARDS", "3")
    assert worker_scheduler.durable_shard_writers() == 3


# ── CONCEPT:KG-2.281 — resolve the engine's REAL K (split-storage) ────────────


class _FakeResharding:
    def __init__(self, k):
        self._k = k

    def rebalance_plan(self):
        # The engine reports one entry per shard ("all K shards represented"),
        # so len(shards) == K — exactly what the real engine returned live.
        return {"moves": [], "shards": [{"shard": i} for i in range(self._k)]}


class _FakeClient:
    def __init__(self, k):
        self.resharding = _FakeResharding(k)


class _FakeEngine:
    """Mimics a GraphComputeEngine handle (carries ``_client``)."""

    def __init__(self, k):
        self._client = _FakeClient(k)


class _FakeBackend:
    """Mimics a backend that wraps the engine on ``._graph`` (the daemon path)."""

    def __init__(self, k):
        self._graph = _FakeEngine(k)


def test_resolve_engine_shard_writers_uses_engine_K(monkeypatch):
    """Split-storage: the host has 16 cpus (→ cpu estimate 8) but the REMOTE engine
    only has K=4 writers. The floor must reflect the engine's actual K (CONCEPT:KG-2.281)."""
    from agent_utilities.knowledge_graph.core import worker_scheduler

    worker_scheduler.set_engine_shard_writers(None)
    monkeypatch.delenv("EPISTEMIC_GRAPH_REDB_SHARDS", raising=False)
    # Pretend this scheduling host has many cpus → the old cpu estimate over-counts.
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert worker_scheduler.durable_shard_writers() == 8  # the WRONG host-cpu value

    # Now resolve from the (remote) engine — K=4 — and it wins + is cached.
    k = worker_scheduler.resolve_engine_shard_writers(_FakeBackend(4))
    assert k == 4
    assert worker_scheduler.durable_shard_writers() == 4
    worker_scheduler.set_engine_shard_writers(None)


def test_resolve_engine_shard_writers_accepts_raw_engine_handle():
    from agent_utilities.knowledge_graph.core import worker_scheduler

    worker_scheduler.set_engine_shard_writers(None)
    assert worker_scheduler.resolve_engine_shard_writers(_FakeEngine(6)) == 6
    worker_scheduler.set_engine_shard_writers(None)


def test_resolve_engine_shard_writers_none_when_unavailable():
    """A handle with no resharding surface (or an erroring engine) yields None so
    the caller degrades to the cpu/env estimate — never crashes scheduling."""
    from agent_utilities.knowledge_graph.core import worker_scheduler

    worker_scheduler.set_engine_shard_writers(None)
    assert worker_scheduler.resolve_engine_shard_writers(object()) is None
    assert worker_scheduler.resolve_engine_shard_writers(None) is None

    class _Boom:
        class resharding:  # noqa: N801
            @staticmethod
            def rebalance_plan():
                raise RuntimeError("non-redb build")

    assert worker_scheduler.resolve_engine_shard_writers(_Boom()) is None
    worker_scheduler.set_engine_shard_writers(None)


def test_derived_cap_floors_at_one():
    policy, _reg = _policy(worker_count=2, reserved=1, per_lane_min=1)
    # 2-1-(many) would go negative → floored to 1 so codebase still progresses.
    pending = {INGEST_LANE: 5, QUERIES_LANE: 1, "research": 1, "connectors": 1}
    assert policy.codebase_cap(pending) == 1


# --- THE core regression: content_url admitted while codebase saturated -----
def test_content_url_admitted_while_codebase_saturates_pool():
    """The headline regression: long codebase jobs occupy workers, but a freshly
    enqueued content_url is ADMITTED immediately (reservation + min-coverage),
    never queued behind the heavy jobs."""
    # 4 workers, derived cap. 3 codebase jobs are running (heavy, long).
    policy, reg = _policy(worker_count=4, reserved=1, per_lane_min=1)
    reg.start("w0", INGEST_LANE, "codebase")
    reg.start("w1", INGEST_LANE, "codebase")
    reg.start("w2", INGEST_LANE, "codebase")
    # 1 worker free. content_url shares the ingestion lane, which IS covered, but
    # a MORE bursty interactive query lane just got work too. First prove a new
    # codebase task is BLOCKED (cap/steer), then a content_url is admitted.
    pending = {INGEST_LANE: 50}  # all ingestion (codebase backlog + 1 content_url)
    # Another codebase would exceed the derived cap (4-1-0 = 3) → blocked.
    assert policy.admit(INGEST_LANE, "codebase", pending) is False
    # The content_url (non-heavy) in the same covered lane: with reserved=1 and 1
    # free, a routine claim is held as spare — UNLESS it's covering an uncovered
    # lane. content_url shares the (covered) ingestion lane, so the spare holds.
    # That is correct: the spare exists precisely to grab the NEXT bursty task.
    assert policy.admit(INGEST_LANE, "content_url", pending) is False
    # Now a worker frees (a codebase job ends): the content_url is admitted.
    reg.finish("w2")
    assert policy.admit(INGEST_LANE, "content_url", pending) is True


def test_content_url_in_own_lane_jumps_saturated_codebase():
    """A dedicated bursty channel (its OWN lane, uncovered + pending) jumps ahead of
    saturated codebase — admitted from spare capacity above the interactive floor
    (CONCEPT:KG-2.289), rather than waiting behind heavy jobs."""
    policy, reg = _policy(worker_count=5, reserved=1, per_lane_min=1)
    for i in range(3):
        reg.start(f"w{i}", INGEST_LANE, "codebase")
    # connectors lane stands in for a dedicated bursty channel with pending work
    # and zero coverage; 2 workers free (one above the interactive floor).
    pending = {INGEST_LANE: 50, "connectors": 1}
    assert reg.free_count(5) == 2
    # Uncovered pending connectors lane → admitted from the non-reserved spare.
    assert policy.admit("connectors", "connector_sync", pending) is True
    # But a fresh codebase task is still capped/steered away.
    assert policy.admit(INGEST_LANE, "codebase", pending) is False


def test_never_pull_last_worker_off_sole_covered_pending_lane():
    """If every busy worker is the sole cover of a pending lane, zero-spare is
    allowed: the policy admits work rather than starve a channel."""
    policy, reg = _policy(worker_count=2, reserved=1, per_lane_min=1)
    reg.start("w0", INGEST_LANE, "codebase")  # 1 busy, 1 free
    # ingestion covered; queries lane pending + uncovered.
    pending = {INGEST_LANE: 5, QUERIES_LANE: 3}
    # The single free worker covers the uncovered queries lane (spends spare).
    assert policy.admit(QUERIES_LANE, "conversation", pending) is True


# --- Best-effort lane cap (CONCEPT:ORCH-1.82) ------------------------------
def test_maint_lane_capped_at_floor_never_expands():
    """A best-effort lane (maint) gets its floor coverage but is refused beyond it,
    so a periodic-tick backlog can't expand into the many free workers."""
    policy, reg = _policy(worker_count=8, reserved=1, per_lane_min=1)
    pending = {MAINT_LANE: 1000}  # huge maint backlog, nothing else pending
    # First maint claim is admitted (covers the lane at its floor).
    assert policy.admit(MAINT_LANE, "scheduled_job", pending) is True
    reg.start("w0", MAINT_LANE, "scheduled_job")
    # 1 running == floor(1): further maint refused though 7 workers are free —
    # the CAP blocks it, not the spare.
    assert policy.admit(MAINT_LANE, "scheduled_job", pending) is False


def test_maint_cap_tracks_per_lane_min():
    """The floor/cap follows ``per_lane_min`` — raising it lets maint use more."""
    policy, reg = _policy(worker_count=8, reserved=1, per_lane_min=2)
    pending = {MAINT_LANE: 1000}
    reg.start("w0", MAINT_LANE, "scheduled_job")
    assert policy.admit(MAINT_LANE, "scheduled_job", pending) is True  # 1 < floor 2
    reg.start("w1", MAINT_LANE, "scheduled_job")
    assert policy.admit(MAINT_LANE, "scheduled_job", pending) is False  # 2 == floor 2


def test_maint_backlog_does_not_starve_throughput_lanes():
    """The headline: with a massive maint backlog pinned at its floor, the
    throughput lanes still get the free workers."""
    policy, reg = _policy(worker_count=8, reserved=1, per_lane_min=1)
    reg.start("w0", MAINT_LANE, "scheduled_job")  # maint at its cap (1)
    pending = {MAINT_LANE: 1000, INGEST_LANE: 100, "worldview": 12}
    assert policy.admit(MAINT_LANE, "scheduled_job", pending) is False  # capped
    assert policy.admit(INGEST_LANE, "document", pending) is True
    assert policy.admit("worldview", "feed_ingest", pending) is True


# --- Dedicated enrichment lane (CONCEPT:KG-2.153) --------------------------
ENRICH_LANE = lane_for_task_type("enrichment_backfill")  # "enrichment"


def test_enrichment_lane_is_its_own_non_besteffort_lane():
    """OWL card backfill lands in the dedicated ``enrichment`` lane, not maint —
    the root-cause fix for ~0% card coverage."""
    from agent_utilities.knowledge_graph.core.task_lanes import (
        BEST_EFFORT_LANES,
        LANE_NAMES,
        lane_model_role,
    )

    assert ENRICH_LANE == "enrichment"
    assert ENRICH_LANE != MAINT_LANE
    assert ENRICH_LANE in LANE_NAMES
    # The whole point: it is NOT best-effort, so it is not floor-capped.
    assert ENRICH_LANE not in BEST_EFFORT_LANES
    assert lane_model_role(ENRICH_LANE) == "lite"


def test_enrichment_lane_expands_into_spare_workers_unlike_maint():
    """The throughput contrast: a huge enrichment backlog DRAINS across many
    workers (bounded only by the hot spare), whereas the same-size maint backlog
    is pinned at its floor of 1. This is why card coverage now climbs at scale."""
    policy, reg = _policy(worker_count=8, reserved=1, per_lane_min=1)
    pending = {ENRICH_LANE: 1000}  # huge backlog, nothing else pending
    admitted = 0
    for i in range(8):
        if policy.admit(ENRICH_LANE, "enrichment_backfill", pending):
            reg.start(f"w{i}", ENRICH_LANE, "enrichment_backfill")
            admitted += 1
        else:
            break
    # Drains into all-but-the-reserved-spare (7), not capped at the maint floor (1).
    assert admitted == 8 - 1
    # Contrast: maint with the identical backlog stops at its floor of 1.
    _policy_m, reg_m = _policy(worker_count=8, reserved=1, per_lane_min=1)
    assert _policy_m.admit(MAINT_LANE, "scheduled_job", {MAINT_LANE: 1000}) is True
    reg_m.start("m0", MAINT_LANE, "scheduled_job")
    assert _policy_m.admit(MAINT_LANE, "scheduled_job", {MAINT_LANE: 1000}) is False


# --- env config ------------------------------------------------------------
def test_scheduler_config_from_env_defaults():
    with mock.patch.dict(os.environ, {}, clear=False):
        for k in (
            "KG_SCHED_RESERVED",
            "KG_SCHED_PER_LANE_MIN",
            "KG_SCHED_CODEBASE_CAP",
        ):
            os.environ.pop(k, None)
        cfg = scheduler_config_from_env(8)
    assert cfg.worker_count == 8
    assert cfg.reserved == 1
    assert cfg.per_lane_min == 1
    assert cfg.codebase_cap is None


def test_scheduler_config_from_env_overrides_and_clamps():
    with mock.patch.dict(
        os.environ,
        {
            "KG_SCHED_RESERVED": "2",
            "KG_SCHED_PER_LANE_MIN": "1",
            "KG_SCHED_CODEBASE_CAP": "100",  # clamped to worker_count
        },
    ):
        cfg = scheduler_config_from_env(6)
    assert cfg.reserved == 2
    assert cfg.codebase_cap == 6  # clamped to worker_count
