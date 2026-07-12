"""Functional task lanes — fair scheduling + observability (CONCEPT:AU-ORCH.execution.two-level-fair-rotation)."""

from __future__ import annotations

from types import SimpleNamespace


def test_lane_for_task_type_maps_functional_domains():
    from agent_utilities.knowledge_graph.core.task_lanes import (
        DEFAULT_LANE,
        lane_for_task_type,
        lane_model_role,
    )

    assert lane_for_task_type("codebase") == "ingestion"
    assert lane_for_task_type("document") == "ingestion"
    assert lane_for_task_type("conversation") == "queries"
    assert lane_for_task_type("research_paper_fetch") == "research"
    assert lane_for_task_type("deep_extract") == "extraction"  # ORCH-1.76 own lane
    assert lane_for_task_type("connector_sync") == "connectors"  # ORCH-1.77 own lane
    assert lane_for_task_type("scheduled_job") == "maint"
    # KG-2.153 — OWL card backfill is its OWN throughput lane, not capped maint.
    assert lane_for_task_type("enrichment_backfill") == "enrichment"
    assert lane_for_task_type("totally_unknown") == DEFAULT_LANE
    assert lane_model_role("ingestion") == "lite"
    assert lane_model_role("enrichment") == "lite"


def test_two_pool_partition_and_helpers():
    """CONCEPT:AU-ORCH.dispatch.two-pool — lanes partition into two worker pools:
    acquisition (I/O-bound source intake) and memory-gen (write-lock-bound KG
    materialization). Un-pooled lanes (queries/maint) resolve to None."""
    from agent_utilities.knowledge_graph.core.task_lanes import (
        ACQUISITION_POOL,
        MEMORY_GEN_POOL,
        POOL_NAMES,
        pool_for,
        pool_for_lane,
        pool_for_task_type,
    )

    assert set(POOL_NAMES) == {ACQUISITION_POOL, MEMORY_GEN_POOL}

    # Lane → pool.
    assert pool_for_lane("connectors") == ACQUISITION_POOL
    assert pool_for_lane("ingestion") == MEMORY_GEN_POOL
    assert pool_for_lane("extraction") == MEMORY_GEN_POOL
    assert pool_for_lane("worldview") == MEMORY_GEN_POOL
    # Interactive / best-effort lanes are un-pooled.
    assert pool_for_lane("queries") is None
    assert pool_for_lane("maint") is None

    # Task type → pool (via its lane).
    assert pool_for_task_type("connector_sync") == ACQUISITION_POOL
    assert pool_for_task_type("codebase") == MEMORY_GEN_POOL
    assert pool_for_task_type("feed_ingest") == MEMORY_GEN_POOL

    # content_url rides the ingestion (memory-gen) lane but is a raw FETCH, so the
    # per-type override budgets it as acquisition.
    assert lane_of("content_url") == "ingestion"
    assert pool_for_task_type("content_url") == ACQUISITION_POOL
    assert pool_for("ingestion", "content_url") == ACQUISITION_POOL
    assert pool_for("ingestion", "codebase") == MEMORY_GEN_POOL


def lane_of(task_type: str) -> str:
    from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type

    return lane_for_task_type(task_type)


def test_sweep_all_sources_enqueues_laned_connector_tasks():
    """ORCH-1.77 — the fleet sweep enqueues one connector_sync task per connector (parallel,
    laned) instead of syncing them sequentially inline."""
    from agent_utilities.knowledge_graph.core.source_sync import sweep_all_sources

    calls: list[tuple] = []

    class _Eng:
        def submit_task(self, target_path, provenance, task_type=None, **kw):
            calls.append((target_path, task_type, provenance))
            return f"job-{target_path}"

    res = sweep_all_sources(_Eng(), mode="delta", include_materialize=False)
    assert res["status"] == "enqueued"
    assert res["enqueued"] == len(calls) > 0
    assert all(t == "connector_sync" for (_, t, _) in calls)
    assert all(p == {"sync_mode": "delta"} for (_, _, p) in calls)


def test_select_pending_task_is_type_fair_within_lane():
    """ORCH-1.76 — within a lane, claiming rotates across its task TYPES, so a fast type
    (diff/document) isn't stuck behind a slow one (a big codebase batch) sharing the lane."""
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
    from agent_utilities.knowledge_graph.core.task_lanes import lane_task_types

    tks: list[str] = []

    def fake_query(cypher, params=None):
        if params and "tk" in params:
            tks.append(params["tk"])
        return []

    stub = SimpleNamespace(query_cypher=fake_query, _control_cypher=fake_query)
    ing = set(lane_task_types("ingestion"))
    TaskManagerMixin._select_pending_task(stub)
    assert ing.issubset(set(tks))  # every ingestion type was tried
    first_ing_a = next(t for t in tks if t in ing)
    tks.clear()
    TaskManagerMixin._select_pending_task(stub)
    first_ing_b = next(t for t in tks if t in ing)
    assert first_ing_a != first_ing_b  # type cursor advanced → per-type fairness


def test_select_pending_task_is_lane_fair():
    """Each claim rotates which lane gets first dibs, so no lane head-of-line-blocks another."""
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
    from agent_utilities.knowledge_graph.core.task_lanes import LANE_NAMES

    seen: list[str] = []

    def fake_query(cypher, params=None):
        if params and "lane" in params:
            seen.append(params["lane"])
        return []  # nothing pending → exercises the full rotation + legacy fallthrough

    stub = SimpleNamespace(query_cypher=fake_query, _control_cypher=fake_query)
    TaskManagerMixin._select_pending_task(stub)
    first_lane_a = seen[0]
    assert set(seen) == set(LANE_NAMES)  # every lane was tried
    seen.clear()
    TaskManagerMixin._select_pending_task(stub)
    first_lane_b = seen[0]
    assert first_lane_a != first_lane_b  # cursor advanced → fairness


def test_record_lane_metrics_is_a_safe_noop_without_prometheus():
    """Phase-0 daemon telemetry (CONCEPT:AU-ORCH.execution.two-level-fair-rotation) — publishes per-lane queue-depth
    + in-flight gauges to the existing gateway_metrics registry. Must never raise
    (best-effort), whether or not the optional ``metrics`` extra is installed, and
    must set every LANE_NAMES series even for lanes absent from the input maps."""
    from agent_utilities.knowledge_graph.core.task_lanes import (
        LANE_NAMES,
        record_lane_metrics,
    )

    # Missing lanes default to 0; no exception either way.
    record_lane_metrics({"maint": 3}, {"maint": 1})
    record_lane_metrics({}, {})
    assert True  # reaching here means no exception propagated


def test_record_lane_metrics_calls_gateway_metrics_gauges(monkeypatch):
    """With the metrics registry mocked, confirm every lane is set with the right values."""
    import agent_utilities.observability.gateway_metrics as gm
    from agent_utilities.knowledge_graph.core.task_lanes import (
        LANE_NAMES,
        record_lane_metrics,
    )

    calls: dict[str, dict[str, float]] = {"depth": {}, "in_flight": {}}

    class _FakeGauge:
        def __init__(self, sink: dict[str, float]):
            self._sink = sink
            self._lane = None

        def labels(self, lane: str):
            self._lane = lane
            return self

        def set(self, value: float):
            self._sink[self._lane] = value

    monkeypatch.setattr(gm, "LANE_QUEUE_DEPTH", _FakeGauge(calls["depth"]))
    monkeypatch.setattr(gm, "LANE_IN_FLIGHT", _FakeGauge(calls["in_flight"]))

    record_lane_metrics({"maint": 3, "queries": 1}, {"maint": 2})
    assert calls["depth"]["maint"] == 3
    assert calls["depth"]["queries"] == 1
    assert calls["depth"].get("ingestion", 0) == 0
    assert calls["in_flight"]["maint"] == 2
    assert calls["in_flight"].get("queries", 0) == 0
    assert set(calls["depth"]) == set(LANE_NAMES)


def test_lane_metrics_reports_per_lane_congestion():
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
    from agent_utilities.knowledge_graph.core.task_lanes import LANE_NAMES

    def fake_query(cypher, params=None):
        # pending lane queries → 4; everything else → 0
        if params and params.get("l") and "pending" in cypher:
            return [{"c": 4}]
        return [{"c": 0}]

    stub = SimpleNamespace(query_cypher=fake_query, _control_cypher=fake_query)
    m = TaskManagerMixin.lane_metrics(stub)
    for lane in LANE_NAMES:
        assert m[lane]["pending"] == 4
        assert "running" in m[lane] and "model_role" in m[lane]
    assert "lane_less" in m
