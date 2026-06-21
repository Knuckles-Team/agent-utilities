"""Functional task lanes — fair scheduling + observability (CONCEPT:ORCH-1.75)."""

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
    assert lane_for_task_type("totally_unknown") == DEFAULT_LANE
    assert lane_model_role("ingestion") == "lite"


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

    stub = SimpleNamespace(query_cypher=fake_query)
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

    stub = SimpleNamespace(query_cypher=fake_query)
    TaskManagerMixin._select_pending_task(stub)
    first_lane_a = seen[0]
    assert set(seen) == set(LANE_NAMES)  # every lane was tried
    seen.clear()
    TaskManagerMixin._select_pending_task(stub)
    first_lane_b = seen[0]
    assert first_lane_a != first_lane_b  # cursor advanced → fairness


def test_lane_metrics_reports_per_lane_congestion():
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
    from agent_utilities.knowledge_graph.core.task_lanes import LANE_NAMES

    def fake_query(cypher, params=None):
        # pending lane queries → 4; everything else → 0
        if params and params.get("l") and "pending" in cypher:
            return [{"c": 4}]
        return [{"c": 0}]

    stub = SimpleNamespace(query_cypher=fake_query)
    m = TaskManagerMixin.lane_metrics(stub)
    for lane in LANE_NAMES:
        assert m[lane]["pending"] == 4
        assert "running" in m[lane] and "model_role" in m[lane]
    assert "lane_less" in m
