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
    assert lane_for_task_type("scheduled_job") == "maint"
    assert lane_for_task_type("totally_unknown") == DEFAULT_LANE
    assert lane_model_role("ingestion") == "lite"


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
