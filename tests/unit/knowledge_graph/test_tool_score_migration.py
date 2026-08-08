#!/usr/bin/python
from __future__ import annotations

"""Regression tests for D-CDX-53's write-side fix: resyncing persisted
``Tool.relevance_score`` values onto one canonical domain
(agent_utilities/knowledge_graph/core/tool_score_migration.py).
"""

import pytest

from agent_utilities.knowledge_graph.core.tool_score_migration import (
    plan_tool_relevance_resync,
    resync_tool_relevance_scores,
)


class _FakeEngine:
    """Minimal stand-in for the real Cypher engine: serves a fixed row set
    for the initial ``MATCH`` and records every ``SET`` write it receives."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.writes: list[dict] = []
        self.store: dict[str, object] = {r["id"]: r["relevance_score"] for r in rows}

    def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        if query.strip().startswith("MATCH (t:Tool) RETURN"):
            # Project current values from ``store`` (not the original fixed
            # row list) so a later call in the same test observes any prior
            # writes — matching how a real engine would re-read live state.
            return [
                {"id": node_id, "relevance_score": self.store[node_id]}
                for node_id in self.store
            ]
        if "SET t.relevance_score" in query:
            assert params is not None
            self.writes.append(dict(params))
            self.store[params["id"]] = params["new"]
            return []
        raise AssertionError(f"unexpected query: {query}")


# ---------------------------------------------------------------------------
# plan_tool_relevance_resync (pure)
# ---------------------------------------------------------------------------


def test_already_canonical_rows_are_left_alone() -> None:
    plan = plan_tool_relevance_resync(
        [{"id": "t1", "relevance_score": 80}, {"id": "t2", "relevance_score": 0}]
    )
    assert {r["id"] for r in plan["already_canonical"]} == {"t1", "t2"}
    assert plan["to_migrate"] == []
    assert plan["quarantined"] == []


def test_legacy_float_rows_are_planned_for_migration() -> None:
    plan = plan_tool_relevance_resync(
        [{"id": "t1", "relevance_score": 0.9}, {"id": "t2", "relevance_score": 0.5}]
    )
    assert plan["already_canonical"] == []
    by_id = {e["id"]: e for e in plan["to_migrate"]}
    assert by_id["t1"] == {"id": "t1", "old": 0.9, "new": 90}
    assert by_id["t2"] == {"id": "t2", "old": 0.5, "new": 50}
    assert plan["quarantined"] == []


def test_corrupt_values_are_quarantined_not_coerced() -> None:
    plan = plan_tool_relevance_resync(
        [
            {"id": "t_neg", "relevance_score": -5},
            {"id": "t_over", "relevance_score": 150},
            {"id": "t_frac", "relevance_score": 1.9},
            {"id": "t_bool", "relevance_score": True},
            {"id": "t_str", "relevance_score": "50"},
            {"id": "t_missing", "relevance_score": None},
        ]
    )
    assert plan["already_canonical"] == []
    assert plan["to_migrate"] == []
    assert {q["id"] for q in plan["quarantined"]} == {
        "t_neg",
        "t_over",
        "t_frac",
        "t_bool",
        "t_str",
        "t_missing",
    }


def test_mixed_rows_partition_correctly() -> None:
    plan = plan_tool_relevance_resync(
        [
            {"id": "canon", "relevance_score": 42},
            {"id": "legacy", "relevance_score": 0.42},
            {"id": "corrupt", "relevance_score": -1},
        ]
    )
    assert [r["id"] for r in plan["already_canonical"]] == ["canon"]
    assert [r["id"] for r in plan["to_migrate"]] == ["legacy"]
    assert [r["id"] for r in plan["quarantined"]] == ["corrupt"]


# ---------------------------------------------------------------------------
# resync_tool_relevance_scores (engine-facing driver)
# ---------------------------------------------------------------------------


def test_dry_run_by_default_writes_nothing() -> None:
    engine = _FakeEngine([{"id": "t1", "relevance_score": 0.9}])
    report = resync_tool_relevance_scores(engine)
    assert report["executed"] is False
    assert report["to_migrate"] == 1
    assert report["migrated"] == 0
    assert engine.writes == []
    # the stored value on the fake engine is untouched
    assert engine.store["t1"] == 0.9


def test_execute_true_writes_canonical_values() -> None:
    engine = _FakeEngine(
        [
            {"id": "t1", "relevance_score": 0.9},
            {"id": "t2", "relevance_score": 80},  # already canonical, no write
        ]
    )
    report = resync_tool_relevance_scores(engine, execute=True)
    assert report["migrated"] == 1
    assert report["already_canonical"] == 1
    assert len(engine.writes) == 1
    assert engine.writes[0] == {"id": "t1", "new": 90}
    assert engine.store["t1"] == 90
    assert engine.store["t2"] == 80


def test_resync_is_idempotent() -> None:
    """Running the resync twice must migrate once, then find nothing left to
    migrate on the second pass — proving the fix converges rather than
    oscillating or re-writing on every run."""
    engine = _FakeEngine([{"id": "t1", "relevance_score": 0.9}])
    first = resync_tool_relevance_scores(engine, execute=True)
    assert first["migrated"] == 1

    second = resync_tool_relevance_scores(engine, execute=True)
    assert second["to_migrate"] == 0
    assert second["migrated"] == 0
    assert second["already_canonical"] == 1


def test_quarantined_rows_reported_and_never_written() -> None:
    engine = _FakeEngine([{"id": "t_bad", "relevance_score": -5}])
    report = resync_tool_relevance_scores(engine, execute=True)
    assert report["quarantined"] == [{"id": "t_bad", "value": -5}]
    assert report["migrated"] == 0
    assert engine.writes == []


def test_one_write_failure_does_not_abort_the_whole_run() -> None:
    class _FlakyEngine(_FakeEngine):
        def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
            if params and params.get("id") == "fails":
                raise RuntimeError("boom")
            return super().query_cypher(query, params)

    engine = _FlakyEngine(
        [
            {"id": "fails", "relevance_score": 0.9},
            {"id": "ok", "relevance_score": 0.5},
        ]
    )
    report = resync_tool_relevance_scores(engine, execute=True)
    assert report["migrated"] == 1
    assert len(report["write_errors"]) == 1
    assert report["write_errors"][0]["id"] == "fails"
    assert engine.store["ok"] == 50
    assert engine.store["fails"] == 0.9  # untouched after the failed write


def test_limit_is_forwarded_to_the_query() -> None:
    captured: dict[str, str] = {}

    class _CapturingEngine(_FakeEngine):
        def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
            if query.strip().startswith("MATCH (t:Tool) RETURN"):
                captured["query"] = query
            return super().query_cypher(query, params)

    engine = _CapturingEngine([])
    resync_tool_relevance_scores(engine, limit=17)
    assert "LIMIT 17" in captured["query"]
