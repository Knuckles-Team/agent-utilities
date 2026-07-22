"""``artifact_evolution_summary`` — the cross-vector artifact-version lineage read
(CONCEPT:AU-AHE.evolution.unified-artifact-lineage). Scans every known
artifact-version node label (``skill_version``, ``prompt_version``) and
aggregates by status/kind — the query `grep -rn "evolution_matrix"` finds
nothing for today.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.research.evolution_state import (
    artifact_evolution_summary,
)


class _StubEngine:
    def __init__(self, rows_by_label: dict[str, list[dict[str, Any]]]):
        self._rows_by_label = rows_by_label

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        for label, rows in self._rows_by_label.items():
            if f"MATCH (v:{label})" in q:
                return rows
        return []


def test_none_engine_returns_empty_shaped_reading():
    summary = artifact_evolution_summary(None)
    assert summary == {"total": 0, "by_status": {}, "by_kind": {}, "versions": []}


def test_aggregates_across_known_labels():
    engine = _StubEngine(
        {
            "skill_version": [
                {"id": "sv1", "status": "active", "artifact_kind": "skill"},
                {"id": "sv2", "status": "proposal", "artifact_kind": "skill"},
            ],
            "prompt_version": [
                {"id": "pv1", "status": "proposal", "artifact_kind": "prompt"},
            ],
        }
    )
    summary = artifact_evolution_summary(engine)
    assert summary["total"] == 3
    assert summary["by_status"] == {"active": 1, "proposal": 2}
    assert summary["by_kind"] == {"skill": 2, "prompt": 1}
    assert len(summary["versions"]) == 3


def test_filters_by_artifact_kind():
    engine = _StubEngine(
        {
            "skill_version": [{"id": "sv1", "status": "active", "artifact_kind": "skill"}],
            "prompt_version": [{"id": "pv1", "status": "proposal", "artifact_kind": "prompt"}],
        }
    )
    summary = artifact_evolution_summary(engine, artifact_kind="prompt")
    assert summary["total"] == 1
    assert summary["by_kind"] == {"prompt": 1}


def test_missing_artifact_kind_falls_back_to_label_derived_kind():
    # A row that predates the artifact_kind field (e.g. an old SkillVersion)
    # still classifies correctly from its own label.
    engine = _StubEngine({"skill_version": [{"id": "sv1", "status": "active"}]})
    summary = artifact_evolution_summary(engine)
    assert summary["by_kind"] == {"skill": 1}


def test_one_label_query_failure_does_not_block_the_others():
    class _PartlyBroken:
        def query_cypher(self, q, params=None):
            if "skill_version" in q:
                raise RuntimeError("backend down")
            return [{"id": "pv1", "status": "active", "artifact_kind": "prompt"}]

    summary = artifact_evolution_summary(_PartlyBroken())
    assert summary["total"] == 1
    assert summary["by_kind"] == {"prompt": 1}


def test_limit_bounds_returned_versions_but_not_totals():
    rows = [{"id": f"sv{i}", "status": "active", "artifact_kind": "skill"} for i in range(5)]
    engine = _StubEngine({"skill_version": rows})
    summary = artifact_evolution_summary(engine, limit=2)
    assert summary["total"] == 5
    assert len(summary["versions"]) == 2
