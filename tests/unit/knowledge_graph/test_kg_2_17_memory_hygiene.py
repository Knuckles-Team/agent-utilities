"""CONCEPT:KG-2.17 — Memory Hygiene.

Covers decay classification (archive/alert/exempt/keep), the importance-tiered half-life, the
semantic-merge grouping with length-ratio pre-filter, and soft-archival (valid_to set, never
deleted) via a fake backend.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.knowledge_graph.memory.hygiene import (
    CONFIDENCE_EXEMPT,
    MemoryHygiene,
    classify_node,
    decay_score,
    half_life_days,
    plan_decay,
    semantic_merge_groups,
)

NOW = datetime(2026, 6, 5, tzinfo=UTC)


def _old(days: int) -> str:
    return (NOW - timedelta(days=days)).isoformat()


@pytest.mark.concept(id="KG-2.17")
def test_half_life_tiers():
    assert half_life_days(0.5) == 90.0  # important → slow decay
    assert half_life_days(0.1) == 30.0  # low importance → fast decay


@pytest.mark.concept(id="KG-2.17")
def test_decay_score_monotonic():
    fresh = decay_score(0, 0.5)
    old = decay_score(365, 0.5)
    assert fresh == pytest.approx(1.0)
    assert 0.0 < old < fresh


@pytest.mark.concept(id="KG-2.17")
def test_classify_exempt_human_and_procedural():
    assert (
        classify_node(
            {"id": "h", "source_type": "human", "created_at": _old(9999)}, NOW
        )
        == "exempt"
    )
    assert (
        classify_node(
            {"id": "p", "source_type": "procedural", "created_at": _old(9999)}, NOW
        )
        == "exempt"
    )


@pytest.mark.concept(id="KG-2.17")
def test_classify_archive_vs_alert_vs_keep():
    # Fresh AI memory → keep.
    assert (
        classify_node(
            {
                "id": "k",
                "source_type": "ai",
                "importance_score": 0.2,
                "created_at": _old(1),
            },
            NOW,
        )
        == "keep"
    )
    # Very old, low importance, low confidence → archive.
    assert (
        classify_node(
            {
                "id": "a",
                "source_type": "ai",
                "importance_score": 0.1,
                "confidence": 0.2,
                "created_at": _old(400),
            },
            NOW,
        )
        == "archive"
    )
    # Very old but high confidence → alert (never silently archived).
    assert (
        classify_node(
            {
                "id": "x",
                "source_type": "ai",
                "importance_score": 0.1,
                "confidence": CONFIDENCE_EXEMPT,
                "created_at": _old(400),
            },
            NOW,
        )
        == "alert"
    )


@pytest.mark.concept(id="KG-2.17")
def test_plan_decay_buckets():
    nodes = [
        {"id": "h", "source_type": "human", "created_at": _old(999)},
        {
            "id": "a",
            "source_type": "ai",
            "importance_score": 0.1,
            "confidence": 0.1,
            "created_at": _old(400),
        },
        {
            "id": "k",
            "source_type": "ai",
            "importance_score": 0.2,
            "created_at": _old(1),
        },
    ]
    plan = plan_decay(nodes, NOW)  # type: ignore[arg-type]
    assert (
        plan["exempt"] == ["h"] and plan["archive"] == ["a"] and plan["keep"] == ["k"]
    )


@pytest.mark.concept(id="KG-2.17")
def test_semantic_merge_groups_and_length_prefilter():
    a = {"id": "a", "content": "x" * 100, "embedding": [1.0, 0.0]}
    b = {
        "id": "b",
        "content": "y" * 100,
        "embedding": [1.0, 0.0],
    }  # identical vector → merge
    c = {
        "id": "c",
        "content": "z" * 10,
        "embedding": [1.0, 0.0],
    }  # 10x shorter → length pre-filter skips
    d = {
        "id": "d",
        "content": "w" * 100,
        "embedding": [0.0, 1.0],
    }  # orthogonal → no merge
    groups = semantic_merge_groups([a, b, c, d])
    assert groups == [["a", "b"]]


class _FakeBackend:
    def __init__(self, rows):
        self._rows = rows
        self.writes = []

    def execute(self, q, params=None):
        if q.strip().upper().startswith("MATCH (N:MEMORY)"):
            return self._rows
        self.writes.append((q, params))
        return []


class _FakeEngine:
    def __init__(self, rows):
        self.backend = _FakeBackend(rows)


@pytest.mark.concept(id="KG-2.17")
def test_run_archives_via_valid_to_not_delete():
    rows = [
        {
            "id": "a",
            "data": {
                "source_type": "ai",
                "importance_score": 0.1,
                "confidence": 0.1,
                "created_at": _old(400),
            },
        },
        {
            "id": "k",
            "data": {
                "source_type": "ai",
                "importance_score": 0.2,
                "created_at": _old(1),
            },
        },
    ]
    eng = _FakeEngine(rows)
    out = MemoryHygiene(eng).run(now=NOW)
    assert out["archived"] == 1 and out["kept"] == 1
    # The archive write sets valid_to + status ARCHIVED — never a DELETE.
    writes = eng.backend.writes
    assert writes and "valid_to" in writes[0][0] and "ARCHIVED" in writes[0][0]
    assert not any("DELETE" in w[0].upper() for w in writes)


@pytest.mark.concept(id="KG-2.17")
def test_dry_run_does_not_write():
    rows = [
        {
            "id": "a",
            "data": {
                "source_type": "ai",
                "importance_score": 0.1,
                "confidence": 0.1,
                "created_at": _old(400),
            },
        }
    ]
    eng = _FakeEngine(rows)
    out = MemoryHygiene(eng).run(now=NOW, dry_run=True)
    assert out["archived"] == 1 and out["dry_run"] is True
    assert eng.backend.writes == []


# ── CONCEPT:KG-2.17 — semantic-merge APPLY (not just count) ──────────────────────

from agent_utilities.knowledge_graph.memory.hygiene import merge_plan  # noqa: E402


@pytest.mark.concept(id="KG-2.17")
def test_merge_plan_unions_tags_and_max_importance():
    nodes = [
        {"id": "a", "tags": ["x"], "importance_score": 0.4},
        {"id": "b", "tags": ["y"], "importance_score": 0.9},
    ]
    plans = merge_plan(nodes, [["a", "b"]])
    assert len(plans) == 1
    p = plans[0]
    assert p["survivor"] == "a" and p["retired"] == ["b"]
    assert set(p["tags"]) == {"x", "y"} and p["importance"] == 0.9


class _MergeBackend:
    def __init__(self, rows):
        self._rows = rows
        self.writes = []

    def execute(self, q, params=None):
        if q.strip().upper().startswith("MATCH (N:MEMORY)"):
            return self._rows
        self.writes.append((q, params))
        return []


class _MergeEngine:
    def __init__(self, rows):
        self.backend = _MergeBackend(rows)
        self.edges = []

    def link_nodes(self, src, dst, rel, properties=None):
        self.edges.append((src, dst, rel))


@pytest.mark.concept(id="KG-2.17")
def test_run_applies_merge_soft_retire_and_edge():
    # Two identical-vector, similar-length AI memories → a merge group.
    rows = [
        {
            "id": "a",
            "data": {
                "source_type": "ai",
                "importance_score": 0.4,
                "created_at": _old(1),
                "content": "x" * 50,
                "tags": ["t1"],
                "embedding": [1.0, 0.0],
            },
        },
        {
            "id": "b",
            "data": {
                "source_type": "ai",
                "importance_score": 0.9,
                "created_at": _old(1),
                "content": "y" * 50,
                "tags": ["t2"],
                "embedding": [1.0, 0.0],
            },
        },
    ]
    eng = _MergeEngine(rows)
    out = MemoryHygiene(eng).run(now=NOW)
    assert out["merge_groups"] == 1 and out["merged"] == 1
    # Survivor 'a' absorbed tags + max importance; dup 'b' soft-retired (MERGED, valid_to), not deleted.
    sets = " ".join(w[0] for w in eng.backend.writes)
    assert "importance_score" in sets and "status = 'MERGED'" in sets
    assert not any("DELETE" in w[0].upper() for w in eng.backend.writes)
    assert ("b", "a", "MERGED_INTO") in eng.edges
