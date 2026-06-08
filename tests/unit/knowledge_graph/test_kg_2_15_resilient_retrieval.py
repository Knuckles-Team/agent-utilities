"""CONCEPT:KG-2.15 — Resilient Retrieval.

Covers the social-closer triviality gate and the lexical fallback cascade (degradation when the
vector path returns nothing), using a fake retriever/backend so no embeddings are needed.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever
from agent_utilities.knowledge_graph.retrieval.hyde_planner import is_trivial_query

# ── triviality gate ─────────────────────────────────────────────────────────────


@pytest.mark.concept(id="KG-2.15")
def test_is_trivial_query():
    for t in ["", "ok", "thanks", "Thanks!", "ok.", "👍", "yep", "bye"]:
        assert is_trivial_query(t) is True, t
    for t in [
        "what is the capital of France",
        "deploy the staging cluster",
        "fix the bug",
    ]:
        assert is_trivial_query(t) is False, t


# ── plan_and_retrieve gate + cascade with a fake retriever ──────────────────────


class _FakeReport:
    gate_passed = True


class _FakeBackend:
    def __init__(self, rows):
        self._rows = rows
        self.queries = []

    def execute(self, q, params=None):
        self.queries.append((q, params))
        return self._rows


class _FakeEngine:
    def __init__(self, backend):
        self.backend = backend


class _FakeRetriever:
    plan_and_retrieve = HybridRetriever.plan_and_retrieve
    usage_telemetry = HybridRetriever.usage_telemetry
    _lexical_fallback = HybridRetriever._lexical_fallback

    def __init__(self, hybrid_results, backend_rows=None):
        self._hybrid_results = hybrid_results
        self.engine = _FakeEngine(_FakeBackend(backend_rows or []))
        self._report = _FakeReport()
        self.hybrid_calls = 0

    @property
    def last_quality_report(self):
        return self._report

    def retrieve_hybrid(self, query, **kw):
        self.hybrid_calls += 1
        return list(self._hybrid_results)


@pytest.mark.concept(id="KG-2.15")
def test_trivial_query_skips_retrieval_entirely():
    r = _FakeRetriever(hybrid_results=[{"id": "1", "_score": 0.9}])
    out = r.plan_and_retrieve("thanks!", mode="standard")  # type: ignore[misc]
    assert out == []
    assert r.hybrid_calls == 0  # planner + retrieval skipped


@pytest.mark.concept(id="KG-2.15")
def test_lexical_fallback_fires_when_vector_empty():
    # Vector path returns nothing; backend has a lexical match.
    rows = [
        {
            "id": "kb1",
            "data": {"name": "Staging deploy guide", "content": "deploy staging"},
        }
    ]
    r = _FakeRetriever(hybrid_results=[], backend_rows=rows)
    out = r.plan_and_retrieve("how to deploy staging", mode="standard")  # type: ignore[misc]
    assert out and out[0]["id"] == "kb1"  # type: ignore[index]
    assert out[0]["_fallback"] == "lexical"  # type: ignore[index]
    assert out[0]["_score"] == 0.2  # type: ignore[index]


@pytest.mark.concept(id="KG-2.15")
def test_no_fallback_when_vector_returns_results():
    r = _FakeRetriever(
        hybrid_results=[{"id": "v1", "_score": 0.8}], backend_rows=[{"id": "x"}]
    )
    out = r.plan_and_retrieve("real query here", mode="standard")  # type: ignore[misc]
    assert [n["id"] for n in out] == ["v1"]  # type: ignore[index]
    assert r.engine.backend.queries == []  # lexical fallback not invoked


@pytest.mark.concept(id="KG-2.15")
def test_lexical_fallback_no_backend_returns_empty():
    r = _FakeRetriever(hybrid_results=[])
    r.engine.backend = None
    out = r.plan_and_retrieve("query with no backend available", mode="standard")  # type: ignore[misc]
    assert out == []
