"""CONCEPT:AU-KG.retrieval.memory-first-retrieval — Memory-First Retrieval.

Covers the pure HyDE-planner helpers (plan parse/fallback, dual thresholds, merge, evidence
ledger) and the ``plan_and_retrieve`` orchestration (HyDE multi-query, dual-threshold, and the
evidence-gated self-correcting second pass) using a lightweight fake retriever.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval.hyde_planner import (
    HYDE_THRESHOLDS,
    HydePlan,
    build_evidence_ledger,
    merge_retrievals,
    parse_hyde_plan,
    threshold_for_mode,
)

# ── pure helpers ──────────────────────────────────────────────────────────────


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_dual_thresholds_match_quarq_constants():
    assert HYDE_THRESHOLDS["standard"] == 0.38
    assert HYDE_THRESHOLDS["deep"] == 0.28
    assert threshold_for_mode("deep") == 0.28
    assert threshold_for_mode("unknown") == 0.38  # safe default


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_parse_hyde_plan_valid_json():
    raw = (
        '{"vector_queries": ["a", "b"], "keywords": "Foo, Bar", "search_mode": "deep"}'
    )
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["a", "b"]
    assert plan.keywords == ["Foo", "Bar"]  # comma-string tolerated
    assert plan.search_mode == "deep"


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_parse_hyde_plan_fallback_on_garbage():
    plan = parse_hyde_plan("not json at all", original_query="orig", mode_hint="deep")
    assert plan.vector_queries == ["orig"]
    assert plan.search_mode == "deep"


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_effective_queries_dedups_and_includes_original():
    plan = HydePlan(vector_queries=["a", "a", ""], search_mode="standard")
    assert plan.effective_queries("orig") == ["a", "orig"]


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_merge_retrievals_dedups_keeps_max_score_and_sorts():
    a = [{"id": "1", "_score": 0.4}, {"id": "2", "_score": 0.9}]
    b = [{"id": "1", "_score": 0.7}, {"id": "3", "_score": 0.5}]
    merged = merge_retrievals([a, b], context_window=10)
    assert [n["id"] for n in merged] == ["2", "1", "3"]
    assert next(n for n in merged if n["id"] == "1")["_score"] == 0.7  # max kept


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_evidence_ledger_accepts_above_threshold_and_extracts_numbers():
    nodes = [
        {"id": "1", "_score": 0.5, "content": "User paid $40 for the monitor"},
        {"id": "2", "_score": 0.1, "content": "unrelated near-miss"},
    ]
    ledger = build_evidence_ledger("how much for the monitor?", nodes)
    assert ledger["accept_count"] == 1
    assert ledger["reject_count"] == 1
    assert ledger["accepted_ids"] == ["1"]
    assert "$40" in ledger["accepted_numbers"]


# ── orchestration with a fake retriever ─────────────────────────────────────────


class _FakeReport:
    def __init__(self, gate_passed: bool):
        self.gate_passed = gate_passed


class _FakeRetriever:
    """Minimal stand-in exercising plan_and_retrieve without a backend or LLM."""

    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    plan_and_retrieve = HybridRetriever.plan_and_retrieve
    usage_telemetry = (
        HybridRetriever.usage_telemetry
    )  # KG-2.18 recall telemetry on the live path

    def __init__(self, gate_passed: bool = True):
        self._report = _FakeReport(gate_passed)
        self.calls: list[float | None] = []

    # plan_and_retrieve uses self._generate_hyde_plan only in "hyde" mode; we test
    # standard/deep modes (single query) so no LLM is needed.
    @property
    def last_quality_report(self):
        return self._report

    def retrieve_hybrid(self, query, context_window=10, relevance_threshold=None, **kw):
        self.calls.append(relevance_threshold)
        # Return one node whose score encodes which threshold was used.
        return [{"id": f"{query}@{relevance_threshold}", "_score": 0.5}]


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_plan_and_retrieve_standard_uses_038_threshold():
    r = _FakeRetriever(gate_passed=True)
    nodes = r.plan_and_retrieve("q", mode="standard")  # type: ignore[misc]
    assert r.calls == [0.38]
    assert nodes and nodes[0]["id"] == "q@0.38"  # type: ignore[index]


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_plan_and_retrieve_deep_uses_028_threshold():
    r = _FakeRetriever(gate_passed=True)
    r.plan_and_retrieve("q", mode="deep")  # type: ignore[misc]
    assert r.calls == [0.28]


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_self_correct_triggers_second_pass_only_on_gate_failure():
    # Gate passes → no second pass.
    ok = _FakeRetriever(gate_passed=True)
    ok.plan_and_retrieve("q", mode="standard", self_correct=True)  # type: ignore[misc]
    assert ok.calls == [0.38]

    # Gate fails → a deep-threshold second pass is added.
    bad = _FakeRetriever(gate_passed=False)
    bad.plan_and_retrieve("q", mode="standard", self_correct=True)  # type: ignore[misc]
    assert bad.calls == [0.38, 0.28]


@pytest.mark.concept(id="AU-KG.retrieval.memory-first-retrieval")
def test_with_ledger_returns_structured_payload():
    r = _FakeRetriever(gate_passed=True)
    out = r.plan_and_retrieve("q", mode="standard", with_ledger=True)  # type: ignore[misc]
    assert set(out) == {"nodes", "ledger", "plan"}
    assert out["plan"]["search_mode"] == "standard"  # type: ignore[call-overload]
