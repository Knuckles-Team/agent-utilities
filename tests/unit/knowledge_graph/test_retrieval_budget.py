"""Retrieval-time context budget + task-scoped retrieval tests (CONCEPT:AU-KG.memory.tiered-memory-caching)."""

from __future__ import annotations

import asyncio

from agent_utilities.knowledge_graph.retrieval.budget import (
    RetrievalBudgetManager,
    fit_within,
)


def _text(n_words: int) -> str:
    return " ".join(["word"] * n_words)


def test_budget_never_exceeds_and_keeps_best_first():
    # estimate_tokens ~ 1.33 * words. 30-word items ~ 39 tokens each.
    cands = [{"id": i, "content": _text(30)} for i in range(10)]
    res = RetrievalBudgetManager(token_budget=100).fit(
        cands, text_of=lambda c: c["content"]
    )
    assert res.tokens_used <= 100
    assert res.truncated
    # Order preserved (best-first): first kept item is index 0.
    assert res.kept[0]["id"] == 0


def test_budget_abstains_when_no_candidate_fits():
    cands = [{"id": 0, "content": _text(1000)}]
    res = RetrievalBudgetManager(token_budget=10).fit(
        cands, text_of=lambda c: c["content"]
    )
    assert res.kept == []
    assert res.tokens_used == 0
    assert res.dropped == 1
    assert res.truncated


def test_fit_within_no_budget_is_passthrough():
    cands = [1, 2, 3]
    assert fit_within(cands, None) == cands
    assert fit_within(cands, 0) == cands


def test_fit_within_reports_kept_subset():
    cands = [{"c": _text(50)} for _ in range(5)]
    kept = fit_within(cands, 100, text_of=lambda c: c["c"])
    assert 0 < len(kept) < 5


# ── task-scoped router ────────────────────────────────────────────────────────
class FakeEngine:
    def __init__(self, results):
        self._results = results
        self.last_query = None

    def search_hybrid(self, query, top_k=3):
        self.last_query = query
        return self._results[:top_k]


def test_route_context_explicit_namespaces_short_circuit():
    from agent_utilities.graph.routing.strategies.workflow_context import (
        WorkflowContextRouter,
    )

    router = WorkflowContextRouter(engine=FakeEngine([]))
    res = asyncio.run(
        router.route_context("q", {"namespaces": ["ns-a", "ns-b"], "goal_id": "g1"})
    )
    assert res.workflow_id == "g1"
    assert res.allowed_namespaces == ["ns-a", "ns-b"]


def test_route_context_biases_query_with_caps_and_budgets():
    from agent_utilities.graph.routing.strategies.workflow_context import (
        WorkflowContextRouter,
    )

    results = [{"id": f"n{i}", "content": " ".join(["w"] * 40)} for i in range(5)]
    engine = FakeEngine(results)
    router = WorkflowContextRouter(engine=engine)
    res = asyncio.run(
        router.route_context(
            "draft email",
            {"required_caps": ["email"], "top_k": 5, "token_budget": 80},
        )
    )
    # capabilities biased the query
    assert engine.last_query and "caps=email" in engine.last_query
    # budget trimmed the 5 results down
    assert 0 < len(res.allowed_namespaces) < 5


# ── retriever wrapper (thin) ──────────────────────────────────────────────────
def test_retrieve_hybrid_budgeted_applies_budget():
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    # Bypass heavy __init__; we only exercise the budgeted wrapper.
    r = object.__new__(HybridRetriever)
    nodes = [{"id": i, "content": " ".join(["w"] * 40)} for i in range(6)]
    r.retrieve_hybrid = lambda query, **kw: nodes  # type: ignore[method-assign]

    out = r.retrieve_hybrid_budgeted("q", token_budget=80)
    assert 0 < len(out) < 6
    # no budget -> all
    assert len(r.retrieve_hybrid_budgeted("q")) == 6


def test_budget_skips_oversized_candidates_but_keeps_smaller_ranked_items():
    cands = [
        {"id": "large", "content": _text(1000)},
        {"id": "small-first", "content": _text(2)},
        {"id": "large-again", "content": _text(1000)},
        {"id": "small-second", "content": _text(2)},
    ]
    result = RetrievalBudgetManager(20).fit(cands, text_of=lambda c: c["content"])
    assert [item["id"] for item in result.kept] == ["small-first", "small-second"]
    assert result.tokens_used <= result.token_budget
    assert result.dropped == 2


def test_explicit_zero_manager_capacity_keeps_no_positive_cost_item():
    result = RetrievalBudgetManager(0).fit(["some evidence"])
    assert result.kept == []
    assert result.tokens_used == 0
    assert result.dropped == 1
