#!/usr/bin/python
"""Tests for reasoning-aware reranking and its wiring into HybridRetriever.

CONCEPT:AU-KG.research.research-pipeline-runner
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.retrieval.reasoning_reranker import (
    LexicalRelevanceScorer,
    ReasoningAwareReranker,
    calibrate,
)

pytestmark = pytest.mark.concept("AU-KG.research.research-pipeline-runner")


# --- calibrate -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,level",
    [(0.0, 0.0), (0.1, 0.0), (0.2, 0.25), (0.4, 0.5), (0.7, 0.75), (0.95, 1.0)],
)
def test_calibrate_five_levels(raw, level):
    assert calibrate(raw) == level


# --- LexicalRelevanceScorer ------------------------------------------------


def test_relevant_scores_higher_than_irrelevant():
    s = LexicalRelevanceScorer()
    q = "graph database vector index tuning"
    rel = s.score(q, "A guide to graph database vector index tuning and performance.")
    irrel = s.score(q, "A recipe for cooking pasta with tomato sauce.")
    assert rel > irrel
    assert 0.0 <= irrel <= rel <= 1.0


def test_instruction_awareness_shifts_score():
    s = LexicalRelevanceScorer()
    q = "deployment"
    text = "Kubernetes rollout strategy and canary deployment steps."
    base = s.score(q, text)
    with_instr = s.score(q, text, instruction="kubernetes canary rollout")
    assert with_instr > base


def test_empty_inputs_score_zero():
    s = LexicalRelevanceScorer()
    assert s.score("", "anything") == 0.0
    assert s.score("anything", "") == 0.0


# --- ReasoningAwareReranker ------------------------------------------------


def _cand(node_id, score, text):
    return {"id": node_id, "_score": score, "content": text}


def test_rerank_promotes_relevant_over_high_prior_irrelevant():
    reranker = ReasoningAwareReranker()
    query = "graph database vector index"
    candidates = [
        _cand("A", 0.9, "an essay about cooking pasta and tomato sauce"),
        _cand("B", 0.5, "graph database vector index tuning and HNSW guide"),
    ]
    out = reranker.rerank(query, candidates)
    assert out[0]["id"] == "B"  # relevance beats raw vector prior
    assert out[0]["_rerank_score"] >= out[1]["_rerank_score"]
    assert "_rerank_level" in out[0] and out[0]["_rerank_level"] in {
        0,
        0.25,
        0.5,
        0.75,
        1.0,
    }


def test_rerank_annotations_and_top_k():
    reranker = ReasoningAwareReranker()
    cands = [
        _cand("A", 0.8, "alpha beta gamma"),
        _cand("B", 0.6, "vector index graph"),
        _cand("C", 0.4, "graph database vector index search"),
    ]
    out = reranker.rerank("graph vector index", cands, top_k=2)
    assert len(out) == 2
    for c in out:
        assert "_rerank_score" in c and "_rerank_relevance" in c


def test_rerank_single_candidate_passthrough():
    reranker = ReasoningAwareReranker()
    cands = [_cand("A", 0.5, "only one")]
    assert reranker.rerank("q", cands) == cands


def test_rerank_stable_tiebreak_preserves_prior_order():
    # Two candidates with identical text+prior → original order preserved.
    reranker = ReasoningAwareReranker()
    cands = [_cand("A", 0.5, "same text"), _cand("B", 0.5, "same text")]
    out = reranker.rerank("unrelated query", cands)
    assert [c["id"] for c in out] == ["A", "B"]


# --- Live path: HybridRetriever._rerank_candidates -------------------------


def _retriever(enable_rerank=True):
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    # __init__ does not touch the (lazy) embedding model, so a MagicMock engine
    # is sufficient to exercise the rerank helper on the live class.
    return HybridRetriever(MagicMock(), enable_rerank=enable_rerank)


def test_live_path_rerank_reorders_and_caps():
    r = _retriever(enable_rerank=True)
    scored = [
        _cand("A", 0.9, "an essay about cooking pasta"),
        _cand("B", 0.5, "graph database vector index tuning guide"),
        _cand("C", 0.3, "weather forecast for tomorrow"),
    ]
    base = r._rerank_candidates("graph database vector index", scored, context_window=1)
    assert len(base) == 1
    assert base[0]["id"] == "B"  # relevant node promoted above higher-prior A


def test_live_path_rerank_disabled_is_plain_vector_slice():
    r = _retriever(enable_rerank=False)
    scored = [
        _cand("A", 0.9, "an essay about cooking pasta"),
        _cand("B", 0.5, "graph database vector index tuning guide"),
    ]
    base = r._rerank_candidates("graph database vector index", scored, context_window=1)
    assert [c["id"] for c in base] == ["A"]  # unchanged vector order
    assert "_rerank_score" not in base[0]
