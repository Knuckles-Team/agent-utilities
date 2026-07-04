#!/usr/bin/python
"""Unit tests for the neural cross-encoder reranker (CONCEPT:AU-KG.retrieval.unset-dependency-free).

These tests must run with NO ML dependencies installed (no torch /
sentence-transformers). The cross-encoder is exercised through an INJECTED fake
model exposing ``predict(pairs) -> list[float]`` of raw logits, so we verify the
sigmoid squashing, batching, instruction-awareness, auto-detection factory, and
determinism without touching any heavy library.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.retrieval.neural_reranker import (
    NeuralCrossEncoderReranker,
    build_rerank_scorer,
    sigmoid,
)
from agent_utilities.knowledge_graph.retrieval.reasoning_reranker import (
    LexicalRelevanceScorer,
    RerankScorer,
)


class FakeCrossEncoder:
    """Stub cross-encoder returning canned raw logits per (query, passage) pair.

    Mimics sentence-transformers' ``CrossEncoder.predict``: it accepts a list of
    ``(query, passage)`` pairs and returns one raw logit each. Logit selection is
    a simple, deterministic rule so tests can assert relevance ordering.
    """

    def __init__(self, *, logit_for: dict[str, float] | None = None) -> None:
        self.logit_for = logit_for or {}
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> list[float]:
        self.calls.append(list(pairs))
        out: list[float] = []
        for _query, passage in pairs:
            # Default: high logit when "relevant" appears, low/negative otherwise.
            if passage in self.logit_for:
                out.append(self.logit_for[passage])
            elif "relevant" in passage:
                out.append(8.0)
            else:
                out.append(-8.0)
        return out


def test_module_imports_without_ml_deps() -> None:
    # The import at module top-level already succeeded; assert the symbols exist.
    assert callable(sigmoid)
    assert callable(build_rerank_scorer)


def test_sigmoid_squashes_extremes() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(50.0) > 0.999
    assert sigmoid(-50.0) < 0.001
    # Numerically stable on large magnitudes (no overflow; underflow to the
    # asymptote is fine — what matters is no exception and a valid [0, 1] value).
    assert 0.0 <= sigmoid(-1000.0) < 0.001
    assert 0.999 < sigmoid(1000.0) <= 1.0


def test_score_high_logit_near_one_low_near_zero() -> None:
    model = FakeCrossEncoder(logit_for={"hot": 10.0, "cold": -10.0})
    scorer = NeuralCrossEncoderReranker(model=model)

    high = scorer.score("q", "hot")
    low = scorer.score("q", "cold")

    assert high > 0.99
    assert low < 0.01


def test_score_batch_maps_each_pair() -> None:
    model = FakeCrossEncoder()
    scorer = NeuralCrossEncoderReranker(model=model)

    scores = scorer.score_batch("q", ["a relevant passage", "noise", "more relevant"])

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Relevant passages outscore the irrelevant one.
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
    # A single batched predict call covered all pairs.
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 3


def test_relevant_pair_outscores_irrelevant() -> None:
    model = FakeCrossEncoder()
    scorer = NeuralCrossEncoderReranker(model=model)

    rel = scorer.score("query", "this is the relevant document")
    irr = scorer.score("query", "totally unrelated content")

    assert rel > irr


def test_instruction_incorporated_into_query_side() -> None:
    model = FakeCrossEncoder()
    scorer = NeuralCrossEncoderReranker(model=model)

    scorer.score("base query", "passage", instruction="focus on safety")

    # The instruction must appear on the query side of the (query, passage) pair.
    last_pairs = model.calls[-1]
    query_text, passage_text = last_pairs[0]
    assert "focus on safety" in query_text
    assert "base query" in query_text
    assert passage_text == "passage"


def test_score_batch_empty_returns_empty() -> None:
    model = FakeCrossEncoder()
    scorer = NeuralCrossEncoderReranker(model=model)
    assert scorer.score_batch("q", []) == []


def test_is_available_returns_bool_and_does_not_crash() -> None:
    result = NeuralCrossEncoderReranker.is_available()
    assert isinstance(result, bool)
    # With no ML deps installed this should be False, but we only contract a bool.


def test_build_rerank_scorer_with_injected_model_returns_neural() -> None:
    model = FakeCrossEncoder()
    scorer = build_rerank_scorer(prefer_neural=True, model=model)
    assert isinstance(scorer, NeuralCrossEncoderReranker)
    # And it actually scores through the injected model.
    assert scorer.score("q", "a relevant doc") > 0.9


def test_build_rerank_scorer_prefer_false_returns_lexical() -> None:
    scorer = build_rerank_scorer(prefer_neural=False)
    assert isinstance(scorer, LexicalRelevanceScorer)
    # Satisfies the RerankScorer protocol: has a callable score.
    assert hasattr(scorer, "score")
    assert callable(scorer.score)
    assert isinstance(scorer, RerankScorer)


def test_build_rerank_scorer_no_model_no_lib_falls_back() -> None:
    # No injected model + no ML library present -> lexical fallback.
    scorer = build_rerank_scorer(prefer_neural=True, model=None)
    if NeuralCrossEncoderReranker.is_available():
        assert isinstance(scorer, NeuralCrossEncoderReranker)
    else:
        assert isinstance(scorer, LexicalRelevanceScorer)


def test_determinism_same_inputs_same_score() -> None:
    model = FakeCrossEncoder(logit_for={"doc": 3.0})
    scorer = NeuralCrossEncoderReranker(model=model)
    first = scorer.score("query", "doc")
    second = scorer.score("query", "doc")
    assert first == second


def test_neural_scorer_satisfies_protocol() -> None:
    scorer = NeuralCrossEncoderReranker(model=FakeCrossEncoder())
    assert isinstance(scorer, RerankScorer)
