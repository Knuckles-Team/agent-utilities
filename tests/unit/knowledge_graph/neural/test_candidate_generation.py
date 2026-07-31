"""Blocking-before-ANN candidate generation (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.neural.candidate_generation import (
    CALIBRATION_REF,
    generate_entity_resolution_proposals,
)


def test_exact_name_match_never_calls_ann():
    """Two identical normalized names resolve via the ladder alone — ANN untouched."""
    engine = MagicMock()
    engine.semantic_search = MagicMock(
        side_effect=AssertionError("ANN must not be called")
    )

    proposals = generate_entity_resolution_proposals(
        engine,
        tenant="acme",
        items=[("a", "Acme Corp"), ("b", "Acme Corp")],
    )

    assert len(proposals) == 1
    assert proposals[0].blocking_tier == "exact"
    assert proposals[0].decision_status == "proposed"
    assert proposals[0].calibration_ref == CALIBRATION_REF
    engine.semantic_search.assert_not_called()


def test_low_entropy_generic_names_produce_no_proposal_without_embeddings():
    """Ambiguous residual with no embedding supplied → no ANN escalation, no crash."""
    engine = MagicMock()
    engine.semantic_search = MagicMock(side_effect=AssertionError("must not be called"))

    proposals = generate_entity_resolution_proposals(
        engine,
        tenant="acme",
        items=[("a", "the"), ("b", "the")],  # generic/low-entropy → residual
        embeddings=None,
    )
    assert proposals == []
    engine.semantic_search.assert_not_called()


def test_ann_escalation_only_for_residual_and_only_with_embeddings():
    engine = MagicMock()
    calls = []

    def _search(vector, top_k):
        calls.append((tuple(vector), top_k))
        return [("c", 0.9), ("a", 1.0), ("d", 0.5)]  # self-match + below-floor filtered

    engine.semantic_search = _search

    proposals = generate_entity_resolution_proposals(
        engine,
        tenant="acme",
        items=[("a", "xyz1"), ("b", "totally different unrelated name")],
        embeddings={"a": [0.1, 0.2], "b": [0.3, 0.4]},
        ann_floor=0.7,
    )

    ann_proposals = [p for p in proposals if p.blocking_tier == "ann"]
    assert len(ann_proposals) >= 1
    assert all(p.raw_similarity >= 0.7 for p in ann_proposals)
    assert all(p.candidate.node_id != p.mention.node_id for p in ann_proposals)
    assert len(calls) >= 1


def test_ann_search_failure_is_swallowed_not_raised():
    engine = MagicMock()
    engine.semantic_search = MagicMock(side_effect=RuntimeError("engine down"))

    proposals = generate_entity_resolution_proposals(
        engine,
        tenant="acme",
        items=[("a", "xyz1"), ("b", "totally different unrelated name")],
        embeddings={"a": [0.1, 0.2], "b": [0.3, 0.4]},
    )
    assert all(p.blocking_tier != "ann" for p in proposals)


def test_calibration_discounts_raw_ann_similarity():
    engine = MagicMock()
    engine.semantic_search = lambda vector, top_k: [("b", 0.9)]

    proposals = generate_entity_resolution_proposals(
        engine,
        tenant="acme",
        items=[("a", "xyz1"), ("b", "totally different unrelated name")],
        embeddings={"a": [0.1], "b": [0.2]},
    )
    ann = next(p for p in proposals if p.blocking_tier == "ann")
    assert ann.raw_similarity == 0.9
    assert ann.score < ann.raw_similarity  # discounted, never inflated
