"""Per-tier precision/recall/calibration-error/drift/cost evaluation (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6)."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_utilities.knowledge_graph.neural.evaluation import (
    accept_rate_drift,
    evaluate_entity_resolution,
    time_block,
)
from agent_utilities.knowledge_graph.neural.models import (
    EntityResolutionProposal,
    GraphNodeRef,
    ReviewOutcome,
)


def _proposal(pid, tier, score):
    return EntityResolutionProposal(
        proposal_id=pid,
        tenant="acme",
        mention=GraphNodeRef(node_id=f"m{pid}"),
        candidate=GraphNodeRef(node_id=f"c{pid}"),
        score=score,
        raw_similarity=score,
        blocking_tier=tier,
        calibration_ref="default-v1-uncalibrated",
        evidence_refs=(f"e:{pid}",),
    )


def _outcome(pid, decision):
    return ReviewOutcome(
        outcome_id=f"o{pid}",
        proposal_id=pid,
        tenant="acme",
        decision=decision,
        reviewer="alice",
        reviewed_at=datetime.now(UTC),
    )


def test_no_reviews_yet_is_honestly_empty_not_fabricated():
    proposals = [_proposal("p1", "exact", 0.98)]
    result = evaluate_entity_resolution(proposals, outcomes=[])
    ev = result["exact"]
    assert ev.n_pending == 1
    assert ev.n_reviewed == 0
    assert ev.precision is None
    assert ev.calibration_error is None


def test_precision_and_calibration_error_computed_from_reviews():
    proposals = [
        _proposal("p1", "ann", 0.9),  # accepted → |0.9 - 1.0| = 0.1
        _proposal("p2", "ann", 0.9),  # rejected → |0.9 - 0.0| = 0.9
    ]
    outcomes = [_outcome("p1", "accepted"), _outcome("p2", "rejected")]
    result = evaluate_entity_resolution(proposals, outcomes)
    ev = result["ann"]
    assert ev.n_reviewed == 2
    assert ev.n_accepted == 1
    assert ev.precision == 0.5
    assert abs(ev.calibration_error - 0.5) < 1e-9  # mean(0.1, 0.9)


def test_recall_none_without_ground_truth():
    proposals = [_proposal("p1", "exact", 0.98)]
    outcomes = [_outcome("p1", "accepted")]
    result = evaluate_entity_resolution(proposals, outcomes)
    assert result["exact"].recall is None


def test_recall_computed_with_known_positive_pairs():
    proposals = [_proposal("p1", "exact", 0.98)]
    outcomes = [_outcome("p1", "accepted")]
    result = evaluate_entity_resolution(
        proposals, outcomes, known_positive_pairs={("mp1", "cp1"), ("mX", "cX")}
    )
    assert result["exact"].recall == 0.5


def test_accept_rate_drift_none_when_a_window_is_empty():
    assert accept_rate_drift([], [_outcome("p1", "accepted")]) is None


def test_accept_rate_drift_measures_change():
    earlier = [_outcome("p1", "rejected"), _outcome("p2", "rejected")]
    later = [_outcome("p3", "accepted"), _outcome("p4", "accepted")]
    assert accept_rate_drift(earlier, later) == 1.0


def test_time_block_measures_elapsed_ms():
    with time_block("embed", call_count=3) as sample:
        pass
    assert sample.operation == "embed"
    assert sample.call_count == 3
    assert sample.elapsed_ms >= 0.0
