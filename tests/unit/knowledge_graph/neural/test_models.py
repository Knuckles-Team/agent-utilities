"""Governed neural-graph model contracts (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.neural.models import (
    EntityResolutionProposal,
    GraphNodeRef,
    RelationLinkPrediction,
    ReviewOutcome,
    TenantScopedEmbedding,
)


def test_entity_resolution_proposal_decision_status_is_pinned():
    proposal = EntityResolutionProposal(
        proposal_id="p1",
        tenant="acme",
        mention=GraphNodeRef(node_id="a", node_type="Person"),
        candidate=GraphNodeRef(node_id="b", node_type="Person"),
        score=0.9,
        raw_similarity=0.95,
        blocking_tier="exact",
        calibration_ref="default-v1-uncalibrated",
        evidence_refs=("blocking:exact:alice",),
    )
    assert proposal.decision_status == "proposed"
    with pytest.raises(ValidationError):
        EntityResolutionProposal.model_validate(
            {**proposal.model_dump(), "decision_status": "accepted"}
        )


def test_entity_resolution_proposal_requires_evidence():
    with pytest.raises(ValidationError):
        EntityResolutionProposal(
            proposal_id="p1",
            tenant="acme",
            mention=GraphNodeRef(node_id="a"),
            candidate=GraphNodeRef(node_id="b"),
            score=0.9,
            raw_similarity=0.9,
            blocking_tier="lsh",
            calibration_ref="x",
            evidence_refs=(),
        )


def test_models_are_frozen():
    ref = GraphNodeRef(node_id="a")
    with pytest.raises(ValidationError):
        ref.node_id = "b"


def test_tenant_scoped_embedding_content_hash_must_be_sha256_hex():
    with pytest.raises(ValidationError):
        TenantScopedEmbedding(
            representation_id="r1",
            tenant="acme",
            target=GraphNodeRef(node_id="a"),
            encoder_id="bge-m3",
            encoder_version="1",
            dimension=4,
            artifact_ref="a",
            graph_epoch=0,
            content_hash="not-a-hash",
        )


def test_review_outcome_decision_is_accepted_or_rejected():
    outcome = ReviewOutcome(
        outcome_id="o1",
        proposal_id="p1",
        tenant="acme",
        decision="accepted",
        reviewer="alice",
        reviewed_at=datetime.now(UTC),
    )
    assert outcome.decision == "accepted"
    with pytest.raises(ValidationError):
        ReviewOutcome.model_validate({**outcome.model_dump(), "decision": "maybe"})


def test_relation_link_prediction_also_pinned_to_proposed():
    pred = RelationLinkPrediction(
        prediction_id="rp1",
        tenant="acme",
        subject=GraphNodeRef(node_id="a"),
        predicate="cites",
        object=GraphNodeRef(node_id="b"),
        score=0.5,
        uncertainty=0.5,
        model_ref="complex-v0",
        trained_on_outcomes_through="2026-01-01",
    )
    assert pred.decision_status == "proposed"
