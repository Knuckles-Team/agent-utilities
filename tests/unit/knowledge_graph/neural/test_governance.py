"""The explicit promotion policy (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.neural.governance import (
    review_entity_resolution_proposal,
)
from agent_utilities.knowledge_graph.neural.models import (
    EntityResolutionProposal,
    GraphNodeRef,
)
from agent_utilities.models.knowledge_graph import RegistryEdgeType


def _proposal(**overrides):
    base = dict(
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
    base.update(overrides)
    return EntityResolutionProposal(**base)


@pytest.fixture(autouse=True)
def _envelope_commit(monkeypatch):
    committed = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        lambda engine, env: committed.append(env) or {"status": "success"},
    )
    return committed


def test_reviewer_is_required():
    with pytest.raises(ValueError, match="reviewer"):
        review_entity_resolution_proposal(
            MagicMock(), proposal=_proposal(), decision="accepted", reviewer=""
        )


def test_rejected_writes_outcome_but_no_edge(_envelope_commit):
    engine = MagicMock()
    engine.link_nodes = MagicMock(side_effect=AssertionError("must not link on reject"))

    outcome = review_entity_resolution_proposal(
        engine, proposal=_proposal(), decision="rejected", reviewer="alice"
    )

    assert outcome.decision == "rejected"
    assert outcome.proposal_id == "p1"
    engine.link_nodes.assert_not_called()
    assert len(_envelope_commit) == 1  # the outcome IS committed either way
    assert "_links" not in _envelope_commit[0].typed_payload


def test_accepted_commits_outcome_and_edge_in_one_envelope(_envelope_commit):
    engine = MagicMock()

    outcome = review_entity_resolution_proposal(
        engine,
        proposal=_proposal(),
        decision="accepted",
        reviewer="alice",
        rationale="confirmed duplicate",
    )

    assert outcome.decision == "accepted"
    assert len(_envelope_commit) == 1
    env = _envelope_commit[0]
    assert env.tenant == "acme"
    assert env.typed_payload["id"] == outcome.outcome_id
    assert env.typed_payload["type"] == "EntityResolutionReviewOutcome"
    edge = env.typed_payload["_links"]
    assert edge == [
        {
            "source": "a",
            "target": "b",
            "relationship": RegistryEdgeType.SIMILAR_TO.value,
            "_rel": "SIMILAR_TO",
            "score": 0.9,
            "governed": True,
            "review_outcome_id": outcome.outcome_id,
            "blocking_tier": "exact",
        }
    ]
    engine.link_nodes.assert_not_called()


def test_accept_uses_envelope_without_direct_link_writer(_envelope_commit):
    engine = MagicMock(spec=[])
    outcome = review_entity_resolution_proposal(
        engine, proposal=_proposal(), decision="accepted", reviewer="alice"
    )
    assert len(_envelope_commit) == 1
    assert (
        _envelope_commit[0].typed_payload["_links"][0]["review_outcome_id"]
        == outcome.outcome_id
    )


def test_failed_native_apply_does_not_call_direct_link(monkeypatch):
    engine = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        lambda engine, env: {"status": "error"},
    )
    with pytest.raises(RuntimeError, match="ChangeEnvelope failed"):
        review_entity_resolution_proposal(
            engine, proposal=_proposal(), decision="accepted", reviewer="alice"
        )
    engine.link_nodes.assert_not_called()


def test_review_rejects_foreign_tenant_before_write(_envelope_commit):
    engine = MagicMock()
    with pytest.raises(PermissionError, match="tenant"):
        review_entity_resolution_proposal(
            engine,
            proposal=_proposal(tenant="foreign"),
            decision="accepted",
            reviewer="alice",
        )
    engine.link_nodes.assert_not_called()
    assert _envelope_commit == []
