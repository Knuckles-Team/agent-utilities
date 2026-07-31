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


def test_accepted_writes_outcome_and_promotes_edge(_envelope_commit):
    engine = MagicMock()
    calls = []
    engine.link_nodes = lambda *a, **k: calls.append((a, k))

    outcome = review_entity_resolution_proposal(
        engine,
        proposal=_proposal(),
        decision="accepted",
        reviewer="alice",
        rationale="confirmed duplicate",
    )

    assert outcome.decision == "accepted"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:3] == ("a", "b", RegistryEdgeType.SIMILAR_TO)
    props = kwargs["properties"]
    assert props["governed"] is True
    assert props["review_outcome_id"] == outcome.outcome_id
    assert props["blocking_tier"] == "exact"
    assert (
        len(_envelope_commit) == 1
    )  # only the outcome envelope; edge is a direct link_nodes call


def test_accept_without_link_nodes_support_raises(_envelope_commit):
    engine = MagicMock(spec=[])  # no link_nodes attribute at all
    with pytest.raises(RuntimeError, match="link_nodes"):
        review_entity_resolution_proposal(
            engine, proposal=_proposal(), decision="accepted", reviewer="alice"
        )
