"""Exact-first, probabilistic-separate result composition (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.neural.candidate_generation import CALIBRATION_REF
from agent_utilities.knowledge_graph.neural.models import (
    EntityResolutionProposal,
    GraphNodeRef,
)
from agent_utilities.knowledge_graph.neural.probabilistic_completion import (
    compose_result,
)


def _proposal():
    return EntityResolutionProposal(
        proposal_id="p1",
        tenant="acme",
        mention=GraphNodeRef(node_id="a"),
        candidate=GraphNodeRef(node_id="b"),
        score=0.8,
        raw_similarity=0.8,
        blocking_tier="ann",
        calibration_ref=CALIBRATION_REF,
        evidence_refs=("ann:x",),
    )


def test_probabilistic_key_absent_by_default():
    result = compose_result([{"id": "1"}], probabilistic=[_proposal()])
    assert "probabilistic" not in result
    assert result["exact"] == [{"id": "1"}]


def test_probabilistic_key_absent_even_if_supplied_when_flag_false():
    result = compose_result(
        [{"id": "1"}], allow_probabilistic=False, probabilistic=[_proposal()]
    )
    assert "probabilistic" not in result


def test_probabilistic_attached_only_when_opted_in():
    proposal = _proposal()
    result = compose_result(
        [{"id": "1"}], allow_probabilistic=True, probabilistic=[proposal]
    )
    assert result["probabilistic"] == [proposal]
    assert result["probabilistic"][0].decision_status == "proposed"


def test_no_probabilistic_key_when_opted_in_but_nothing_to_attach():
    result = compose_result([{"id": "1"}], allow_probabilistic=True, probabilistic=[])
    assert "probabilistic" not in result
