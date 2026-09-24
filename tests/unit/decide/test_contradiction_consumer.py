"""EH-034: contradiction handling is a suggestion from EG; nothing is retracted."""

from __future__ import annotations

from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (
    Claim,
    ContradictionDetector,
)
from tests.unit.decide.fakes import FakeTransport, abstained, acted

NEW = Claim("c-new", "The billing service is not deprecated.")
OLD = Claim("c-old", "The billing service is deprecated.")


def test_eg_suggests_a_handling_but_the_finding_stays_a_proposal(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("retract:c-old")
    [finding] = ContradictionDetector().check(NEW, [OLD])
    assert finding.suggestion == "retract:c-old"
    assert finding.decision_record is not None
    assert eg.requests[0]["question"]["safety"] == "irreversible"


def test_an_abstention_surfaces_the_friction_for_a_human(eg: FakeTransport) -> None:
    eg.answer = abstained()
    [finding] = ContradictionDetector().check(NEW, [OLD])
    assert finding.suggestion == "keep_both"
    assert eg.op_names() == ["commit"]
