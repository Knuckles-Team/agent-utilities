#!/usr/bin/python
"""Tests for the provenance-completeness critic gate.

CONCEPT:AU-AHE.harness.pre-emit-quality-gate
"""

import pytest

from agent_utilities.harness.provenance_gate import (
    ProvenanceCriticGate,
    ProvenanceVerdict,
)

pytestmark = pytest.mark.concept("AU-AHE.harness.pre-emit-quality-gate")


def test_fully_grounded_answer_accepted():
    gate = ProvenanceCriticGate()
    answer = "Revenue grew to 42 last quarter [s1]. Margins held steady [s2]."
    v = gate.evaluate(answer, sources=["s1", "s2"], tool_values=[42])
    assert isinstance(v, ProvenanceVerdict)
    assert v.decision == "accept"
    assert v.completeness == 1.0
    assert v.ungrounded_numbers == [] and v.uncited_claims == []


def test_ungrounded_number_triggers_revise():
    gate = ProvenanceCriticGate()
    # 99 is not in tool_values and its sentence has no citation.
    answer = "The system achieved 99 percent accuracy on the benchmark."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[], attempt=0)
    assert v.decision == "revise"
    assert "99" in v.ungrounded_numbers
    assert v.numeric_grounded < 1.0


def test_number_grounded_by_tool_value():
    gate = ProvenanceCriticGate()
    answer = "The calculator returned 4 for the sum [s1]."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[4])
    assert "4" not in v.ungrounded_numbers
    assert v.decision == "accept"


def test_uncited_substantive_claim_flagged():
    gate = ProvenanceCriticGate()
    answer = (
        "Our platform is the most secure enterprise solution available anywhere today."
    )
    v = gate.evaluate(answer, sources=["s1"], tool_values=[])
    assert v.uncited_claims
    assert v.claim_grounded < 1.0
    assert v.decision in {"revise", "escalate"}


def test_invalid_citation_blocks_accept():
    gate = ProvenanceCriticGate()
    # Cites s9 which is not a known source.
    answer = "Latency dropped to 5 ms [s9]."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[5])
    assert v.invalid_citations == ["s9"]
    assert v.decision != "accept"


def test_escalation_after_revise_budget_exhausted():
    gate = ProvenanceCriticGate(max_revise=2)
    answer = "We are 100 percent certain this is the best option overall for everyone."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[], attempt=2)
    assert v.decision == "escalate"


def test_short_non_substantive_sentences_exempt():
    gate = ProvenanceCriticGate()
    answer = "Yes. Done."  # trivial, number-free, short → no citation required
    v = gate.evaluate(answer, sources=[], tool_values=[])
    assert v.decision == "accept"
    assert v.completeness == 1.0
