"""Characterization tests for ProvenanceCriticGate.evaluate (CCN 26),
agent_utilities/harness/provenance_gate.py.

tests/harness/test_provenance_gate.py already covers the broad
accept/revise/escalate decision flow, numeric/claim grounding, tool-value
grounding, and invalid-citation blocking. This file targets branches not
covered there: empty-input degenerate cases, dict-shaped `sources`, the
`reasons` list content/ordering, uncited-claim truncation to 120 chars, and
the exact accept_threshold boundary.

Pins OBSERVED behaviour before a complexity-reduction refactor. Must stay
byte-identical across the refactor commit.
"""

from __future__ import annotations

from agent_utilities.harness.provenance_gate import ProvenanceCriticGate


def test_empty_answer_has_perfect_completeness_and_is_accepted():
    gate = ProvenanceCriticGate()
    v = gate.evaluate("", sources=[], tool_values=[])
    assert v.decision == "accept"
    assert v.completeness == 1.0
    assert v.numeric_grounded == 1.0
    assert v.claim_grounded == 1.0


def test_sources_accepts_dict_and_uses_its_keys():
    gate = ProvenanceCriticGate()
    answer = "Revenue grew to 42 last quarter [s1]."
    v = gate.evaluate(answer, sources={"s1": "Some Report"}, tool_values=[42])
    assert v.decision == "accept"


def test_sources_ids_are_normalized_to_strings():
    gate = ProvenanceCriticGate()
    # A citation "[1]" should match a source id 1 (int) once stringified.
    answer = "Latency dropped to 5 ms [1]."
    v = gate.evaluate(answer, sources=[1], tool_values=[5])
    assert v.invalid_citations == []


def test_reasons_lists_ungrounded_uncited_and_invalid_in_order():
    gate = ProvenanceCriticGate()
    answer = (
        "The system achieved 99 percent accuracy on the benchmark. "
        "Our platform is the most secure enterprise solution available today. "
        "Latency dropped to 5 ms [s9]."
    )
    v = gate.evaluate(answer, sources=["s1"], tool_values=[])
    assert len(v.reasons) == 3
    assert "ungrounded numeric claim" in v.reasons[0]
    assert "uncited substantive claim" in v.reasons[1]
    assert "unknown sources" in v.reasons[2]


def test_no_ungrounded_or_uncited_gives_empty_reasons():
    gate = ProvenanceCriticGate()
    answer = "Revenue grew to 42 last quarter [s1]."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[42])
    assert v.reasons == []


def test_uncited_claim_text_is_truncated_to_120_chars():
    gate = ProvenanceCriticGate()
    long_sentence = (
        "This is a very long uncited claim sentence that goes on and on " * 3
    )
    v = gate.evaluate(long_sentence.strip() + ".", sources=[], tool_values=[])
    assert v.uncited_claims
    assert len(v.uncited_claims[0]) <= 120


def test_invalid_citations_are_deduplicated_and_sorted():
    gate = ProvenanceCriticGate()
    answer = "Value one [zz]. Value two [aa]. Value three [zz]."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[])
    assert v.invalid_citations == ["aa", "zz"]


def test_completeness_exactly_at_accept_threshold_is_accepted():
    gate = ProvenanceCriticGate(accept_threshold=0.5)
    # One grounded, one ungrounded numeric sentence -> numeric_grounded = 0.5;
    # no substantive claims beyond the numeric ones (both count, short) so
    # claim_grounded contribution is exercised via the numeric-only formula.
    answer = "Total was 4 [s1]. Total was 99."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[4], attempt=0)
    assert v.completeness >= 0.5
    assert v.decision == "accept"


def test_completeness_just_below_threshold_triggers_revise_not_escalate():
    gate = ProvenanceCriticGate(accept_threshold=0.99, max_revise=3)
    answer = "The system achieved 99 percent accuracy on the benchmark."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[], attempt=0)
    assert v.decision == "revise"


def test_attempt_at_max_revise_boundary_escalates():
    gate = ProvenanceCriticGate(max_revise=2)
    answer = "The system achieved 99 percent accuracy on the benchmark."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[], attempt=2)
    assert v.decision == "escalate"


def test_attempt_below_max_revise_boundary_revises():
    gate = ProvenanceCriticGate(max_revise=2)
    answer = "The system achieved 99 percent accuracy on the benchmark."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[], attempt=1)
    assert v.decision == "revise"


def test_invalid_citation_blocks_accept_even_when_completeness_clears_threshold():
    gate = ProvenanceCriticGate(accept_threshold=0.7)
    answer = "Revenue grew to 42 last quarter [s1]. Latency dropped to 5 ms [s9]."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[42, 5], attempt=0)
    assert v.completeness == 0.75
    assert v.completeness >= gate.accept_threshold  # would qualify on score alone
    assert v.decision == "revise"  # blocked purely by the invalid_citations gate


def test_invalid_citation_forces_non_accept_even_with_grounded_number():
    gate = ProvenanceCriticGate()
    # The number is grounded via tool_values (numeric_grounded=1.0), but the
    # citation targets an unknown source, so the sentence has no VALID cite:
    # it also counts as an uncited substantive claim (claim_grounded=0.0).
    # completeness = 0.5*1.0 + 0.5*0.0 = 0.5, and invalid_citations alone is
    # enough to block accept regardless of the completeness score.
    answer = "Latency dropped to 5 ms [unknown-source]."
    v = gate.evaluate(answer, sources=["s1"], tool_values=[5], attempt=0)
    assert v.numeric_grounded == 1.0
    assert v.completeness == 0.5
    assert v.decision != "accept"
