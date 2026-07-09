"""Confidence propagation + light TMS over BeliefNode graphs
(CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision, workstream C2).

Covers the propose-only contract (never mutates a BeliefNode, never violates the
support/contradict mutex), the explainable log-odds formula (more support raises
confidence, more contradiction lowers it, no evidence leaves it ~unchanged),
last_reviewed bumping, the itemized reasoning trace, contradicted_by_node_ids
population from fresh ContradictionDetector friction, and the engine-delegation
seam's graceful fallback. No LLM, no network, fully deterministic.

@pytest.mark.concept("AU-KG.maintenance.confidence-propagation-belief-revision")
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.adaptation.belief_revision import (
    BeliefRevision,
    BeliefRevisionPass,
    apply_revision,
    explain_revision,
    recompute_confidence,
)
from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (
    ContradictionDetector,
)
from agent_utilities.models.knowledge_graph import BeliefNode, RegistryNodeType

pytestmark = pytest.mark.concept(
    "AU-KG.maintenance.confidence-propagation-belief-revision"
)


def _belief(
    belief_id: str,
    statement: str,
    confidence: float,
    *,
    supported_by: list[str] | None = None,
    contradicted_by: list[str] | None = None,
    last_reviewed: str = "2026-01-01T00:00:00+00:00",
) -> BeliefNode:
    return BeliefNode(
        id=belief_id,
        type=RegistryNodeType.BELIEF,
        name=belief_id,
        statement=statement,
        confidence=confidence,
        supported_by_node_ids=supported_by or [],
        contradicted_by_node_ids=contradicted_by or [],
        last_reviewed=last_reviewed,
    )


# ── recompute_confidence() ────────────────────────────────────────────────


def test_recompute_confidence_no_evidence_is_a_no_op() -> None:
    belief = _belief("b1", "caching improves performance", 0.6)
    new_confidence = recompute_confidence(belief, [], [])
    assert new_confidence == pytest.approx(0.6, abs=1e-9)


def test_recompute_confidence_more_support_raises_confidence() -> None:
    belief = _belief("b1", "caching improves performance", 0.5)
    one_support = [_belief("s1", "caching is fast", 0.8)]
    two_support = [
        _belief("s1", "caching is fast", 0.8),
        _belief("s2", "caching reduces latency", 0.8),
    ]
    baseline = recompute_confidence(belief, [], [])
    with_one = recompute_confidence(belief, one_support, [])
    with_two = recompute_confidence(belief, two_support, [])
    assert baseline < with_one < with_two <= 1.0


def test_recompute_confidence_more_contradiction_lowers_confidence() -> None:
    belief = _belief("b1", "caching improves performance", 0.7)
    one_contra = [_belief("c1", "caching degrades performance", 0.8)]
    two_contra = [
        _belief("c1", "caching degrades performance", 0.8),
        _belief("c2", "caching adds overhead", 0.8),
    ]
    baseline = recompute_confidence(belief, [], [])
    with_one = recompute_confidence(belief, [], one_contra)
    with_two = recompute_confidence(belief, [], two_contra)
    assert 0.0 <= with_two < with_one < baseline


def test_recompute_confidence_stronger_evidence_moves_more() -> None:
    belief = _belief("b1", "x is true", 0.5)
    weak_support = [_belief("s1", "y supports x", 0.55)]
    strong_support = [_belief("s1", "y supports x", 0.95)]
    weak_result = recompute_confidence(belief, weak_support, [])
    strong_result = recompute_confidence(belief, strong_support, [])
    assert weak_result < strong_result


def test_recompute_confidence_stays_in_bounds_under_heavy_contradiction() -> None:
    belief = _belief("b1", "x is true", 0.9)
    contradicting = [
        _belief(f"c{i}", "x is false", 0.99) for i in range(20)
    ]
    result = recompute_confidence(belief, [], contradicting)
    assert 0.0 <= result < 0.05


def test_recompute_confidence_stays_in_bounds_under_heavy_support() -> None:
    belief = _belief("b1", "x is true", 0.1)
    supporting = [_belief(f"s{i}", "x is true", 0.99) for i in range(20)]
    result = recompute_confidence(belief, supporting, [])
    assert 0.95 < result <= 1.0


def test_recompute_confidence_is_deterministic() -> None:
    belief = _belief("b1", "x is true", 0.5)
    support = [_belief("s1", "y", 0.7)]
    contra = [_belief("c1", "z", 0.6)]
    results = {recompute_confidence(belief, support, contra) for _ in range(5)}
    assert len(results) == 1


# ── explain_revision() ────────────────────────────────────────────────────


def test_explain_revision_itemizes_each_edge_plus_summary() -> None:
    belief = _belief("b1", "x is true", 0.5)
    support = [_belief("s1", "y", 0.8)]
    contra = [_belief("c1", "z", 0.6)]
    new_confidence = recompute_confidence(belief, support, contra)
    trace = explain_revision(belief, support, contra, new_confidence)

    assert len(trace) == 3  # 1 support row + 1 contradict row + 1 summary row
    roles = [row["role"] for row in trace]
    assert roles == ["support", "contradict", "summary"]

    support_row = trace[0]
    assert support_row["node_id"] == "s1"
    assert support_row["log_odds_contribution"] > 0

    contradict_row = trace[1]
    assert contradict_row["node_id"] == "c1"
    assert contradict_row["log_odds_contribution"] < 0

    summary_row = trace[2]
    assert summary_row["node_id"] == "b1"
    assert summary_row["old_confidence"] == pytest.approx(0.5)
    assert summary_row["new_confidence"] == pytest.approx(new_confidence)
    assert summary_row["delta"] == pytest.approx(new_confidence - 0.5, abs=1e-6)


def test_explain_revision_is_json_serializable() -> None:
    belief = _belief("b1", "x is true", 0.5)
    support = [_belief("s1", "y", 0.8)]
    new_confidence = recompute_confidence(belief, support, [])
    trace = explain_revision(belief, support, [], new_confidence)
    json.dumps(trace)  # must not raise


# ── apply_revision() ──────────────────────────────────────────────────────


def test_apply_revision_never_mutates_the_input_belief() -> None:
    belief = _belief("b1", "x is true", 0.5)
    revision = BeliefRevision(
        belief_id="b1",
        old_confidence=0.5,
        new_confidence=0.9,
        new_contradicted_by_node_ids=[],
        last_reviewed="2026-02-02T00:00:00+00:00",
        reasoning_trace=[],
    )
    revised = apply_revision(belief, revision)
    assert belief.confidence == 0.5  # original untouched
    assert belief.last_reviewed == "2026-01-01T00:00:00+00:00"
    assert revised.confidence == 0.9
    assert revised is not belief


def test_apply_revision_bumps_last_reviewed() -> None:
    belief = _belief("b1", "x is true", 0.5, last_reviewed="2026-01-01T00:00:00+00:00")
    revision = BeliefRevision(
        belief_id="b1",
        old_confidence=0.5,
        new_confidence=0.5,
        new_contradicted_by_node_ids=[],
        last_reviewed="2099-12-31T00:00:00+00:00",
        reasoning_trace=[],
    )
    revised = apply_revision(belief, revision)
    assert revised.last_reviewed == "2099-12-31T00:00:00+00:00"


def test_apply_revision_never_violates_support_contradict_mutex() -> None:
    belief = _belief(
        "b1",
        "x is true",
        0.5,
        supported_by=["n1", "n2"],
    )
    # n1 flips from supporting to contradicting in this revision.
    revision = BeliefRevision(
        belief_id="b1",
        old_confidence=0.5,
        new_confidence=0.3,
        new_contradicted_by_node_ids=["n1"],
        last_reviewed="2026-01-02T00:00:00+00:00",
        reasoning_trace=[],
    )
    revised = apply_revision(belief, revision)
    assert revised.contradicted_by_node_ids == ["n1"]
    assert revised.supported_by_node_ids == ["n2"]
    assert not (
        set(revised.supported_by_node_ids) & set(revised.contradicted_by_node_ids)
    )
    # Pydantic's own mutex validator ran successfully (no raise) — belt and suspenders.


# ── BeliefRevisionPass.check() ────────────────────────────────────────────


def test_pass_check_bumps_last_reviewed_via_injected_clock() -> None:
    belief = _belief("b1", "x is true", 0.5, last_reviewed="2020-01-01T00:00:00+00:00")
    revision_pass = BeliefRevisionPass(now_fn=lambda: "2030-06-15T00:00:00+00:00")
    revision = revision_pass.check(belief, [], [])
    assert revision.last_reviewed == "2030-06-15T00:00:00+00:00"
    assert revision.last_reviewed != belief.last_reviewed


def test_pass_check_records_contradicting_ids() -> None:
    belief = _belief("b1", "x is true", 0.6)
    contra = [_belief("c1", "x is false", 0.7), _belief("c2", "x is wrong", 0.7)]
    revision = BeliefRevisionPass().check(belief, [], contra)
    assert revision.new_contradicted_by_node_ids == ["c1", "c2"]
    assert revision.new_confidence < revision.old_confidence


# ── BeliefRevisionPass.scan() ──────────────────────────────────────────────


def test_scan_populates_contradicted_by_from_fresh_friction() -> None:
    # High topical overlap ⇒ "high" severity, comfortably above the default
    # "medium" threshold.
    beliefs = [
        _belief(
            "a", "the new caching layer clearly improves database performance", 0.8
        ),
        _belief(
            "b", "the new caching layer clearly degrades database performance", 0.8
        ),
    ]
    revisions = BeliefRevisionPass().scan(beliefs)
    by_id = {r.belief_id: r for r in revisions}
    assert by_id["a"].new_contradicted_by_node_ids == ["b"]
    assert by_id["b"].new_contradicted_by_node_ids == ["a"]
    # Mutual contradiction of similar-strength beliefs should lower both.
    assert by_id["a"].new_confidence < 0.8
    assert by_id["b"].new_confidence < 0.8


def test_scan_leaves_agreeing_beliefs_uncontradicted() -> None:
    beliefs = [
        _belief("a", "Caching improves performance", 0.8),
        _belief("b", "Caching reduces database load", 0.8),
    ]
    revisions = BeliefRevisionPass().scan(beliefs)
    for revision in revisions:
        assert revision.new_contradicted_by_node_ids == []
        assert revision.new_confidence == pytest.approx(0.8, abs=1e-9)


def test_scan_uses_recorded_support_edges() -> None:
    beliefs = [
        _belief("a", "x is true", 0.5, supported_by=["s"]),
        _belief("s", "y confirms x", 0.9),
    ]
    revisions = BeliefRevisionPass().scan(beliefs)
    by_id = {r.belief_id: r for r in revisions}
    assert by_id["a"].new_confidence > 0.5


def test_scan_never_produces_overlapping_support_and_contradiction() -> None:
    # 'a' both nominally supported_by 'x' AND fresh-friction-contradicted by 'x'
    # (contrived, but exercises the mutex-preservation branch in scan()). Uses a
    # lenient severity_threshold so the (low-severity) friction is picked up.
    beliefs = [
        _belief("a", "Caching improves performance", 0.6, supported_by=["x"]),
        _belief("x", "Caching degrades performance", 0.6),
    ]
    revisions = BeliefRevisionPass(severity_threshold="low").scan(beliefs)
    by_id = {r.belief_id: r for r in revisions}
    a_revision = by_id["a"]
    # 'x' must land as a contradiction, never simultaneously counted as support.
    assert "x" in a_revision.new_contradicted_by_node_ids
    revised_a = apply_revision(beliefs[0], a_revision)
    assert not (
        set(revised_a.supported_by_node_ids)
        & set(revised_a.contradicted_by_node_ids)
    )


def test_scan_respects_severity_threshold() -> None:
    # Short, sparse-overlap opposing pair ⇒ "low" severity (confirmed via
    # ContradictionDetector directly: similarity 0.35, band "low"). A "high"
    # threshold must NOT populate contradicted_by; "low" must.
    beliefs = [
        _belief("a", "Caching improves performance", 0.7),
        _belief("b", "Caching degrades performance", 0.7),
    ]
    lenient = BeliefRevisionPass(severity_threshold="low").scan(beliefs)
    strict = BeliefRevisionPass(severity_threshold="high").scan(beliefs)
    lenient_by_id = {r.belief_id: r for r in lenient}
    strict_by_id = {r.belief_id: r for r in strict}
    assert lenient_by_id["a"].new_contradicted_by_node_ids == ["b"]
    assert strict_by_id["a"].new_contradicted_by_node_ids == []


def test_scan_ignores_unrelated_beliefs() -> None:
    beliefs = [
        _belief("a", "Caching improves performance", 0.8),
        _belief("b", "Caching degrades performance", 0.8),
        _belief("e", "lithium powers electric vehicles", 0.5),
    ]
    revisions = BeliefRevisionPass().scan(beliefs)
    by_id = {r.belief_id: r for r in revisions}
    assert by_id["e"].new_contradicted_by_node_ids == []
    assert by_id["e"].new_confidence == pytest.approx(0.5, abs=1e-9)


def test_scan_is_deterministic_and_sorted_by_belief_id() -> None:
    beliefs = [
        _belief("z", "Caching improves performance", 0.8),
        _belief("a", "Caching degrades performance", 0.8),
        _belief("m", "lithium powers electric vehicles", 0.5),
    ]
    runs = [
        [(r.belief_id, r.new_confidence) for r in BeliefRevisionPass().scan(beliefs)]
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs)
    ids = [belief_id for belief_id, _ in runs[0]]
    assert ids == sorted(ids)


def test_scan_never_escapes_no_exception_for_empty_input() -> None:
    assert BeliefRevisionPass().scan([]) == []


def test_scan_reuses_injected_contradiction_detector() -> None:
    calls: list[tuple[str, str]] = []

    class _RecordingDetector(ContradictionDetector):
        def scan(self, claims):  # type: ignore[override]
            calls.append(tuple(c.id for c in claims))
            return super().scan(claims)

    beliefs = [
        _belief("a", "Caching improves performance", 0.8),
        _belief("b", "Caching degrades performance", 0.8),
    ]
    BeliefRevisionPass(contradiction_detector=_RecordingDetector()).scan(beliefs)
    assert calls, "injected detector.scan() was not called"


# ── engine-delegation seam (graceful fallback) ────────────────────────────


def test_use_engine_falls_back_to_local_formula_when_surface_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No 'epistemic' engine surface exists yet — the delegation seam must
    degrade to the local Python formula, never raise, never return None
    confidence."""
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    def fake_invoke(*, surface, action, graph, candidates, params):
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "degraded": True,
                "error": f"engine surface {surface!r} is not available",
            }
        )

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    belief = _belief("b1", "x is true", 0.5)
    support = [_belief("s1", "y", 0.8)]
    local_only = BeliefRevisionPass(use_engine=False).check(belief, support, [])
    delegated_but_degraded = BeliefRevisionPass(use_engine=True).check(
        belief, support, []
    )
    assert delegated_but_degraded.new_confidence == pytest.approx(
        local_only.new_confidence
    )


def test_use_engine_prefers_live_surface_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the engine DOES answer with a well-formed payload, the pass uses it
    instead of the local formula."""
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    def fake_invoke(*, surface, action, graph, candidates, params):
        assert surface == "epistemic"
        assert action == "propagate"
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "result": {
                    "new_confidence": 0.42,
                    "reasoning_trace": [{"role": "summary", "new_confidence": 0.42}],
                },
            }
        )

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    belief = _belief("b1", "x is true", 0.5)
    revision = BeliefRevisionPass(use_engine=True).check(belief, [], [])
    assert revision.new_confidence == pytest.approx(0.42)
    assert revision.reasoning_trace == [{"role": "summary", "new_confidence": 0.42}]
