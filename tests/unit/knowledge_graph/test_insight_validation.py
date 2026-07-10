"""Insight Engine closed loop — the ``_run_insight_validation`` loop-controller
stage (CONCEPT:AU-KG.evolution.insight-engine-closed-loop, workstream C4).

Mine → CandidateInsight → EvidenceBundle → Claim → Validation (reuses
promotion_governance + capability_ratchet as-is) → Action gate (reuses
action_policy.decide(), kind="promote_mined_claim"). Mirrors the ``_mine_*``/
belief-revision sub-step best-effort tolerance already established in
``test_loop_controller.py``.

@pytest.mark.concept("AU-KG.evolution.insight-engine-closed-loop")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.research.loop_controller import LoopController

pytestmark = pytest.mark.concept("AU-KG.evolution.insight-engine-closed-loop")


class _InsightStubEngine:
    """Minimal engine double: records ``add_node`` calls, canned ``query_cypher``.

    ``governance_rule`` rows (scope='action_policy') let a test relax BOTH the
    ``promote_mined_claim`` action-policy kind AND the ``merge_promotion`` kind
    the reused ``GovernedAutoMerger`` separately consults — X3 autonomy only
    ever promotes when both are exercised.
    """

    def __init__(self, *, governance_rules: list[dict[str, Any]] | None = None):
        # Upsert-keyed-by-id, matching a real engine: a second ``add_node`` for the
        # same id (e.g. the promoter flipping proposal → active) overwrites the
        # node in place rather than accumulating a duplicate.
        self.nodes: dict[str, dict[str, Any]] = {}
        self.add_node_calls = 0
        self.backend = object()
        self._governance_rules = governance_rules or []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.add_node_calls += 1
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "governance_rule" in q:
            return [{"r": dict(r)} for r in self._governance_rules]
        # RegressionGateResult / CapabilityRatchetResult / ConstitutionRule /
        # Policy lookups all fall through to "no recorded verdict" ⇒ pass.
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


def _mine_result(
    *, association_confidence: float = 0.95, anomaly_score: float = 6.0
) -> dict[str, Any]:
    return {
        "association_rules": {
            "count": 1,
            "examples": [
                {
                    "antecedent": ["concept:cA"],
                    "consequent": ["capability:capZ"],
                    "confidence": association_confidence,
                    "lift": 1.5,
                }
            ],
        },
        "anomalies": {
            "count": 1,
            "examples": [
                {
                    "capability": "cap:weak",
                    "covered_concepts": 0.0,
                    "anomaly_score": anomaly_score,
                }
            ],
        },
        "predicted_edges": {"count": 0, "examples": []},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# (a) below-floor findings never become claims — propose-only is preserved
# ---------------------------------------------------------------------------


def test_below_floor_findings_never_become_claims():
    eng = _InsightStubEngine()
    # association confidence well below CONFIDENCE_FLOOR (0.6)
    mine_result = _mine_result(association_confidence=0.1, anomaly_score=0.1)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["eligible"] == 0
    assert rep["below_floor"] == 2
    assert rep["persisted_claims"] == 0
    assert rep["promoted"] == 0
    assert eng.by_type("Claim") == []


def test_mixed_floor_findings_only_persist_the_eligible_one():
    eng = _InsightStubEngine()
    # association clears the floor (0.95 >= 0.6), anomaly does not (score 0.1
    # saturates to 0.02 confidence).
    mine_result = _mine_result(association_confidence=0.95, anomaly_score=0.1)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["candidates"] == 2
    assert rep["eligible"] == 1
    assert rep["below_floor"] == 1
    assert rep["persisted_claims"] == 1
    claims = eng.by_type("Claim")
    assert len(claims) == 1
    assert claims[0]["status"] == "proposal"
    assert claims[0]["is_verified"] is False


# ---------------------------------------------------------------------------
# (b) action_policy.decide() is consulted before any promotion
# ---------------------------------------------------------------------------


def test_action_policy_is_consulted_for_every_eligible_claim(monkeypatch):
    import agent_utilities.orchestration.action_policy as ap_mod

    calls: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        calls.append(request.kind)
        return real_decide(self, request)

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)

    eng = _InsightStubEngine()
    mine_result = _mine_result(association_confidence=0.95, anomaly_score=6.0)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["eligible"] == 2
    assert calls.count("promote_mined_claim") == 2  # one per eligible finding
    # Default shipped tier ⇒ queued, never silently allowed.
    for ex in rep["examples"]:
        assert ex["action_decision"] == "queue_approval"
        assert ex["promoted"] is False


def test_shipped_default_never_promotes_even_with_autonomy_flag_alone(monkeypatch):
    """Turning on KG_INSIGHT_AUTONOMY alone (no relaxed ActionPolicy tier) must
    NOT promote anything — action_policy.decide() still queues by default."""
    from agent_utilities.core.config import config as _cfg

    monkeypatch.setattr(_cfg, "kg_insight_autonomy", True)

    eng = _InsightStubEngine()  # no governance_rule overrides ⇒ shipped default
    mine_result = _mine_result(association_confidence=0.95, anomaly_score=6.0)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["autonomy_enabled"] is True
    assert rep["promoted"] == 0
    for ex in rep["examples"]:
        assert ex["action_decision"] == "queue_approval"
        assert ex["promoted"] is False
    # Claims are still persisted as proposals — propose-only floor intact.
    claims = eng.by_type("Claim")
    assert len(claims) == 2
    assert all(c["status"] == "proposal" for c in claims)


# ---------------------------------------------------------------------------
# (c) the autonomous tier only auto-promotes when EXPLICITLY configured
# (autonomy flag ON *and* the action-policy tier relaxed)
# ---------------------------------------------------------------------------


def test_autonomy_off_by_default_never_promotes():
    """Default config: KG_INSIGHT_AUTONOMY is False, so even a relaxed policy
    tier must not promote anything."""
    eng = _InsightStubEngine(
        governance_rules=[
            {"scope": "action_policy", "kind": "*", "target": "*", "tier": "auto"}
        ]
    )
    mine_result = _mine_result(association_confidence=0.95, anomaly_score=6.0)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["autonomy_enabled"] is False
    assert rep["promoted"] == 0
    assert all(c["status"] == "proposal" for c in eng.by_type("Claim"))


def test_autonomy_on_and_relaxed_policy_together_promote(monkeypatch):
    """Both keys must turn: KG_INSIGHT_AUTONOMY=1 AND an operator-relaxed
    promote_mined_claim (+ merge_promotion, which GovernedAutoMerger separately
    re-checks) tier. Only then does the reused GovernedAutoMerger flip a claim
    proposal → active."""
    from agent_utilities.core.config import config as _cfg

    monkeypatch.setattr(_cfg, "kg_insight_autonomy", True)

    eng = _InsightStubEngine(
        governance_rules=[
            {"scope": "action_policy", "kind": "*", "target": "*", "tier": "auto"}
        ]
    )
    mine_result = _mine_result(association_confidence=0.95, anomaly_score=6.0)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["autonomy_enabled"] is True
    assert rep["promoted"] == 2
    claims = eng.by_type("Claim")
    assert len(claims) == 2
    assert all(c["status"] == "active" for c in claims)
    assert all(c["is_verified"] is True for c in claims)


# ---------------------------------------------------------------------------
# Best-effort tolerance (mirrors the _mine_*/belief-revision sub-step pattern)
# ---------------------------------------------------------------------------


def test_persist_failure_is_tolerated_per_candidate():
    class _FailingEngine(_InsightStubEngine):
        def add_node(self, node_id, node_type, properties=None):
            if node_type == "Claim":
                raise RuntimeError("kg unreachable")
            super().add_node(node_id, node_type, properties)

    eng = _FailingEngine()
    mine_result = _mine_result(association_confidence=0.95, anomaly_score=6.0)
    rep = LoopController(eng)._run_insight_validation(mine_result)  # must not raise

    assert rep["persisted_claims"] == 0
    assert any("insight_validation:persist" in e for e in rep["errors"])


def test_empty_mine_result_is_a_clean_no_op():
    eng = _InsightStubEngine()
    rep = LoopController(eng)._run_insight_validation({})
    assert rep["candidates"] == 0
    assert rep["persisted_claims"] == 0
    assert rep["errors"] == []


# ---------------------------------------------------------------------------
# run_one_cycle wiring: gated stage placed after mine_discovery
# ---------------------------------------------------------------------------


def test_run_one_cycle_wires_insight_validation_after_mine_discovery(monkeypatch):
    import agent_utilities.knowledge_graph.research.loop_controller as loop_controller

    def fake_mine_discovery(self):
        return _mine_result(association_confidence=0.95, anomaly_score=6.0)

    monkeypatch.setattr(
        loop_controller.LoopController, "_run_mine_discovery", fake_mine_discovery
    )

    eng = _InsightStubEngine()
    rep = LoopController(eng).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        belief_revision=False,
    )
    assert rep["insight_validation"] is not None
    assert rep["insight_validation"]["eligible"] == 2

    # Explicitly disabled ⇒ stage does not run at all.
    rep_disabled = LoopController(eng).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        belief_revision=False,
        insight_validation=False,
    )
    assert rep_disabled["insight_validation"] is None
