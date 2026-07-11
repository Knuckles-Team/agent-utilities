"""End-to-end epistemic mining flywheel loops (X-3, CONCEPT:AU-KG.evolution.
mining-flywheel), wired through the REAL ``_run_insight_validation`` /
``_run_trace_mining`` loop-controller stages (no reimplemented governance —
these exercise the SAME promotion_governance + action_policy + GovernedAutoMerger
+ OutcomeRouter path ``test_insight_validation.py``/``test_trace_pattern_miner.py``
already cover, plus the NEW flywheel lifecycle overlay + the two closed loops):

* LOOP 1 — ontology-gap (``PredictedEdge``) claim → validated → accepted →
  MATERIALIZED as a real KG edge → outcome fed back through the durable bandit.
* LOOP 2 — process/routing quality (``SequentialPattern``) claim → validated →
  accepted → outcome fed back to the durable bandit (CONCEPT:AU-P1-3), on top
  of the existing in-process ``OutcomeRouter`` feedback.
* A claim denied by ``action_policy`` is RETRACTED and never re-proposed on a
  later cycle over the identical (content-addressed) finding.
* The lifecycle is queryable via ``ClaimFlywheel``.

@pytest.mark.concept("AU-KG.evolution.mining-flywheel")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.research.claim_flywheel import (
    ClaimFlywheel,
    ClaimLifecycleState,
)
from agent_utilities.knowledge_graph.research.loop_controller import LoopController

pytestmark = pytest.mark.concept("AU-KG.evolution.mining-flywheel")


class _FlywheelLoopStubEngine:
    """Combines the existing insight/trace-mining stub shape (``governance_rule``
    lookups relax the ActionPolicy tier) with a REAL ``ClaimLifecycleEvent``
    round-trip (unlike the minimal stubs in the sibling test files) so the
    flywheel's cross-cycle retracted-memory is exercised for real, plus
    ``add_edge`` so LOOP 1's materialization has somewhere to land."""

    def __init__(self, *, governance_rules: list[dict[str, Any]] | None = None):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []
        self.backend = None
        self._governance_rules = governance_rules or []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        self.edges.append((source, target, rel_type, properties))

    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        return {"id": derived_id, "depends_on": [], "generating_activity": None}

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if "governance_rule" in query:
            return [{"r": dict(r)} for r in self._governance_rules]
        if "ClaimLifecycleEvent" in query:
            cid = (params or {}).get("id")
            rows = [
                n
                for n in self.nodes.values()
                if n.get("type") == "ClaimLifecycleEvent" and n.get("claim_id") == cid
            ]
            return [
                {
                    "from_state": r.get("from_state"),
                    "to_state": r.get("to_state"),
                    "reason": r.get("reason"),
                    "actor": r.get("actor"),
                    "governance_valid": r.get("governance_valid"),
                    "action_decision": r.get("action_decision"),
                    "timestamp": r.get("timestamp"),
                }
                for r in rows
            ]
        # RegressionGateResult / CapabilityRatchetResult / ConstitutionRule /
        # Policy lookups all fall through to "no recorded verdict" ⇒ pass.
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


_RELAXED_RULES = [
    {"scope": "action_policy", "kind": "*", "target": "*", "tier": "auto"}
]


def _predicted_edge_mine_result(*, score: float = 0.95) -> dict[str, Any]:
    return {
        "association_rules": {"count": 0, "examples": []},
        "anomalies": {"count": 0, "examples": []},
        "predicted_edges": {
            "count": 1,
            "examples": [
                {
                    "source": "concept:mining",
                    "target": "concept:calibration",
                    "score": score,
                }
            ],
        },
        "errors": [],
    }


# ---------------------------------------------------------------------------
# LOOP 1 — ontology-gap (PredictedEdge) → accept → materialize → outcome
# ---------------------------------------------------------------------------


def test_ontology_gap_claim_accepts_materializes_and_feeds_outcome(monkeypatch):
    from agent_utilities.core.config import config as _cfg

    monkeypatch.setattr(_cfg, "kg_insight_autonomy", True)

    eng = _FlywheelLoopStubEngine(governance_rules=_RELAXED_RULES)
    mine_result = _predicted_edge_mine_result(score=0.95)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["eligible"] == 1
    assert rep["promoted"] == 1

    # the claim's OWN Claim node still flips proposal -> active exactly like
    # every other insight-validation claim (unchanged, well-tested contract).
    claims = eng.by_type("Claim")
    assert len(claims) == 1
    assert claims[0]["status"] == "active"
    assert claims[0]["is_verified"] is True
    claim_id = claims[0]["id"]

    # LOOP 1 closes: the predicted edge is MATERIALIZED as a real KG edge.
    # (X-6 / Seam 3, CONCEPT:EG-KG.epistemic.truth-maintenance: the claim ALSO
    # writes 2 `:DerivedFrom` provenance edges to its own source_ids at propose
    # time -- unconditional, unrelated to this promotion-gated materialize step
    # -- so `eng.edges` carries those too; asserted separately below.)
    materialized = [e for e in eng.edges if e[2] == "PREDICTED_RELATION"]
    assert len(materialized) == 1
    src, dst, rel_type, props = materialized[0]
    assert (src, dst, rel_type) == (
        "concept:mining",
        "concept:calibration",
        "PREDICTED_RELATION",
    )
    assert props["claim_id"] == claim_id

    # The propose-time provenance edges landed too, from the claim to its own
    # mined source_ids (the SAME src/dst the predicted edge itself connects).
    derived_from = {
        (s, d)
        for s, d, _, props in eng.edges
        if s == claim_id and props.get("relationship_type") == "DERIVED_FROM"
    }
    assert derived_from == {
        (claim_id, "concept:mining"),
        (claim_id, "concept:calibration"),
    }

    # the flywheel's OWN lifecycle overlay independently agrees: accepted.
    flywheel = ClaimFlywheel(eng)
    assert flywheel.current_state(claim_id) == ClaimLifecycleState.ACCEPTED
    history = [e["to_state"] for e in flywheel.history(claim_id)]
    assert history == ["proposed", "validated", "accepted"]

    # the acceptance's outcome was captured as an observation.
    outcomes = eng.by_type("ClaimOutcome")
    assert len(outcomes) == 1
    assert outcomes[0]["claim_id"] == claim_id
    assert outcomes[0]["reward"] == pytest.approx(0.95)


def test_predicted_edge_not_promoted_stays_out_of_materialize_path():
    """The shipped default (no relaxed policy, autonomy off) never promotes —
    so LOOP 1's materialize step must not fire at all."""
    eng = _FlywheelLoopStubEngine()  # shipped default: approval_required
    mine_result = _predicted_edge_mine_result(score=0.95)
    rep = LoopController(eng)._run_insight_validation(mine_result)

    assert rep["promoted"] == 0
    # LOOP 1's materialize step (the `PREDICTED_RELATION` edge) must not fire.
    assert not any(rel == "PREDICTED_RELATION" for _, _, rel, _ in eng.edges)
    assert eng.by_type("ClaimOutcome") == []
    claims = eng.by_type("Claim")
    assert claims[0]["status"] == "proposal"
    claim_id = claims[0]["id"]

    # The propose-time `:DerivedFrom` provenance edges (X-6 / Seam 3) still land
    # regardless -- unconditional, unrelated to promotion.
    assert {
        (s, d)
        for s, d, _, props in eng.edges
        if props.get("relationship_type") == "DERIVED_FROM"
    } == {
        (claim_id, "concept:mining"),
        (claim_id, "concept:calibration"),
    }


# ---------------------------------------------------------------------------
# LOOP 2 — process/routing quality (SequentialPattern) → accept → outcome →
# durable bandit
# ---------------------------------------------------------------------------


def _patch_trace_mine_result(monkeypatch, *, support: float = 0.9) -> None:
    import agent_utilities.knowledge_graph.research.trace_pattern_miner as tpm

    monkeypatch.setattr(
        tpm,
        "mine_trace_patterns",
        lambda engine, **kw: {
            "patterns": {
                "patterns": [
                    {"items": ["Read", "Edit"], "support": support, "count": 4}
                ]
            },
            "failure_episodes": 4,
            "sequences_mined": 4,
            "errors": [],
        },
    )


def test_routing_claim_accepts_and_feeds_durable_bandit(monkeypatch):
    _patch_trace_mine_result(monkeypatch, support=0.9)
    eng = _FlywheelLoopStubEngine(governance_rules=_RELAXED_RULES)
    rep = LoopController(eng)._run_trace_mining()

    assert rep["routed"] == 1
    claims = eng.by_type("Claim")
    assert claims[0]["status"] == "proposal"  # unchanged legacy contract
    claim_id = claims[0]["id"]

    flywheel = ClaimFlywheel(eng)
    assert flywheel.current_state(claim_id) == ClaimLifecycleState.ACCEPTED

    # LOOP 2 closes: the outcome was captured, and the claim's OWN reward
    # (support=0.9 — a well-evidenced claim) is independent of the bandit's
    # deliberately negative durable_reward (the routed-away-from tool choice).
    outcomes = eng.by_type("ClaimOutcome")
    assert len(outcomes) == 1
    assert outcomes[0]["reward"] == pytest.approx(0.9)
    assert outcomes[0]["durable_reward"] == pytest.approx(0.0)
    # a confidently-mined claim does not auto-deprecate itself.
    assert flywheel.current_state(claim_id) == ClaimLifecycleState.ACCEPTED


def test_routing_claim_not_routed_by_default_never_reaches_flywheel_accept(
    monkeypatch,
):
    _patch_trace_mine_result(monkeypatch, support=0.9)
    eng = _FlywheelLoopStubEngine()  # shipped default: approval_required

    rep = LoopController(eng)._run_trace_mining()

    assert rep["routed"] == 0
    claim_id = eng.by_type("Claim")[0]["id"]
    flywheel = ClaimFlywheel(eng)
    # validated (governance passes with support 0.9), but never accepted —
    # the shipped route_policy_update tier queues, never auto-allows.
    assert flywheel.current_state(claim_id) == ClaimLifecycleState.VALIDATED
    assert eng.by_type("ClaimOutcome") == []


# ---------------------------------------------------------------------------
# A retracted claim is never re-proposed across cycles
# ---------------------------------------------------------------------------


def test_denied_claim_is_retracted_and_not_reprocessed_on_a_later_cycle(monkeypatch):
    import agent_utilities.orchestration.action_policy as ap_mod
    from agent_utilities.orchestration.action_policy import ActionDecision

    # Force a hard DENY (e.g. a blast-radius/rate-limit breach) for this
    # cycle's promote_mined_claim decision, regardless of the shipped tier.
    def _deny(self, request):
        return ActionDecision(
            decision="deny",
            tier="approval_required",
            request=request,
            reason="rate limit exceeded",
        )

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", _deny)

    eng = _FlywheelLoopStubEngine()
    mine_result = _predicted_edge_mine_result(score=0.95)

    rep1 = LoopController(eng)._run_insight_validation(mine_result)
    assert rep1["persisted_claims"] == 1
    claim_id = eng.by_type("Claim")[0]["id"]

    flywheel = ClaimFlywheel(eng)
    assert flywheel.current_state(claim_id) == ClaimLifecycleState.RETRACTED

    claims_before = len(eng.by_type("Claim"))
    events_before = len(eng.by_type("ClaimLifecycleEvent"))

    # A SECOND, independent cycle re-mines the IDENTICAL (content-addressed)
    # finding — a fresh LoopController, fresh in-process state, same engine.
    rep2 = LoopController(eng)._run_insight_validation(mine_result)

    assert rep2["persisted_claims"] == 0  # never re-persisted as a fresh proposal
    assert any(ex.get("skipped") == "retracted" for ex in rep2["examples"])
    # no new Claim / lifecycle event was minted for the refused re-proposal.
    assert len(eng.by_type("Claim")) == claims_before
    assert len(eng.by_type("ClaimLifecycleEvent")) == events_before
    assert flywheel.current_state(claim_id) == ClaimLifecycleState.RETRACTED
