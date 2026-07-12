"""The epistemic mining flywheel — the claim lifecycle state machine (X-3,
CONCEPT:AU-KG.evolution.mining-flywheel).

Unit tests for :mod:`agent_utilities.knowledge_graph.research.claim_flywheel`
in isolation: the five-state machine (proposed → validated → accepted →
deprecated → retracted), the queryable transition history, the durable
rejection memory (a retracted claim is never re-proposed), and the
outcome→observation→durable-bandit feedback closing.

@pytest.mark.concept("AU-KG.evolution.mining-flywheel")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.research.claim_flywheel import (
    ClaimFlywheel,
    ClaimLifecycleState,
    IllegalTransition,
)

pytestmark = pytest.mark.concept("AU-KG.evolution.mining-flywheel")


class _FlywheelStubEngine:
    """A competent-enough double: ``ClaimLifecycleEvent``/``ClaimOutcome`` nodes
    round-trip through ``query_cypher`` (unlike the minimal stubs elsewhere in
    this test suite), so cross-instance (cross-cycle) state reads are exercised
    for real, not just the in-process cache."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []
        self.backend = None  # no durable capability-reward backend wired

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        self.edges.append((source, target, rel_type, properties))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


# ---------------------------------------------------------------------------
# The state machine itself
# ---------------------------------------------------------------------------


def test_fresh_claim_defaults_to_proposed():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    assert fw.current_state("claim:1") == ClaimLifecycleState.PROPOSED
    assert fw.history("claim:1") == []


def test_propose_records_initial_event():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    record = fw.propose("claim:1", reason="mined AssociationRule finding")
    assert record is not None
    assert record.from_state == ""
    assert record.to_state == "proposed"
    events = eng.by_type("ClaimLifecycleEvent")
    assert len(events) == 1
    assert events[0]["claim_id"] == "claim:1"


def test_full_lifecycle_advances_in_order():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True, reason="governance passed")
    assert fw.current_state("claim:1") == ClaimLifecycleState.VALIDATED
    fw.accept("claim:1", reason="promoted", action_decision="allow")
    assert fw.current_state("claim:1") == ClaimLifecycleState.ACCEPTED
    fw.deprecate("claim:1", reason="drifted")
    assert fw.current_state("claim:1") == ClaimLifecycleState.DEPRECATED
    fw.retract("claim:1", reason="finally removed")
    assert fw.current_state("claim:1") == ClaimLifecycleState.RETRACTED

    # queryable: the full chronological history is visible.
    history = fw.history("claim:1")
    assert [e["to_state"] for e in history] == [
        "proposed",
        "validated",
        "accepted",
        "deprecated",
        "retracted",
    ]


def test_illegal_transition_raises_and_never_silently_skips():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    # PROPOSED cannot jump straight to ACCEPTED — must pass through VALIDATED.
    with pytest.raises(IllegalTransition):
        fw.accept("claim:1")
    # And the illegal attempt did not silently change the recorded state.
    assert fw.current_state("claim:1") == ClaimLifecycleState.PROPOSED


def test_validate_false_holds_without_retracting():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    result = fw.validate("claim:1", False, reason="shacl: conforms=false")
    assert result is None  # no state-advancing transition returned
    assert fw.current_state("claim:1") == ClaimLifecycleState.PROPOSED
    history = fw.history("claim:1")
    # the held attempt IS recorded (audit-visible), not silent.
    assert any(e["reason"] == "shacl: conforms=false" for e in history)


def test_reject_retracts_from_proposed_or_validated():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.reject("claim:1", reason="constitution forbid rule matched")
    assert fw.current_state("claim:1") == ClaimLifecycleState.RETRACTED


# ---------------------------------------------------------------------------
# Durable rejection memory: a retracted claim is never re-proposed
# ---------------------------------------------------------------------------


def test_propose_refuses_to_reopen_a_retracted_claim_same_instance():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.reject("claim:1", reason="denied")
    before = len(eng.by_type("ClaimLifecycleEvent"))

    result = fw.propose("claim:1", reason="re-mined identical finding")
    assert result is None
    assert fw.current_state("claim:1") == ClaimLifecycleState.RETRACTED
    # no new event was minted for the refused re-proposal.
    assert len(eng.by_type("ClaimLifecycleEvent")) == before


def test_propose_refuses_to_reopen_a_retracted_claim_new_cycle():
    """A FRESH ClaimFlywheel instance (a new mining cycle) must still see the
    durable rejection — this is the cross-process/cross-cycle path, which
    depends on the engine's query_cypher reflecting the earlier writes (this
    stub engine does; a real engine always does)."""
    eng = _FlywheelStubEngine()
    ClaimFlywheel(eng).propose("claim:1")
    ClaimFlywheel(eng).reject("claim:1", reason="denied cycle 1")

    fresh = ClaimFlywheel(eng)
    assert fresh.is_retracted("claim:1") is True
    assert fresh.propose("claim:1", reason="re-mined cycle 2") is None
    assert fresh.current_state("claim:1") == ClaimLifecycleState.RETRACTED


# ---------------------------------------------------------------------------
# Outcome → observation → durable feedback
# ---------------------------------------------------------------------------


def test_record_outcome_persists_a_queryable_observation():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    outcome = fw.record_outcome("claim:1", reward=0.9, note="materialized")
    assert outcome["reward"] == pytest.approx(0.9)
    outcomes = eng.by_type("ClaimOutcome")
    assert len(outcomes) == 1
    assert outcomes[0]["claim_id"] == "claim:1"
    assert outcomes[0]["note"] == "materialized"
    # a good outcome does not deprecate an accepted claim.
    assert outcome["deprecated"] is False
    assert fw.current_state("claim:1") == ClaimLifecycleState.ACCEPTED


def test_record_outcome_auto_deprecates_accepted_claim_on_bad_reward():
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    outcome = fw.record_outcome("claim:1", reward=0.05, note="drifted")
    assert outcome["deprecated"] is True
    assert fw.current_state("claim:1") == ClaimLifecycleState.DEPRECATED


def test_record_outcome_reward_and_durable_reward_are_independent():
    """A well-supported claim (high ``reward``) must not auto-deprecate just
    because it teaches the bandit something negative (low ``durable_reward``,
    e.g. a repeated-failure routing pattern the claim itself is confidently
    mined from)."""
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    outcome = fw.record_outcome(
        "claim:1", reward=0.9, durable_reward=0.0, note="routed away from"
    )
    assert outcome["reward"] == pytest.approx(0.9)
    assert outcome["bandit_reward"] == pytest.approx(0.0)
    assert outcome["deprecated"] is False
    assert fw.current_state("claim:1") == ClaimLifecycleState.ACCEPTED
    outcomes = eng.by_type("ClaimOutcome")
    assert outcomes[0]["durable_reward"] == pytest.approx(0.0)


def test_record_outcome_never_raises_on_a_bare_engine():
    """Best-effort: an engine with no query_cypher/backend surface at all must
    not crash outcome recording."""

    class _BareEngine:
        def add_node(self, *a, **kw):
            raise RuntimeError("kg unreachable")

        def query_cypher(self, *a, **kw):
            raise RuntimeError("kg unreachable")

    fw = ClaimFlywheel(_BareEngine())
    fw.propose("claim:1")  # tolerated
    outcome = fw.record_outcome("claim:1", reward=0.9)  # must not raise
    assert outcome["reward"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): record_outcome's
# durable capability-reward writeback registers off the SAME ClaimOutcome node
# it just persisted, and that materialization ACTUALLY goes Stale when the
# observation is later revised.
# ---------------------------------------------------------------------------


class _RewardCypherBackend:
    """Minimal in-memory Cypher executor for the durable-outcome write path
    (mirrors ``tests/unit/graph/test_capability_designation.py``'s
    ``_FakeCypherBackend``)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        nid = str(params.get("id"))
        if "SET" in query:
            node = self.nodes.setdefault(nid, {})
            node["capability_reward"] = params.get("r")
            node["capability_reward_count"] = params.get("c")
            return []
        node = self.nodes.get(nid, {})
        return [
            {
                "reward": node.get("capability_reward"),
                "count": node.get("capability_reward_count"),
            }
        ]


class _TmsAwareFlywheelEngine(_FlywheelStubEngine):
    """``_FlywheelStubEngine`` + a real durable-reward Cypher backend + a
    minimal, faithful in-process TruthMaintenance index (same contract as
    ``test_insight_validation.py``'s ``_TmsAwareInsightStubEngine``): every
    ``add_node`` bumps that id's version; ``register_materialization``
    snapshots the CURRENT version of every ``:DerivedFrom`` target;
    ``materialization_status`` reports "Stale" the instant a snapshotted
    dependency's version has since moved."""

    def __init__(self) -> None:
        super().__init__()
        self.backend = _RewardCypherBackend()
        self._versions: dict[str, int] = {}
        self._materializations: dict[str, dict[str, int]] = {}

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        super().add_node(node_id, node_type, properties)
        self._versions[node_id] = self._versions.get(node_id, 0) + 1

    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        deps = {
            target
            for source, target, _rel_type, props in self.edges
            if source == derived_id and props.get("relationship_type") == "DERIVED_FROM"
        }
        self._materializations[derived_id] = {d: self._versions.get(d, 0) for d in deps}
        return {
            "id": derived_id,
            "depends_on": sorted(deps),
            "generating_activity": None,
        }

    def materialization_status(self, derived_id: str) -> str | None:
        snapshot = self._materializations.get(derived_id)
        if snapshot is None:
            return None
        for dep, ver in snapshot.items():
            if self._versions.get(dep, 0) != ver:
                return "Stale"
        return "Fresh"


def test_record_outcome_registers_capability_reward_materialization_and_invalidates():
    eng = _TmsAwareFlywheelEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    fw.record_outcome("claim:1", reward=0.9, note="materialized")
    outcome_id = eng.by_type("ClaimOutcome")[0]["id"]

    # The durable capability reward (persisted onto "claim:1", the default
    # durable_key) is registered as a live TMS materialization off the SAME
    # ClaimOutcome node record_outcome just wrote — never a fabricated id.
    assert eng.materialization_status("claim:1") == "Fresh"

    # The observation is later revised through the normal write path.
    eng.add_node(outcome_id, "ClaimOutcome", properties={"reward": 0.1})
    assert eng.materialization_status("claim:1") == "Stale"


def test_record_outcome_with_durable_key_registers_on_the_routing_entity():
    """``durable_key`` (e.g. an ``OutcomeRouter`` bandit key) is the entity the
    materialization registers on — not the claim id — mirroring where the
    durable reward itself is written."""
    eng = _TmsAwareFlywheelEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    fw.record_outcome(
        "claim:1",
        reward=0.9,
        durable_reward=0.0,
        durable_key="trace_pattern_miner:failure_tool_sequence:Read",
    )

    assert (
        eng.materialization_status("trace_pattern_miner:failure_tool_sequence:Read")
        == "Fresh"
    )
    assert eng.materialization_status("claim:1") is None  # never registered
