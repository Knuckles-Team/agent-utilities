"""Closed-loop agent mining (CONCEPT:AU-KG.evolution.insight-engine-closed-loop, workstream C6).

Mine (Episode/OutcomeEvaluation/ToolCall provenance → repeated FAILURE
tool-call sequences) → CandidateInsight → EvidenceBundle → Claim → Validation
(reuses promotion_governance as-is) → Action gate (reuses action_policy.decide(),
kind="route_policy_update") → governed routing/prompt/tool change +
OutcomeRouter.record() — gated STRICTLY after the action-policy decision.

@pytest.mark.concept("AU-KG.evolution.insight-engine-closed-loop")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.research.candidate_insight import (
    CONFIDENCE_FLOOR,
    candidates_from_sequential_patterns,
)
from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.knowledge_graph.research.trace_pattern_miner import (
    gather_failure_tool_sequences,
    mine_trace_patterns,
)

pytestmark = pytest.mark.concept("AU-KG.evolution.insight-engine-closed-loop")


# ---------------------------------------------------------------------------
# gather_failure_tool_sequences
# ---------------------------------------------------------------------------


class _TraceStubEngine:
    """Minimal engine double: canned Episode/OutcomeEvaluation/ToolCall rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self._rows = rows or []
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "Episode" in q and "USED_TOOL" in q:
            return list(self._rows)
        return []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


def test_gather_failure_tool_sequences_groups_by_episode_in_order():
    engine = _TraceStubEngine(
        rows=[
            {"episode_id": "ep1", "tool_name": "Read", "ts": 1},
            {"episode_id": "ep1", "tool_name": "Edit", "ts": 2},
            {"episode_id": "ep2", "tool_name": "Bash", "ts": 1},
            {"episode_id": "ep2", "tool_name": "Bash", "ts": 2},
        ]
    )
    ids, sequences = gather_failure_tool_sequences(engine)
    assert ids == ["ep1", "ep2"]
    assert sequences == [["Read", "Edit"], ["Bash", "Bash"]]


def test_gather_failure_tool_sequences_drops_single_tool_episodes():
    """A one-tool-call episode carries no ORDERED subsequence — excluded."""
    engine = _TraceStubEngine(rows=[{"episode_id": "ep1", "tool_name": "Read", "ts": 1}])
    ids, sequences = gather_failure_tool_sequences(engine)
    assert ids == []
    assert sequences == []


def test_gather_failure_tool_sequences_handles_missing_engine():
    assert gather_failure_tool_sequences(None) == ([], [])


def test_gather_failure_tool_sequences_degrades_on_query_failure():
    class _BoomEngine:
        def query_cypher(self, q, params=None):
            raise RuntimeError("engine unreachable")

    assert gather_failure_tool_sequences(_BoomEngine()) == ([], [])


# ---------------------------------------------------------------------------
# mine_trace_patterns — delegates to the engine's graph_mine "sequence" surface
# ---------------------------------------------------------------------------


def test_mine_trace_patterns_no_failures_is_a_clean_empty_result():
    engine = _TraceStubEngine(rows=[])
    result = mine_trace_patterns(engine)
    assert result["patterns"] == {"patterns": []}
    assert result["sequences_mined"] == 0
    assert result["errors"] == []


def test_mine_trace_patterns_invokes_graph_mine_sequence_surface(monkeypatch):
    import json

    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    captured: dict[str, Any] = {}

    def fake_invoke(*, surface, action, graph, candidates, params):
        captured["surface"] = surface
        captured["action"] = action
        captured["sequences"] = params["sequences"]
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "result": {
                    "patterns": [{"items": ["Read", "Edit"], "support": 0.8, "count": 4}],
                    "n_sequences": 2,
                    "n_patterns": 1,
                },
            }
        )

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    engine = _TraceStubEngine(
        rows=[
            {"episode_id": "ep1", "tool_name": "Read", "ts": 1},
            {"episode_id": "ep1", "tool_name": "Edit", "ts": 2},
            {"episode_id": "ep2", "tool_name": "Read", "ts": 1},
            {"episode_id": "ep2", "tool_name": "Edit", "ts": 2},
        ]
    )
    result = mine_trace_patterns(engine)
    assert captured["surface"] == "mining"
    assert captured["action"] == "sequence"
    assert captured["sequences"] == [["Read", "Edit"], ["Read", "Edit"]]
    assert result["patterns"]["patterns"] == [
        {"items": ["Read", "Edit"], "support": 0.8, "count": 4}
    ]
    assert result["failure_episodes"] == 2
    assert result["sequences_mined"] == 2


def test_mine_trace_patterns_degrades_cleanly_on_no_mining_engine_build(monkeypatch):
    import json

    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    monkeypatch.setattr(
        engine_surface_tools,
        "_invoke",
        lambda **kw: json.dumps({"degraded": True, "error": "no mining surface"}),
    )
    engine = _TraceStubEngine(
        rows=[
            {"episode_id": "ep1", "tool_name": "Read", "ts": 1},
            {"episode_id": "ep1", "tool_name": "Edit", "ts": 2},
        ]
    )
    result = mine_trace_patterns(engine)
    assert result["patterns"] == {"patterns": []}
    assert result["errors"]  # recorded, not raised


# ---------------------------------------------------------------------------
# candidates_from_sequential_patterns
# ---------------------------------------------------------------------------


def test_candidates_from_sequential_patterns_maps_support_to_confidence():
    result = {"patterns": [{"items": ["Read", "Edit"], "support": 0.8, "count": 4}]}
    candidates = candidates_from_sequential_patterns(result)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.finding_type == "SequentialPattern"
    assert cand.confidence == pytest.approx(0.8)
    assert cand.clears_floor is (0.8 >= CONFIDENCE_FLOOR)
    assert cand.source_ids == ["Read", "Edit"]


def test_candidates_from_sequential_patterns_empty_on_no_patterns():
    assert candidates_from_sequential_patterns({"patterns": []}) == []
    assert candidates_from_sequential_patterns(None) == []


def test_candidates_from_sequential_patterns_skips_patterns_with_no_items():
    result = {"patterns": [{"items": [], "support": 0.9, "count": 1}]}
    assert candidates_from_sequential_patterns(result) == []


# ---------------------------------------------------------------------------
# LoopController._run_trace_mining — the C4-reused closed-loop pipeline
# ---------------------------------------------------------------------------


class _TraceMiningStubEngine:
    """Same shape as ``test_insight_validation.py``'s ``_InsightStubEngine`` —
    empty governance-adjacent query results ⇒ ``PromotionGovernanceValidator``
    passes by default; ``governance_rules`` relaxes the ActionPolicy tier."""

    def __init__(self, *, governance_rules: list[dict[str, Any]] | None = None):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()
        self._governance_rules = governance_rules or []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "governance_rule" in q:
            return [{"r": dict(r)} for r in self._governance_rules]
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


def _patch_mine_result(monkeypatch, *, support: float = 0.8) -> None:
    """Patch ``trace_pattern_miner.mine_trace_patterns`` to a canned one-pattern result."""
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


def test_below_floor_pattern_never_becomes_a_claim(monkeypatch):
    _patch_mine_result(monkeypatch, support=0.1)  # well below CONFIDENCE_FLOOR
    engine = _TraceMiningStubEngine()
    rep = LoopController(engine)._run_trace_mining()

    assert rep["eligible"] == 0
    assert rep["below_floor"] == 1
    assert rep["persisted_claims"] == 0
    assert rep["routed"] == 0
    assert engine.by_type("Claim") == []


def test_eligible_pattern_persists_as_an_unverified_proposal_claim(monkeypatch):
    _patch_mine_result(monkeypatch, support=0.8)
    engine = _TraceMiningStubEngine()
    rep = LoopController(engine)._run_trace_mining()

    assert rep["eligible"] == 1
    assert rep["persisted_claims"] == 1
    claims = engine.by_type("Claim")
    assert len(claims) == 1
    assert claims[0]["status"] == "proposal"
    assert claims[0]["is_verified"] is False


def test_shipped_default_never_routes_or_records(monkeypatch):
    """route_policy_update is approval_required by default — the gate must
    queue, and OutcomeRouter.record() must never fire."""
    import agent_utilities.orchestration.outcome_router as router_mod

    record_calls: list[tuple] = []
    monkeypatch.setattr(
        router_mod.OutcomeRouter,
        "record",
        lambda self, tc, choice, reward: record_calls.append((tc, choice, reward)),
    )
    _patch_mine_result(monkeypatch, support=0.8)
    engine = _TraceMiningStubEngine()  # no governance_rule override ⇒ shipped default
    rep = LoopController(engine)._run_trace_mining()

    assert rep["routed"] == 0
    assert record_calls == []
    for ex in rep["examples"]:
        assert ex["action_decision"] == "queue_approval"
        assert ex["routed"] is False


def test_action_policy_consulted_with_route_policy_update_kind(monkeypatch):
    import agent_utilities.orchestration.action_policy as ap_mod

    calls: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        calls.append(request.kind)
        return real_decide(self, request)

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)
    _patch_mine_result(monkeypatch, support=0.8)
    engine = _TraceMiningStubEngine()
    LoopController(engine)._run_trace_mining()

    assert calls == ["route_policy_update"]


def test_route_policy_update_default_never_auto():
    """SAFETY: the shipped ActionPolicy default for route_policy_update must
    never be auto/auto_notify (mirrors test_promote_mined_claim_default_never_auto)."""
    from agent_utilities.orchestration.action_policy import (
        DEFAULT_POLICY,
        TIER_APPROVAL,
    )

    rule = next(r for r in DEFAULT_POLICY["rules"] if r["kind"] == "route_policy_update")
    assert rule["tier"] == TIER_APPROVAL


def test_relaxed_policy_routes_and_records_after_decide(monkeypatch):
    """Relaxing route_policy_update to auto lets the routed branch fire — and
    it still only fires AFTER the action-policy decision."""
    import agent_utilities.orchestration.outcome_router as router_mod

    record_calls: list[tuple] = []
    monkeypatch.setattr(
        router_mod.OutcomeRouter,
        "record",
        lambda self, tc, choice, reward: record_calls.append((tc, choice, reward)),
    )
    # support=0.9 clears BOTH the CandidateInsight floor (0.6) and the
    # PromotionGovernanceValidator's own quality threshold (0.85) — needed for
    # the routed branch (gated on verdict.valid too) to actually fire.
    _patch_mine_result(monkeypatch, support=0.9)
    engine = _TraceMiningStubEngine(
        governance_rules=[
            {
                "scope": "action_policy",
                "kind": "route_policy_update",
                "target": "*",
                "tier": "auto",
            }
        ]
    )
    rep = LoopController(engine)._run_trace_mining()

    assert rep["routed"] == 1
    assert record_calls == [("failure_tool_sequence", "Read", 0.0)]
    for ex in rep["examples"]:
        assert ex["action_decision"] == "allow"
        assert ex["routed"] is True


# ---------------------------------------------------------------------------
# THE safety test: OutcomeRouter.record() must never precede action_policy.decide()
# ---------------------------------------------------------------------------


def test_gate_runs_before_any_outcome_record(monkeypatch):
    """Review-grep-proof: instrument BOTH ``ActionPolicy.decide`` and
    ``OutcomeRouter.record`` to append to one shared, ordered call log. Force
    the decision to ``allow`` (via a relaxed policy) so the routed branch
    actually executes — proving the invariant holds on the LIVE path, not just
    the (trivially safe) default-denied path."""
    import agent_utilities.orchestration.action_policy as ap_mod
    import agent_utilities.orchestration.outcome_router as router_mod

    call_log: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        call_log.append(f"decide:{request.kind}")
        return real_decide(self, request)

    def spy_record(self, task_class, choice, reward):
        call_log.append(f"record:{task_class}:{choice}")

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)
    monkeypatch.setattr(router_mod.OutcomeRouter, "record", spy_record)

    _patch_mine_result(monkeypatch, support=0.9)
    engine = _TraceMiningStubEngine(
        governance_rules=[
            {
                "scope": "action_policy",
                "kind": "route_policy_update",
                "target": "*",
                "tier": "auto",
            }
        ]
    )
    rep = LoopController(engine)._run_trace_mining()

    assert rep["routed"] == 1  # the routed branch DID execute this run
    record_entries = [c for c in call_log if c.startswith("record:")]
    assert record_entries  # sanity: record really was called

    # THE invariant: every "record" entry has a "decide" entry at an earlier
    # index in the shared call log — record() is never observed before decide().
    for idx, entry in enumerate(call_log):
        if entry.startswith("record:"):
            assert any(
                call_log[j].startswith("decide:") for j in range(idx)
            ), f"record() at position {idx} has no preceding decide() call: {call_log}"

    # And specifically: the very first call in this stage is always decide(),
    # never record() — no path reaches record() cold.
    assert call_log[0].startswith("decide:")


def test_gate_runs_before_record_even_when_decide_denies(monkeypatch):
    """Default (denied) path: decide() still runs, record() must never run."""
    import agent_utilities.orchestration.action_policy as ap_mod
    import agent_utilities.orchestration.outcome_router as router_mod

    call_log: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        call_log.append(f"decide:{request.kind}")
        return real_decide(self, request)

    def spy_record(self, task_class, choice, reward):
        call_log.append(f"record:{task_class}:{choice}")

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)
    monkeypatch.setattr(router_mod.OutcomeRouter, "record", spy_record)

    _patch_mine_result(monkeypatch, support=0.8)
    engine = _TraceMiningStubEngine()  # shipped default ⇒ approval_required ⇒ denied-to-auto
    rep = LoopController(engine)._run_trace_mining()

    assert rep["routed"] == 0
    assert "decide:route_policy_update" in call_log
    assert not any(c.startswith("record:") for c in call_log)
