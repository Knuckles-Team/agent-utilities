"""SkillOpt-native ReflACT skill evolution — the ``_run_skill_evolution`` loop-
controller stage + ``skill_evolution.run_reflact_cycle`` (CONCEPT:AU-AHE.optimization.
skillopt-native-reflact). Ground truth: SkillOpt (arXiv:2605.23904), ported ReflACT
loop (Rollout->Reflect->Aggregate/Select/Update->Evaluate) with a benchmark-gated
promotion decision run through the EXISTING ``action_policy.decide()`` veto gate.
Mirrors ``test_insight_validation.py``'s stub-engine + best-effort-tolerance pattern.

@pytest.mark.concept("AU-AHE.optimization.skillopt-native-reflact")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.knowledge_graph.research.skill_evolution import (
    InternalCorpusSignalProvider,
    SkillTask,
    run_reflact_cycle,
)

pytestmark = pytest.mark.concept("AU-AHE.optimization.skillopt-native-reflact")


class _SkillEvoStubEngine:
    """Minimal engine double: records ``add_node``/``add_edge``, canned ``query_cypher``.

    Same shape as ``test_insight_validation.py``'s ``_InsightStubEngine`` (upsert-
    keyed-by-id, empty governance_rule/regression rows by default) so
    ``action_policy.decide()`` behaves exactly as it does against a real engine with
    no operator overrides — i.e. the shipped conservative default.
    """

    def __init__(self, *, governance_rules: list[dict[str, Any]] | None = None):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()
        self._governance_rules = governance_rules or []
        self.edges: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **properties}))

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "governance_rule" in q:
            return [{"r": dict(r)} for r in self._governance_rules]
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


# ---------------------------------------------------------------------------
# helpers — a deterministic winning / losing scenario built purely from the
# SHIPPED defaults (default_static_executor + default_edit_fn): no LLM, no
# custom executor/edit_fn needed to prove the cycle actually runs.
# ---------------------------------------------------------------------------

_INCUMBENT = "Base skill instructions.\n"


def _winning_signal() -> InternalCorpusSignalProvider:
    """Train task fails on the incumbent (triggers a Reflect pattern); holdout task
    only passes once the Update stage's auto-appended marker section is present —
    so the candidate strictly beats the incumbent using nothing but shipped defaults.
    """
    train = [
        SkillTask(id="t1", prompt="p1", check=lambda out: "nonexistent_term" in out)
    ]
    holdout = [
        SkillTask(
            id="h1", prompt="h1", check=lambda out: "Known failure patterns" in out
        )
    ]
    return InternalCorpusSignalProvider(train, holdout)


def _tying_signal() -> InternalCorpusSignalProvider:
    """Train task still fails (produces a pattern, so Update actually runs), but the
    holdout check passes on BOTH incumbent and candidate (the original text is never
    removed by the append-only default edit) — a tie, which must NOT promote.
    """
    train = [
        SkillTask(id="t1", prompt="p1", check=lambda out: "nonexistent_term" in out)
    ]
    holdout = [
        SkillTask(
            id="h1", prompt="h1", check=lambda out: "Base skill instructions" in out
        )
    ]
    return InternalCorpusSignalProvider(train, holdout)


# ---------------------------------------------------------------------------
# (a) the cycle actually runs: Rollout -> Reflect -> Update -> Evaluate
# ---------------------------------------------------------------------------


def test_reflact_cycle_runs_and_extracts_failure_patterns():
    eng = _SkillEvoStubEngine()
    signal = _winning_signal()
    rep = run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=signal)

    assert rep["train_n"] == 1
    assert rep["holdout_n"] == 1
    assert rep["failure_patterns"], "Reflect stage must extract at least one pattern"
    assert rep["candidate_version_id"] is not None
    assert rep["candidate_score"] == 1.0
    assert rep["incumbent_score"] == 0.0
    assert rep["gate_action"] == "accept"


def test_no_train_failures_is_a_clean_skip():
    eng = _SkillEvoStubEngine()
    train = [SkillTask(id="t1", prompt="p1", check=lambda out: "Base" in out)]  # passes
    holdout = [SkillTask(id="h1", prompt="h1", check=lambda out: True)]
    signal = InternalCorpusSignalProvider(train, holdout)
    rep = run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=signal)

    assert rep["gate_action"] == "skip_no_patches"
    assert rep["candidate_version_id"] is None
    assert eng.by_type("skill_version") == []


# ---------------------------------------------------------------------------
# (b) a losing/tying candidate is NEVER promoted (benchmark gate blocks it)
# ---------------------------------------------------------------------------


def test_losing_candidate_is_persisted_rejected_and_never_reaches_action_policy(
    monkeypatch,
):
    import agent_utilities.orchestration.action_policy as ap_mod

    calls: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        calls.append(request.kind)
        return real_decide(self, request)

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)

    eng = _SkillEvoStubEngine()
    signal = _tying_signal()
    rep = run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=signal)

    assert rep["gate_action"] == "reject"
    assert rep["promoted"] is False
    assert rep["action_decision"] is None
    assert (
        "promote_skill_version" not in calls
    )  # a benchmark loss skips the gate entirely

    versions = eng.by_type("skill_version")
    assert len(versions) == 1
    assert versions[0]["status"] == "rejected"


# ---------------------------------------------------------------------------
# (c) a winning candidate IS proposed through action_policy — shipped default
# NEVER auto-promotes (approval_required tier, mirroring promote_mined_claim)
# ---------------------------------------------------------------------------


def test_winning_candidate_default_never_auto_promotes(monkeypatch):
    import agent_utilities.orchestration.action_policy as ap_mod

    calls: list[str] = []
    real_decide = ap_mod.ActionPolicy.decide

    def spy_decide(self, request):
        calls.append(request.kind)
        return real_decide(self, request)

    monkeypatch.setattr(ap_mod.ActionPolicy, "decide", spy_decide)

    eng = _SkillEvoStubEngine()  # no governance_rule overrides => shipped default
    signal = _winning_signal()
    rep = run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=signal)

    assert rep["gate_action"] == "accept"
    assert calls == ["promote_skill_version"]  # the gate WAS consulted
    assert (
        rep["action_decision"] == "queue_approval"
    )  # shipped tier => queued, not allowed
    assert rep["promoted"] is False

    versions = eng.by_type("skill_version")
    assert len(versions) == 1
    assert versions[0]["status"] == "proposal"  # NOT active
    assert versions[0]["benchmark_score"] == 1.0
    # no SUPERSEDES edge written — promotion never happened
    assert eng.edges == []


def test_winning_candidate_promotes_when_policy_relaxed():
    """Two-key discipline satisfied in the OTHER direction: an operator-relaxed
    action-policy tier lets a benchmark-winning candidate actually flip active +
    write the SUPERSEDES edge — proving the promotion PATH itself works, not just
    that it's blocked by default."""
    eng = _SkillEvoStubEngine(
        governance_rules=[
            {"scope": "action_policy", "kind": "*", "target": "*", "tier": "auto"}
        ]
    )
    signal = _winning_signal()
    rep = run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=signal)

    assert rep["action_decision"] == "allow"
    assert rep["promoted"] is True

    versions = eng.by_type("skill_version")
    assert len(versions) == 1
    assert versions[0]["status"] == "active"
    assert any(
        target.endswith(":" + eng.by_type("skill_version")[0]["parent_hash"])
        for _, target, props in eng.edges
        if props.get("relationship_type") == "SUPERSEDES"
    )


# ---------------------------------------------------------------------------
# (d) trajectories persist as OutcomeEvaluationNodes — the shared durable store
# VariantPool.evaluate_fitness already reads back, not a private one.
# ---------------------------------------------------------------------------


def test_rollout_trajectories_persist_as_outcome_evaluation_nodes():
    eng = _SkillEvoStubEngine()
    train = [
        SkillTask(id="t1", prompt="p1", check=lambda out: "nonexistent_term" in out)
    ]
    holdout = [
        SkillTask(
            id="h1", prompt="h1", check=lambda out: "Known failure patterns" in out
        )
    ]
    signal = InternalCorpusSignalProvider(train, holdout, engine=eng)
    run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=signal)

    outcomes = eng.by_type("outcome_evaluation")
    assert outcomes, "rollout episodes must be persisted as OutcomeEvaluationNodes"


# ---------------------------------------------------------------------------
# (e) run_one_cycle wiring — gated stage placed after distill_skills, injectable
# discovery hook, clean no-op with nothing registered.
# ---------------------------------------------------------------------------


def test_no_registered_targets_is_a_clean_no_op():
    eng = _SkillEvoStubEngine()
    rep = LoopController(eng)._run_skill_evolution()
    assert rep["skipped"] is True
    assert rep["targets"] == 0


def test_run_one_cycle_wires_skill_evolution_with_injected_targets():
    eng = _SkillEvoStubEngine()

    def provider():
        return [
            {
                "skill_id": "skill:demo",
                "content": _INCUMBENT,
                "signal": _winning_signal(),
            }
        ]

    controller = LoopController(eng, skill_eval_targets_provider=provider)
    rep = controller.run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        belief_revision=False,
        mine_discovery=False,
        insight_validation=False,
        trace_mining=False,
    )

    assert rep["skill_evolution"] is not None
    assert rep["skill_evolution"]["targets"] == 1
    assert (
        rep["skill_evolution"]["promoted"] == 0
    )  # shipped default never auto-promotes
    assert len(rep["skill_evolution"]["results"]) == 1
    assert rep["skill_evolution"]["results"][0]["gate_action"] == "accept"

    # Explicitly disabled => stage does not run at all.
    rep_disabled = LoopController(
        eng, skill_eval_targets_provider=provider
    ).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        belief_revision=False,
        mine_discovery=False,
        insight_validation=False,
        trace_mining=False,
        skill_evolution=False,
    )
    assert rep_disabled["skill_evolution"] is None


# ---------------------------------------------------------------------------
# (f) best-effort tolerance — one failing target never blocks the others
# ---------------------------------------------------------------------------


def test_one_failing_target_does_not_block_the_others():
    eng = _SkillEvoStubEngine()

    def provider():
        return [
            {"skill_id": "skill:broken"},  # missing required "content"/"signal" keys
            {
                "skill_id": "skill:demo",
                "content": _INCUMBENT,
                "signal": _winning_signal(),
            },
        ]

    controller = LoopController(eng, skill_eval_targets_provider=provider)
    rep = controller._run_skill_evolution()

    assert rep["targets"] == 2
    assert len(rep["results"]) == 1  # only the well-formed target produced a report
    assert any("skill:broken" in e for e in rep["errors"])
