"""Transparent + steerable self-evolution flywheel.

CONCEPT:KG-2.290 — live EvolutionState surface + per-stage progress beacon.
CONCEPT:KG-2.291 — saturation gauge (open_gaps trend + velocity + coverage) + request-more.
CONCEPT:KG-2.292 — distill→develop seam: a distilled spec becomes a develop-able node.
CONCEPT:OS-5.73 — spec-level review/veto checkpoint before develop.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.research import evolution_state as es
from agent_utilities.knowledge_graph.research import spec_proposals as sp


class FakeEngine:
    """In-memory engine answering the label queries these modules issue."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, node_type, properties=None):
        node = self.nodes.get(node_id, {})
        node.update({"id": node_id, "type": node_type, **(properties or {})})
        self.nodes[node_id] = node

    def add_edge(self, source, target, rel_type="", **kw):
        self.edges.append((source, target, rel_type))

    def _by_label(self, label):
        return [n for n in self.nodes.values() if n.get("type") == label]

    def query_cypher(self, query, params=None):
        params = params or {}
        q = query
        if "EvolutionBeacon" in q:
            return [
                {
                    "status": n.get("status"),
                    "stage": n.get("stage"),
                    "timestamp": n.get("timestamp"),
                    "metadata": n.get("metadata"),
                }
                for n in self._by_label("EvolutionBeacon")
            ][:1]
        if "EvolutionCycle" in q:
            return [
                {"ts": n.get("created_at"), "metadata": n.get("metadata")}
                for n in self._by_label("EvolutionCycle")
            ]
        if "ProposalPublication" in q:
            return [
                {"kind": n.get("kind"), "ok": n.get("ok"), "ts": n.get("published_at")}
                for n in self._by_label("ProposalPublication")
            ]
        if "CapabilityRatchetResult" in q:
            return [
                {"result": n.get("result"), "ts": n.get("recorded_at")}
                for n in self._by_label("CapabilityRatchetResult")
            ]
        if "SpecProposal" in q:
            specs = self._by_label("SpecProposal")
            if "n.id" in q:
                specs = [n for n in specs if n.get("id") == params.get("id")]
            return [{"n": dict(n)} for n in specs]
        if "ActionApproval" in q:
            return [
                {"id": n["id"]}
                for n in self._by_label("ActionApproval")
                if n.get("kind") == params.get("kind")
                and n.get("target") == params.get("target")
                and n.get("status") == "pending"
            ]
        if "ActionDecision" in q:
            return [
                {
                    "target": n.get("target"),
                    "decision": n.get("decision"),
                    "ts": n.get("decided_unix"),
                }
                for n in self._by_label("ActionDecision")
            ]
        if "governance_rule" in q:
            return []
        if "Concept" in q:
            if "ADDRESSED_BY" in q:
                return []
            return [
                {
                    "id": n["id"],
                    "name": n.get("name"),
                    "loop_kind": n.get("loop_kind"),
                    "status": n.get("status"),
                    "objective": n.get("objective"),
                    "validation_cmd": n.get("validation_cmd"),
                    "skill_ref": n.get("skill_ref"),
                    "end_state": n.get("end_state"),
                    "spec_id": n.get("spec_id"),
                    "prio_bucket": n.get("prio_bucket"),
                }
                for n in self._by_label("Concept")
            ]
        return []


class _Spec:
    """A SpecDraft-shaped object (duck-typed)."""

    def __init__(self, title, concept_ids=(), value_score=1.0):
        self.title = title
        self.problem = "the problem"
        self.approach = "the approach"
        self.value = "the value"
        self.concept_ids = list(concept_ids)
        self.value_score = value_score
        self.target_codebase = "agent-utilities"


# ── KG-2.290: live beacon ────────────────────────────────────────────────
@pytest.mark.concept("KG-2.290")
def test_beacon_reflects_live_stage():
    eng = FakeEngine()
    beacon = es.StageBeacon(eng, cycle_id="evo_cycle_x", why="mine open gaps")
    beacon.enter("assimilate", detail="open_gaps=12")
    live = es.read_beacon(eng)
    assert live["stage"] == "assimilate"
    assert live["status"] == "running"
    assert live["why"] == "mine open gaps"
    assert live["detail"] == "open_gaps=12"
    # finishing flips it idle + stamps the closing summary
    beacon.finish(open_gaps=5, errors=0, saturation=0.4)
    done = es.read_beacon(eng)
    assert done["status"] == "idle" and done["stage"] == "idle"
    assert done["open_gaps"] == 5 and done["saturation"] == 0.4


# ── KG-2.291: saturation gauge ───────────────────────────────────────────
@pytest.mark.concept("KG-2.291")
def test_gauge_fires_request_more_when_saturated_and_stalling():
    g = es.saturation_gauge(
        coverage_pct=95.0, velocity_verdict="stalling", gaps_recent=10, gaps_prior=10
    )
    assert g["saturated"] is True
    assert g["request_more"] is True
    assert g["recommendation"]
    assert g["gauge"] >= es.SATURATION_THRESHOLD


@pytest.mark.concept("KG-2.291")
def test_gauge_quiet_while_improving():
    g = es.saturation_gauge(
        coverage_pct=20.0, velocity_verdict="improving", gaps_recent=2, gaps_prior=20
    )
    assert g["request_more"] is False
    assert g["gauge"] < es.SATURATION_THRESHOLD


@pytest.mark.concept("KG-2.291")
def test_gauge_does_not_request_more_without_stalling_even_if_high():
    # High coverage + flat gaps but velocity NOT stalling → no request-more.
    g = es.saturation_gauge(
        coverage_pct=100.0, velocity_verdict="steady", gaps_recent=8, gaps_prior=8
    )
    assert g["request_more"] is False


@pytest.mark.concept("KG-2.291")
def test_emit_saturation_signal_only_when_requested():
    eng = FakeEngine()
    quiet = es.saturation_gauge(
        coverage_pct=10.0, velocity_verdict="improving", gaps_recent=1, gaps_prior=10
    )
    assert es.emit_saturation_signal(eng, quiet) is None
    hot = es.saturation_gauge(
        coverage_pct=95.0, velocity_verdict="stalling", gaps_recent=9, gaps_prior=9
    )
    sid = es.emit_saturation_signal(eng, hot)
    assert sid and eng._by_label("EvolutionSaturationSignal")


# ── KG-2.292: distill→develop seam ───────────────────────────────────────
@pytest.mark.concept("KG-2.292")
def test_distilled_spec_becomes_developable_node_with_provenance():
    eng = FakeEngine()
    sid = sp.persist_spec_proposal(
        eng, _Spec("Add a thing", concept_ids=["concept:a", "concept:b"]),
        spec_path="/repo/.specify/specs/kg-distilled/add-a-thing.md",
    )
    assert sid == "spec_proposal:add-a-thing"
    got = sp.get_spec(eng, sid)
    assert got["status"] == sp.STATUS_PENDING
    assert got["problem"] == "the problem"
    assert got["spec_path"].endswith("add-a-thing.md")
    # DISTILLED_FROM provenance edges to both source concepts
    assert (sid, "concept:a", "DISTILLED_FROM") in eng.edges
    assert (sid, "concept:b", "DISTILLED_FROM") in eng.edges
    # backlog summary sees it as pending
    summ = sp.specs_summary(eng)
    assert summ["counts"][sp.STATUS_PENDING] == 1


@pytest.mark.concept("KG-2.292")
def test_spec_to_proposal_shape_feeds_promotion_path():
    from agent_utilities.knowledge_graph.research.change_synthesis import (
        synthesize_change_set,
    )

    spec = {
        "id": "spec_proposal:x",
        "title": "Improve foo",
        "problem": "foo is slow",
        "approach": "cache it",
        "value": "faster",
        "concept_ids": ["concept:foo"],
    }
    proposal = sp._spec_to_proposal(spec)
    # The existing promotion pipeline accepts the proposal and (no target_file →)
    # produces the SDD spec+tasks skeleton change set — distilled spec is no dead end.
    change = synthesize_change_set(proposal, validate=False)
    assert change.kind == "sdd_plan"
    assert any(f.path.endswith("spec.md") for f in change.files)
    assert any(f.path.endswith("tasks.md") for f in change.files)
    assert change.concept_ids == ["concept:foo"]


# ── OS-5.73: spec review / veto checkpoint ───────────────────────────────
@pytest.mark.concept("OS-5.73")
def test_review_approve_binds_develop_loop():
    eng = FakeEngine()
    sid = sp.persist_spec_proposal(eng, _Spec("Build X"))
    res = sp.review_spec(eng, sid, "approve")
    assert res["status"] == sp.STATUS_APPROVED
    assert sp.get_spec(eng, sid)["status"] == sp.STATUS_APPROVED
    # a develop Loop bound to the spec now exists (steerable via graph_loops)
    loop_id = f"loop:develop:{sid}"
    assert loop_id in eng.nodes
    assert eng.nodes[loop_id]["spec_id"] == sid
    assert eng.nodes[loop_id]["loop_kind"] == "develop"


@pytest.mark.concept("OS-5.73")
def test_review_reject_vetoes_spec():
    eng = FakeEngine()
    sid = sp.persist_spec_proposal(eng, _Spec("Bad idea"))
    res = sp.review_spec(eng, sid, "reject")
    assert res["status"] == sp.STATUS_REJECTED
    # a rejected spec never develops
    assert sp.get_spec(eng, sid)["status"] == sp.STATUS_REJECTED
    assert sp.develop_spec(eng, sid)["status"] == "not_approved"


@pytest.mark.concept("OS-5.73")
def test_review_edit_holds_spec_pending():
    eng = FakeEngine()
    sid = sp.persist_spec_proposal(eng, _Spec("Tweak me"))
    res = sp.review_spec(eng, sid, "edit", edits={"approach": "a better approach"})
    assert res["status"] == sp.STATUS_PENDING
    got = sp.get_spec(eng, sid)
    assert got["approach"] == "a better approach"
    assert got["status"] == sp.STATUS_PENDING  # still held for review


@pytest.mark.concept("OS-5.73")
def test_develop_holds_for_approval_default_review_first(monkeypatch):
    # An approved spec is fed into governed_publish, which (default merge_promotion
    # tier = approval_required) QUEUES a human approval — never auto-merges.
    import agent_utilities.knowledge_graph.research.change_publisher as cp

    captured = {}

    def fake_governed_publish(engine, proposal, *, source="loop_engine", **kw):
        captured["proposal_id"] = proposal["id"]
        return {"status": "approval_queued", "approval_id": "appr:1"}

    monkeypatch.setattr(cp, "governed_publish", fake_governed_publish)
    eng = FakeEngine()
    sid = sp.persist_spec_proposal(eng, _Spec("Ship it"))
    sp.review_spec(eng, sid, "approve", submit_develop=False)
    rep = sp.develop_spec(eng, sid)
    assert rep["status"] == "approval_queued"
    assert captured["proposal_id"] == sid
    # spec is now 'developing' (handed to the publish gate, awaiting human grant)
    assert sp.get_spec(eng, sid)["status"] == sp.STATUS_DEVELOPING


@pytest.mark.concept("OS-5.73")
def test_auto_advance_holds_specs_under_default_policy():
    # KG_LOOP_AUTO_DEVELOP path: with the shipped spec_promotion=approval_required
    # tier the auto path does NOT approve — it queues for the human (review-first).
    eng = FakeEngine()
    sid = sp.persist_spec_proposal(eng, _Spec("Auto candidate"))
    out = sp.auto_advance_specs(eng)
    assert out["approved"] == 0
    assert out["queued"] == 1
    assert sp.get_spec(eng, sid)["status"] == sp.STATUS_PENDING


# ── integration: _advance_develop routes a spec-bound loop ───────────────
@pytest.mark.concept("KG-2.292")
def test_advance_develop_routes_spec_bound_loop(monkeypatch):
    from agent_utilities.knowledge_graph.research import spec_proposals as sp_mod
    from agent_utilities.knowledge_graph.research.loop_controller import LoopController

    monkeypatch.setattr(
        sp_mod, "develop_spec", lambda engine, spec_id: {"status": "approval_queued"}
    )
    lc = LoopController(FakeEngine())
    out = lc._advance_develop({"id": "loop:develop:s", "spec_id": "spec_proposal:s"})
    assert out["status"] == "completed" and out["done"] is True


# ── KG-2.290: aggregated EvolutionState read ─────────────────────────────
@pytest.mark.concept("KG-2.290")
def test_evolution_state_aggregates_everything():
    eng = FakeEngine()
    # a finished beacon
    b = es.StageBeacon(eng, cycle_id="c1", why="mine")
    b.enter("distill", detail="2 specs")
    # an open spec backlog
    sp.persist_spec_proposal(eng, _Spec("A"))
    sp.persist_spec_proposal(eng, _Spec("B"))
    # a couple EvolutionCycle nodes for the gaps trend
    eng.add_node("evo_cycle_1", "EvolutionCycle", {
        "created_at": "2026-06-28T01", "metadata": json.dumps({"open_gaps": 9})})
    eng.add_node("evo_cycle_2", "EvolutionCycle", {
        "created_at": "2026-06-28T02", "metadata": json.dumps({"open_gaps": 9})})

    state = es.read_evolution_state(eng, include_coverage=False)
    assert state["beacon"]["stage"] == "distill"
    assert state["specs"]["counts"][sp.STATUS_PENDING] == 2
    assert "gauge" in state["saturation"]
    assert state["open_gaps"]["series"] == [9, 9]
    # steering hints are present (the operator's how-to-steer surface)
    assert "review_spec" in state["steering"]
