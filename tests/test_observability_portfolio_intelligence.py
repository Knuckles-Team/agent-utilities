"""Regression tests for the portfolio comparative-intelligence engine
(``agent_utilities.observability.portfolio_intelligence``) — the ServiceNow
TRM / ERPNext new-software request -> weighted multi-criteria assessment ->
adopt/reject/consolidate/migrate flow
(``reports/enterprise-comparative-intelligence-design.md``). Mirrors
``test_observability_incident_router.py``/``test_observability_lifecycle_orchestrator.py``'s
fake-KG style.
"""

from __future__ import annotations

from typing import Any

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
import agent_utilities.observability.health_ingest as hi
from agent_utilities.observability import portfolio_intelligence as pi


class _NodesView:
    def __init__(self, props: dict[str, dict[str, Any]]) -> None:
        self._props = props

    def get(self, node_id: str, default: Any = None) -> Any:
        return self._props.get(node_id, default)


class _FakeEngine:
    """Serves out_edges/in_edges/get_nodes_by_label/personalized_pagerank from
    a flat edge list (``(src, rel_type, tgt)``) + per-node property table."""

    def __init__(
        self,
        edges: list[tuple[str, str, str]] | None = None,
        node_props: dict[str, dict[str, Any]] | None = None,
        by_label: dict[str, list[tuple[str, dict[str, Any]]]] | None = None,
        pagerank: dict[str, float] | None = None,
    ) -> None:
        self._edges = edges or []
        self._by_label = by_label or {}
        self.nodes = _NodesView(node_props or {})
        self._pagerank = pagerank or {}

    def out_edges(self, node_id: str, data: bool = False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if s == node_id]

    def in_edges(self, node_id: str, data: bool = False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if t == node_id]

    def get_nodes_by_label(self, label: str, limit: int = 0):
        rows = self._by_label.get(label, [])
        return rows[:limit] if limit else rows

    def personalized_pagerank(
        self, seed_nodes, damping: float = 0.85, iterations: int = 100
    ):
        return dict(self._pagerank)


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, entities, relationships=None, *, source, domain, **kw):
        self.calls.append(
            {
                "entities": entities,
                "relationships": relationships or [],
                "source": source,
                "domain": domain,
            }
        )
        return {"nodes": len(entities), "edges": len(relationships or [])}


# ── 1. hard gate: PHI candidate missing HIPAA is rejected regardless of score ──


def test_compliance_gate_rejects_phi_candidate_missing_hipaa():
    engine = _FakeEngine(
        edges=[
            # a great candidate on paper: cheap, unique capability, no peers
            ("prod-a", "incursCost", "cost-a"),
        ],
        node_props={"cost-a": {"annualCost": 1.0}},
        by_label={
            "ComplianceGate": [
                (
                    "hipaa-gate",
                    {
                        "name": "HIPAA",
                        "required": True,
                        "appliesToSector": "healthcare",
                        "appliesToDataClass": "phi",
                    },
                )
            ]
        },
    )
    request = {"candidateId": "prod-a", "sector": "healthcare", "dataClass": "phi"}

    outcome = pi.assess_candidate(request, engine=engine)

    assert outcome["verdict"] == "reject"
    assert "compliance_gate" in outcome["rationale"]
    assert "HIPAA" in outcome["rationale"]
    # gate tier ran before any scoring — no criteria were computed
    assert outcome["criteria"] == []
    assert outcome["assessmentScore"] == 0.0


def test_compliance_gate_passes_when_candidate_is_governed_by_the_gate():
    engine = _FakeEngine(
        edges=[("prod-a", "governedBy", "hipaa-gate")],
        by_label={
            "ComplianceGate": [
                (
                    "hipaa-gate",
                    {
                        "name": "HIPAA",
                        "required": True,
                        "appliesToSector": "healthcare",
                        "appliesToDataClass": "phi",
                    },
                )
            ]
        },
    )
    request = {"candidateId": "prod-a", "sector": "healthcare", "dataClass": "phi"}

    check = pi._check_compliance_gates("prod-a", request, engine)
    assert check.passed is True


def test_unaffected_sector_does_not_trigger_the_hipaa_gate():
    engine = _FakeEngine(
        by_label={
            "ComplianceGate": [
                (
                    "hipaa-gate",
                    {
                        "name": "HIPAA",
                        "required": True,
                        "appliesToSector": "healthcare",
                        "appliesToDataClass": "phi",
                    },
                )
            ]
        },
    )
    request = {"candidateId": "prod-a", "sector": "retail", "dataClass": "pii"}
    check = pi._check_compliance_gates("prod-a", request, engine)
    assert check.passed is True
    assert "no REQUIRED" in check.reason


# ── 2. weighted score ranks peers + weights unique capabilities higher ────────


def _capability_edges() -> list[tuple[str, str, str]]:
    return [
        ("prod-a", "providesCapability", "cap-common"),
        ("prod-a", "providesCapability", "cap-unique"),
        ("prod-b", "providesCapability", "cap-common"),
    ]


def test_unique_capability_scores_higher_than_a_pure_overlap_peer():
    engine = _FakeEngine(edges=_capability_edges())
    criteria = pi.score_criteria("prod-a", ["prod-b"], {}, engine, pi.DEFAULT_WEIGHTS)
    by_kind = {c.kind: c for c in criteria}
    assert by_kind["unique-capability"].score > 0.0
    # prod-a covers the full peer-union capability set (cap-common)
    assert by_kind["functionality"].score == 1.0

    peer_criteria = pi.score_criteria(
        "prod-b", ["prod-a"], {}, engine, pi.DEFAULT_WEIGHTS
    )
    peer_by_kind = {c.kind: c for c in peer_criteria}
    assert peer_by_kind["unique-capability"].score == 0.0


def test_assess_candidate_ranks_the_peer_group_and_adopts_the_winner():
    engine = _FakeEngine(edges=_capability_edges())
    outcome = pi.assess_candidate({"candidateId": "prod-a"}, engine=engine)

    assert outcome["verdict"] in ("adopt", "migrate", "consolidate")
    ids = [r["id"] for r in outcome["peerRanking"]]
    assert set(ids) == {"prod-a", "prod-b"}
    # prod-a's extra unique capability (weighted at 0.20, the highest default
    # weight) must make it outrank the pure-overlap peer.
    assert outcome["peerRanking"][0]["id"] == "prod-a"


# ── 3. consolidation sweep flags a redundant product ──────────────────────────


def _redundant_portfolio_engine() -> _FakeEngine:
    return _FakeEngine(
        edges=[
            ("prod-x", "providesCapability", "cap-shared"),
            ("prod-y", "providesCapability", "cap-shared"),
            ("prod-x", "swappableWith", "prod-y"),
            ("prod-y", "swappableWith", "prod-x"),
            ("prod-x", "incursCost", "cost-x"),
            ("prod-y", "incursCost", "cost-y"),
        ],
        node_props={
            "cost-x": {"annualCost": 100.0},
            "cost-y": {"annualCost": 900.0},
        },
        by_label={
            "TechnologyProduct": [("prod-x", {}), ("prod-y", {})],
        },
    )


def test_rationalize_portfolio_flags_a_redundant_cluster():
    engine = _redundant_portfolio_engine()
    result = pi.rationalize_portfolio(engine=engine, write=False)

    assert result["clusters"] == 1
    assert len(result["recommendations"]) == 1
    rec = result["recommendations"][0]
    assert rec["verdict"] == "consolidate"
    # the cheaper product (prod-x) wins and proposes retiring prod-y
    assert rec["candidateId"] == "prod-x"
    assert "prod-y" in rec["consolidates"]
    assert rec["financialDelta"] > 0  # retiring the pricier peer is a net saving


def test_rationalize_portfolio_writes_recommendations_when_write_true(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)
    engine = _redundant_portfolio_engine()

    result = pi.rationalize_portfolio(engine=engine, write=True)

    assert result["clusters"] == 1
    assert cap.calls, "expected the consolidate recommendation to be persisted"
    written_types = {e["type"] for call in cap.calls for e in call["entities"]}
    assert "Recommendation" in written_types
    assert "Assessment" in written_types


# ── 4. adopt/reject/consolidate/migrate verdict via the SHACL shapes ──────────


def test_shacl_shape_conforms_for_every_valid_verdict():
    for verdict in ("adopt", "reject", "consolidate", "migrate"):
        report = pi.validate_verdict_shape(
            {
                "candidateId": "prod-a",
                "verdict": verdict,
                "rationale": "some rationale text",
                "assessmentScore": 0.75,
            }
        )
        assert report["conforms"] is True, (verdict, report["violations"])


def test_shacl_shape_rejects_an_invalid_verdict_value():
    report = pi.validate_verdict_shape(
        {
            "candidateId": "prod-a",
            "verdict": "maybe",
            "rationale": "some rationale text",
            "assessmentScore": 0.75,
        }
    )
    assert report["conforms"] is False


def test_shacl_shape_rejects_a_missing_rationale():
    report = pi.validate_verdict_shape(
        {
            "candidateId": "prod-a",
            "verdict": "adopt",
            "rationale": "",
            "assessmentScore": 0.5,
        }
    )
    assert report["conforms"] is False


# ── 5. writeback is dry-run-first ──────────────────────────────────────────────


def test_writeback_defaults_to_graph_only(monkeypatch):
    monkeypatch.delenv("TRM_WRITEBACK_BACKEND", raising=False)
    out = pi._route_writeback({"id": "trm:1"}, {"verdict": "adopt", "rationale": "r"})
    assert out == {"backend": "none", "status": "graph-only"}


def test_writeback_servicenow_is_dry_run_by_default(monkeypatch):
    monkeypatch.setenv("TRM_WRITEBACK_BACKEND", "servicenow")
    monkeypatch.delenv("SERVICENOW_ENABLE_WRITE", raising=False)
    captured: dict[str, Any] = {}

    def fake_run_writeback(target, *, dry_run, **ops):
        captured["target"] = target
        captured["dry_run"] = dry_run
        captured["work_notes"] = ops.get("work_notes")
        return {"status": "completed", "proposals": [{"op": "work_notes"}]}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.run_writeback",
        fake_run_writeback,
    )

    outcome = {
        "verdict": "adopt",
        "rationale": "wins its peer group",
        "confidence": 0.9,
        "assessmentScore": 0.8,
        "financialDelta": 0.0,
    }
    out = pi._route_writeback(
        {"id": "trm:1", "table": "u_trm_request", "sysId": "sys123"}, outcome
    )

    assert captured["target"] == "servicenow"
    assert captured["dry_run"] is True
    assert captured["work_notes"][0]["sys_id"] == "sys123"
    assert "adopt" in captured["work_notes"][0]["note"]
    assert out["backend"] == "servicenow"


def test_writeback_servicenow_live_only_when_write_enabled(monkeypatch):
    monkeypatch.setenv("TRM_WRITEBACK_BACKEND", "servicenow")
    monkeypatch.setenv("SERVICENOW_ENABLE_WRITE", "true")
    captured: dict[str, Any] = {}

    def fake_run_writeback(target, *, dry_run, **ops):
        captured["dry_run"] = dry_run
        return {"status": "completed"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.run_writeback",
        fake_run_writeback,
    )
    pi._route_writeback(
        {"id": "trm:1", "sysId": "sys123"}, {"verdict": "adopt", "rationale": "r"}
    )
    assert captured["dry_run"] is False


def test_run_trm_assessment_writes_graph_and_routes_writeback(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)
    monkeypatch.delenv("TRM_WRITEBACK_BACKEND", raising=False)
    engine = _FakeEngine(edges=_capability_edges())

    out = pi.run_trm_assessment(
        {"id": "trm:req:1", "candidateId": "prod-a"}, engine=engine
    )

    assert out["requestId"] == "trm:req:1"
    assert out["writeback"] == {"backend": "none", "status": "graph-only"}
    assert cap.calls, "expected Assessment/Recommendation nodes to be written"
    written_types = {e["type"] for call in cap.calls for e in call["entities"]}
    assert {"Assessment", "Recommendation", "TRMRequest"} <= written_types


# ── 6. guarded no-engine no-op ─────────────────────────────────────────────────


def test_assess_candidate_degrades_without_an_engine(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    out = pi.assess_candidate({"candidateId": "prod-a"})
    assert out["verdict"] == "reject"
    assert "no engine" in out["rationale"]


def test_rationalize_portfolio_degrades_without_an_engine(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    out = pi.rationalize_portfolio()
    assert out == {"clusters": 0, "recommendations": []}


def test_assess_candidate_requires_a_candidate_id():
    out = pi.assess_candidate({}, engine=_FakeEngine())
    assert out["verdict"] == "reject"
    assert "candidateId" in out["rationale"]


# ── weight resolution ───────────────────────────────────────────────────────


def test_resolve_weights_applies_goal_type_policy_and_normalizes():
    weights = pi.resolve_weights(goal_types=["efficiency"])
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # efficiency raises cost + consolidation-benefit relative to the defaults
    baseline = pi.resolve_weights(goal_types=[])
    assert weights["cost"] > baseline["cost"]
    assert weights["consolidation-benefit"] > baseline["consolidation-benefit"]


def test_resolve_weights_reads_goal_types_from_the_engine():
    engine = _FakeEngine(
        by_label={"StrategicGoal": [("goal-1", {"goalType": "compliance"})]}
    )
    weights = pi.resolve_weights(engine)
    baseline = pi.resolve_weights(goal_types=[])
    assert weights["compliance-ato"] > baseline["compliance-ato"]


# ── gates: EOL + gov ATO ────────────────────────────────────────────────────


def test_eol_gate_rejects_a_product_past_its_end_of_life():
    engine = _FakeEngine(node_props={"prod-a": {"endOfLifeDate": "2020-01-01"}})
    check = pi._check_eol("prod-a", engine)
    assert check.passed is False


def test_eol_gate_passes_when_no_eol_recorded():
    engine = _FakeEngine()
    check = pi._check_eol("prod-a", engine)
    assert check.passed is True


def test_ato_gate_not_applicable_for_commercial_profile():
    engine = _FakeEngine()
    check = pi._check_ato("prod-a", {"profile": "commercial"}, engine)
    assert check.passed is True


def test_ato_gate_rejects_gov_candidate_without_authorization():
    engine = _FakeEngine()
    check = pi._check_ato("prod-a", {"profile": "gov"}, engine)
    assert check.passed is False


def test_ato_gate_passes_gov_candidate_with_authorized_ato():
    engine = _FakeEngine(
        edges=[("prod-a", "authorizedBy", "ato-1")],
        node_props={"ato-1": {"atoStatus": "authorized"}},
    )
    check = pi._check_ato("prod-a", {"profile": "gov"}, engine)
    assert check.passed is True
