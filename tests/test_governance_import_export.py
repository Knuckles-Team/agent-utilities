"""Governance-process importers + the export_workflow exporter
(``reports/autonomous-sdlc-loop-design.md`` §6.0/§7.1, decision #4).

Each external governance model (Camunda BPMN, ARIS EPC, ArchiMate business
process, OneTrust/ERPNext approval) imports into the ONE executable
``:WorkflowDefinition``/``:WorkflowStep`` target with its approval constructs as
``kind="gate"`` steps; ``export_workflow`` round-trips a stored workflow back out
to BPMN / SKILL.md.

@pytest.mark.concept("AU-KG.ontology.connector-agnostic-proposal")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.governance_import import (
    GovernanceImporter,
    export_workflow,
    looks_like_gate,
)

pytestmark = pytest.mark.concept("AU-KG.ontology.connector-agnostic-proposal")


class FakeGraph:
    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str, dict]] = []

    def add_node(self, node_id, props):
        self._nodes[node_id] = dict(props)

    def add_edge(self, src, tgt, **props):
        self._edges.append((src, tgt, props))

    @property
    def nodes(self):
        outer = self

        class _View(dict):
            def __call__(self, data=False):
                if data:
                    return list(outer._nodes.items())
                return list(outer._nodes)

        return _View(outer._nodes)

    def out_edges(self, node_id, data=False):
        rows = [(s, t, p) for s, t, p in self._edges if s == node_id]
        return rows if data else [(s, t) for s, t, _ in rows]

    def in_edges(self, node_id, data=False):
        rows = [(s, t, p) for s, t, p in self._edges if t == node_id]
        return rows if data else [(s, t) for s, t, _ in rows]


class FakeBackend:
    """Honors the ORCH-1.23 agent-matching query surface (for BPMN delegate)."""

    def execute(self, query, params=None):
        return []


class FakeEngine:
    def __init__(self, with_backend=False):
        self.graph = FakeGraph()
        self.backend = FakeBackend() if with_backend else None

    def add_node(self, node_id, node_type, properties=None, **props):
        self.graph.add_node(node_id, {"type": node_type, **(properties or props or {})})

    def link_nodes(self, source, target, rel_type, properties=None):
        self.graph.add_edge(source, target, type=rel_type, **(properties or {}))

    def search_hybrid(self, query, top_k=3):
        return []


# --------------------------------------------------------------------------- #
# fixtures: seed each source model into the fake KG
# --------------------------------------------------------------------------- #
def _seed_epc(engine: FakeEngine) -> str:
    """ARIS EPC: start → Validate Request → Manager Approval → end."""
    model = "aris:model:m1"
    engine.add_node(model, "ProcessModel", name="Onboarding")
    objs = {
        "aris:object:e0": ("EPCEvent", "Request Received"),
        "aris:object:f1": ("EPCFunction", "Validate Request"),
        "aris:object:e1": ("EPCEvent", "Validated"),
        "aris:object:f2": ("EPCFunction", "Manager Approval"),
        "aris:object:e2": ("EPCEvent", "Done"),
    }
    for oid, (t, n) in objs.items():
        engine.add_node(oid, t, name=n, objectType=t)
        engine.link_nodes(model, oid, "hasObject")
    seq = [
        "aris:object:e0",
        "aris:object:f1",
        "aris:object:e1",
        "aris:object:f2",
        "aris:object:e2",
    ]
    for a, b in zip(seq, seq[1:], strict=False):
        engine.link_nodes(a, b, "flowsTo")
    return model


def _seed_archimate(engine: FakeEngine) -> str:
    """ArchiMate: Intake (BusinessProcess) -Triggering-> Compliance Review -Triggering-> Fulfil."""
    p = "archimate:BusinessProcess:p1"
    engine.add_node(p, "BusinessProcess", name="Intake")
    r = "archimate:BusinessProcess:p2"
    engine.add_node(r, "BusinessProcess", name="Compliance Review")
    f = "archimate:BusinessProcess:p3"
    engine.add_node(f, "BusinessProcess", name="Fulfil Order")
    engine.link_nodes(p, r, "Triggering")
    engine.link_nodes(r, f, "Triggering")
    return p


# --------------------------------------------------------------------------- #
def test_gate_heuristic():
    assert looks_like_gate({"name": "Manager Approval"})
    assert looks_like_gate({"name": "DPIA assessment"})
    assert looks_like_gate({"is_gate": True})
    assert not looks_like_gate({"name": "Ship Order"})


def test_import_epc_yields_workflow_with_gate_step():
    engine = FakeEngine()
    model = _seed_epc(engine)
    rep = GovernanceImporter(engine).import_epc(model)

    assert "workflow_id" in rep, rep
    assert rep["step_count"] == 2  # two functions; events collapsed
    assert rep["gate_count"] == 1  # "Manager Approval" → gate
    # the WorkflowDefinition + a gate WorkflowStep + REALIZES edge landed
    steps = [
        d for _n, d in engine.graph.nodes(data=True) if d.get("type") == "WorkflowStep"
    ]
    kinds = {s["kind"] for s in steps}
    assert "gate" in kinds
    assert any(p.get("type") == "REALIZES" for _s, _t, p in engine.graph._edges)


def test_import_archimate_walks_triggering_chain():
    engine = FakeEngine()
    p = _seed_archimate(engine)
    rep = GovernanceImporter(engine).import_archimate(p)

    assert rep["step_count"] == 3
    assert rep["gate_count"] == 1  # "Compliance Review" → gate
    # dependency order preserved: fulfil depends on review depends on intake
    steps = sorted(
        (
            d
            for _n, d in engine.graph.nodes(data=True)
            if d.get("type") == "WorkflowStep"
        ),
        key=lambda s: s["step_order"],
    )
    assert [s["node_id"] for s in steps] == [
        "intake",
        "compliance_review",
        "fulfil_order",
    ]


def test_import_onetrust_single_gate():
    engine = FakeEngine()
    engine.add_node("onetrust:assessment:a1", "Assessment", name="Vendor DPIA")
    rep = GovernanceImporter(engine).import_approval_gate(
        "onetrust:assessment:a1", "onetrust"
    )
    assert rep["step_count"] == 1
    assert rep["gate_count"] == 1
    step = next(
        d for _n, d in engine.graph.nodes(data=True) if d.get("type") == "WorkflowStep"
    )
    assert step["kind"] == "gate"
    assert step["boundCapability"] == "onetrust_assessments"


def test_import_erpnext_single_gate():
    engine = FakeEngine()
    rep = GovernanceImporter(engine).import_approval_gate(
        "erpnext:workflow:purchase", "erpnext", name="Purchase Approval"
    )
    step = next(
        d for _n, d in engine.graph.nodes(data=True) if d.get("type") == "WorkflowStep"
    )
    assert step["kind"] == "gate"
    assert step["boundCapability"] == "erpnext_workflow"
    assert rep["gate_count"] == 1


async def test_import_bpmn_delegates_to_process_compiler():
    """Camunda BPMN import reuses ProcessPlanCompiler end-to-end from the lifted
    BusinessProcess subgraph (the extractor fixture)."""
    from agent_utilities.knowledge_graph.enrichment.extractors.camunda import extract
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch
    from tests.unit.knowledge_graph.enrichment.bpmn_fixtures import XmlCapableClient

    engine = FakeEngine(with_backend=True)

    class _Writer:
        def add_node(self, node_id, **props):
            engine.graph.add_node(
                node_id, {"type": props.pop("type", "Thing"), **props}
            )

        def add_edge(self, src, tgt, **props):
            engine.graph.add_edge(
                src, tgt, type=props.pop("rel_type", "RELATES_TO"), **props
            )

        def execute(self, q, p=None):
            return []

        def execute_batch(self, q, batch):
            if "MERGE (n:" in q:
                for row in batch:
                    self.add_node(
                        row["id"], **{k: v for k, v in row.items() if k != "id"}
                    )
            elif "MERGE (s)-[r:" in q:
                for row in batch:
                    self.add_edge(
                        row["source"],
                        row["target"],
                        rel_type=row.get("type"),
                        **{
                            k: v
                            for k, v in row.items()
                            if k not in ("source", "target", "type")
                        },
                    )
            return []

    batch = extract({"client": XmlCapableClient()})
    write_batch(_Writer(), batch)

    rep = await GovernanceImporter(engine).import_bpmn("bpmn_process:invoice:1:abc")
    assert rep.get("translator") == "camunda-bpmn"
    assert "workflow_id" in rep, rep


# --------------------------------------------------------------------------- #
# exporter round-trip
# --------------------------------------------------------------------------- #
def test_export_workflow_to_bpmn_marks_gate_as_usertask():
    engine = FakeEngine()
    model = _seed_epc(engine)
    rep = GovernanceImporter(engine).import_epc(model)
    out = export_workflow(engine, rep["name"], fmt="bpmn")

    assert out["format"] == "bpmn"
    xml = out["content"]
    assert xml.startswith("<?xml")
    assert "<bpmn2:userTask" in xml  # the gate step
    assert "<bpmn2:serviceTask" in xml  # the ordinary step
    assert "sequenceFlow" in xml


def test_export_workflow_to_skill_md_round_trips_gate(tmp_path):
    engine = FakeEngine()
    model = _seed_epc(engine)
    rep = GovernanceImporter(engine).import_epc(model)
    out = export_workflow(engine, rep["name"], fmt="skill")

    md = tmp_path / "SKILL.md"
    md.write_text(out["content"], encoding="utf-8")

    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        parse_workflow_skill,
    )

    parsed = parse_workflow_skill(md)
    assert parsed is not None
    kinds = {s["kind"] for s in parsed["steps"]}
    assert "gate" in kinds  # the gate survives the KG → SKILL.md round-trip


def test_export_missing_workflow_errors():
    out = export_workflow(FakeEngine(), "nonexistent")
    assert "error" in out


def test_export_workflow_json():
    engine = FakeEngine()
    rep = GovernanceImporter(engine).import_approval_gate(
        "onetrust:a1", "onetrust", name="DPIA"
    )
    out = export_workflow(engine, rep["name"], fmt="json")
    import json as _json

    data = _json.loads(out["content"])
    assert data["steps"][0]["kind"] == "gate"
