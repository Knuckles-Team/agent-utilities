"""Tests for the connector → skill synthesis distiller (CONCEPT:AU-KG.ontology.connector-agnostic-proposal/2.83).

Covers: classify (single-action→atomic, flowsTo-chain→workflow), dedup (existing
skill ⇒ no proposal), proposal node shape + AUTOMATES/DERIVED_FROM edges, the
dual-mode artifact, per-connector coverage (camunda/aris/egeria/leanix), and the
LIVE-PATH (LoopController.run_one_cycle populates report["skill_proposals"] and
SkillProposal nodes land in the graph).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.distillation.skill_synthesizer import (
    ConnectorSkillDistiller,
    SkillCandidate,
    render_workflow_skill_md,
)


# --------------------------------------------------------------------------- #
# A NodeView-compatible fake graph (callable + subscriptable, like the engine).
# --------------------------------------------------------------------------- #
class _NodeView:
    def __init__(self, graph: "FakeGraph") -> None:
        self._g = graph

    def __call__(self, data: bool = False):
        if data:
            return list(self._g._nodes.items())
        return list(self._g._nodes)

    def __getitem__(self, nid: str) -> dict:
        return self._g._nodes[nid]

    def __iter__(self):
        return iter(self._g._nodes)

    def __contains__(self, nid: str) -> bool:
        return nid in self._g._nodes


class FakeGraph:
    """Minimal NX-compatible compute-graph mirror used by the distiller."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str, dict]] = []

    @property
    def nodes(self) -> _NodeView:
        return _NodeView(self)

    def add_node(self, nid: str, props: dict) -> None:
        self._nodes[nid] = dict(props)

    def add_edge(self, src: str, tgt: str, **props: Any) -> None:
        self._edges.append((src, tgt, props))

    def in_edges(self, nid: str, data: bool = False):
        rows = [(s, t, p) for s, t, p in self._edges if t == nid]
        return rows if data else [(s, t) for s, t, _ in rows]

    def out_edges(self, nid: str, data: bool = False):
        rows = [(s, t, p) for s, t, p in self._edges if s == nid]
        return rows if data else [(s, t) for s, t, _ in rows]


class FakeEngine:
    """Seeded in-memory engine — add_node / link_nodes land in the FakeGraph."""

    def __init__(self) -> None:
        self.graph = FakeGraph()
        self.backend = None

    def add_node(self, node_id, node_type, properties=None, **props) -> None:
        self.graph.add_node(node_id, {"type": node_type, **(properties or props or {})})

    def link_nodes(self, source, target, rel_type, properties=None) -> None:
        self.graph.add_edge(source, target, type=rel_type, **(properties or {}))

    # the distiller never calls this on the fake path, but compiler/reasoner may
    def query_cypher(self, *a, **k):  # noqa: D401 - inert fake for tests
        return []


# --------------------------------------------------------------------------- #
# seeding helpers — one ontology, many connectors
# --------------------------------------------------------------------------- #
def _seed_process(
    engine: FakeEngine,
    *,
    proc_id: str,
    proc_name: str,
    task_labels: list[str],
    id_prefix: str,
) -> None:
    """Seed a BusinessProcess with a flowsTo-chain of BusinessTasks.

    Mirrors how every connector lifts into the SAME ontology classes
    (BusinessProcess / BusinessTask PART_OF / FLOWS_TO).
    """
    engine.add_node(proc_id, "BusinessProcess", {"name": proc_name})
    task_ids = []
    for i, label in enumerate(task_labels):
        tid = f"{id_prefix}_task:{proc_id}:{i}"
        engine.add_node(
            tid,
            "BusinessTask",
            {"name": label, "element_id": f"el{i}", "task_type": "serviceTask"},
        )
        engine.link_nodes(tid, proc_id, "PART_OF")
        task_ids.append(tid)
    for a, b in zip(task_ids, task_ids[1:], strict=False):
        engine.link_nodes(a, b, "FLOWS_TO")


def _camunda_engine() -> FakeEngine:
    e = FakeEngine()
    _seed_process(
        e,
        proc_id="bpmn_process:invoice",
        proc_name="Invoice Approval",
        task_labels=["receive invoice", "validate amount", "approve payment"],
        id_prefix="bpmn",
    )
    return e


def _aris_engine() -> FakeEngine:
    e = FakeEngine()
    _seed_process(
        e,
        proc_id="aris_model:onboarding",
        proc_name="Employee Onboarding",
        task_labels=["create account", "assign equipment", "schedule training"],
        id_prefix="aris",
    )
    return e


def _egeria_engine() -> FakeEngine:
    e = FakeEngine()
    _seed_process(
        e,
        proc_id="egeria_process:ingest",
        proc_name="Data Ingest Pipeline",
        task_labels=["fetch dataset", "profile columns", "publish catalog"],
        id_prefix="egeria",
    )
    return e


def _leanix_engine() -> FakeEngine:
    """LeanIX maps capabilities + processes; seed a capability + a process."""
    e = FakeEngine()
    e.add_node(
        "leanix_cap:reporting", "BusinessCapability", {"name": "Financial Reporting"}
    )
    _seed_process(
        e,
        proc_id="leanix_process:close",
        proc_name="Month End Close",
        task_labels=["lock ledgers", "reconcile accounts", "generate report"],
        id_prefix="leanix",
    )
    return e


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
def test_classify_flowsto_chain_becomes_workflow_and_atomic_steps():
    engine = _camunda_engine()
    d = ConnectorSkillDistiller(engine)
    candidates = d.classify(d.discover())
    workflows = [c for c in candidates if c.kind == "workflow"]
    atomics = [c for c in candidates if c.kind == "atomic"]
    assert len(workflows) == 1, "a 3-task flowsTo chain is one workflow candidate"
    wf = workflows[0]
    assert wf.name == "invoice-approval"
    assert len(wf.steps) == 3
    # each step maps to an atomic candidate created alongside it
    assert {s["name"] for s in wf.steps} == {
        "receive-invoice",
        "validate-amount",
        "approve-payment",
    }
    assert len(atomics) == 3
    # the workflow automates the source process
    assert wf.automates == "bpmn_process:invoice"
    # dependency wiring: step 2 depends on step 1, step 3 on step 2
    deps = {s["name"]: s.get("depends_on", []) for s in wf.steps}
    assert deps["validate-amount"] == [1]
    assert deps["approve-payment"] == [2]


def test_classify_single_action_becomes_atomic():
    engine = FakeEngine()
    _seed_process(
        engine,
        proc_id="bpmn_process:single",
        proc_name="Single Step Proc",
        task_labels=["send notification"],
        id_prefix="bpmn",
    )
    d = ConnectorSkillDistiller(engine)
    candidates = d.classify(d.discover())
    assert all(c.kind == "atomic" for c in candidates)
    names = {c.name for c in candidates}
    assert "send-notification" in names
    # no workflow for a lone action
    assert not [c for c in candidates if c.kind == "workflow"]


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def test_dedup_existing_skill_is_not_reproposed():
    engine = _camunda_engine()
    # an existing atomic skill that already covers one of the steps (by name)
    engine.add_node("skill:validate-amount", "skill", {"name": "validate-amount"})
    d = ConnectorSkillDistiller(engine)
    candidates = d.classify(d.discover())
    kept = d.dedup(candidates)
    kept_names = {c.name for c in kept}
    assert "validate-amount" not in kept_names, "covered by an existing skill name"
    # the other novel candidates survive
    assert "receive-invoice" in kept_names
    covered = [c for c in candidates if c.novelty == "covered"]
    assert {c.name for c in covered} == {"validate-amount"}


# --------------------------------------------------------------------------- #
# propose — node shape + provenance edges
# --------------------------------------------------------------------------- #
def test_propose_writes_nodes_with_automates_and_derived_from_edges():
    engine = _camunda_engine()
    d = ConnectorSkillDistiller(engine)
    report = d.run()
    assert report.proposed > 0
    nodes = dict(engine.graph.nodes(data=True))
    # the workflow proposal node exists with the right type + propose-only status
    wf_id = "skill_workflow_proposal:invoice-approval"
    assert wf_id in nodes
    wf = nodes[wf_id]
    assert wf["type"] == "skill_workflow_proposal"
    assert wf["proposal_status"] == "proposal"
    assert wf["provenance"].startswith("camunda:")
    assert isinstance(wf["trigger_patterns"], list) and wf["trigger_patterns"]
    # AUTOMATES + DERIVED_FROM edges from the proposal
    edges = engine.graph._edges
    automates = [
        (s, t) for s, t, p in edges if p.get("type") == "AUTOMATES" and s == wf_id
    ]
    derived = [
        (s, t) for s, t, p in edges if p.get("type") == "DERIVED_FROM" and s == wf_id
    ]
    composes = [
        (s, t) for s, t, p in edges if p.get("type") == "COMPOSES" and s == wf_id
    ]
    assert automates == [(wf_id, "bpmn_process:invoice")]
    assert derived == [(wf_id, "bpmn_process:invoice")]
    assert len(composes) == 3, "workflow COMPOSES its 3 atomic step proposals"
    # an atomic proposal exists too
    assert "skill_proposal:receive-invoice" in nodes


# --------------------------------------------------------------------------- #
# dual-mode artifact
# --------------------------------------------------------------------------- #
def test_dual_mode_artifact_has_dag_and_execution_and_footer():
    engine = _camunda_engine()
    d = ConnectorSkillDistiller(engine)
    candidates = d.dedup(d.classify(d.discover()))
    wf = next(c for c in candidates if c.kind == "workflow")
    md = render_workflow_skill_md(wf)
    # frontmatter contract
    assert md.startswith("---\n")
    assert "name: invoice-approval" in md
    assert "team_config:" in md
    assert "specialist_ids:" in md
    assert "concept: KG-2.90" in md
    # machine-readable DAG: ### Step N with depends_on
    assert "### Step 1: receive-invoice" in md
    assert "[depends_on: Step 1]" in md
    assert "[depends_on: Step 2]" in md
    # Claude-executable Execution section
    assert "## Execution" in md
    assert "Run every step with NO `depends_on` in parallel" in md
    # standard graph-os delegation footer
    assert "graph_orchestrate action=execute_workflow" in md
    assert "kg-delegate" in md
    assert "otherwise execute steps natively in dependency order" in md


def test_draft_artifact_writes_to_staging_not_repo(tmp_path):
    engine = _camunda_engine()
    d = ConnectorSkillDistiller(engine, staging_root=tmp_path)
    candidates = d.dedup(d.classify(d.discover()))
    wf = next(c for c in candidates if c.kind == "workflow")
    path = d.draft_artifact(wf)
    assert str(tmp_path) in path
    assert path.endswith("SKILL.md")
    text = (tmp_path / wf.name / "SKILL.md").read_text()
    assert "## Steps" in text


# --------------------------------------------------------------------------- #
# connector coverage — connector-agnostic over the ontology
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("factory", "expect_name", "expect_system"),
    [
        (_camunda_engine, "invoice-approval", "camunda"),
        (_aris_engine, "employee-onboarding", "aris"),
        (_egeria_engine, "data-ingest-pipeline", "egeria"),
        (_leanix_engine, "month-end-close", "leanix"),
    ],
)
def test_every_connector_yields_a_correct_candidate(
    factory, expect_name, expect_system
):
    engine = factory()
    d = ConnectorSkillDistiller(engine)
    candidates = d.classify(d.discover())
    wf = [c for c in candidates if c.kind == "workflow"]
    assert wf, f"{expect_system}: a flowsTo chain must yield a workflow candidate"
    assert any(c.name == expect_name for c in wf)
    assert all(c.source_system == expect_system for c in wf)


def test_leanix_capability_becomes_atomic_skill():
    engine = _leanix_engine()
    d = ConnectorSkillDistiller(engine)
    candidates = d.classify(d.discover())
    names = {c.name for c in candidates if c.kind == "atomic"}
    assert "financial-reporting" in names, "a Capability → atomic skill candidate"


# --------------------------------------------------------------------------- #
# materialize — human-approved proposal → physical SKILL.md (staging)
# --------------------------------------------------------------------------- #
def test_materialize_approved_proposal_writes_skill_md_and_stamps_node(tmp_path):
    engine = _camunda_engine()
    d = ConnectorSkillDistiller(engine, staging_root=tmp_path)
    d.run()
    pid = "skill_proposal:receive-invoice"
    res = d.materialize(pid)
    assert res["proposal_id"] == pid
    assert res["status"] in ("approved", "drafted")
    # the SKILL.md exists in the staging dir (never a repo)
    assert str(tmp_path) in res["skill_md"]
    assert (tmp_path / "receive-invoice" / "SKILL.md").exists()
    # the node is stamped approved
    nodes = dict(engine.graph.nodes(data=True))
    assert nodes[pid]["proposal_status"] == "approved"


def test_materialize_unknown_proposal_is_not_found():
    engine = _camunda_engine()
    d = ConnectorSkillDistiller(engine)
    res = d.materialize("skill_proposal:does-not-exist")
    assert res["status"] == "not_found"


# --------------------------------------------------------------------------- #
# LIVE-PATH — LoopController.run_one_cycle invokes the distiller
# --------------------------------------------------------------------------- #
def test_loop_cycle_populates_skill_proposals_live_path():
    from agent_utilities.knowledge_graph.research.loop_controller import LoopController

    engine = _camunda_engine()
    controller = LoopController(engine)
    # keep the heavy/external stages off; the distill stage is default-ON and
    # must run + populate the report + land nodes as a side effect of the cycle.
    report = controller.run_one_cycle(
        assimilate=False,
        reason=False,
        synthesize=False,
        breadth=False,
        distill=False,
        standardize=False,
    )
    assert "skill_proposals" in report
    sp = report["skill_proposals"]
    assert sp is not None and sp.get("proposed", 0) >= 1
    # nodes actually exist in the graph
    nodes = dict(engine.graph.nodes(data=True))
    assert any(
        n.get("type") in ("skill_proposal", "skill_workflow_proposal")
        for n in nodes.values()
    )
    # timing recorded in metrics
    assert "distill_skills" in report["metrics"]["stage_ms"]
