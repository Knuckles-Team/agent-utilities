"""ProcessPlanCompiler — ontology→workflow bridge (CONCEPT:ORCH-1.41).

End-to-end: the BPMN fixture from the KG-2.53 extractor lift seeds a
BusinessProcess/BusinessTask/FLOWS_TO subgraph in a fake engine; the compiler
turns it into a stored WorkflowDefinition with a REALIZES bridge edge,
sequence-flow-derived dependencies, gateway-collapse parallelism, cycle
rejection, and explicit manual steps for unresolvable tasks.

@pytest.mark.concept("ORCH-1.41")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.extractors.camunda import extract
from agent_utilities.knowledge_graph.enrichment.registry import write_batch
from agent_utilities.knowledge_graph.process_plan_compiler import (
    ProcessCompilationError,
    ProcessPlanCompiler,
)

from .enrichment.bpmn_fixtures import XmlCapableClient

pytestmark = pytest.mark.concept("ORCH-1.41")

PROC = "bpmn_process:invoice:1:abc"


class FakeGraph:
    """Minimal compute-graph mirror (nodes / in_edges / out_edges)."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str, dict]] = []

    def add_node(self, node_id, props):
        self.nodes[node_id] = dict(props)

    def add_edge(self, src, tgt, **props):
        self._edges.append((src, tgt, props))

    def in_edges(self, node_id, data=False):
        rows = [(s, t, p) for s, t, p in self._edges if t == node_id]
        return rows if data else [(s, t) for s, t, _ in rows]

    def out_edges(self, node_id, data=False):
        rows = [(s, t, p) for s, t, p in self._edges if s == node_id]
        return rows if data else [(s, t) for s, t, _ in rows]


class FakeBackend:
    """Honors the ORCH-1.23 agent-matching query surface."""

    def __init__(self, servers=()):
        self.servers = list(servers)

    def execute(self, query, params=None):
        params = params or {}
        if "PROVIDES" in query:
            text = str(params.get("text", "")).lower()
            for name in self.servers:
                if name.lower() in text:
                    return [{"server": name, "tools": [f"{name}_tool"]}]
            return []
        return []


class FakeEngine:
    def __init__(self, servers=()):
        self.graph = FakeGraph()
        self.backend = FakeBackend(servers)

    def add_node(self, node_id, node_type, properties=None, **props):
        self.graph.add_node(node_id, {"type": node_type, **(properties or props or {})})

    def link_nodes(self, source, target, rel_type, properties=None):
        self.graph.add_edge(source, target, type=rel_type, **(properties or {}))

    def search_hybrid(self, query, top_k=3):
        return []


class _BatchWriter:
    """write_batch-compatible adapter that lands batches in the FakeGraph."""

    def __init__(self, engine):
        self.engine = engine

    def add_node(self, node_id, **props):
        node_type = props.pop("type", "Thing")
        self.engine.graph.add_node(node_id, {"type": node_type, **props})

    def add_edge(self, src, tgt, **props):
        rel = props.pop("rel_type", "RELATES_TO")
        self.engine.graph.add_edge(src, tgt, type=rel, **props)

    # The one writer (write_entities) persists via UNWIND execute_batch; route
    # those rows into the FakeGraph through the add_node/add_edge above.
    def execute(self, query, params=None):
        return []

    def execute_batch(self, query, batch):
        if "MERGE (n:" in query:
            for row in batch:
                self.add_node(row["id"], **{k: v for k, v in row.items() if k != "id"})
        elif "MERGE (s)-[r:" in query:
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


def _seeded_engine(servers=("review", "archive")):
    """Extractor fixture (KG-2.53) → KG subgraph → compiler input."""
    engine = FakeEngine(servers)
    batch = extract({"client": XmlCapableClient()})
    write_batch(_BatchWriter(engine), batch)
    return engine


class TestCompile:
    async def test_dependencies_derived_from_sequence_flows(self):
        engine = _seeded_engine()
        plan = await ProcessPlanCompiler(engine).compile(PROC)

        by_task = {s.refined_subtask: s for s in plan.steps}
        review = by_task["Review Invoice"]
        rework = by_task["Request Rework"]
        archive = by_task["Archive Invoice"]

        assert review.depends_on == []
        # rework depends on review *through* the collapsed gateway
        assert rework.depends_on == [review.id]
        # archive joins both branches: review (via gateway) and rework
        assert sorted(archive.depends_on) == sorted([review.id, rework.id])
        # the gateway itself is structure — never a step
        assert "Approved?" not in by_task
        assert len(plan.steps) == 3

    async def test_agents_resolved_via_kg_semantic_matching(self):
        engine = _seeded_engine()
        plan = await ProcessPlanCompiler(engine).compile(PROC)
        ids = {s.refined_subtask: s.id for s in plan.steps}
        assert ids["Review Invoice"] == "review"
        assert ids["Archive Invoice"] == "archive"

    async def test_branch_conditions_preserved_in_plan_metadata(self):
        engine = _seeded_engine()
        plan = await ProcessPlanCompiler(engine).compile(PROC)
        conds = {
            (c["from"], c["to"]): c["condition"]
            for c in plan.metadata["branch_conditions"]
        }
        review = "bpmn_task:invoice:1:abc:review"
        assert conds[(review, "bpmn_task:invoice:1:abc:archive")] == (
            "${approved == true}"
        )
        assert conds[(review, "bpmn_task:invoice:1:abc:rework")] == (
            "${approved == false}"
        )

    async def test_unresolved_task_kept_as_manual_step(self):
        engine = _seeded_engine(servers=("review", "archive"))
        plan = await ProcessPlanCompiler(engine).compile(PROC)
        (manual,) = [s for s in plan.steps if s.id.startswith("manual:")]
        assert manual.refined_subtask == "Request Rework"
        assert manual.metadata["unresolved"] is True
        assert plan.metadata["unresolved_tasks"] == ["Request Rework"]

    async def test_require_resolved_fails_listing_unmatched_tasks(self):
        engine = _seeded_engine(servers=("review", "archive"))
        with pytest.raises(ProcessCompilationError) as exc:
            await ProcessPlanCompiler(engine).compile(PROC, require_resolved=True)
        assert "Request Rework" in str(exc.value)

    async def test_missing_structure_is_actionable_error(self):
        engine = FakeEngine()
        with pytest.raises(ProcessCompilationError) as exc:
            await ProcessPlanCompiler(engine).compile("bpmn_process:ghost")
        assert "BPMN step-level lift" in str(exc.value)


class TestGatewayParallelism:
    async def test_gateway_branches_become_parallel_steps(self):
        """A → gateway → (B, C): B and C share deps {A} and run in parallel."""
        engine = FakeEngine(servers=("alpha", "beta", "gamma"))
        engine.add_node("proc:p1", "BusinessProcess", {"name": "P1"})
        for el, name, gw in (
            ("a", "alpha intake", False),
            ("gw", "split", True),
            ("b", "beta enrich", False),
            ("c", "gamma notify", False),
        ):
            engine.graph.add_node(
                f"task:{el}",
                {
                    "type": "BusinessTask",
                    "name": name,
                    "element_id": el,
                    "task_type": "parallelGateway" if gw else "serviceTask",
                    "is_gateway": gw,
                },
            )
            engine.graph.add_edge(f"task:{el}", "proc:p1", type="PART_OF")
        engine.graph.add_edge("task:a", "task:gw", type="FLOWS_TO")
        engine.graph.add_edge("task:gw", "task:b", type="FLOWS_TO")
        engine.graph.add_edge("task:gw", "task:c", type="FLOWS_TO")

        plan = await ProcessPlanCompiler(engine).compile("proc:p1")
        by_task = {s.refined_subtask: s for s in plan.steps}
        assert by_task["beta enrich"].depends_on == [by_task["alpha intake"].id]
        assert by_task["gamma notify"].depends_on == [by_task["alpha intake"].id]
        assert by_task["beta enrich"].parallel is True
        assert by_task["gamma notify"].parallel is True
        assert by_task["alpha intake"].parallel is False


class TestCycleRejection:
    async def test_loop_back_rejected_with_clear_message(self):
        engine = FakeEngine(servers=("alpha", "beta"))
        engine.add_node("proc:loop", "BusinessProcess", {"name": "Loop"})
        for el, name in (("a", "alpha step"), ("b", "beta step")):
            engine.graph.add_node(
                f"task:{el}",
                {
                    "type": "BusinessTask",
                    "name": name,
                    "element_id": el,
                    "task_type": "task",
                    "is_gateway": False,
                },
            )
            engine.graph.add_edge(f"task:{el}", "proc:loop", type="PART_OF")
        engine.graph.add_edge("task:a", "task:b", type="FLOWS_TO")
        engine.graph.add_edge("task:b", "task:a", type="FLOWS_TO")

        with pytest.raises(ProcessCompilationError) as exc:
            await ProcessPlanCompiler(engine).compile("proc:loop")
        message = str(exc.value)
        assert "cycle" in message
        assert "alpha step" in message and "beta step" in message


class TestCompileAndStore:
    async def test_realizes_edge_links_workflow_to_process(self):
        engine = _seeded_engine()
        report = await ProcessPlanCompiler(engine).compile_and_store(PROC)

        workflow_id = report["workflow_id"]
        assert workflow_id in engine.graph.nodes
        assert engine.graph.nodes[workflow_id]["type"] == "WorkflowDefinition"
        realizes = [
            (s, t) for s, t, p in engine.graph._edges if p.get("type") == "REALIZES"
        ]
        assert (workflow_id, PROC) in realizes

    async def test_report_carries_name_steps_and_unresolved(self):
        engine = _seeded_engine()
        report = await ProcessPlanCompiler(engine).compile_and_store(PROC)
        assert report["name"] == "process_invoice_receipt"
        assert report["step_count"] == 3
        assert report["unresolved_tasks"] == ["Request Rework"]
        assert report["process_id"] == PROC

    async def test_workflow_steps_persisted_with_dependencies(self):
        engine = _seeded_engine()
        report = await ProcessPlanCompiler(engine).compile_and_store(PROC)
        workflow_id = report["workflow_id"]
        step_nodes = [
            engine.graph.nodes[t]
            for s, t, p in engine.graph._edges
            if s == workflow_id and p.get("type") == "HAS_STEP"
        ]
        assert len(step_nodes) == 3
        join = next(n for n in step_nodes if n.get("node_id") == "archive")
        assert "review" in join["depends_on_json"]


class TestMcpSurface:
    async def test_compile_process_rest_twin_forwards_args(self):
        from unittest.mock import AsyncMock, patch

        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from agent_utilities.mcp.kg_server import (
            graph_orchestrate_compile_process_endpoint,
        )

        app = Starlette(
            routes=[
                Route(
                    "/graph/orchestrate/compile-process",
                    graph_orchestrate_compile_process_endpoint,
                    methods=["POST"],
                )
            ]
        )
        with patch(
            "agent_utilities.mcp.kg_server._execute_tool",
            new=AsyncMock(return_value='{"status": "compiled"}'),
        ) as mock_tool:
            client = TestClient(app)
            resp = client.post(
                "/graph/orchestrate/compile-process",
                json={"process_id": PROC, "name": "invoice_flow"},
            )
        assert resp.status_code == 200
        mock_tool.assert_awaited_once_with(
            "graph_orchestrate",
            action="compile_process",
            task=PROC,
            agent_name="invoice_flow",
        )
