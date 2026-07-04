"""Execution-time workflow ontology gate (CONCEPT:AU-ORCH.execution.ontology-validation-execution-path).

A stored WorkflowDefinition is SHACL-validated (WorkflowDefinitionShape /
WorkflowStepShape in governance.shapes.ttl) before dispatch — malformed
definitions are refused with a structured report; with KG_BRAIN_ENFORCE on,
the ontology permissioning row gate (markings + ACLs, fail-closed) is applied
to the workflow node for the current actor.

@pytest.mark.concept("AU-ORCH.execution.ontology-validation-execution-path")
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")
pytest.importorskip("rdflib")

from agent_utilities.knowledge_graph.core.workflow_gate import (
    gate_workflow_execution,
    workflow_shape_gate_enabled,
)

pytestmark = pytest.mark.concept("AU-ORCH.execution.ontology-validation-execution-path")


class FakeGraph:
    """Compute-mirror fake honoring nodes(data=True) / out_edges(data=True)."""

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


class FakeEngine:
    def __init__(self):
        self.graph = FakeGraph()
        self.backend = None


def _seed_workflow(engine, name="invoice_flow", step_count=2, steps=None):
    wid = f"workflow:{name}:abc123"
    engine.graph.add_node(
        wid,
        {"type": "WorkflowDefinition", "name": name, "step_count": step_count},
    )
    if steps is None:
        steps = [{"node_id": "review"}, {"node_id": "archive"}]
    for i, step in enumerate(steps):
        sid = f"{wid}:step:{i}"
        engine.graph.add_node(sid, {"type": "WorkflowStep", "step_order": i, **step})
        engine.graph.add_edge(wid, sid, type="HAS_STEP", step_order=i)
    return wid


class TestShapeGate:
    def test_valid_workflow_passes(self):
        engine = FakeEngine()
        wid = _seed_workflow(engine)
        gate = gate_workflow_execution(engine, "invoice_flow")
        assert gate["allowed"] is True
        assert gate["workflow_id"] == wid
        assert gate["violations"] == []

    def test_zero_step_workflow_refused(self):
        engine = FakeEngine()
        _seed_workflow(engine, name="empty_flow", step_count=0, steps=[])
        gate = gate_workflow_execution(engine, "empty_flow")
        assert gate["allowed"] is False
        assert any(
            "step" in str(v.get("message", "")).lower() for v in gate["violations"]
        )

    def test_step_missing_target_refused(self):
        engine = FakeEngine()
        _seed_workflow(
            engine,
            name="broken_flow",
            step_count=2,
            steps=[{"node_id": "review"}, {"node_id": ""}],  # unresolvable target
        )
        gate = gate_workflow_execution(engine, "broken_flow")
        assert gate["allowed"] is False
        assert any("node_id" in str(v.get("message", "")) for v in gate["violations"])

    def test_unstored_workflow_passes_through(self):
        engine = FakeEngine()
        gate = gate_workflow_execution(engine, "dynamic_adhoc_flow")
        assert gate["allowed"] is True
        assert gate["workflow_id"] is None

    def test_gate_off_bypasses_shape_validation(self, monkeypatch):
        from agent_utilities.core.config import config as cfg

        monkeypatch.setattr(cfg, "kg_workflow_shape_gate", False)
        assert workflow_shape_gate_enabled() is False
        engine = FakeEngine()
        _seed_workflow(engine, name="empty_flow", step_count=0, steps=[])
        gate = gate_workflow_execution(engine, "empty_flow")
        assert gate["allowed"] is True

    def test_default_flag_is_on(self):
        from agent_utilities.core.config import AgentConfig

        assert AgentConfig().kg_workflow_shape_gate is True


class TestPermissionGate:
    def _actor(self, actor_id="agent:intern", roles=()):
        from agent_utilities.models.company_brain import ActorType
        from agent_utilities.security.brain_context import ActorContext

        return ActorContext(
            actor_id=actor_id, actor_type=ActorType.AI_AGENT, roles=tuple(roles)
        )

    @pytest.fixture(autouse=True)
    def _fresh_brain(self, monkeypatch):
        # Isolate the process-wide CompanyBrain so ACLs don't leak across tests.
        import agent_utilities.knowledge_graph.core.company_brain_runtime as cbr

        monkeypatch.setattr(cbr, "_BRAIN", None)
        yield
        monkeypatch.setattr(cbr, "_BRAIN", None)

    def test_enforcement_off_skips_acl_check(self, monkeypatch):
        monkeypatch.delenv("KG_BRAIN_ENFORCE", raising=False)
        engine = FakeEngine()
        _seed_workflow(engine)
        gate = gate_workflow_execution(engine, "invoice_flow", actor=self._actor())
        assert gate["allowed"] is True

    def test_enforcement_on_acl_deny_raises_permission_error(self, monkeypatch):
        monkeypatch.setenv("KG_BRAIN_ENFORCE", "1")
        from agent_utilities.knowledge_graph.ontology.permissioning import build_acl
        from agent_utilities.models.company_brain import DataClassification

        engine = FakeEngine()
        wid = _seed_workflow(engine)
        build_acl(
            wid,
            DataClassification.RESTRICTED,
            read_roles=["workflow_operator"],
            data_owner="ops",
        )
        with pytest.raises(PermissionError) as exc:
            gate_workflow_execution(engine, "invoice_flow", actor=self._actor())
        assert "invoice_flow" in str(exc.value)

    def test_enforcement_on_acl_allow_passes(self, monkeypatch):
        monkeypatch.setenv("KG_BRAIN_ENFORCE", "1")
        from agent_utilities.knowledge_graph.ontology.permissioning import build_acl
        from agent_utilities.models.company_brain import DataClassification

        engine = FakeEngine()
        wid = _seed_workflow(engine)
        build_acl(
            wid,
            DataClassification.INTERNAL,
            read_roles=["workflow_operator"],
            data_owner="ops",
        )
        gate = gate_workflow_execution(
            engine,
            "invoice_flow",
            actor=self._actor(roles=("workflow_operator",)),
        )
        assert gate["allowed"] is True


class TestExecuteWorkflowWiring:
    def test_execute_workflow_action_gates_before_dispatch(self):
        """The MCP execute_workflow branch calls the gate before orch dispatch."""
        import inspect

        from agent_utilities.mcp.tools import analysis_tools

        source = inspect.getsource(analysis_tools)
        gate_idx = source.find("gate_workflow_execution(engine, gate_name)")
        dispatch_idx = source.find("await orch.execute_workflow(")
        assert gate_idx != -1, "execute_workflow must run the ORCH-1.42 gate"
        assert dispatch_idx != -1
        assert gate_idx < dispatch_idx, "gate must run BEFORE dispatch"


class TestDispatchWorkflowWiring:
    """The background twin (action='dispatch_workflow') runs the SAME gate.

    execute_workflow was gated at ORCH-1.42 ship time; dispatch_workflow (the
    fire-and-forget background dispatch, also the REST twin
    /api/graph/orchestrate/dispatch-workflow) was a documented follow-up —
    these tests pin the closed gap: malformed stored definitions are refused
    BEFORE any background task is created, valid ones dispatch, and the
    KG_WORKFLOW_SHAPE_GATE flag keeps the same default-on / off-bypass
    semantics as the foreground path.
    """

    @pytest.fixture()
    def dispatch(self, monkeypatch):
        """(engine, name) -> tool output, with a recording fake runner."""
        import asyncio

        import agent_utilities.orchestration as orch_mod
        from agent_utilities.mcp import kg_server

        kg_server.ensure_tools_registered()

        class _FakeRunner:
            instances: list = []

            def __init__(self):
                type(self).instances.append(self)
                self.calls: list[dict] = []

            async def execute_workflow(self, **kwargs):
                self.calls.append(kwargs)
                return {"ok": True}

        _FakeRunner.instances = []
        monkeypatch.setattr(orch_mod, "AgentOrchestrationEngine", _FakeRunner)

        async def _run(engine, name):
            monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
            out = await kg_server._execute_tool(
                "graph_orchestrate", action="dispatch_workflow", agent_name=name
            )
            await asyncio.sleep(0)  # let any created background task start
            return out

        return _run, _FakeRunner

    async def test_malformed_workflow_refused_before_background_dispatch(
        self, dispatch
    ):
        import json

        run, runner = dispatch
        engine = FakeEngine()
        _seed_workflow(engine, name="empty_flow", step_count=0, steps=[])
        out = await run(engine, "empty_flow")
        payload = json.loads(out)
        assert "background dispatch refused" in payload["error"]
        assert payload["violations"]
        assert runner.instances == [], "no background task for a refused workflow"

    async def test_valid_workflow_dispatches_in_background(self, dispatch):
        run, runner = dispatch
        engine = FakeEngine()
        _seed_workflow(engine)
        out = await run(engine, "invoice_flow")
        assert "Workflow dispatched in background" in out
        assert runner.instances and runner.instances[0].calls
        assert runner.instances[0].calls[0]["workflow_id"] == "invoice_flow"

    async def test_gate_off_bypasses_shape_validation(self, dispatch, monkeypatch):
        from agent_utilities.core.config import config as cfg

        monkeypatch.setattr(cfg, "kg_workflow_shape_gate", False)
        run, runner = dispatch
        engine = FakeEngine()
        _seed_workflow(engine, name="empty_flow", step_count=0, steps=[])
        out = await run(engine, "empty_flow")
        assert "Workflow dispatched in background" in out
        assert runner.instances, "gate off must fall through to dispatch"

    def test_dispatch_workflow_action_gates_before_background_task(self):
        """Source order: the gate runs BEFORE asyncio.create_task in the branch."""
        import inspect

        from agent_utilities.mcp.tools import analysis_tools

        source = inspect.getsource(analysis_tools)
        branch_idx = source.find('elif action == "dispatch_workflow":')
        assert branch_idx != -1
        gate_idx = source.find("gate_workflow_execution(engine, gate_name)", branch_idx)
        task_idx = source.find("asyncio.create_task(", branch_idx)
        assert gate_idx != -1, "dispatch_workflow must run the ORCH-1.42 gate"
        assert task_idx != -1
        assert gate_idx < task_idx, "gate must run BEFORE background dispatch"
