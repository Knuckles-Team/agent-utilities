"""Execution-time workflow ontology gate (CONCEPT:AU-ORCH.execution.ontology-validation-execution-path).

A stored WorkflowDefinition is SHACL-validated (WorkflowDefinitionShape /
WorkflowStepShape in governance.shapes.ttl) before dispatch — malformed
definitions are refused with a structured report; the mandatory ontology
permissioning row gate (markings + ACLs, fail-closed) is applied
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


def _seed_workflow(engine, name="invoice_flow", step_count=2, steps=None, *, acl=True):
    wid = f"workflow:{name}:abc123"
    engine.graph.add_node(
        wid,
        {
            "node_type": "WorkflowDefinition",
            "name": name,
            "step_count": step_count,
        },
    )
    if acl:
        from agent_utilities.knowledge_graph.ontology.permissioning import build_acl
        from agent_utilities.models.company_brain import DataClassification

        build_acl(wid, DataClassification.PUBLIC)
    if steps is None:
        steps = [{"node_id": "review"}, {"node_id": "archive"}]
    for i, step in enumerate(steps):
        sid = f"{wid}:step:{i}"
        engine.graph.add_node(
            sid, {"node_type": "WorkflowStep", "step_order": i, **step}
        )
        engine.graph.add_edge(wid, sid, relationship="HAS_STEP", step_order=i)
    return wid


@pytest.fixture(autouse=True)
def _governed_permission_state(monkeypatch):
    import agent_utilities.knowledge_graph.core.company_brain_runtime as cbr
    from agent_utilities.knowledge_graph.ontology.permissioning import (
        clear_markings,
        set_marking_store,
    )

    class Store:
        @staticmethod
        def execute(_query, _params):
            return []

    monkeypatch.setattr(cbr, "_BRAIN", None)
    clear_markings()
    set_marking_store(Store())
    yield
    clear_markings()
    monkeypatch.setattr(cbr, "_BRAIN", None)


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

    def test_unstored_workflow_is_refused(self):
        engine = FakeEngine()
        gate = gate_workflow_execution(engine, "dynamic_adhoc_flow")
        assert gate["allowed"] is False
        assert gate["workflow_id"] is None
        assert gate["violations"] == [
            {
                "code": "workflow_definition_missing",
                "message": "A persisted WorkflowDefinition is required for execution.",
            }
        ]

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
            actor_id=actor_id,
            actor_type=ActorType.AI_AGENT,
            roles=tuple(roles),
            tenant_id="tenant-a",
            authenticated=True,
        )

    def test_missing_acl_denies(self):
        engine = FakeEngine()
        _seed_workflow(engine, acl=False)
        with pytest.raises(PermissionError):
            gate_workflow_execution(engine, "invoice_flow", actor=self._actor())

    def test_acl_deny_raises_permission_error(self):
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
        with pytest.raises(PermissionError):
            gate_workflow_execution(engine, "invoice_flow", actor=self._actor())

    def test_acl_allow_passes(self):
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
        """The focused workflow execute branch gates before orchestration."""
        import inspect

        from agent_utilities.mcp.tools import workflow_tools

        source = inspect.getsource(workflow_tools)
        gate_idx = source.find("gate = _workflow_gate(engine, workflow)")
        dispatch_idx = source.find("await orchestrator.execute_workflow(")
        assert gate_idx != -1, "execute must run the ORCH-1.42 gate"
        assert dispatch_idx != -1
        assert gate_idx < dispatch_idx, "gate must run BEFORE dispatch"


class TestDispatchWorkflowWiring:
    """The focused background dispatch runs the same gate and preserves run ids."""

    @pytest.fixture()
    def dispatch(self, monkeypatch):
        """(engine, name) -> tool output, with a recording fake runner."""
        import asyncio

        from agent_utilities.mcp import kg_server
        from agent_utilities.mcp.tools.workflow_tools import register_workflow_tools
        from agent_utilities.workflows import runner as runner_mod

        class _FakeMCP:
            def tool(self, **_kwargs):
                return lambda fn: fn

        register_workflow_tools(_FakeMCP())

        class _FakeRunner:
            instances: list = []

            def __init__(self):
                type(self).instances.append(self)
                self.calls: list[dict] = []

            async def execute_by_name(self, workflow, engine, **kwargs):
                del engine
                self.calls.append({"workflow": workflow, **kwargs})

                class _Result:
                    def to_dict(self):
                        return {"status": "completed", **kwargs}

                return _Result()

        _FakeRunner.instances = []
        monkeypatch.setattr(runner_mod, "WorkflowRunner", _FakeRunner)

        async def _run(engine, name):
            monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
            out = await kg_server._execute_tool(
                "graph_workflows", action="dispatch", workflow=name
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
        assert "execution refused" in payload["error"]
        assert payload["violations"]
        assert runner.instances == [], "no background task for a refused workflow"

    async def test_valid_workflow_dispatches_in_background(self, dispatch):
        import json

        from agent_utilities.mcp import kg_server

        run, runner = dispatch
        engine = FakeEngine()
        _seed_workflow(engine)
        out = await run(engine, "invoice_flow")
        payload = json.loads(out)
        assert payload["status"] == "dispatched"
        assert runner.instances and runner.instances[0].calls
        call = runner.instances[0].calls[0]
        assert call["workflow"] == "invoice_flow"
        assert call["trace_session"] == payload["session_id"]
        status = json.loads(
            await kg_server._execute_tool(
                "graph_workflows",
                action="status",
                workflow=payload["session_id"],
            )
        )
        assert status["trace_session"] == payload["session_id"]

    async def test_gate_off_bypasses_shape_validation(self, dispatch, monkeypatch):
        import json

        from agent_utilities.core.config import config as cfg

        monkeypatch.setattr(cfg, "kg_workflow_shape_gate", False)
        run, runner = dispatch
        engine = FakeEngine()
        _seed_workflow(engine, name="empty_flow", step_count=0, steps=[])
        out = await run(engine, "empty_flow")
        assert json.loads(out)["status"] == "dispatched"
        assert runner.instances, "gate off must fall through to dispatch"

    def test_dispatch_workflow_action_gates_before_background_task(self):
        """Source order: the gate runs BEFORE asyncio.create_task in the branch."""
        import inspect

        from agent_utilities.mcp.tools import workflow_tools

        source = inspect.getsource(workflow_tools)
        branch_idx = source.find('if action in {"execute", "dispatch"}:')
        assert branch_idx != -1
        gate_idx = source.find("gate = _workflow_gate(engine, workflow)", branch_idx)
        task_idx = source.find("asyncio.create_task(", branch_idx)
        assert gate_idx != -1, "dispatch_workflow must run the ORCH-1.42 gate"
        assert task_idx != -1
        assert gate_idx < task_idx, "gate must run BEFORE background dispatch"
