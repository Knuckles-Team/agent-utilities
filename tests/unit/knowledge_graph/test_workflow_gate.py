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


class FakeBackend:
    """Minimal Cypher-shaped stand-in for the backend/persistent-store branch of
    :func:`workflow_gate._find_workflow` (D-WS-6) — a real deployment's ``execute()``
    on a connected backend, honoring the exact two queries that function issues
    (``MATCH (w:WorkflowDefinition) WHERE w.name = $name ...`` and the
    ``-[:HAS_STEP]->`` step lookup), never the NX/compute-graph branch
    :class:`FakeGraph` exercises.
    """

    def __init__(self):
        self._definitions: dict[str, dict] = {}
        self._steps: dict[str, list[dict]] = {}

    def add_workflow(self, wid, name, step_count, steps=None):
        self._definitions[name] = {"wid": wid, "name": name, "step_count": step_count}
        self._steps[wid] = steps or []

    def execute(self, query, params):
        if "WorkflowDefinition) WHERE w.name" in query:
            row = self._definitions.get(params.get("name"))
            return [row] if row else []
        if "HAS_STEP" in query:
            return self._steps.get(params.get("wid"), [])
        return []


class FakeBackendEngine:
    """Engine stand-in whose ``backend`` is set — routes ``_find_workflow`` down
    the backend branch, never the ``graph`` (compute-mirror) branch, exactly as a
    real connected-backend deployment does.
    """

    def __init__(self):
        self.backend = FakeBackend()
        self.graph = None


def _seed_backend_workflow(
    engine, name="invoice_flow", step_count=2, steps=None, *, acl=True
):
    wid = f"workflow:{name}:abc123"
    if acl:
        from agent_utilities.knowledge_graph.ontology.permissioning import build_acl
        from agent_utilities.models.company_brain import DataClassification

        build_acl(wid, DataClassification.PUBLIC)
    if steps is None:
        steps = [{"step_id": "review"}, {"step_id": "archive"}]
    step_rows = [
        {"sid": f"{wid}:step:{i}", "step_order": i, **step}
        for i, step in enumerate(steps)
    ]
    engine.backend.add_workflow(wid, name, step_count, step_rows)
    return wid


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


class TestShapeGateBackendPath:
    """D-WS-6 — mirrors :class:`TestShapeGate` against the backend/persistent-store
    branch of ``_find_workflow`` (``engine.backend`` set, real Cypher-shaped
    queries), not just the NX/compute-graph branch every other test in this file
    exercises. A real deployment uses the backend branch exclusively — this
    file previously had zero coverage proving it behaves identically.
    """

    def test_valid_workflow_passes(self):
        engine = FakeBackendEngine()
        wid = _seed_backend_workflow(engine)
        gate = gate_workflow_execution(engine, "invoice_flow")
        assert gate["allowed"] is True
        assert gate["workflow_id"] == wid
        assert gate["violations"] == []

    def test_zero_step_workflow_refused(self):
        engine = FakeBackendEngine()
        _seed_backend_workflow(engine, name="empty_flow", step_count=0, steps=[])
        gate = gate_workflow_execution(engine, "empty_flow")
        assert gate["allowed"] is False
        assert any(
            "step" in str(v.get("message", "")).lower() for v in gate["violations"]
        )

    def test_unstored_workflow_is_refused(self):
        engine = FakeBackendEngine()
        gate = gate_workflow_execution(engine, "dynamic_adhoc_flow")
        assert gate["allowed"] is False
        assert gate["workflow_id"] is None


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

    def test_durable_governance_rehydrates_acl_after_restart(self, monkeypatch):
        from agent_utilities.knowledge_graph.core import secured_reads
        from agent_utilities.models.company_brain import DataClassification
        from agent_utilities.protocols.source_connectors.base import ExternalAccess

        engine = FakeEngine()
        wid = _seed_workflow(engine, acl=False)
        monkeypatch.setattr(
            secured_reads,
            "_durable_access_rows",
            lambda node_ids: {
                wid: {
                    "tenant_id": "tenant-a",
                    "classification": DataClassification.PUBLIC.value,
                    "external_access": ExternalAccess(is_public=True).model_dump(
                        mode="json"
                    ),
                }
                if wid in node_ids
                else {},
            },
        )

        gate = gate_workflow_execution(engine, "invoice_flow", actor=self._actor())

        assert gate["allowed"] is True
        assert gate["workflow_id"] == wid

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
        gate_idx = source.find(
            "gate = await asyncio.to_thread(_workflow_gate, engine, workflow)"
        )
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
        branch_idx = source.find(
            'if action in {"execute", "execute_dynamic", "dispatch"}:'
        )
        assert branch_idx != -1
        gate_idx = source.find(
            "gate = await asyncio.to_thread(_workflow_gate, engine, workflow)",
            branch_idx,
        )
        task_idx = source.find("asyncio.create_task(", branch_idx)
        assert gate_idx != -1, "dispatch_workflow must run the ORCH-1.42 gate"
        assert task_idx != -1
        assert gate_idx < task_idx, "gate must run BEFORE background dispatch"


class TestOrchestratorExecuteWorkflowGatesAtTheChokepoint:
    """D-WS-8 — the gate now runs INSIDE ``Orchestrator.execute_workflow`` itself.

    Before this fix only the ``graph_workflows`` MCP handler called
    ``gate_workflow_execution`` before dispatch; four production callers
    (``ticket_playbooks._dispatch_workflow``,
    ``loop_controller._default_skill_runner``,
    ``weights_distillation._dispatch_train_workflow``, ``schedule_engine``)
    invoke ``Orchestrator.execute_workflow`` directly and skipped SHACL/ACL
    validation entirely. Gating at this single chokepoint means every caller
    inherits it without touching any of the four call sites.
    """

    async def _orchestrator(self, engine):
        from agent_utilities.orchestration.manager import Orchestrator

        return Orchestrator(engine)

    async def test_unstored_workflow_raises_gate_denied_before_any_dispatch(self):
        from agent_utilities.knowledge_graph.core.workflow_gate import (
            WorkflowGateDeniedError,
        )

        engine = FakeEngine()
        orchestrator = await self._orchestrator(engine)

        with pytest.raises(WorkflowGateDeniedError) as excinfo:
            await orchestrator.execute_workflow("never_stored_flow")

        assert excinfo.value.gate["workflow_id"] is None
        assert excinfo.value.gate["violations"][0]["code"] == (
            "workflow_definition_missing"
        )

    async def test_zero_step_workflow_raises_gate_denied_before_any_dispatch(
        self, monkeypatch
    ):
        """A step_count=0 WorkflowDefinition must be refused BY THE GATE, not
        merely by the deeper zero-step guard in ``WorkflowRunner`` (D-FSR-1) —
        proving the gate itself is now reached from this entrypoint.
        """
        from agent_utilities.knowledge_graph.core.workflow_gate import (
            WorkflowGateDeniedError,
        )
        from agent_utilities.workflows.runner import WorkflowRunner

        engine = FakeEngine()
        _seed_workflow(engine, name="empty_flow", step_count=0, steps=[])
        orchestrator = await self._orchestrator(engine)

        called = {"execute_by_name": False}

        async def _must_not_be_called(*_a, **_k):
            called["execute_by_name"] = True
            raise AssertionError("WorkflowRunner must not run past a gate denial")

        monkeypatch.setattr(WorkflowRunner, "execute_by_name", _must_not_be_called)

        with pytest.raises(WorkflowGateDeniedError):
            await orchestrator.execute_workflow("empty_flow")

        assert called["execute_by_name"] is False

    async def test_valid_workflow_passes_the_gate_and_reaches_the_runner(
        self, monkeypatch
    ):
        """The gate is not a blanket refusal — a conformant, ACL-permitted
        workflow still reaches ``WorkflowRunner.execute_by_name`` as before.
        """
        from agent_utilities.workflows.runner import WorkflowResult, WorkflowRunner

        engine = FakeEngine()
        _seed_workflow(engine, name="invoice_flow")
        orchestrator = await self._orchestrator(engine)

        seen: dict[str, object] = {}

        async def _fake_execute_by_name(self, *, workflow_name, engine, **kwargs):
            seen["workflow_name"] = workflow_name
            return WorkflowResult(
                workflow_name=workflow_name,
                session_id="sess-1",
                step_results=[],
                status="completed",
            )

        monkeypatch.setattr(WorkflowRunner, "execute_by_name", _fake_execute_by_name)

        payload = await orchestrator.execute_workflow("invoice_flow")

        assert seen["workflow_name"] == "invoice_flow"
        assert payload["run_id"] == "sess-1"

    def test_gate_call_precedes_the_workflow_runner_import_in_source(self):
        """Source order: the chokepoint gate must run BEFORE WorkflowRunner is
        even constructed — a caller must never be able to reach dispatch on a
        denial (belt-and-suspenders alongside the behavioral tests above).
        """
        import inspect

        from agent_utilities.orchestration import manager

        source = inspect.getsource(manager.Orchestrator.execute_workflow)
        gate_idx = source.find("gate_workflow_execution(self.engine, workflow_id)")
        runner_idx = source.find("runner = WorkflowRunner(")
        assert gate_idx != -1, (
            "execute_workflow must call gate_workflow_execution (D-WS-8)"
        )
        assert runner_idx != -1
        assert gate_idx < runner_idx, "gate must run BEFORE the runner is built"
