"""Execution-time workflow ontology gate (CONCEPT:ORCH-1.42).

A stored WorkflowDefinition is SHACL-validated (WorkflowDefinitionShape /
WorkflowStepShape in governance.shapes.ttl) before dispatch — malformed
definitions are refused with a structured report; with KG_BRAIN_ENFORCE on,
the ontology permissioning row gate (markings + ACLs, fail-closed) is applied
to the workflow node for the current actor.

@pytest.mark.concept("ORCH-1.42")
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")
pytest.importorskip("rdflib")

from agent_utilities.knowledge_graph.core.workflow_gate import (
    gate_workflow_execution,
    workflow_shape_gate_enabled,
)

pytestmark = pytest.mark.concept("ORCH-1.42")


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
        engine.graph.add_node(
            sid, {"type": "WorkflowStep", "step_order": i, **step}
        )
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
        assert any(
            "node_id" in str(v.get("message", "")) for v in gate["violations"]
        )

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

        from agent_utilities.mcp import kg_server

        source = inspect.getsource(kg_server)
        gate_idx = source.find("gate_workflow_execution(engine, gate_name)")
        dispatch_idx = source.find("await orch.execute_workflow(")
        assert gate_idx != -1, "execute_workflow must run the ORCH-1.42 gate"
        assert dispatch_idx != -1
        assert gate_idx < dispatch_idx, "gate must run BEFORE dispatch"
