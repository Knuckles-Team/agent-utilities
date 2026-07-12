"""Gate/approval step-kind + suspend/resume in WorkflowRunner
(``reports/autonomous-sdlc-loop-design.md`` §7.1 delta 2-3).

A ``kind="gate"`` step is not agent-executed: WorkflowRunner consults the
``gate_checker``. Pending → the run SUSPENDS (persists state, returns a
``status="suspended"`` result) rather than blocking; once the gate's
``:satisfiedBy`` edge is recorded, :meth:`WorkflowRunner.resume` continues the
DAG idempotently. Approved-inline proceeds; rejected skips the on-success
downstream but runs an ``on_reject`` branch.

@pytest.mark.concept("AU-ORCH.execution.gate-step-suspend-resume")
"""

from __future__ import annotations

import pytest

from agent_utilities.models.graph import ExecutionStep, GraphPlan
from agent_utilities.workflows import runner as runner_mod
from agent_utilities.workflows.runner import WorkflowRunner

pytestmark = pytest.mark.concept("AU-ORCH.execution.gate-step-suspend-resume")


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


class FakeEngine:
    def __init__(self):
        self.graph = FakeGraph()
        self.backend = None

    def add_node(self, node_id, node_type, properties=None, **props):
        self.graph.add_node(node_id, {"type": node_type, **(properties or props or {})})

    def link_nodes(self, source, target, rel_type, properties=None):
        self.graph.add_edge(source, target, type=rel_type, **(properties or {}))


def _plan_with_gate():
    """prep -> [gate] approve_gate -> deploy."""
    return GraphPlan(
        steps=[
            ExecutionStep(id="prep", refined_subtask="prepare"),
            ExecutionStep(
                id="approve_gate",
                kind="gate",
                depends_on=["prep"],
                refined_subtask="owner sign-off",
            ),
            ExecutionStep(id="deploy", depends_on=["approve_gate"]),
        ]
    )


@pytest.fixture
def _fake_agent(monkeypatch):
    """Stub run_agent so 'task' steps 'complete' without a live LLM. Records runs."""
    ran: list[str] = []

    async def _fake_run_agent(agent_name, task, engine=None, **kw):
        ran.append(agent_name)
        return f"ok:{agent_name}"

    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_runner.run_agent", _fake_run_agent
    )
    return ran


async def test_gate_pending_suspends_the_run(_fake_agent):
    """With the gate unsatisfied, the run stops AT the gate: prep runs, deploy does
    not, and the result is status='suspended'."""
    engine = FakeEngine()
    runner = WorkflowRunner()  # default gate_checker → no :satisfiedBy edge = pending
    result = await runner._execute_plan_via_agents(
        _plan_with_gate(), engine, "governed_deploy", trace_session="run-gate-1"
    )

    assert result.status == "suspended"
    assert "prep" in _fake_agent  # upstream ran
    assert "deploy" not in _fake_agent  # blocked by the gate
    by_id = {r.node_id: r for r in result.step_results}
    assert by_id["approve_gate"].status == runner_mod.STATUS_BLOCKED
    # state was persisted for resume
    assert "workflowrun:run-gate-1" in engine.graph.nodes


async def test_gate_resumes_on_approval(_fake_agent, monkeypatch):
    """Once the gate's :satisfiedBy edge is recorded, resume drives the DAG to
    completion WITHOUT re-running the already-completed upstream step."""
    engine = FakeEngine()
    runner = WorkflowRunner()
    r1 = await runner._execute_plan_via_agents(
        _plan_with_gate(), engine, "governed_deploy", trace_session="run-gate-2"
    )
    assert r1.status == "suspended"
    _fake_agent.clear()

    # A human/approval process records the gate as satisfied.
    engine.link_nodes(
        "approve_gate", "approval:owner1", "satisfiedBy", {"decision": "approved"}
    )

    # resume() reloads persisted state + re-drives from the gate. Stub the store load
    # to return the same plan (no live KG).
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.workflow_store.WorkflowStore.load_workflow",
        lambda self, name: _plan_with_gate(),
    )
    r2 = await runner.resume("governed_deploy", engine, "run-gate-2")

    assert r2.status == "completed"
    assert "prep" not in _fake_agent  # NOT re-run — idempotent resume
    assert "deploy" in _fake_agent  # unblocked past the gate
    by_id = {r.node_id: r for r in r2.step_results}
    assert by_id["approve_gate"].status == "completed"
    assert by_id["deploy"].status == "completed"


async def test_gate_approved_inline_proceeds(_fake_agent):
    """When the gate is already satisfied at first reach, the run completes in one
    pass (no suspension)."""
    engine = FakeEngine()
    engine.link_nodes(
        "approve_gate", "approval:owner1", "satisfiedBy", {"decision": "approved"}
    )
    runner = WorkflowRunner()
    result = await runner._execute_plan_via_agents(
        _plan_with_gate(), engine, "governed_deploy", trace_session="run-gate-3"
    )
    assert result.status == "completed"
    assert "deploy" in _fake_agent


async def test_gate_rejected_skips_downstream_and_routes_on_reject(_fake_agent):
    """A rejected gate skips its on-success downstream (deploy) but runs the
    on_reject branch (rollback)."""
    engine = FakeEngine()
    engine.link_nodes(
        "approve_gate", "reject:owner1", "satisfiedBy", {"decision": "rejected"}
    )
    plan = GraphPlan(
        steps=[
            ExecutionStep(id="prep", refined_subtask="prepare"),
            ExecutionStep(
                id="approve_gate",
                kind="gate",
                depends_on=["prep"],
                on_reject="rollback",
            ),
            ExecutionStep(id="deploy", depends_on=["approve_gate"]),
            ExecutionStep(id="rollback", depends_on=["approve_gate"]),
        ]
    )
    runner = WorkflowRunner()
    result = await runner._execute_plan_via_agents(
        plan, engine, "governed_deploy", trace_session="run-gate-4"
    )

    assert "deploy" not in _fake_agent  # on-success downstream skipped
    assert "rollback" in _fake_agent  # on_reject branch ran
    by_id = {r.node_id: r for r in result.step_results}
    assert by_id["approve_gate"].status == runner_mod.STATUS_REJECTED
    assert by_id["deploy"].status == runner_mod.STATUS_SKIPPED


async def test_custom_gate_checker_is_consulted(_fake_agent):
    """A deployment can bind gate satisfaction to its own governance system via the
    gate_checker override."""
    engine = FakeEngine()
    seen: list[str] = []

    def _checker(eng, step):
        seen.append(getattr(step, "id", ""))
        return "approved"

    runner = WorkflowRunner(gate_checker=_checker)
    result = await runner._execute_plan_via_agents(
        _plan_with_gate(), engine, "governed_deploy", trace_session="run-gate-5"
    )
    assert seen == ["approve_gate"]
    assert result.status == "completed"


def test_gate_step_round_trips_through_workflow_store():
    """A gate ExecutionStep persists its kind/condition/on_reject via WorkflowStore
    and reloads with them intact (§7.1 delta 2)."""
    from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

    engine = FakeEngine()
    store = WorkflowStore(engine)
    plan = GraphPlan(
        steps=[
            ExecutionStep(id="prep"),
            ExecutionStep(
                id="g", kind="gate", condition="on_approval", on_reject="prep"
            ),
        ]
    )
    store.save_workflow("wf", plan)
    loaded = store.load_workflow("wf")
    assert loaded is not None
    g = next(s for s in loaded.steps if s.id == "g")
    assert g.kind == "gate"
    assert g.condition == "on_approval"
    assert g.on_reject == "prep"
