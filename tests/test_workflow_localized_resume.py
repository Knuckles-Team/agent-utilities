"""WorkflowRunner.resume_localized — Atomic Task Graph paper idea #1 wired into
the executor (``reports/paper-analysis-2607.01942.md`` §4 Rank 1), plus the
model_tier -> reasoning_effort routing hint (paper idea #3). Mirrors the
fake-engine style of ``test_workflow_gate_suspend_resume.py``.

@pytest.mark.concept("AU-ORCH.execution.workflow-lifecycle-management")
"""

from __future__ import annotations

import pytest

from agent_utilities.models.graph import ExecutionStep, GraphPlan
from agent_utilities.workflows.runner import WorkflowRunner

pytestmark = pytest.mark.concept("AU-ORCH.execution.workflow-lifecycle-management")


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

    def out_edges(self, node_id, data=False):
        return self.graph.out_edges(node_id, data=data)


def _diamond_plan():
    """prep -> [branchA -> mergeA]
    -> [branchB -> mergeB]   (independent sibling branch of branchA)
    """
    return GraphPlan(
        steps=[
            ExecutionStep(id="prep", refined_subtask="prepare"),
            ExecutionStep(id="branchA", depends_on=["prep"]),
            ExecutionStep(id="mergeA", depends_on=["branchA"]),
            ExecutionStep(id="branchB", depends_on=["prep"]),
            ExecutionStep(id="mergeB", depends_on=["branchB"]),
        ]
    )


@pytest.fixture
def _fake_agent(monkeypatch):
    ran: list[str] = []
    seen_efforts: dict[str, str | None] = {}

    async def _fake_run_agent(
        agent_name, task, engine=None, reasoning_effort=None, **kw
    ):
        ran.append(agent_name)
        seen_efforts[agent_name] = reasoning_effort
        return f"ok:{agent_name}"

    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_runner.run_agent", _fake_run_agent
    )
    return ran, seen_efforts


async def test_resume_localized_only_reruns_the_invalidated_branch(
    _fake_agent, monkeypatch
):
    ran, _efforts = _fake_agent
    engine = FakeEngine()
    runner = WorkflowRunner()
    r1 = await runner._execute_plan_via_agents(
        _diamond_plan(), engine, "ci_repair_wf", trace_session="run-localized-1"
    )
    assert r1.status == "completed"
    assert set(ran) == {"prep", "branchA", "mergeA", "branchB", "mergeB"}
    ran.clear()

    # Wire in the TRANSITION_TO edges the localized-repair walk needs (a live
    # engine would have written these when the workflow was ingested).
    engine.link_nodes("branchA", "mergeA", "TRANSITION_TO")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.workflow_store.WorkflowStore.load_workflow",
        lambda self, name: _diamond_plan(),
    )

    r2 = await runner.resume_localized(
        "ci_repair_wf",
        engine,
        "run-localized-1",
        failed_step="branchA",
        prior_result=r1,
    )

    assert r2.status == "completed"
    # ONLY the invalidated region (branchA + its TRANSITION_TO descendant
    # mergeA) re-ran; the sibling branch (branchB/mergeB) and upstream (prep)
    # were PRESERVED — not re-executed.
    assert set(ran) == {"branchA", "mergeA"}
    by_id = {r.node_id: r for r in r2.step_results}
    assert by_id["prep"].status == "completed"
    assert by_id["branchB"].status == "completed"
    assert by_id["mergeB"].status == "completed"


async def test_resume_localized_with_no_dag_edges_reruns_only_the_failed_step(
    _fake_agent, monkeypatch
):
    """A failed_step with no outgoing TRANSITION_TO edges (e.g. a terminal step,
    or the engine simply has no DAG-edge info) invalidates only itself — every
    other completed step, including its own downstream siblings, is preserved."""
    ran, _efforts = _fake_agent
    engine = FakeEngine()
    runner = WorkflowRunner()
    r1 = await runner._execute_plan_via_agents(
        _diamond_plan(), engine, "ci_repair_wf", trace_session="run-localized-2"
    )
    ran.clear()

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.workflow_store.WorkflowStore.load_workflow",
        lambda self, name: _diamond_plan(),
    )
    r2 = await runner.resume_localized(
        "ci_repair_wf",
        engine,
        "run-localized-2",
        failed_step="mergeA",
        prior_result=r1,
    )
    assert r2.status == "completed"
    assert ran == ["mergeA"]


async def test_model_tier_hint_routes_reasoning_effort(_fake_agent):
    """A step tagged model_tier='small' threads a reduced reasoning_effort into
    run_agent (paper idea #3); an untagged step passes None through unchanged."""
    _ran, efforts = _fake_agent
    engine = FakeEngine()
    plan = GraphPlan(
        steps=[
            ExecutionStep(id="cheap_step", model_tier="small"),
            ExecutionStep(id="plain_step"),
        ]
    )
    runner = WorkflowRunner()
    result = await runner._execute_plan_via_agents(
        plan, engine, "tier_wf", trace_session="run-tier-1"
    )
    assert result.status == "completed"
    assert efforts["cheap_step"] == "low"
    assert efforts["plain_step"] is None
    by_id = {r.node_id: r for r in result.step_results}
    assert by_id["cheap_step"].model_tier == "small"
    assert by_id["plain_step"].model_tier is None


async def test_model_id_pin_wins_over_model_tier_hint(_fake_agent):
    """An explicit model_id always wins — the tier hint is only honored when no
    exact model is pinned (mirrors Task.model_id's own docstring contract)."""
    _ran, efforts = _fake_agent
    engine = FakeEngine()
    plan = GraphPlan(
        steps=[
            ExecutionStep(id="pinned_step", model_tier="small", model_id="exact-model")
        ]
    )
    runner = WorkflowRunner()
    await runner._execute_plan_via_agents(
        plan, engine, "tier_wf2", trace_session="run-tier-2"
    )
    assert efforts["pinned_step"] is None
