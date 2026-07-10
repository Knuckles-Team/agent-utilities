#!/usr/bin/python
from __future__ import annotations

"""``TeamComposition.to_durable_task_dag()`` + poll-based dependency firing.

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3a)

Covers the opt-in durable persistence companion to
``TeamComposition.to_graph_plan()`` — it must round-trip through the
EXISTING ``WorkflowStore.save_workflow``/``load_workflow`` (no new
persistence path) while ALSO writing one ``:AgentTask`` per step with
``depends_on_task_ids`` mirroring ``ExecutionStep.depends_on`` — and the
dependency-firing sweep (``fleet_reconciler.fire_ready_agent_tasks``) that
flips a 'blocked' task to 'ready' once every dependency completes. This sweep
is now CDC-gated by ``fleet_reconciler.AgentTaskDepWatcher`` (Phase 3b, D13) —
see ``tests/unit/test_agent_task_dep_watcher.py`` for that layer; these tests
cover the sweep body itself, called directly.

@pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")
"""

import pytest

from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
from agent_utilities.models.knowledge_graph import RegistryEdgeType, TeamComposition
from agent_utilities.orchestration.fleet_reconciler import fire_ready_agent_tasks

pytestmark = pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")


# ---------------------------------------------------------------------------
# Minimal NX-shaped fake engine — mirrors WorkflowStore's ``backend=None``
# (in-memory graph_compute) path, same shape as the fake used by
# tests/unit/test_workflow_lineage_closeout.py for the same store.
# ---------------------------------------------------------------------------


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
    """WorkflowStore-shaped fake with ``backend=None`` (exercises the NX load path)."""

    def __init__(self):
        self.graph = FakeGraph()
        self.backend = None

    def add_node(self, node_id, node_type, properties=None, **props):
        # Real engine.add_node normalizes/overrides "type" AFTER copying
        # properties (props["type"] = node_type), so a pydantic node's own
        # ``type`` enum field never wins over the caller's label — mirror
        # that ordering here.
        data = dict(properties or props or {})
        data["type"] = node_type
        self.graph.add_node(node_id, data)

    def link_nodes(self, source, target, rel_type, properties=None):
        self.graph.add_edge(source, target, type=str(rel_type), **(properties or {}))

    # ── query surface for fire_ready_agent_tasks ────────────────────
    def query_cypher(self, query: str, params: dict | None = None):
        params = params or {}
        nodes = self.graph._nodes
        if "AgentTask {status: 'blocked'}" in query:
            return [
                {"id": nid, "depends_on_task_ids": n.get("depends_on_task_ids") or []}
                for nid, n in nodes.items()
                if n.get("type") == "AgentTask" and n.get("status") == "blocked"
            ]
        if "AgentTask) WHERE t.id IN $ids" in query:
            wanted = set(params.get("ids") or [])
            return [
                {"id": nid, "status": n.get("status")}
                for nid, n in nodes.items()
                if n.get("type") == "AgentTask" and nid in wanted
            ]
        return []


def _team(execution_mode: str = "sequential") -> TeamComposition:
    return TeamComposition(
        team_id="team-1",
        source="composed",
        execution_mode=execution_mode,
        adaptive_agent_router=[
            {
                "role": "researcher",
                "agent_id": "researcher",
                "system_prompt": "research",
            },
            {"role": "writer", "agent_id": "writer", "system_prompt": "write"},
            {"role": "reviewer", "agent_id": "reviewer", "system_prompt": "review"},
        ],
    )


# ---------------------------------------------------------------------------
# to_durable_task_dag() round trip
# ---------------------------------------------------------------------------


def test_to_durable_task_dag_returns_workflow_id():
    engine = FakeEngine()
    store = WorkflowStore(engine)
    team = _team()
    dag_id = team.to_durable_task_dag(store, dag_name="test_dag")
    assert dag_id.startswith("workflow:test_dag:")


def test_to_durable_task_dag_writes_agent_task_nodes_mirroring_depends_on():
    engine = FakeEngine()
    store = WorkflowStore(engine)
    team = _team()
    dag_id = team.to_durable_task_dag(store, dag_name="test_dag")

    researcher_id = f"{dag_id}:task:researcher"
    writer_id = f"{dag_id}:task:writer"
    reviewer_id = f"{dag_id}:task:reviewer"

    researcher = engine.graph._nodes[researcher_id]
    writer = engine.graph._nodes[writer_id]
    reviewer = engine.graph._nodes[reviewer_id]

    assert researcher["type"] == "AgentTask"
    assert researcher["dag_id"] == dag_id
    assert researcher["depends_on_task_ids"] == []
    assert researcher["status"] == "pending"  # no deps -> immediately runnable

    assert writer["depends_on_task_ids"] == [researcher_id]
    assert writer["status"] == "blocked"  # has a dependency -> blocked until it fires

    assert reviewer["depends_on_task_ids"] == [writer_id]
    assert reviewer["status"] == "blocked"

    # TASK_DEPENDS_ON edges: task -> dependency.
    assert (writer_id, researcher_id) in [
        (s, t)
        for s, t, p in engine.graph._edges
        if p.get("type") == str(RegistryEdgeType.TASK_DEPENDS_ON)
    ]
    assert (reviewer_id, writer_id) in [
        (s, t)
        for s, t, p in engine.graph._edges
        if p.get("type") == str(RegistryEdgeType.TASK_DEPENDS_ON)
    ]


def test_to_durable_task_dag_round_trips_depends_on_through_workflow_store():
    """The in-memory GraphPlan persisted via save_workflow round-trips the same
    depends_on the AgentTask DAG mirrors — one write, two consistent views."""
    engine = FakeEngine()
    store = WorkflowStore(engine)
    team = _team()
    dag_id = team.to_durable_task_dag(store, dag_name="roundtrip_dag")

    loaded = store.load_workflow("roundtrip_dag")
    assert loaded is not None
    steps_by_id = {s.id: s for s in loaded.steps}
    assert steps_by_id["researcher"].depends_on == []
    assert steps_by_id["writer"].depends_on == ["researcher"]
    assert steps_by_id["reviewer"].depends_on == ["writer"]

    # The AgentTask DAG's dependency lists agree with the reloaded GraphPlan.
    for step_id, deps in (
        ("researcher", []),
        ("writer", ["researcher"]),
        ("reviewer", ["writer"]),
    ):
        task = engine.graph._nodes[f"{dag_id}:task:{step_id}"]
        assert task["depends_on_task_ids"] == [f"{dag_id}:task:{d}" for d in deps]


def test_to_durable_task_dag_fan_in_has_no_depends_on():
    """A parallel-only team (no sequential dependency) writes unblocked tasks."""
    engine = FakeEngine()
    store = WorkflowStore(engine)
    team = TeamComposition(
        team_id="team-2",
        execution_mode="parallel",
        adaptive_agent_router=[
            {"role": "a", "agent_id": "a"},
            {"role": "b", "agent_id": "b"},
        ],
    )
    dag_id = team.to_durable_task_dag(store, dag_name="parallel_dag")
    for step_id in ("a", "b"):
        task = engine.graph._nodes[f"{dag_id}:task:{step_id}"]
        assert task["depends_on_task_ids"] == []
        assert task["status"] == "pending"


# ---------------------------------------------------------------------------
# fire_ready_agent_tasks — poll-based dependency firing (interim, not CDC)
# ---------------------------------------------------------------------------


def test_fire_ready_agent_tasks_fires_when_all_deps_completed():
    engine = FakeEngine()
    store = WorkflowStore(engine)
    team = _team()
    dag_id = team.to_durable_task_dag(store, dag_name="fire_dag")
    researcher_id = f"{dag_id}:task:researcher"
    writer_id = f"{dag_id}:task:writer"

    # Dependency not yet complete -> writer stays blocked.
    fired = fire_ready_agent_tasks(engine)
    assert writer_id not in fired
    assert engine.graph._nodes[writer_id]["status"] == "blocked"

    # Complete the dependency -> next sweep fires the dependent task.
    engine.graph._nodes[researcher_id]["status"] = "completed"
    fired = fire_ready_agent_tasks(engine)
    assert writer_id in fired
    assert engine.graph._nodes[writer_id]["status"] == "ready"

    # reviewer still blocked on writer, which is 'ready' not 'completed'.
    reviewer_id = f"{dag_id}:task:reviewer"
    assert engine.graph._nodes[reviewer_id]["status"] == "blocked"


def test_fire_ready_agent_tasks_never_fires_on_missing_dependency_evidence():
    """Conservative: a dependency id with no matching node never fires (no
    evidence -> no action, mirroring the reconciler's diff() discipline)."""
    engine = FakeEngine()
    engine.add_node(
        "orphan-task",
        "AgentTask",
        properties={
            "status": "blocked",
            "depends_on_task_ids": ["nonexistent-task"],
        },
    )
    fired = fire_ready_agent_tasks(engine)
    assert fired == []
    assert engine.graph._nodes["orphan-task"]["status"] == "blocked"


def test_fire_ready_agent_tasks_handles_missing_engine():
    assert fire_ready_agent_tasks(None) == []


def test_fire_ready_agent_tasks_wired_into_fleet_reconciler_report(
    tmp_path, monkeypatch
):
    """FleetReconciler.reconcile() includes 'fired_agent_tasks' in its report
    (the leader-only tick this phase extends, per the C3/Phase 3a spec)."""
    from agent_utilities.orchestration import fleet_reconciler as fr
    from agent_utilities.orchestration.action_policy import ActionPolicy
    from agent_utilities.orchestration.fleet_actuation import DryRunActuator
    from agent_utilities.orchestration.fleet_reconciler import FleetReconciler

    from .fleet_autonomy_fakes import FakeEngine as FleetFakeEngine
    from .fleet_autonomy_fakes import FakeObserver

    engine = FleetFakeEngine()
    rec = FleetReconciler(
        engine,
        observer=FakeObserver({}),
        actuator=DryRunActuator(),
        policy=ActionPolicy(engine=engine, policy_path=None),
    )
    monkeypatch.setattr(fr, "load_desired_state", lambda *a, **k: {})
    monkeypatch.setattr(fr, "fire_ready_agent_tasks", lambda eng, **kw: ["orphan-task"])
    report = rec.reconcile()
    assert report["fired_agent_tasks"] == ["orphan-task"]
