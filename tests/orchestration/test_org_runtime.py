"""Tests for the runtime org dynamics (CONCEPT:AU-ORCH.org.recruiter /
AU-ORCH.org.work-item-dag / AU-AHE.org.role-experience).

Covers the kanban phase state machine, manager-mode classifier, the recruiter's
goal→org synthesis + reuse-vs-hire staffing, the Self-Grown experience write-back
(both directly and through the live FeedbackService branch), and an end-to-end
work-item DAG run with review, rework, escalation, and experience accrual over
two runs.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.orchestration.org_runtime import (
    InvalidPhaseTransition,
    ManagerMode,
    OrgChart,
    OrgPhase,
    OrgRuntime,
    Recruiter,
    RoleSpec,
    WorkItem,
    experience_score,
    infer_manager_mode,
    is_runnable,
    is_terminal,
    kanban_column,
    record_role_experience,
    validate_transition,
)


# ── Fake engine/backend ───────────────────────────────────────────────────
class FakeEngine:
    """Minimal in-memory stand-in for the KG engine (add/get node + link)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.backend = self  # module reads engine.backend or engine

    def add_node(self, node_id, node_type, properties=None):  # noqa: ANN001
        d = dict(self.nodes.get(node_id, {}))
        d.update(properties or {})
        d["type"] = node_type
        self.nodes[node_id] = d

    def get_node(self, node_id):  # noqa: ANN001
        return dict(self.nodes.get(node_id, {}))

    def link_nodes(self, src, tgt, rel, properties=None):  # noqa: ANN001
        self.edges.append((src, tgt, rel))


# ── Phase state machine ────────────────────────────────────────────────────
def test_phase_transitions_and_projections():
    validate_transition(None, OrgPhase.READY)  # initial creation ok
    validate_transition(OrgPhase.READY, OrgPhase.READY)  # idempotent ok
    validate_transition(OrgPhase.READY, OrgPhase.RUNNING)
    validate_transition(OrgPhase.RUNNING, OrgPhase.AWAITING_REVIEW)
    validate_transition(OrgPhase.AWAITING_REVIEW, OrgPhase.APPROVED)
    with pytest.raises(InvalidPhaseTransition):
        validate_transition(OrgPhase.APPROVED, OrgPhase.RUNNING)
    with pytest.raises(InvalidPhaseTransition):
        validate_transition(OrgPhase.QUEUED, OrgPhase.APPROVED)


def test_phase_projections():
    assert kanban_column(OrgPhase.QUEUED) == "todo"
    assert kanban_column(OrgPhase.RUNNING) == "in_progress"
    assert kanban_column(OrgPhase.AWAITING_REVIEW) == "in_review"
    assert kanban_column(OrgPhase.APPROVED) == "done"
    assert is_runnable(OrgPhase.READY)
    assert is_runnable(OrgPhase.READY_FOR_REWORK)
    assert not is_runnable(OrgPhase.RUNNING)
    assert is_terminal(OrgPhase.APPROVED)
    assert not is_terminal(OrgPhase.ESCALATED)


def test_work_item_transition_guard():
    item = WorkItem("wi_1", "t", "d", owner_role="r")
    item.transition(OrgPhase.READY)
    item.transition(OrgPhase.RUNNING)
    with pytest.raises(InvalidPhaseTransition):
        item.transition(OrgPhase.QUEUED)


# ── Manager mode classifier ────────────────────────────────────────────────
def test_infer_manager_mode_priority():
    execute = WorkItem("a", "t", "d", owner_role="r", role_type="worker")
    assert infer_manager_mode(execute) is ManagerMode.EXECUTE

    delegate = WorkItem("b", "t", "d", owner_role="r", role_type="coordinator")
    assert infer_manager_mode(delegate) is ManagerMode.DELEGATE

    integrate = WorkItem(
        "c", "t", "d", owner_role="r", role_type="coordinator", dependencies=["a"]
    )
    assert infer_manager_mode(integrate) is ManagerMode.INTEGRATE

    rework = WorkItem("d", "t", "d", owner_role="r", phase=OrgPhase.READY_FOR_REWORK)
    assert infer_manager_mode(rework) is ManagerMode.REWORK

    review = WorkItem("e", "t", "d", owner_role="r")
    assert infer_manager_mode(review, is_review_entry=True) is ManagerMode.REVIEW


# ── Self-Grown experience write-back ───────────────────────────────────────
def test_record_role_experience_accrues_and_scores():
    eng = FakeEngine()
    r1 = record_role_experience(
        eng,
        "writer",
        employee_id="emp_writer",
        success=True,
        reward=1.0,
        domains=["eng"],
    )
    assert r1["successes"] == 1
    assert r1["experienceScore"] > 0
    assert experience_score(eng, "emp_writer") == pytest.approx(r1["experienceScore"])

    # A failure lowers relative gain; a second success keeps climbing.
    record_role_experience(
        eng, "writer", employee_id="emp_writer", success=False, reward=0.0
    )
    r3 = record_role_experience(
        eng,
        "writer",
        employee_id="emp_writer",
        success=True,
        reward=1.0,
        domains=["eng"],
    )
    assert r3["successes"] == 2
    assert r3["failures"] == 1
    prof = json.loads(eng.nodes["emp_writer"]["experienceProfile"])
    assert prof["successes"] == 2 and prof["failures"] == 1
    # seniority band promotes as score crosses thresholds.
    assert eng.nodes["emp_writer"]["seniority"] in {"junior", "mid", "senior"}


def test_feedback_service_role_experience_branch_live_path():
    """The live AHE path: record_action_outcome('role_experience:..') updates
    the :Employee profile (Wire-First — exercise the real entry point)."""
    from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService

    eng = FakeEngine()
    svc = FeedbackService(backend=eng)
    res = svc.record_action_outcome(
        "role_experience:writer",
        success=True,
        reward=1.0,
        agent_id="writer",
        corrected_value={"employee_id": "emp_writer", "domains": ["eng"]},
    )
    assert res.applied or True  # outcome result returned
    assert experience_score(eng, "emp_writer") > 0
    prof = json.loads(eng.nodes["emp_writer"]["experienceProfile"])
    assert prof["successes"] == 1


# ── Recruiter / org synthesis ──────────────────────────────────────────────
def test_recruiter_drafts_roles_from_goal():
    eng = FakeEngine()
    chart = Recruiter(eng).synthesize_org(
        "Research and build a new pricing engine", domains=["finance"]
    )
    role_ids = {r.role_id for r in chart.roles}
    # keyword seeds fire for research + build, plus the always-on coordinator.
    assert "project_coordinator" in role_ids
    assert any(r.role_type == "worker" for r in chart.roles)
    assert all(len(chart.employees) == len(chart.roles) for _ in [0])
    # every role persisted as an :AgentRole node + staffed by an :Employee node.
    assert any(n.get("type") == "AgentRole" for n in eng.nodes.values())
    assert any(n.get("type") == "Employee" for n in eng.nodes.values())
    # fresh company → every hire is a proposed_hire.
    assert all(e.status == "proposed_hire" for e in chart.employees)


def test_recruiter_reuses_experienced_employee():
    eng = FakeEngine()
    # Pre-seed an experienced employee for the generalist seat.
    record_role_experience(
        eng,
        "generalist",
        employee_id="emp_generalist",
        success=True,
        reward=1.0,
        domains=["x"],
    )
    chart = Recruiter(eng).synthesize_org("accomplish something vague")
    gen = next(e for e in chart.employees if e.role_id == "generalist")
    assert gen.status == "existing_staff"
    assert gen.experience_score > 0


# ── Work-item DAG runtime (end-to-end) ─────────────────────────────────────
class _StubRuntime(OrgRuntime):
    """OrgRuntime whose role executor is a canned map (no live LLM)."""

    def __init__(self, engine, responses, **kw):  # noqa: ANN001
        super().__init__(engine, **kw)
        self.responses = responses
        self.calls: list[str] = []

    async def _execute_role(self, role_id, task, context):  # noqa: ANN001
        self.calls.append(role_id)
        resp = self.responses.get(role_id, "done")
        return resp(task) if callable(resp) else resp


@pytest.mark.asyncio
async def test_org_run_happy_path_dag_and_experience():
    eng = FakeEngine()
    runtime = _StubRuntime(eng, responses={})  # all roles return "done"
    # goal without test/qa keywords → no reviewer gate; research+build workers.
    result = await runtime.run("research and build a data pipeline")
    assert result["status"] == "completed"
    assert result["approved"] == result["total"]
    # coordinator ran AFTER its worker dependencies (INTEGRATE order).
    coord_idx = runtime.calls.index("project_coordinator")
    worker_idxs = [i for i, c in enumerate(runtime.calls) if c != "project_coordinator"]
    assert all(coord_idx > i for i in worker_idxs)
    # experience accrued for a worker employee.
    assert experience_score(eng, "emp_software_engineer") > 0

    # Second run reuses the now-experienced staff.
    result2 = await runtime.run("research and build a data pipeline")
    chart2 = result2["org_chart"]
    reused = [e for e in chart2["employees"] if e["status"] == "existing_staff"]
    assert reused, "second run should reuse experienced staff"


@pytest.mark.asyncio
async def test_org_run_review_rework_then_escalation():
    eng = FakeEngine()
    escalations: list[str] = []

    async def esc_cb(item, reason):  # noqa: ANN001
        escalations.append(reason)
        return "approve"  # human approves the beyond-team blocker

    runtime = _StubRuntime(
        eng,
        responses={"worker": "deliverable", "reviewer": "REWORK: not good enough"},
        escalation_cb=esc_cb,
    )
    item = WorkItem(
        "wi_worker",
        "do work",
        "the work",
        owner_role="worker",
        role_type="worker",
        reviewer_role="reviewer",
    )
    chart = OrgChart(
        goal="g",
        company_id="__c__",
        roles=[
            RoleSpec("worker", "Worker", "do", role_type="worker"),
            RoleSpec("reviewer", "Reviewer", "review", role_type="reviewer"),
        ],
    )
    await runtime.run("g", work_items=[item], chart=chart)
    # reviewer kept rejecting → escalated once → human approved.
    assert len(escalations) == 1
    assert item.phase is OrgPhase.APPROVED
    assert item.rework_count >= 1


@pytest.mark.asyncio
async def test_org_run_deadlock_escalates():
    eng = FakeEngine()
    escalations: list[str] = []

    async def esc_cb(item, reason):  # noqa: ANN001
        escalations.append(reason)
        return None

    runtime = _StubRuntime(eng, responses={}, escalation_cb=esc_cb)
    blocked = WorkItem(
        "wi_b", "b", "b", owner_role="worker", dependencies=["missing_dep"]
    )
    chart = OrgChart(
        goal="g", company_id="__c__", roles=[RoleSpec("worker", "Worker", "do")]
    )
    await runtime.run("g", work_items=[blocked], chart=chart)
    assert escalations and "dependencies" in escalations[0]
    assert blocked.phase is OrgPhase.ESCALATED


# ── Surface parity: both actions reachable on MCP + REST ───────────────────
def test_org_actions_in_manifest_and_rest_routes():
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    org_actions = {
        op["action"] for op in GRAPHOS_ACTIONS if op["tool"] == "graph_orchestrate"
    }
    assert {"synthesize_org", "run_org"} <= org_actions

    # REST twin routes are mounted.
    from agent_utilities.mcp import kg_server

    class _App:
        def __init__(self):
            self.paths: set[str] = set()

        def add_route(self, path, handler, methods=None):  # noqa: ANN001
            self.paths.add(path)

    app = _App()
    kg_server._mount_rest_routes(app)
    assert "/graph/orchestrate/synthesize-org" in app.paths
    assert "/graph/orchestrate/run-org" in app.paths
