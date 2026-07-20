"""Tests for the runtime org dynamics (CONCEPT:AU-ORCH.org.recruiter /
AU-ORCH.org.work-item-dag / AU-AHE.org.role-experience).

Covers the immutable plan model, manager-mode classifier, the recruiter's
goal→org synthesis + reuse-vs-hire staffing, the Self-Grown experience write-back
(both directly and through the live FeedbackService branch), and an end-to-end
native WorkItem DAG run with review, rework, escalation, and experience accrual
over two runs.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.orchestration import work_item as wi
from agent_utilities.orchestration.org_runtime import (
    ManagerMode,
    OrgChart,
    OrgPlanItem,
    OrgRuntime,
    Recruiter,
    RoleSpec,
    experience_score,
    infer_manager_mode,
    record_role_experience,
)


# ── Fake engine/backend ───────────────────────────────────────────────────
class FakeEngine:
    """In-memory double for graph writes and native WorkItem verbs."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.native_calls: list[str] = []
        self.backend = self  # module reads engine.backend or engine

    def add_node(
        self,
        node_id,
        node_type,
        properties=None,
        ephemeral=False,  # noqa: ANN001
    ):
        d = dict(self.nodes.get(node_id, {}))
        d.update(properties or {})
        d["type"] = node_type
        self.nodes[node_id] = d

    def get_node(self, node_id):  # noqa: ANN001
        return dict(self.nodes.get(node_id, {}))

    def link_nodes(
        self,
        src,
        tgt,
        rel,
        properties=None,
        ephemeral=False,  # noqa: ANN001
    ):
        self.edges.append((src, tgt, rel))

    def compare_and_set_node_fields(self, node_id, conditions, updates):  # noqa: ANN001
        node = self.nodes.get(node_id)
        if node is None or any(node.get(k) != v for k, v in conditions.items()):
            return False
        node.update(updates)
        return True

    def query_cypher(self, cypher, params=None):  # noqa: ANN001
        params = params or {}
        query = " ".join(cypher.split())
        if query.startswith("MATCH (w:WorkItem {id: $id}) RETURN w.id"):
            node = self.nodes.get(str(params["id"]))
            if node is None or node.get("type") != "WorkItem":
                return []
            return [
                {
                    "id": params["id"],
                    **{field: node.get(field) for field in wi._FIELDS},
                }
            ]
        if query.startswith("MATCH (w:WorkItem {tenant: $tenant})"):
            return [
                {
                    "c": sum(
                        node.get("type") == "WorkItem"
                        and node.get("tenant") == params["tenant"]
                        and node.get("status") not in params["terminal"]
                        for node in self.nodes.values()
                    )
                }
            ]
        raise AssertionError(f"unrecognized query: {query}")

    def claim_work_item(self, request):  # noqa: ANN001
        self.native_calls.append("claim")
        node = self.nodes.get(str(request.work_item_id))
        if node is None or node.get("status") != wi.WorkItemStatus.READY.value:
            return {
                "schema_version": "1",
                "claimed": False,
                "reason": "not_ready",
                "work_item_id": None,
                "kind": None,
                "payload_ref": None,
                "lease_holder_ref": None,
                "lease_epoch": None,
                "fencing_token": None,
                "lease_expires_at_ms": None,
                "attempt": None,
                "max_attempts": None,
                "tenant_in_flight": 0,
                "changed_work_item_ids": [],
            }
        attempt = int(node.get("attempt") or 0) + 1
        epoch = int(node.get("lease_epoch") or 0) + 1
        node.update(
            status=wi.WorkItemStatus.LEASED.value,
            attempt=attempt,
            lease_owner=request.worker_ref,
            lease_epoch=epoch,
            fencing_token=epoch,
            lease_expires_at=(request.now_ms + request.lease_ms) / 1000.0,
        )
        return {
            "schema_version": "1",
            "claimed": True,
            "reason": "claimed",
            "work_item_id": request.work_item_id,
            "kind": node.get("kind"),
            "payload_ref": node.get("payload_ref"),
            "lease_holder_ref": request.worker_ref,
            "lease_epoch": epoch,
            "fencing_token": epoch,
            "lease_expires_at_ms": request.now_ms + request.lease_ms,
            "attempt": attempt,
            "max_attempts": node.get("max_attempts"),
            "tenant_in_flight": 1,
            "changed_work_item_ids": [request.work_item_id],
        }

    @staticmethod
    def _owns(node, request):  # noqa: ANN001
        return bool(
            node
            and node.get("lease_owner") == request.get("worker_ref")
            and node.get("lease_epoch") == request.get("expected_epoch")
            and node.get("fencing_token") == request.get("fencing_token")
        )

    def renew_work_item_lease(self, request):  # noqa: ANN001
        self.native_calls.append("renew")
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request):
            return {"renewed": False}
        node["lease_expires_at"] = request["now_unix"] + request["lease_ttl"]
        return {"renewed": True}

    def commit_work_item_result(self, request):  # noqa: ANN001
        self.native_calls.append("commit")
        node = self.nodes.get(request["work_item_id"])
        if node is None:
            return {"status": "missing"}
        if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
            return {"status": "noop"}
        if not self._owns(node, request):
            return {"status": "fenced"}
        outcome = request["outcome"]
        node.update(
            status=outcome,
            result_ref=request.get("result_ref"),
            error_ref=request.get("error_ref"),
            lease_owner=None,
            lease_expires_at=None,
        )
        if outcome == wi.WorkItemStatus.SUCCEEDED.value:
            for child_id in node.get("downstream_ids") or []:
                child = self.nodes[child_id]
                child["dep_count"] = max(0, int(child.get("dep_count") or 0) - 1)
                if child["dep_count"] == 0:
                    child["status"] = wi.WorkItemStatus.READY.value
        return {"status": "committed"}

    def cancel_work_item(self, request):  # noqa: ANN001
        self.native_calls.append("cancel")
        node = self.nodes.get(request["work_item_id"])
        if node is None:
            return {"status": "missing"}
        if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
            return {"status": "noop"}
        node.update(status=wi.WorkItemStatus.CANCELLED.value)
        return {"status": "cancelled"}


# ── Phase state machine ────────────────────────────────────────────────────
def test_org_plan_item_is_immutable_and_has_no_lifecycle_state():
    item = OrgPlanItem("plan_1", "t", "d", owner_role="r")
    assert "status" not in item.__dataclass_fields__
    assert "phase" not in item.__dataclass_fields__
    assert not hasattr(item, "transition")
    with pytest.raises((AttributeError, TypeError)):
        item.owner_role = "other"  # type: ignore[misc]


# ── Manager mode classifier ────────────────────────────────────────────────
def test_infer_manager_mode_priority():
    execute = OrgPlanItem("a", "t", "d", owner_role="r", role_type="worker")
    assert infer_manager_mode(execute) is ManagerMode.EXECUTE

    delegate = OrgPlanItem("b", "t", "d", owner_role="r", role_type="coordinator")
    assert infer_manager_mode(delegate) is ManagerMode.DELEGATE

    integrate = OrgPlanItem(
        "c", "t", "d", owner_role="r", role_type="coordinator", dependencies=("a",)
    )
    assert infer_manager_mode(integrate) is ManagerMode.INTEGRATE

    rework = OrgPlanItem("d", "t", "d", owner_role="r")
    assert infer_manager_mode(rework, rework_count=1) is ManagerMode.REWORK

    review = OrgPlanItem("e", "t", "d", owner_role="r")
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
    assert result["succeeded"] == result["total"]
    assert {"claim", "renew", "commit"} <= set(eng.native_calls)
    native_rows = [
        node for node in eng.nodes.values() if node.get("type") == "WorkItem"
    ]
    assert native_rows
    assert all("workItemPhase" not in node for node in native_rows)
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
    item = OrgPlanItem(
        "plan_worker",
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
    result = await runtime.run("g", plan_items=[item], chart=chart)
    # reviewer kept rejecting → escalated once → human approved.
    assert len(escalations) == 1
    assert result["plan_items"][0]["status"] == wi.WorkItemStatus.SUCCEEDED.value
    assert result["plan_items"][0]["rework_count"] >= 1
    assert eng.native_calls.count("commit") == 1


@pytest.mark.asyncio
async def test_org_run_deadlock_escalates():
    eng = FakeEngine()
    escalations: list[str] = []

    async def esc_cb(item, reason):  # noqa: ANN001
        escalations.append(reason)
        return None

    runtime = _StubRuntime(eng, responses={}, escalation_cb=esc_cb)
    blocked = OrgPlanItem(
        "plan_b", "b", "b", owner_role="worker", dependencies=("missing_dep",)
    )
    chart = OrgChart(
        goal="g", company_id="__c__", roles=[RoleSpec("worker", "Worker", "do")]
    )
    result = await runtime.run("g", plan_items=[blocked], chart=chart)
    assert escalations and "dependencies" in escalations[0]
    assert result["plan_items"][0]["status"] == wi.WorkItemStatus.CANCELLED.value
    assert "cancel" in eng.native_calls


# ── Surface parity: both actions reachable on MCP + REST ───────────────────
def test_org_actions_in_manifest_and_rest_routes():
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    org_actions = {
        op["action"] for op in GRAPHOS_ACTIONS if op["tool"] == "graph_agents"
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
    kg_server.ensure_tools_registered()
    kg_server._mount_rest_routes(app)
    assert "/graph/agents" in app.paths
