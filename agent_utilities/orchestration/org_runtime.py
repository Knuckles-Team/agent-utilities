"""Runtime org dynamics — recruiter, work-item DAG, and Self-Grown experience.

Ported (shape, not code) from OpenOPC's ``opc/layer2_organization`` — the
"One-Person Company" (autonomous AI-native company) runtime. Our
``ontology_company.ttl`` already models the static org STRUCTURE
(``:Company``/``:Department``/``:AgentRole``/``:Employee``); this module adds the
missing runtime DYNAMICS, built as a **workflow over the existing orchestrator**
(``Orchestrator.execute_agent`` → ``run_agent``) rather than a new service
(the platform's one-core rule).

Three capabilities, each concept-anchored:

* **Recruiter / org-synthesis** (CONCEPT:AU-ORCH.org.recruiter). From a goal,
  :class:`Recruiter` drafts an org chart (departments → roles) and *fills* each
  role — reusing an experienced :Employee where one exists, else hiring a fresh
  template — instantiating ``:AgentRole``/``:Employee`` nodes in the KG and
  reusing existing instances.
* **Engine-native work-item DAG** (CONCEPT:AU-ORCH.org.work-item-dag).
  :class:`OrgRuntime` derives immutable :class:`OrgPlanItem` definitions, then
  submits, claims, renews, and commits the executable DAG through the sole
  native ``orchestration.work_item`` authority. Manager modes
  (execute/delegate/review/integrate/rework) are turn-local execution context,
  never a second durable lifecycle.
* **Self-Grown** (CONCEPT:AU-AHE.org.role-experience). Each item's outcome is
  written back through the AHE reward loop
  (:meth:`FeedbackService.record_action_outcome` with a ``role_experience:``
  ``action_id``), updating the :Employee's ``experienceProfile`` /
  ``experienceScore`` so the next recruiter run reuses proven staff.

The whole runtime is exposed as ``graph_agents(action='synthesize_org')``
and ``graph_agents(action='run_org')`` on both the MCP server and the REST
gateway (surface parity).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from agent_utilities.orchestration.work_item import (
    WorkItemStatus,
    cancel_work_item,
    claim_specific,
    commit_result,
    get_work_item,
    heartbeat,
    mark_running,
    new_work_item_id,
    submit_work_item,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Manager modes (CONCEPT:AU-ORCH.org.work-item-dag)
# ─────────────────────────────────────────────────────────────────────────
class ManagerMode(StrEnum):
    """How the owning role acts on a work item this turn.

    Priority-ordered classifier ported from OpenOPC ``turn_mode.py``.
    """

    REWORK = "rework"  # reviewer rejected a prior turn; address feedback
    REVIEW = "review"  # evaluate a subordinate deliverable, emit a verdict
    INTEGRATE = "integrate"  # parent resumes after children approved; roll up
    DELEGATE = "delegate"  # manager role, subordinates but nothing spawned yet
    EXECUTE = "execute"  # default: a leaf role does the work itself


# ─────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class RoleSpec:
    """A drafted org-chart seat (adapts OpenOPC ``RoleConfig``)."""

    role_id: str
    name: str
    responsibility: str
    department: str = "Operations"
    role_type: str = "worker"  # worker | coordinator | reviewer
    reports_to: str | None = None
    domains: list[str] = field(default_factory=list)
    reused: bool = False  # True when bound to a pre-existing :AgentRole node

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "responsibility": self.responsibility,
            "department": self.department,
            "role_type": self.role_type,
            "reports_to": self.reports_to,
            "domains": self.domains,
            "reused": self.reused,
        }


@dataclass
class EmployeeSpec:
    """A staffed employee filling a role (adapts OpenOPC ``EmployeeConfig``)."""

    employee_id: str
    name: str
    role_id: str
    status: str  # existing_staff | proposed_hire
    experience_score: float = 0.0
    seniority: str = "junior"
    domains: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "role_id": self.role_id,
            "status": self.status,
            "experience_score": round(self.experience_score, 3),
            "seniority": self.seniority,
            "domains": self.domains,
            "rationale": self.rationale,
        }


@dataclass
class OrgChart:
    """A synthesized org: the goal, its roles, and who staffs each."""

    goal: str
    company_id: str
    roles: list[RoleSpec] = field(default_factory=list)
    employees: list[EmployeeSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "company_id": self.company_id,
            "roles": [r.to_dict() for r in self.roles],
            "employees": [e.to_dict() for e in self.employees],
        }


@dataclass(frozen=True)
class OrgPlanItem:
    """Immutable organization-plan input for one native executable WorkItem.

    This object deliberately has no status, lease, attempt, result, or
    transition method. Those fields belong exclusively to the engine-native
    ``WorkItem`` record created when :meth:`OrgRuntime.run` starts.
    """

    plan_item_id: str
    title: str
    description: str
    owner_role: str
    dependencies: tuple[str, ...] = ()
    reviewer_role: str | None = None
    role_type: str = "worker"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_item_id": self.plan_item_id,
            "title": self.title,
            "owner_role": self.owner_role,
            "reviewer_role": self.reviewer_role,
            "dependencies": list(self.dependencies),
            "role_type": self.role_type,
        }


@dataclass
class _ExecutionState:
    """Process-local projection of one native WorkItem execution.

    It is never persisted. ``status`` is refreshed from the native record or
    set only after a successful fenced commit/cancel response.
    """

    plan: OrgPlanItem
    work_item_id: str
    status: str = WorkItemStatus.SUBMITTED.value
    manager_mode: ManagerMode = ManagerMode.EXECUTE
    output: str = ""
    rework_count: int = 0
    review_feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_item_id": self.plan.plan_item_id,
            "work_item_id": self.work_item_id,
            "title": self.plan.title,
            "owner_role": self.plan.owner_role,
            "reviewer_role": self.plan.reviewer_role,
            "dependencies": list(self.plan.dependencies),
            "status": self.status,
            "manager_mode": self.manager_mode.value,
            "rework_count": self.rework_count,
            "output": self.output[:500],
        }


def infer_manager_mode(
    item: OrgPlanItem,
    *,
    rework_count: int = 0,
    review_feedback: str = "",
    is_review_entry: bool = False,
) -> ManagerMode:
    """Pure classifier for the owning role's mode this turn.

    Priority-ordered, mirroring OpenOPC ``infer_turn_mode``:
    REWORK → REVIEW → INTEGRATE → DELEGATE → EXECUTE.
    """
    if review_feedback or rework_count > 0:
        return ManagerMode.REWORK
    if is_review_entry or item.metadata.get("review_target_work_item_id"):
        return ManagerMode.REVIEW
    if item.dependencies and item.role_type in ("coordinator", "reviewer"):
        return ManagerMode.INTEGRATE
    if item.role_type == "coordinator" and not item.dependencies:
        return ManagerMode.DELEGATE
    return ManagerMode.EXECUTE


# ─────────────────────────────────────────────────────────────────────────
# Self-Grown experience profiles (CONCEPT:AU-AHE.org.role-experience)
# ─────────────────────────────────────────────────────────────────────────
#: Experience thresholds → seniority band (accrues as outcomes compound).
_SENIORITY_BANDS = ((8.0, "senior"), (3.0, "mid"), (0.0, "junior"))


def _seniority_for_score(score: float) -> str:
    for threshold, band in _SENIORITY_BANDS:
        if score >= threshold:
            return band
    return "junior"


def _read_node(backend: Any, node_id: str) -> dict[str, Any]:
    """Best-effort read of a node's property dict from the engine/backend."""
    for getter in ("get_node", "node"):
        fn = getattr(backend, getter, None)
        if callable(fn):
            try:
                data = fn(node_id)
                if isinstance(data, dict):
                    return dict(data)
            except Exception:  # noqa: BLE001 — best-effort read
                pass
    graph = getattr(backend, "graph", None)
    try:
        if graph is not None and node_id in graph.nodes:
            return dict(graph.nodes[node_id])
    except Exception:  # noqa: BLE001
        pass
    return {}


def experience_score(backend: Any, employee_id: str) -> float:
    """Return the current experience score for an employee (0.0 if unseen)."""
    node = _read_node(backend, employee_id)
    try:
        return float(node.get("experienceScore", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def record_role_experience(
    backend: Any,
    role_id: str,
    *,
    employee_id: str = "",
    success: bool = True,
    reward: float = 0.0,
    domains: list[str] | None = None,
) -> dict[str, Any]:
    """Accrue one outcome into an employee's experience profile.

    The Self-Grown write-back. Called from
    :meth:`FeedbackService.record_action_outcome` via the ``role_experience:``
    ``action_id`` prefix (mirroring the ``trust:`` / ``model_route:`` seams), so
    the AHE reward loop is the ONE path that grows staff. Updates a JSON
    ``experienceProfile`` (success/partial/failure counters + per-domain counts)
    and a scalar ``experienceScore`` on the ``:Employee`` node, then re-bands
    seniority. The recruiter reads these back when it staffs the next run.

    Returns the updated ``{experienceScore, seniority, successes, ...}`` summary.
    """
    emp = employee_id or role_id
    node = _read_node(backend, emp)
    raw = node.get("experienceProfile")
    profile: dict[str, Any]
    if isinstance(raw, str) and raw:
        try:
            profile = json.loads(raw)
        except Exception:  # noqa: BLE001 — corrupt profile → restart clean
            profile = {}
    elif isinstance(raw, dict):
        profile = dict(raw)
    else:
        profile = {}

    profile.setdefault("successes", 0)
    profile.setdefault("partials", 0)
    profile.setdefault("failures", 0)
    profile.setdefault("role_id", role_id)
    dom_counts: dict[str, int] = dict(profile.get("domains", {}) or {})

    r = max(0.0, min(1.0, float(reward)))
    if success and r >= 0.75:
        profile["successes"] = int(profile["successes"]) + 1
    elif success or r > 0.0:
        profile["partials"] = int(profile["partials"]) + 1
    else:
        profile["failures"] = int(profile["failures"]) + 1
    for dom in domains or []:
        dom_counts[dom] = dom_counts.get(dom, 0) + 1
    profile["domains"] = dom_counts

    # Score: successes reward, partials half, failures penalize; domain breadth
    # is a small bonus (adapts EmployeeEvolutionManager.get_experience_score).
    score = (
        int(profile["successes"])
        + 0.5 * int(profile["partials"])
        - 0.25 * int(profile["failures"])
        + 0.5 * len(dom_counts)
    )
    score = max(0.0, score)
    seniority = _seniority_for_score(score)
    profile["experience_score"] = round(score, 3)
    profile["seniority"] = seniority

    props = {
        "experienceProfile": json.dumps(profile, sort_keys=True),
        "experienceScore": round(score, 3),
        "seniority": seniority,
        "role_id": role_id,
    }
    _write_node(backend, emp, "Employee", props)
    return {
        "employee_id": emp,
        "role_id": role_id,
        "experienceScore": round(score, 3),
        "seniority": seniority,
        "successes": profile["successes"],
        "partials": profile["partials"],
        "failures": profile["failures"],
    }


def _write_node(
    backend: Any, node_id: str, node_type: str, props: dict[str, Any]
) -> None:
    """Best-effort upsert of node properties (engine ``add_node`` semantics)."""
    fn = getattr(backend, "add_node", None)
    if callable(fn):
        try:
            fn(node_id, node_type, properties=props)
            return
        except TypeError:
            try:
                fn(node_id, node_type, props)
                return
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    graph = getattr(backend, "graph", None)
    try:
        if graph is not None:
            existing = (
                dict(graph.nodes.get(node_id, {})) if node_id in graph.nodes else {}
            )
            existing.update(props)
            existing["type"] = node_type
            graph.add_node(node_id, **existing)
    except Exception:  # noqa: BLE001
        logger.debug("org_runtime: node write skipped for %s", node_id)


def _link(backend: Any, src: str, tgt: str, rel: str, **props: Any) -> None:
    for name in ("link_nodes", "add_edge"):
        fn = getattr(backend, name, None)
        if callable(fn):
            try:
                if name == "link_nodes":
                    fn(src, tgt, rel, properties=props or None)
                else:
                    fn(src, tgt, rel, **props)
                return
            except Exception:  # noqa: BLE001 — edges are best-effort provenance
                continue


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "role"


# ─────────────────────────────────────────────────────────────────────────
# Recruiter / org-synthesis (CONCEPT:AU-ORCH.org.recruiter)
# ─────────────────────────────────────────────────────────────────────────
#: Keyword → (role name, department, role_type) seeds for deterministic drafting.
#: Deliberately dependency-free (no LLM) so org synthesis works zero-infra; an
#: LLM refinement is layered on top when an engine + model are available.
_ROLE_SEEDS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (
        ("research", "investigate", "discover", "analyze", "study"),
        "Research Analyst",
        "Research",
        "worker",
    ),
    (
        ("design", "architecture", "architect", "plan", "spec"),
        "Solution Architect",
        "Engineering",
        "worker",
    ),
    (
        ("build", "implement", "code", "develop", "engineer", "program"),
        "Software Engineer",
        "Engineering",
        "worker",
    ),
    (("test", "qa", "verify", "validate", "quality"), "QA Engineer", "QA", "reviewer"),
    (
        ("deploy", "release", "ship", "operate", "ops", "infra"),
        "DevOps Engineer",
        "IT",
        "worker",
    ),
    (
        ("write", "document", "content", "copy", "docs"),
        "Content Writer",
        "Product",
        "worker",
    ),
    (
        ("market", "growth", "campaign", "launch", "brand"),
        "Marketing Lead",
        "Product",
        "worker",
    ),
    (
        ("finance", "budget", "cost", "revenue", "pricing"),
        "Finance Analyst",
        "Operations",
        "worker",
    ),
)


class Recruiter:
    """Synthesize an org chart from a goal and staff it (Self-Built).

    CONCEPT:AU-ORCH.org.recruiter. Drafts departments → roles from the goal
    (deterministic keyword seeds, optionally refined by the local LLM), then for
    each role *fills the seat*: reuse the highest-experience existing :Employee if
    one exists (``existing_staff``), else hire a fresh template
    (``proposed_hire``). Reuses any pre-existing ``:AgentRole``/``:Employee``
    nodes, and persists new ones. The reuse-vs-hire decision reads the
    ``experienceScore`` grown by :func:`record_role_experience` — closing the
    Self-Grown loop.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    # -- drafting -------------------------------------------------------
    def _draft_roles(self, goal: str, domains: list[str]) -> list[RoleSpec]:
        text = goal.lower()
        roles: list[RoleSpec] = []
        seen: set[str] = set()
        for keywords, name, dept, rtype in _ROLE_SEEDS:
            if any(k in text for k in keywords):
                rid = _slug(name)
                if rid in seen:
                    continue
                seen.add(rid)
                roles.append(
                    RoleSpec(
                        role_id=rid,
                        name=name,
                        responsibility=f"Own the '{name}' contribution toward: {goal}",
                        department=dept,
                        role_type=rtype,
                        domains=list(domains),
                    )
                )
        if not roles:
            # Every goal needs at least one doer.
            roles.append(
                RoleSpec(
                    role_id="generalist",
                    name="Generalist",
                    responsibility=f"Accomplish the goal end to end: {goal}",
                    department="Operations",
                    role_type="worker",
                    domains=list(domains),
                )
            )
        # Always add a coordinator to own the overall goal + integrate results.
        coord = RoleSpec(
            role_id="project_coordinator",
            name="Project Coordinator",
            responsibility=f"Decompose, delegate, and integrate delivery of: {goal}",
            department="Operations",
            role_type="coordinator",
            domains=list(domains),
        )
        for r in roles:
            r.reports_to = coord.role_id
        return [coord, *roles]

    def _existing_role_ids(self) -> set[str]:
        """Role ids already present as ``:AgentRole`` nodes (reuse pool)."""
        out: set[str] = set()
        backend = getattr(self.engine, "backend", None) or self.engine
        graph = getattr(self.engine, "graph", None)
        try:
            if graph is not None:
                for _nid, data in graph.nodes(data=True):
                    if str(data.get("type", "")).endswith("AgentRole"):
                        rid = str(data.get("id") or data.get("role_id") or "").strip()
                        if rid:
                            out.add(_slug(rid))
        except Exception:  # noqa: BLE001 — reuse pool is best-effort
            logger.debug("recruiter: existing-role scan skipped")
        _ = backend
        return out

    # -- staffing -------------------------------------------------------
    def _staff_role(self, role: RoleSpec) -> EmployeeSpec:
        """Reuse the experienced employee for this role, else hire fresh."""
        backend = getattr(self.engine, "backend", None) or self.engine
        candidate_emp = f"emp_{role.role_id}"
        score = experience_score(backend, candidate_emp)
        if score > 0.0:
            node = _read_node(backend, candidate_emp)
            return EmployeeSpec(
                employee_id=candidate_emp,
                name=str(node.get("name") or role.name),
                role_id=role.role_id,
                status="existing_staff",
                experience_score=score,
                seniority=str(node.get("seniority") or _seniority_for_score(score)),
                domains=role.domains,
                rationale=f"reuse experienced staff (score={score:.2f})",
            )
        return EmployeeSpec(
            employee_id=candidate_emp,
            name=role.name,
            role_id=role.role_id,
            status="proposed_hire",
            experience_score=0.0,
            seniority="junior",
            domains=role.domains,
            rationale="no experienced staff — hire fresh template",
        )

    # -- persistence ----------------------------------------------------
    def _persist(self, chart: OrgChart) -> None:
        backend = getattr(self.engine, "backend", None) or self.engine
        _write_node(
            backend,
            chart.company_id,
            "Company",
            {"id": chart.company_id, "goal": chart.goal},
        )
        for role in chart.roles:
            node_id = f"role_{role.role_id}"
            _write_node(
                backend,
                node_id,
                "AgentRole",
                {
                    "id": role.role_id,
                    "role": role.name,
                    "role_type": role.role_type,
                    "department": role.department,
                    "responsibility": role.responsibility,
                },
            )
            _link(backend, chart.company_id, node_id, "hasAgentRole")
            if role.reports_to:
                _link(backend, node_id, f"role_{role.reports_to}", "reportsTo")
        for emp in chart.employees:
            _write_node(
                backend,
                emp.employee_id,
                "Employee",
                {
                    "id": emp.employee_id,
                    "name": emp.name,
                    "role_id": emp.role_id,
                    "seniority": emp.seniority,
                    "experienceScore": round(emp.experience_score, 3),
                },
            )
            _link(backend, emp.employee_id, f"role_{emp.role_id}", "staffsRole")

    def synthesize_org(
        self,
        goal: str,
        *,
        domains: list[str] | None = None,
        company_id: str = "__company__",
    ) -> OrgChart:
        """Draft an org chart from *goal* and staff every role.

        Reuses existing ``:AgentRole`` nodes (marks ``reused=True``) and existing
        experienced ``:Employee`` staff; persists the result to the KG.
        """
        doms = list(domains or [])
        roles = self._draft_roles(goal, doms)
        existing = self._existing_role_ids()
        for r in roles:
            r.reused = r.role_id in existing
        chart = OrgChart(goal=goal, company_id=company_id, roles=roles)
        chart.employees = [self._staff_role(r) for r in roles]
        self._persist(chart)
        logger.info(
            "recruiter: synthesized org for goal=%r — %d roles (%d reused), %d staffed",
            goal[:60],
            len(roles),
            sum(1 for r in roles if r.reused),
            len(chart.employees),
        )
        return chart


# ─────────────────────────────────────────────────────────────────────────
# Work-item DAG runtime (CONCEPT:AU-ORCH.org.work-item-dag)
# ─────────────────────────────────────────────────────────────────────────
#: An escalation is handed an immutable plan item + reason. Returning
#: ``"approve"`` accepts the current output; any other result fails closed.
EscalationCallback = Callable[[OrgPlanItem, str], Awaitable[str | None]]

_MAX_REWORK = 1  # rework rounds before a beyond-team blocker escalates to human


class OrgRuntime:
    """Execute an organization DAG through the sole native WorkItem authority.

    Plan definitions are immutable. Native submission, dependency release,
    lease ownership, renewal, fencing, and terminal commit are the only durable
    execution state. Independent items may run concurrently after native claim.
    """

    def __init__(
        self,
        engine: Any,
        *,
        escalation_cb: EscalationCallback | None = None,
        max_steps: int = 20,
    ) -> None:
        self.engine = engine
        self.escalation_cb = escalation_cb or self._default_escalation
        self.max_steps = max_steps
        self._backend = getattr(engine, "backend", None) or engine

    # -- default escalation seam ---------------------------------------
    async def _default_escalation(self, item: OrgPlanItem, reason: str) -> str | None:
        """No human wired: log the blocker and fail closed.

        The callback never writes task state. Its resolution is committed by
        the live native lease holder, or the blocked WorkItem is cancelled by
        the native authority when it cannot be claimed.
        """
        logger.warning("org escalation [%s]: %s", item.plan_item_id, reason)
        return None

    # -- executor seam (overridable for tests) -------------------------
    async def _execute_role(self, role_id: str, task: str, context: str | None) -> str:
        """Run one role turn through the core orchestrator.

        Routes through :meth:`Orchestrator.execute_agent` → ``run_agent`` — the
        SAME executor the rest of the platform uses (the one-core rule), so each
        turn's ``RunTrace``/``:ToolCall`` provenance is written for free.
        """
        from agent_utilities.orchestration.manager import Orchestrator

        orch = Orchestrator(self.engine)
        return await orch.execute_agent(
            agent_name=role_id,
            task=task,
            max_steps=self.max_steps,
            context=context,
        )

    # -- work-item derivation ------------------------------------------
    def derive_plan(self, goal: str, chart: OrgChart) -> list[OrgPlanItem]:
        """Derive immutable organization-plan inputs from an org chart.

        One plan item is produced per worker plus a coordinator integration
        item. No lifecycle state is created until :meth:`run` submits native
        WorkItems.
        """
        coord = next((r for r in chart.roles if r.role_type == "coordinator"), None)
        reviewer = next((r for r in chart.roles if r.role_type == "reviewer"), None)
        workers = [r for r in chart.roles if r.role_type == "worker"]
        items: list[OrgPlanItem] = []
        worker_ids: list[str] = []
        for r in workers:
            wid = f"plan_{r.role_id}"
            worker_ids.append(wid)
            items.append(
                OrgPlanItem(
                    plan_item_id=wid,
                    title=f"{r.name}: contribute to goal",
                    description=r.responsibility,
                    owner_role=r.role_id,
                    role_type=r.role_type,
                    reviewer_role=reviewer.role_id if reviewer else None,
                    metadata={"domains": tuple(r.domains)},
                )
            )
        if coord:
            items.append(
                OrgPlanItem(
                    plan_item_id=f"plan_{coord.role_id}",
                    title=f"{coord.name}: integrate deliverables",
                    description=f"Integrate all worker deliverables into the goal: {goal}",
                    owner_role=coord.role_id,
                    role_type="coordinator",
                    dependencies=tuple(worker_ids),
                    metadata={"domains": tuple(coord.domains)},
                )
            )
        return items

    # -- scheduling ----------------------------------------------------
    def _refresh_status(self, state: _ExecutionState) -> str:
        row = get_work_item(self.engine, state.work_item_id)
        state.status = str((row or {}).get("status") or "missing")
        return state.status

    async def _run_item(
        self,
        state: _ExecutionState,
        outputs: dict[str, str],
    ) -> None:
        """Claim, renew, and commit one item through native WorkItem verbs."""
        import asyncio  # local import keeps module import light

        item = state.plan
        claim = claim_specific(self.engine, state.work_item_id)
        if claim is None:
            self._refresh_status(state)
            return
        if not mark_running(self.engine, state.work_item_id, claim):
            self._refresh_status(state)
            return
        state.status = WorkItemStatus.RUNNING.value
        state.manager_mode = infer_manager_mode(item)
        ctx = "\n\n".join(
            f"Output of dependency {index + 1}:\n{outputs.get(dep, '')}"
            for index, dep in enumerate(item.dependencies)
            if outputs.get(dep)
        )

        async def fail(error_ref: str, reward: float = 0.0) -> None:
            result = commit_result(
                self.engine,
                state.work_item_id,
                claim,
                outcome=WorkItemStatus.FAILED.value,
                error_ref=error_ref,
                retryable=False,
            )
            if result not in {"committed", "noop"}:
                logger.warning(
                    "org native failure commit rejected [%s]: %s",
                    state.work_item_id,
                    result,
                )
            self._refresh_status(state)
            self._record_experience(item, success=False, reward=reward)

        while True:
            state.manager_mode = infer_manager_mode(
                item,
                rework_count=state.rework_count,
                review_feedback=state.review_feedback,
            )
            framed = self._frame_task(item, state.manager_mode)
            feedback_ctx = ctx
            if state.review_feedback:
                feedback_ctx = (
                    f"{ctx}\n\nReviewer feedback to address:\n{state.review_feedback}"
                ).strip()
            try:
                out = await self._execute_role(
                    item.owner_role, framed, feedback_ctx or None
                )
            except Exception as exc:  # noqa: BLE001 — isolate one DAG item
                state.output = f"error: {type(exc).__name__}"
                await fail("org-execution-error")
                return
            state.output = str(out)
            if state.output.startswith("Agent execution failed"):
                await fail("org-agent-execution-failed")
                return
            if not heartbeat(self.engine, state.work_item_id, claim):
                self._refresh_status(state)
                logger.warning("org native WorkItem lease lost before review/commit")
                return

            if item.reviewer_role and state.manager_mode is not ManagerMode.REVIEW:
                verdict, feedback = await self._review(item, state.output)
                if verdict != "approve":
                    if state.rework_count >= _MAX_REWORK:
                        resolution = await self.escalation_cb(
                            item,
                            f"review rejected {state.rework_count + 1}x: "
                            f"{feedback[:200]}",
                        )
                        if resolution != "approve":
                            await fail("org-review-escalated", reward=0.25)
                            return
                        reward = 0.75
                    else:
                        state.rework_count += 1
                        state.review_feedback = feedback
                        self._record_experience(item, success=False, reward=0.4)
                        await asyncio.sleep(0)
                        continue
                else:
                    reward = 1.0
            else:
                reward = 1.0

            result = commit_result(
                self.engine,
                state.work_item_id,
                claim,
                outcome=WorkItemStatus.SUCCEEDED.value,
                result_ref=f"org-result:{state.work_item_id}",
                retryable=False,
            )
            if result not in {"committed", "noop"}:
                logger.warning(
                    "org native success commit rejected [%s]: %s",
                    state.work_item_id,
                    result,
                )
                self._refresh_status(state)
                return
            self._refresh_status(state)
            self._record_experience(item, success=True, reward=reward)
            return

    @staticmethod
    def _frame_task(item: OrgPlanItem, mode: ManagerMode) -> str:
        """Frame the item's task text for the owning role's mode."""
        base = item.description
        if mode == ManagerMode.DELEGATE:
            return f"[DELEGATE] As coordinator, plan and delegate the work for: {base}"
        if mode == ManagerMode.INTEGRATE:
            return f"[INTEGRATE] Roll up the subordinate deliverables into the final result for: {base}"
        if mode == ManagerMode.REWORK:
            return f"[REWORK] Address the reviewer feedback and redo: {base}"
        if mode == ManagerMode.REVIEW:
            return f"[REVIEW] Evaluate the deliverable and return a verdict for: {base}"
        return base

    async def _review(self, item: OrgPlanItem, output: str) -> tuple[str, str]:
        """Run the reviewer role and parse an approve/rework verdict."""
        assert item.reviewer_role is not None
        task = (
            f"[REVIEW] Evaluate this deliverable for '{item.title}'. "
            f"Reply with 'APPROVE' if acceptable or 'REWORK: <reason>' otherwise.\n\n"
            f"Deliverable:\n{output[:2000]}"
        )
        try:
            verdict_out = await self._execute_role(item.reviewer_role, task, None)
        except Exception as exc:  # noqa: BLE001 — a failed reviewer defaults to rework
            return "rework", f"reviewer error: {exc}"
        text = str(verdict_out).strip()
        if re.search(r"\bapprove\b", text, re.IGNORECASE) and not re.search(
            r"\brework\b", text, re.IGNORECASE
        ):
            return "approve", ""
        m = re.search(r"rework[:\-\s]+(.*)", text, re.IGNORECASE | re.DOTALL)
        return "rework", (m.group(1).strip() if m else text)[:400]

    # -- experience ----------------------------------------------------
    def _record_experience(
        self, item: OrgPlanItem, *, success: bool, reward: float
    ) -> None:
        """Write the item's outcome back through the AHE reward loop.

        Uses :meth:`FeedbackService.record_action_outcome` with a
        ``role_experience:<role_id>`` ``action_id`` so the SAME reward substrate
        that trains routing/retrieval grows the org's staff. Falls back to a
        direct profile write if the feedback service is unavailable.
        """
        emp_id = f"emp_{item.owner_role}"
        domains = list(item.metadata.get("domains", []) or [])
        try:
            from agent_utilities.knowledge_graph.adaptation.feedback import (
                FeedbackService,
            )

            svc = FeedbackService.from_engine(self.engine)
            svc.record_action_outcome(
                f"role_experience:{item.owner_role}",
                success=success,
                reward=reward,
                agent_id=item.owner_role,
                reason="organization_native_work_item",
                corrected_value={"employee_id": emp_id, "domains": domains},
            )
        except Exception as exc:  # noqa: BLE001 — never fail the run on write-back
            logger.debug("org experience write-back via feedback failed: %s", exc)
            record_role_experience(
                self._backend,
                item.owner_role,
                employee_id=emp_id,
                success=success,
                reward=reward,
                domains=domains,
            )

    def _submit_plan(
        self, items: Sequence[OrgPlanItem], *, run_id: str
    ) -> dict[str, _ExecutionState]:
        """Materialize immutable plan inputs as native WorkItems.

        Parent definitions are submitted before children where possible so the
        native reverse dependency index is complete. Missing/cyclic references
        remain conservatively blocked and are handled through native cancel.
        """
        by_id = {item.plan_item_id: item for item in items}
        if len(by_id) != len(items) or any(not key for key in by_id):
            raise ValueError("organization plan item ids must be unique and non-empty")
        native_ids = {key: new_work_item_id() for key in by_id}
        missing_ids: dict[str, str] = {}
        pending = list(items)
        ordered: list[OrgPlanItem] = []
        submitted: set[str] = set()
        while pending:
            ready = [
                item
                for item in pending
                if all(
                    dep not in by_id or dep in submitted for dep in item.dependencies
                )
            ]
            if not ready:
                ordered.extend(pending)
                break
            for item in ready:
                ordered.append(item)
                submitted.add(item.plan_item_id)
                pending.remove(item)

        states: dict[str, _ExecutionState] = {}
        for item in ordered:
            dependencies = [
                native_ids.get(dep) or missing_ids.setdefault(dep, new_work_item_id())
                for dep in item.dependencies
            ]
            native_id = native_ids[item.plan_item_id]
            submit_work_item(
                self.engine,
                kind="organization_task",
                queue="organization",
                payload_ref=f"org-plan:{native_id}",
                depends_on=dependencies,
                max_attempts=1,
                idempotency_key=native_id,
                description="organization plan task",
                dag_id=run_id,
                metadata={"plan_schema_version": "1"},
                work_item_id=native_id,
            )
            state = _ExecutionState(plan=item, work_item_id=native_id)
            self._refresh_status(state)
            states[item.plan_item_id] = state
        return states

    # -- top-level run -------------------------------------------------
    async def run(
        self,
        goal: str,
        *,
        plan_items: Sequence[OrgPlanItem] | None = None,
        domains: list[str] | None = None,
        chart: OrgChart | None = None,
    ) -> dict[str, Any]:
        """Synthesize an org, submit its native WorkItem DAG, and execute it.

        The return value is only a process-local presentation. Durable status is
        read from the engine-native WorkItems identified in ``plan_items``.
        """
        import asyncio

        if chart is None:
            chart = Recruiter(self.engine).synthesize_org(goal, domains=domains)
        items = (
            list(plan_items)
            if plan_items is not None
            else self.derive_plan(goal, chart)
        )
        outputs: dict[str, str] = {}
        run_id = f"org-{uuid.uuid4().hex}"
        states = self._submit_plan(items, run_id=run_id)
        terminal = {
            WorkItemStatus.SUCCEEDED.value,
            WorkItemStatus.FAILED.value,
            WorkItemStatus.CANCELLED.value,
            WorkItemStatus.DEAD_LETTER.value,
            "missing",
        }
        remaining = list(states.values())
        guard = 0
        while remaining:
            guard += 1
            if guard > len(items) * (2 + _MAX_REWORK) + 5:
                logger.error("org run %s: scheduler guard tripped", run_id)
                for state in remaining:
                    cancel_work_item(
                        self.engine,
                        state.work_item_id,
                        reason="org-scheduler-guard",
                    )
                    self._refresh_status(state)
                break
            ready = [
                state
                for state in remaining
                if self._refresh_status(state) == WorkItemStatus.READY.value
            ]
            if not ready:
                blocked = [state for state in remaining if state.status not in terminal]
                for state in blocked:
                    await self.escalation_cb(
                        state.plan, "unsatisfiable dependencies (DAG deadlock)"
                    )
                    cancel_work_item(
                        self.engine,
                        state.work_item_id,
                        reason="org-unsatisfiable-dependencies",
                    )
                    self._refresh_status(state)
                break

            await asyncio.gather(*(self._run_item(state, outputs) for state in ready))
            for state in ready:
                self._refresh_status(state)
                if state.status == WorkItemStatus.SUCCEEDED.value:
                    outputs[state.plan.plan_item_id] = state.output
            remaining = [
                state
                for state in remaining
                if self._refresh_status(state) not in terminal
            ]

        succeeded = sum(
            self._refresh_status(state) == WorkItemStatus.SUCCEEDED.value
            for state in states.values()
        )
        status = (
            "completed"
            if succeeded == len(items)
            else ("partial" if succeeded else "failed")
        )
        return {
            "run_id": run_id,
            "goal": goal,
            "status": status,
            "org_chart": chart.to_dict(),
            "plan_items": [states[item.plan_item_id].to_dict() for item in items],
            "succeeded": succeeded,
            "total": len(items),
        }


__all__ = [
    "ManagerMode",
    "RoleSpec",
    "EmployeeSpec",
    "OrgChart",
    "OrgPlanItem",
    "Recruiter",
    "OrgRuntime",
    "infer_manager_mode",
    "record_role_experience",
    "experience_score",
    "EscalationCallback",
]
