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
* **Work-item state machine + dependency DAG + kanban phases**
  (CONCEPT:AU-ORCH.org.work-item-dag). :class:`OrgRuntime` derives a
  :class:`WorkItem` DAG from the org chart, runs independent items in parallel
  and dependents once their predecessors are approved, drives each item through
  :class:`OrgPhase` transitions under a :class:`ManagerMode`
  (execute/delegate/review/integrate/rework), and escalates beyond-team blockers
  to a human via an escalation seam.
* **Self-Grown** (CONCEPT:AU-AHE.org.role-experience). Each item's outcome is
  written back through the AHE reward loop
  (:meth:`FeedbackService.record_action_outcome` with a ``role_experience:``
  ``action_id``), updating the :Employee's ``experienceProfile`` /
  ``experienceScore`` so the next recruiter run reuses proven staff.

The whole runtime is exposed as ``graph_orchestrate(action='synthesize_org')``
and ``graph_orchestrate(action='run_org')`` on both the MCP server and the REST
gateway (surface parity).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Kanban phase state machine (CONCEPT:AU-ORCH.org.work-item-dag)
# ─────────────────────────────────────────────────────────────────────────
class OrgPhase(StrEnum):
    """The single authoritative state of a work item.

    A compact adaptation of OpenOPC ``phase.py`` — one enum value per concrete
    situation, with pure-function projections (kanban column, runnability) and a
    static transition table enforced on every write.
    """

    QUEUED = "queued"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    READY = "ready"
    READY_FOR_REWORK = "ready_for_rework"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    ESCALATED = "escalated"  # awaiting a human decision on a beyond-team blocker
    APPROVED = "approved"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TODO = frozenset(
    {
        OrgPhase.QUEUED,
        OrgPhase.WAITING_DEPENDENCIES,
        OrgPhase.READY,
        OrgPhase.READY_FOR_REWORK,
    }
)
_IN_PROGRESS = frozenset({OrgPhase.RUNNING})
_IN_REVIEW = frozenset({OrgPhase.AWAITING_REVIEW, OrgPhase.ESCALATED})
_DONE = frozenset({OrgPhase.APPROVED, OrgPhase.FAILED, OrgPhase.CANCELLED})

TERMINAL_PHASES: frozenset[OrgPhase] = _DONE
RUNNABLE_PHASES: frozenset[OrgPhase] = frozenset(
    {OrgPhase.READY, OrgPhase.READY_FOR_REWORK}
)

_UNIVERSAL_EXITS = frozenset({OrgPhase.FAILED, OrgPhase.CANCELLED, OrgPhase.ESCALATED})

ALLOWED_TRANSITIONS: dict[OrgPhase, frozenset[OrgPhase]] = {
    OrgPhase.QUEUED: frozenset({OrgPhase.READY, OrgPhase.WAITING_DEPENDENCIES})
    | _UNIVERSAL_EXITS,
    OrgPhase.WAITING_DEPENDENCIES: frozenset({OrgPhase.READY}) | _UNIVERSAL_EXITS,
    OrgPhase.READY: frozenset({OrgPhase.RUNNING, OrgPhase.WAITING_DEPENDENCIES})
    | _UNIVERSAL_EXITS,
    OrgPhase.READY_FOR_REWORK: frozenset({OrgPhase.RUNNING}) | _UNIVERSAL_EXITS,
    OrgPhase.RUNNING: frozenset(
        {OrgPhase.AWAITING_REVIEW, OrgPhase.APPROVED, OrgPhase.READY}
    )
    | _UNIVERSAL_EXITS,
    OrgPhase.AWAITING_REVIEW: frozenset({OrgPhase.APPROVED, OrgPhase.READY_FOR_REWORK})
    | _UNIVERSAL_EXITS,
    OrgPhase.ESCALATED: frozenset({OrgPhase.APPROVED, OrgPhase.READY_FOR_REWORK})
    | frozenset({OrgPhase.FAILED, OrgPhase.CANCELLED}),
    OrgPhase.APPROVED: frozenset(),
    OrgPhase.FAILED: frozenset(),
    OrgPhase.CANCELLED: frozenset(),
}

_PHASE_TO_COLUMN: dict[OrgPhase, str] = {
    **{p: "todo" for p in _TODO},
    **{p: "in_progress" for p in _IN_PROGRESS},
    **{p: "in_review" for p in _IN_REVIEW},
    **{p: "done" for p in _DONE},
}


class InvalidPhaseTransition(ValueError):
    """Raised when a work-item write attempts a transition not in the table."""


def validate_transition(previous: OrgPhase | None, target: OrgPhase) -> None:
    """Raise :class:`InvalidPhaseTransition` when *target* is unreachable.

    Initial creation (``previous is None``) and idempotent writes are allowed.
    """
    if previous is None or previous == target:
        return
    if target not in ALLOWED_TRANSITIONS.get(previous, frozenset()):
        raise InvalidPhaseTransition(
            f"invalid phase transition: {previous.value} -> {target.value}"
        )


def kanban_column(phase: OrgPhase) -> str:
    """Project a phase to one of the four kanban columns."""
    return _PHASE_TO_COLUMN[phase]


def is_runnable(phase: OrgPhase) -> bool:
    """Whether the DAG scheduler should claim + spawn this item next tick."""
    return phase in RUNNABLE_PHASES


def is_terminal(phase: OrgPhase) -> bool:
    return phase in TERMINAL_PHASES


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


@dataclass
class WorkItem:
    """A unit of runtime work owned by a role, flowing through the kanban DAG."""

    work_item_id: str
    title: str
    description: str
    owner_role: str
    dependencies: list[str] = field(default_factory=list)
    reviewer_role: str | None = None
    phase: OrgPhase = OrgPhase.QUEUED
    manager_mode: ManagerMode = ManagerMode.EXECUTE
    role_type: str = "worker"
    output: str = ""
    rework_count: int = 0
    review_feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, target: OrgPhase) -> None:
        """Validated phase write."""
        validate_transition(self.phase, target)
        self.phase = target

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "title": self.title,
            "owner_role": self.owner_role,
            "reviewer_role": self.reviewer_role,
            "dependencies": self.dependencies,
            "phase": self.phase.value,
            "kanban_column": kanban_column(self.phase),
            "manager_mode": self.manager_mode.value,
            "rework_count": self.rework_count,
            "output": self.output[:500],
        }


def infer_manager_mode(item: WorkItem, *, is_review_entry: bool = False) -> ManagerMode:
    """Pure classifier for the owning role's mode this turn.

    Priority-ordered, mirroring OpenOPC ``infer_turn_mode``:
    REWORK → REVIEW → INTEGRATE → DELEGATE → EXECUTE.
    """
    if (
        item.phase == OrgPhase.READY_FOR_REWORK
        or item.review_feedback
        or item.rework_count > 0
    ):
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
#: An escalation is handed a work item + reason; returns a resolution string
#: ("approve" / "rework" / None). Default seam marks the item ESCALATED.
EscalationCallback = Callable[[WorkItem, str], Awaitable[str | None]]

_MAX_REWORK = 1  # rework rounds before a beyond-team blocker escalates to human


class OrgRuntime:
    """Execute a work-item DAG over the existing orchestrator (Self-Run).

    Independent items run in parallel; dependents wait for approval of every
    predecessor. Each item runs under its :class:`ManagerMode`, is optionally
    reviewed by its manager role (approve / rework), and — when a rework budget
    is exhausted or a dependency can never be satisfied — escalated to a human
    through :attr:`escalation_cb`. Every completed item's outcome is written back
    to the :Employee experience profile via the AHE reward loop.
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
    async def _default_escalation(self, item: WorkItem, reason: str) -> str | None:
        """No human wired → record the blocker and leave the item ESCALATED.

        A deployment supplies its own callback (e.g. one that opens an approval
        via :mod:`agent_utilities.orchestration.action_policy`) to actually reach
        a human; the default is fail-safe (never silently auto-approves).
        """
        logger.warning("org escalation [%s]: %s", item.work_item_id, reason)
        _write_node(
            self._backend,
            item.work_item_id,
            "WorkItem",
            {"escalation_reason": reason, "workItemPhase": OrgPhase.ESCALATED.value},
        )
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
    def derive_work_items(self, goal: str, chart: OrgChart) -> list[WorkItem]:
        """Derive a work-item DAG from an org chart.

        One item per worker role (parallelizable), plus a coordinator
        INTEGRATE item that depends on all of them. A reviewer role, if present,
        becomes the coordinator item's ``reviewer_role``.
        """
        coord = next((r for r in chart.roles if r.role_type == "coordinator"), None)
        reviewer = next((r for r in chart.roles if r.role_type == "reviewer"), None)
        workers = [r for r in chart.roles if r.role_type == "worker"]
        items: list[WorkItem] = []
        worker_ids: list[str] = []
        for r in workers:
            wid = f"wi_{r.role_id}"
            worker_ids.append(wid)
            items.append(
                WorkItem(
                    work_item_id=wid,
                    title=f"{r.name}: contribute to goal",
                    description=r.responsibility,
                    owner_role=r.role_id,
                    role_type=r.role_type,
                    reviewer_role=reviewer.role_id if reviewer else None,
                    phase=OrgPhase.QUEUED,
                    metadata={"goal": goal},
                )
            )
        if coord:
            items.append(
                WorkItem(
                    work_item_id=f"wi_{coord.role_id}",
                    title=f"{coord.name}: integrate deliverables",
                    description=f"Integrate all worker deliverables into the goal: {goal}",
                    owner_role=coord.role_id,
                    role_type="coordinator",
                    dependencies=worker_ids,
                    phase=OrgPhase.QUEUED,
                    metadata={"goal": goal},
                )
            )
        return items

    # -- scheduling ----------------------------------------------------
    @staticmethod
    def _runnable(items: list[WorkItem], done: set[str]) -> list[WorkItem]:
        out = []
        for it in items:
            if it.phase not in (
                OrgPhase.QUEUED,
                OrgPhase.WAITING_DEPENDENCIES,
                OrgPhase.READY,
            ):
                continue
            if all(dep in done for dep in it.dependencies):
                out.append(it)
        return out

    async def _run_item(self, item: WorkItem, outputs: dict[str, str]) -> None:
        """Drive one item through RUNNING → (review) → APPROVED / escalation."""
        import asyncio  # local import keeps module import light

        item.manager_mode = infer_manager_mode(item)
        # Thread approved-dependency outputs in as context (INTEGRATE/REWORK use it).
        ctx = "\n\n".join(
            f"Output of '{d}':\n{outputs.get(d, '')}"
            for d in item.dependencies
            if outputs.get(d)
        )

        while True:
            item.transition(OrgPhase.RUNNING)
            mode = item.manager_mode
            framed = self._frame_task(item, mode)
            feedback_ctx = ctx
            if item.review_feedback:
                feedback_ctx = f"{ctx}\n\nReviewer feedback to address:\n{item.review_feedback}".strip()
            try:
                out = await self._execute_role(
                    item.owner_role, framed, feedback_ctx or None
                )
            except Exception as exc:  # noqa: BLE001 — one item must not kill the DAG
                item.transition(OrgPhase.FAILED)
                item.output = f"error: {exc}"
                self._record_experience(item, success=False, reward=0.0)
                self._persist_item(item)
                return
            item.output = str(out)
            failed = str(out).startswith("Agent execution failed")
            if failed:
                item.transition(OrgPhase.FAILED)
                self._record_experience(item, success=False, reward=0.0)
                self._persist_item(item)
                return

            # Review gate: a reviewer role adjudicates the deliverable.
            if item.reviewer_role and mode not in (ManagerMode.REVIEW,):
                item.transition(OrgPhase.AWAITING_REVIEW)
                self._persist_item(item)
                verdict, feedback = await self._review(item)
                if verdict == "approve":
                    item.transition(OrgPhase.APPROVED)
                    self._record_experience(item, success=True, reward=1.0)
                    self._persist_item(item)
                    return
                # rework
                if item.rework_count >= _MAX_REWORK:
                    # beyond-team blocker → escalate to a human
                    item.transition(OrgPhase.ESCALATED)
                    self._persist_item(item)
                    resolution = await self.escalation_cb(
                        item,
                        f"review rejected {item.rework_count + 1}x: {feedback[:200]}",
                    )
                    if resolution == "approve":
                        item.transition(OrgPhase.APPROVED)
                        self._record_experience(item, success=True, reward=0.75)
                    else:
                        item.transition(OrgPhase.FAILED)
                        self._record_experience(item, success=False, reward=0.25)
                    self._persist_item(item)
                    return
                item.rework_count += 1
                item.review_feedback = feedback
                item.manager_mode = ManagerMode.REWORK
                item.transition(OrgPhase.READY_FOR_REWORK)
                self._record_experience(item, success=False, reward=0.4)
                self._persist_item(item)
                await asyncio.sleep(0)  # yield between rework rounds
                continue

            item.transition(OrgPhase.APPROVED)
            self._record_experience(item, success=True, reward=1.0)
            self._persist_item(item)
            return

    @staticmethod
    def _frame_task(item: WorkItem, mode: ManagerMode) -> str:
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

    async def _review(self, item: WorkItem) -> tuple[str, str]:
        """Run the reviewer role and parse an approve/rework verdict."""
        assert item.reviewer_role is not None
        task = (
            f"[REVIEW] Evaluate this deliverable for '{item.title}'. "
            f"Reply with 'APPROVE' if acceptable or 'REWORK: <reason>' otherwise.\n\n"
            f"Deliverable:\n{item.output[:2000]}"
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

    # -- experience + persistence --------------------------------------
    def _record_experience(
        self, item: WorkItem, *, success: bool, reward: float
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
                reason=f"work_item:{item.work_item_id}",
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

    def _persist_item(self, item: WorkItem) -> None:
        _write_node(
            self._backend,
            item.work_item_id,
            "WorkItem",
            {
                "id": item.work_item_id,
                "title": item.title,
                "workItemPhase": item.phase.value,
                "workItemManagerMode": item.manager_mode.value,
                "kanbanColumn": kanban_column(item.phase),
                "rework_count": item.rework_count,
            },
        )
        _link(
            self._backend, item.work_item_id, f"role_{item.owner_role}", "ownedByRole"
        )
        if item.reviewer_role:
            _link(
                self._backend,
                item.work_item_id,
                f"role_{item.reviewer_role}",
                "reviewedByRole",
            )
        for dep in item.dependencies:
            _link(self._backend, item.work_item_id, dep, "dependsOnWorkItem")

    # -- top-level run -------------------------------------------------
    async def run(
        self,
        goal: str,
        *,
        work_items: list[WorkItem] | None = None,
        domains: list[str] | None = None,
        chart: OrgChart | None = None,
    ) -> dict[str, Any]:
        """Synthesize an org (unless *chart* given), derive/accept a work-item
        DAG, and execute it wave by wave.

        Returns a run summary: the org chart, per-item final states, and counts.
        """
        import asyncio

        if chart is None:
            chart = Recruiter(self.engine).synthesize_org(goal, domains=domains)
        items = (
            work_items
            if work_items is not None
            else self.derive_work_items(goal, chart)
        )
        by_id = {it.work_item_id: it for it in items}
        outputs: dict[str, str] = {}
        done: set[str] = set()
        run_id = f"org-{uuid.uuid4().hex[:8]}"

        # Mark initial phases: no-dep items become READY, dependents wait.
        for it in items:
            it.transition(
                OrgPhase.WAITING_DEPENDENCIES if it.dependencies else OrgPhase.READY
            )

        remaining = list(items)
        guard = 0
        while remaining:
            guard += 1
            if guard > len(items) * (2 + _MAX_REWORK) + 5:
                logger.error("org run %s: scheduler guard tripped", run_id)
                break
            # Promote satisfied waiters to READY.
            for it in remaining:
                if it.phase == OrgPhase.WAITING_DEPENDENCIES and all(
                    d in done for d in it.dependencies
                ):
                    it.transition(OrgPhase.READY)
            ready = self._runnable(remaining, done)
            if not ready:
                # No item can run and none is in flight → a beyond-team blocker.
                blocked = [it for it in remaining if not is_terminal(it.phase)]
                for it in blocked:
                    it.transition(OrgPhase.ESCALATED)
                    await self.escalation_cb(
                        it, "unsatisfiable dependencies (DAG deadlock)"
                    )
                    self._persist_item(it)
                break

            await asyncio.gather(*(self._run_item(it, outputs) for it in ready))
            for it in ready:
                if it.phase == OrgPhase.APPROVED:
                    done.add(it.work_item_id)
                    outputs[it.work_item_id] = it.output
            remaining = [it for it in remaining if not is_terminal(it.phase)]

        approved = sum(1 for it in items if it.phase == OrgPhase.APPROVED)
        status = (
            "completed"
            if approved == len(items)
            else ("partial" if approved else "failed")
        )
        return {
            "run_id": run_id,
            "goal": goal,
            "status": status,
            "org_chart": chart.to_dict(),
            "work_items": [by_id[i].to_dict() for i in by_id],
            "approved": approved,
            "total": len(items),
        }


__all__ = [
    "OrgPhase",
    "ManagerMode",
    "RoleSpec",
    "EmployeeSpec",
    "OrgChart",
    "WorkItem",
    "Recruiter",
    "OrgRuntime",
    "InvalidPhaseTransition",
    "validate_transition",
    "kanban_column",
    "is_runnable",
    "is_terminal",
    "infer_manager_mode",
    "record_role_experience",
    "experience_score",
    "EscalationCallback",
]
