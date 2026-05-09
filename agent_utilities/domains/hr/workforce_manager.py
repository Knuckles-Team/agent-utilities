"""HR/Workforce Management Service Layer (CONCEPT:KG-2.80).

Provides workforce management capabilities:
- Org chart traversal and hierarchy queries
- Competency matrix generation
- Succession planning
- Hiring pipeline lifecycle management
- OKR cascading and tracking
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.domains.hr.models import (
    CompetencyNode,
    CredentialNode,
    DepartmentNode,
    EmployeeNode,
    EmploymentStatus,
    HiringPipelineNode,
    OKRNode,
    PerformanceReviewNode,
    PipelineStage,
)

logger = logging.getLogger(__name__)


class WorkforceManager:
    """Enterprise workforce management engine.

    Manages the organizational graph: employees, departments,
    competencies, credentials, and performance reviews.
    Integrates with the Knowledge Graph via node/edge operations.
    """

    def __init__(self) -> None:
        self._employees: dict[str, EmployeeNode] = {}
        self._departments: dict[str, DepartmentNode] = {}
        self._competencies: dict[str, CompetencyNode] = {}
        self._credentials: dict[str, CredentialNode] = {}
        self._reviews: dict[str, PerformanceReviewNode] = {}
        self._okrs: dict[str, OKRNode] = {}
        self._pipelines: dict[str, HiringPipelineNode] = {}

    # --- Employee Management ---

    def add_employee(self, employee: EmployeeNode) -> None:
        """Register an employee in the workforce graph."""
        self._employees[employee.id] = employee
        logger.info("Added employee %s (%s)", employee.id, employee.name)

    def get_employee(self, employee_id: str) -> EmployeeNode | None:
        """Retrieve an employee by ID."""
        return self._employees.get(employee_id)

    def get_direct_reports(self, manager_id: str) -> list[EmployeeNode]:
        """Get all employees who directly report to a manager."""
        return [
            e
            for e in self._employees.values()
            if e.manager_id == manager_id and e.status == EmploymentStatus.ACTIVE
        ]

    def get_report_chain(self, employee_id: str) -> list[EmployeeNode]:
        """Get the full transitive reporting chain (all subordinates).

        Uses BFS to traverse the reportsTo hierarchy.
        """
        chain: list[EmployeeNode] = []
        queue = [employee_id]
        visited: set[str] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for report in self.get_direct_reports(current):
                chain.append(report)
                queue.append(report.id)

        return chain

    def get_management_chain(self, employee_id: str) -> list[EmployeeNode]:
        """Get the upward management chain from employee to CEO."""
        chain: list[EmployeeNode] = []
        current = self._employees.get(employee_id)

        while current and current.manager_id:
            manager = self._employees.get(current.manager_id)
            if manager and manager.id not in {e.id for e in chain}:
                chain.append(manager)
                current = manager
            else:
                break

        return chain

    # --- Department Management ---

    def add_department(self, department: DepartmentNode) -> None:
        """Register a department."""
        self._departments[department.id] = department

    def get_department_headcount(self, department_id: str) -> int:
        """Calculate actual headcount for a department."""
        return sum(
            1
            for e in self._employees.values()
            if e.department_id == department_id and e.status == EmploymentStatus.ACTIVE
        )

    def get_department_hierarchy(self, department_id: str) -> list[DepartmentNode]:
        """Get all sub-departments transitively."""
        hierarchy: list[DepartmentNode] = []
        queue = [department_id]
        visited: set[str] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for dept in self._departments.values():
                if dept.parent_department_id == current:
                    hierarchy.append(dept)
                    queue.append(dept.id)

        return hierarchy

    # --- Competency Matrix ---

    def add_competency(self, competency: CompetencyNode) -> None:
        """Register a competency."""
        self._competencies[competency.id] = competency

    def add_credential(self, credential: CredentialNode) -> None:
        """Register a credential."""
        self._credentials[credential.id] = credential

    def get_competency_matrix(
        self, department_id: str | None = None
    ) -> dict[str, list[CompetencyNode]]:
        """Generate a competency matrix mapping employees to their skills.

        Args:
            department_id: Optional filter by department.

        Returns:
            Dict of employee_id → list of CompetencyNode.
        """
        matrix: dict[str, list[CompetencyNode]] = {}

        for emp in self._employees.values():
            if department_id and emp.department_id != department_id:
                continue
            if emp.status != EmploymentStatus.ACTIVE:
                continue
            matrix[emp.id] = [
                self._competencies[cid]
                for cid in emp.competencies
                if cid in self._competencies
            ]

        return matrix

    def find_employees_with_competency(self, competency_id: str) -> list[EmployeeNode]:
        """Find all active employees with a specific competency."""
        return [
            emp
            for emp in self._employees.values()
            if competency_id in emp.competencies
            and emp.status == EmploymentStatus.ACTIVE
        ]

    def find_employees_with_credential(self, credential_id: str) -> list[EmployeeNode]:
        """Find employees holding a specific credential."""
        return [
            emp
            for emp in self._employees.values()
            if credential_id in emp.credentials
            and emp.status == EmploymentStatus.ACTIVE
        ]

    # --- Succession Planning ---

    def find_succession_candidates(
        self, position_competencies: list[str], department_id: str | None = None
    ) -> list[tuple[EmployeeNode, float]]:
        """Find internal candidates for a position based on competency match.

        Args:
            position_competencies: Required competency IDs.
            department_id: Optional department filter.

        Returns:
            List of (employee, match_score) sorted by score descending.
        """
        candidates: list[tuple[EmployeeNode, float]] = []

        for emp in self._employees.values():
            if emp.status != EmploymentStatus.ACTIVE:
                continue
            if department_id and emp.department_id != department_id:
                continue

            if not position_competencies:
                continue

            match_count = sum(1 for c in position_competencies if c in emp.competencies)
            score = match_count / len(position_competencies)
            if score > 0:
                candidates.append((emp, score))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    # --- Performance Reviews ---

    def add_review(self, review: PerformanceReviewNode) -> None:
        """Record a performance review."""
        self._reviews[review.id] = review

    def get_reviews_for_employee(self, employee_id: str) -> list[PerformanceReviewNode]:
        """Get all reviews for an employee."""
        return [r for r in self._reviews.values() if r.employee_id == employee_id]

    # --- OKR Management ---

    def add_okr(self, okr: OKRNode) -> None:
        """Register an OKR."""
        self._okrs[okr.id] = okr

    def get_cascaded_okrs(self, parent_okr_id: str) -> list[OKRNode]:
        """Get all OKRs cascading from a parent (transitive)."""
        result: list[OKRNode] = []
        queue = [parent_okr_id]
        visited: set[str] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for okr in self._okrs.values():
                if okr.parent_okr_id == current:
                    result.append(okr)
                    queue.append(okr.id)

        return result

    # --- Hiring Pipeline ---

    def add_pipeline(self, pipeline: HiringPipelineNode) -> None:
        """Register a hiring pipeline."""
        self._pipelines[pipeline.id] = pipeline

    def advance_pipeline(
        self, pipeline_id: str, to_stage: PipelineStage
    ) -> HiringPipelineNode | None:
        """Advance a pipeline to the next stage."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline:
            pipeline.stage = to_stage
            logger.info("Pipeline %s advanced to %s", pipeline_id, to_stage.value)
        return pipeline

    # --- Summary Statistics ---

    def get_workforce_summary(self) -> dict[str, Any]:
        """Get high-level workforce statistics."""
        active = [
            e for e in self._employees.values() if e.status == EmploymentStatus.ACTIVE
        ]
        return {
            "total_employees": len(self._employees),
            "active_employees": len(active),
            "departments": len(self._departments),
            "competencies_tracked": len(self._competencies),
            "credentials_tracked": len(self._credentials),
            "open_pipelines": sum(
                1 for p in self._pipelines.values() if p.stage != PipelineStage.CLOSED
            ),
            "active_okrs": sum(1 for o in self._okrs.values() if o.status == "active"),
        }
