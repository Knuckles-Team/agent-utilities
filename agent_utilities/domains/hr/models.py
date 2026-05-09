"""HR/Workforce Management Pydantic Models (CONCEPT:KG-2.80).

Pydantic models for the Human Capital & Workforce Management domain.
Aligned to BFO upper ontology, HR-XML, W3C ORG, and SFIA frameworks.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# HR Enums
# ---------------------------------------------------------------------------


class EmploymentStatus(StrEnum):
    """Employment status of an employee."""

    ACTIVE = "active"
    ON_LEAVE = "on_leave"
    TERMINATED = "terminated"
    CONTRACTOR = "contractor"
    PROBATION = "probation"


class ProficiencyLevel(StrEnum):
    """SFIA-aligned competency proficiency levels."""

    FOLLOW = "1_follow"
    ASSIST = "2_assist"
    APPLY = "3_apply"
    ENABLE = "4_enable"
    ADVISE = "5_advise"


class ReviewCycle(StrEnum):
    """Performance review cycle types."""

    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"
    AD_HOC = "ad_hoc"


class PipelineStage(StrEnum):
    """Hiring pipeline stages."""

    REQUISITION = "requisition"
    SOURCING = "sourcing"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    ONBOARDING = "onboarding"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# HR Node Types (extend RegistryNodeType)
# ---------------------------------------------------------------------------


class HRNodeType(StrEnum):
    """HR-specific node types for the KG."""

    EMPLOYEE = "employee"
    DEPARTMENT = "department"
    POSITION_ROLE = "position_role"
    COMPETENCY = "competency"
    CREDENTIAL = "credential"
    PERFORMANCE_REVIEW = "performance_review"
    COMPENSATION_BAND = "compensation_band"
    OKR = "okr"
    HIRING_PIPELINE = "hiring_pipeline"


class HREdgeType(StrEnum):
    """HR-specific edge types."""

    REPORTS_TO = "reports_to"
    HAS_COMPETENCY = "has_competency"
    HOLDS_CREDENTIAL = "holds_credential"
    ASSIGNED_TO_DEPARTMENT = "assigned_to_department"
    REVIEWED_IN = "reviewed_in"
    COMPENSATED_AT = "compensated_at"
    CERTIFIED_FOR = "certified_for"
    CASCADES_TO = "cascades_to"
    DEPARTMENT_PART_OF = "department_part_of"
    MANAGES = "manages"


# ---------------------------------------------------------------------------
# HR Pydantic Models
# ---------------------------------------------------------------------------


class EmployeeNode(BaseModel):
    """An employee in the enterprise.

    Extends the Person concept with employment metadata.
    BFO: IndependentContinuant. Aligned to foaf:Person + HR-XML.
    """

    id: str = Field(description="Unique employee identifier")
    name: str = Field(description="Full legal name")
    employee_id: str = Field(default="", description="HR system employee ID")
    email: str = Field(default="", description="Corporate email")
    department_id: str = Field(default="", description="Assigned department")
    manager_id: str = Field(default="", description="Direct manager employee ID")
    title: str = Field(default="", description="Job title")
    status: EmploymentStatus = EmploymentStatus.ACTIVE
    hire_date: str = Field(default="", description="ISO date of hire")
    location: str = Field(default="", description="Primary work location")
    competencies: list[str] = Field(default_factory=list, description="Competency IDs")
    credentials: list[str] = Field(default_factory=list, description="Credential IDs")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DepartmentNode(BaseModel):
    """An organizational unit within the enterprise.

    BFO: IndependentContinuant. Aligned to W3C ORG org:OrganizationalUnit.
    """

    id: str = Field(description="Unique department identifier")
    name: str = Field(description="Department name")
    parent_department_id: str = Field(
        default="", description="Parent department for hierarchy"
    )
    head_employee_id: str = Field(default="", description="Department head employee ID")
    cost_center: str = Field(default="", description="Financial cost center code")
    headcount: int = Field(default=0, description="Current headcount")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompetencyNode(BaseModel):
    """A skill or knowledge area with proficiency levels.

    BFO: GenericallyDependentContinuant. Aligned to SFIA Framework.
    """

    id: str = Field(description="Unique competency identifier")
    name: str = Field(description="Competency name")
    category: str = Field(
        default="", description="Category: technical, leadership, domain"
    )
    proficiency_level: ProficiencyLevel = ProficiencyLevel.APPLY
    description: str = Field(default="", description="What this competency covers")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CredentialNode(BaseModel):
    """A certification, license, or qualification.

    BFO: GenericallyDependentContinuant. Aligned to Open Badges.
    """

    id: str = Field(description="Unique credential identifier")
    name: str = Field(description="Credential name (e.g., AWS Solutions Architect)")
    issuer: str = Field(default="", description="Issuing organization")
    issue_date: str = Field(default="", description="ISO date of issue")
    expiry_date: str = Field(default="", description="ISO expiry date")
    is_active: bool = Field(default=True, description="Whether credential is current")
    authorized_activities: list[str] = Field(
        default_factory=list, description="Procedure IDs this credential authorizes"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerformanceReviewNode(BaseModel):
    """A periodic performance evaluation event.

    BFO: Process. Links Employee → Reviewer → Ratings.
    """

    id: str = Field(description="Unique review identifier")
    employee_id: str = Field(description="Employee being reviewed")
    reviewer_id: str = Field(description="Reviewing manager/peer")
    cycle: ReviewCycle = ReviewCycle.ANNUAL
    period: str = Field(default="", description="Review period (e.g., 2026-H1)")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Performance rating")
    strengths: list[str] = Field(default_factory=list)
    growth_areas: list[str] = Field(default_factory=list)
    comments: str = Field(default="")
    completed_at: str = Field(default="", description="ISO timestamp of completion")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompensationBandNode(BaseModel):
    """A salary range tied to position levels and geographies.

    BFO: GenericallyDependentContinuant.
    """

    id: str = Field(description="Unique band identifier")
    name: str = Field(description="Band name (e.g., L5-Engineering)")
    level: str = Field(default="", description="Position level")
    min_salary: float = Field(default=0.0)
    max_salary: float = Field(default=0.0)
    mid_salary: float = Field(default=0.0)
    currency: str = Field(default="USD")
    geography: str = Field(default="", description="Geographic scope")
    metadata: dict[str, Any] = Field(default_factory=dict)


class OKRNode(BaseModel):
    """An Objective & Key Result.

    BFO: SpecificallyDependentContinuant.
    Cascades from org → department → individual.
    """

    id: str = Field(description="Unique OKR identifier")
    objective: str = Field(description="The objective statement")
    key_results: list[str] = Field(
        default_factory=list, description="Measurable key results"
    )
    owner_id: str = Field(
        default="", description="Employee or department owning this OKR"
    )
    parent_okr_id: str = Field(default="", description="Parent OKR for cascading")
    progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Completion 0.0–1.0"
    )
    period: str = Field(default="", description="OKR period (e.g., 2026-Q2)")
    status: str = Field(
        default="active", description="active, at_risk, completed, cancelled"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class HiringPipelineNode(BaseModel):
    """A hiring pipeline lifecycle.

    BFO: Process. Requisition → sourcing → screening → offer → onboarding.
    """

    id: str = Field(description="Unique pipeline identifier")
    position_title: str = Field(description="Job title being hired for")
    department_id: str = Field(default="", description="Hiring department")
    hiring_manager_id: str = Field(default="", description="Hiring manager employee ID")
    stage: PipelineStage = PipelineStage.REQUISITION
    candidates_count: int = Field(default=0)
    opened_at: str = Field(default="", description="ISO date pipeline opened")
    closed_at: str = Field(default="", description="ISO date pipeline closed")
    metadata: dict[str, Any] = Field(default_factory=dict)
