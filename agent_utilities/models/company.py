#!/usr/bin/python
"""Company Operations Pydantic Models.

CONCEPT:KG-2.6 — Company Operations Domain
CONCEPT:KG-2.1 — Company Intelligence Graph

Provides Pydantic models for autonomous company operations including:
- CompanyProfile: The top-level company entity
- StrategicGoal: Measurable goals with KPI tracking
- KPI: Key Performance Indicators
- AgentDepartment: AI-staffed organizational units
- BenefitsPlan: Employee benefits administration
- PayrollRecord: Payroll tracking
- CompanyLicense: Business license management
- CorporateGovernanceDoc: Governance documents
- RegulatoryFiling: Government regulatory filings
- IntellectualProperty: IP asset tracking
- CompanySoftware: Deployed software systems
- DeploymentBlueprint: Docker Swarm deployment templates
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNode,
    RegistryNodeType,
)


class CompanyProfile(RegistryNode):
    """A company entity with all metadata for autonomous AI-driven operations.

    Maps to OWL class :Company in ontology_company.ttl.
    Inheriting companies can provide domain-specific ontologies via SPARQL federation.
    """

    type: RegistryNodeType = RegistryNodeType.COMPANY
    legal_name: str = Field(description="Full legal name of the company")
    dba_name: str | None = Field(
        default=None, description="Doing Business As (trade name)"
    )
    entity_type: Literal["llc", "corp", "s_corp", "sole_prop"] = Field(
        description="Legal entity type"
    )
    state_of_incorporation: str = Field(
        description="US state of incorporation (e.g., 'DE', 'WY')"
    )
    ein: str = Field(description="Employer Identification Number (IRS-assigned)")
    departments: list[str] = Field(
        default_factory=list, description="AgentDepartment node IDs"
    )
    goals: list[str] = Field(default_factory=list, description="StrategicGoal node IDs")
    kpis: list[str] = Field(default_factory=list, description="KPI node IDs")


class StrategicGoal(RegistryNode):
    """A measurable company goal with KPI tracking.

    Maps to OWL class :StrategicGoal in ontology_company.ttl.
    Cascades from org-level to department-level to individual-level
    via the goalCascadesTo transitive property.
    """

    type: RegistryNodeType = RegistryNodeType.STRATEGIC_GOAL
    goal_type: Literal[
        "profit",
        "growth",
        "efficiency",
        "sustainability",
        "employee_satisfaction",
        "compliance",
        "innovation",
    ] = Field(description="Category of the strategic goal")
    target_value: float = Field(description="Target metric value")
    actual_value: float = Field(default=0.0, description="Current metric value")
    measurement_unit: str = Field(
        description="Unit of measurement (e.g., 'USD', 'percent', 'count')"
    )
    deadline: str | None = Field(
        default=None, description="ISO date deadline (YYYY-MM-DD)"
    )
    cascades_to: list[str] = Field(
        default_factory=list,
        description="Child StrategicGoal IDs (transitive hierarchy)",
    )


class KPI(RegistryNode):
    """A Key Performance Indicator tracked in the Knowledge Graph.

    Maps to OWL class :KeyPerformanceIndicator in ontology_company.ttl.
    Linked to StrategicGoal via MEASURES_GOAL edge.
    """

    type: RegistryNodeType = RegistryNodeType.KPI
    kpi_type: str = Field(
        description="KPI category: profit, revenue, employee_satisfaction, "
        "environmental_impact, throughput, efficiency, quality, compliance"
    )
    current_value: float = Field(description="Current measured value")
    target_value: float = Field(description="Target value")
    trend: Literal["improving", "stable", "declining"] = Field(
        description="Current trend direction"
    )
    measurement_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = Field(
        description="How often this KPI is measured"
    )
    goal_id: str | None = Field(
        default=None, description="StrategicGoal this KPI measures"
    )


class AgentDepartment(RegistryNode):
    """An AI-staffed department with team config and automation level.

    Maps to OWL class :AgentDepartment in ontology_company.ttl.
    CONCEPT:ORCH-1.27 — Autonomous Department Orchestration.
    Links to TeamConfigNode for agent team composition and
    to CompanySoftware for the department's tooling stack.
    """

    type: RegistryNodeType = RegistryNodeType.AGENT_DEPARTMENT
    department_type: Literal[
        "finance",
        "legal",
        "hr",
        "ops",
        "strategy",
        "marketing",
        "sales",
        "r_and_d",
        "it",
        "executive",
    ] = Field(description="Department function")
    team_config_id: str = Field(description="TeamConfigNode ID for agent composition")
    automation_level: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentage of tasks handled by AI (0.0–1.0)",
    )
    human_oversight_required: bool = Field(
        default=True,
        description="Whether a human must approve critical decisions",
    )
    workflow_ids: list[str] = Field(
        default_factory=list, description="Workflow IDs this department runs"
    )
    software_stack: list[str] = Field(
        default_factory=list, description="CompanySoftware node IDs"
    )
    prompt_agent: str | None = Field(
        default=None,
        description="Reference to prompts.json agent name for this department",
    )


class BenefitsPlan(RegistryNode):
    """An employee benefits plan (medical, dental, 401k, etc.).

    Maps to OWL class :BenefitsPlan in ontology_company.ttl.
    """

    type: RegistryNodeType = RegistryNodeType.BENEFITS_PLAN
    plan_type: Literal[
        "medical",
        "dental",
        "vision",
        "401k",
        "pto",
        "disability",
        "life_insurance",
    ] = Field(description="Type of benefit")
    provider_name: str = Field(description="Benefits provider organization name")
    coverage_level: Literal["individual", "family", "employee_plus_one"] = Field(
        description="Coverage tier"
    )
    monthly_cost: float = Field(ge=0.0, description="Monthly premium cost in USD")


class PayrollRecord(RegistryNode):
    """A payroll entry for a pay period.

    Maps to OWL class :PayrollRecord in ontology_company.ttl.
    Linked to Employee via FOR_EMPLOYEE edge.
    """

    type: RegistryNodeType = RegistryNodeType.PAYROLL_RECORD
    employee_id: str = Field(description="Employee node ID")
    pay_period_start: str = Field(description="ISO date (YYYY-MM-DD)")
    pay_period_end: str = Field(description="ISO date (YYYY-MM-DD)")
    gross_pay: float = Field(ge=0.0, description="Gross pay in USD")
    deductions: float = Field(ge=0.0, description="Total deductions in USD")
    net_pay: float = Field(ge=0.0, description="Net pay in USD")
    tax_withholding: float = Field(ge=0.0, description="Total tax withholding in USD")


class CompanyLicense(RegistryNode):
    """A business, professional, or regulatory license.

    Maps to OWL class :CompanyLicense in ontology_company.ttl.
    """

    type: RegistryNodeType = RegistryNodeType.COMPANY_LICENSE
    license_type: Literal["business", "professional", "regulatory"] = Field(
        description="Category of license"
    )
    jurisdiction: str = Field(description="Jurisdiction name (state, county, city)")
    expiration_date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD)"
    )
    renewal_required: bool = Field(
        default=True, description="Whether this license requires periodic renewal"
    )


class CorporateGovernanceDoc(RegistryNode):
    """A foundational corporate governance document.

    Maps to OWL class :CorporateGovernanceDoc in ontology_legal.ttl.
    """

    type: RegistryNodeType = RegistryNodeType.CORPORATE_GOVERNANCE_DOC
    doc_type: Literal[
        "articles_of_incorporation",
        "bylaws",
        "operating_agreement",
        "shareholder_agreement",
        "board_resolution",
    ] = Field(description="Type of governance document")
    effective_date: str = Field(description="ISO date (YYYY-MM-DD)")
    last_amended: str | None = Field(
        default=None, description="Last amendment date (YYYY-MM-DD)"
    )


class RegulatoryFiling(RegistryNode):
    """A required filing with a government regulatory body.

    Maps to OWL class :RegulatoryFiling in ontology_legal.ttl.
    """

    type: RegistryNodeType = RegistryNodeType.REGULATORY_FILING
    filing_agency: Literal["SEC", "IRS", "State_AG", "FTC", "DOL", "EEOC"] = Field(
        description="Government agency"
    )
    filing_type: str = Field(
        description="Filing type: annual_report, tax_return, eeo1, osha_log, etc."
    )
    filing_deadline: str = Field(description="ISO date deadline (YYYY-MM-DD)")
    filing_status: Literal[
        "pending", "filed", "overdue", "accepted", "rejected"
    ] = Field(default="pending", description="Current status")


class IntellectualPropertyAsset(RegistryNode):
    """An intellectual property asset owned by the company.

    Maps to OWL class :IntellectualProperty in ontology_legal.ttl.
    """

    type: RegistryNodeType = RegistryNodeType.INTELLECTUAL_PROPERTY
    ip_type: Literal["patent", "trademark", "copyright", "trade_secret"] = Field(
        description="Type of IP"
    )
    registration_number: str | None = Field(
        default=None, description="Government registration number"
    )
    filing_date: str | None = Field(default=None, description="ISO date (YYYY-MM-DD)")
    expiration_date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD)"
    )
    ip_status: Literal[
        "pending", "granted", "active", "expired", "abandoned", "infringed"
    ] = Field(default="pending", description="Current status")


class CompanySoftware(RegistryNode):
    """A software system deployed for company operations.

    Maps to OWL class :CompanySoftware in ontology_company_infra.ttl.
    CONCEPT:ECO-4.3 — Company Infrastructure Orchestration.
    """

    type: RegistryNodeType = RegistryNodeType.COMPANY_SOFTWARE
    software_name: str = Field(
        description="Display name: ERPNext, Akaunting, Twenty CRM, etc."
    )
    version: str | None = Field(default=None, description="Software version")
    function_category: Literal[
        "accounting",
        "hr",
        "legal",
        "crm",
        "project_mgmt",
        "monitoring",
        "ci_cd",
        "knowledge_base",
        "communication",
    ] = Field(description="Company function this software serves")
    api_endpoint: str | None = Field(default=None, description="API endpoint URL")
    mcp_server_id: str | None = Field(
        default=None,
        description="Agent-packages MCP server identifier, if available",
    )
    dns_rewrite: str | None = Field(
        default=None, description="DNS hostname configured via Technitium DNS"
    )
    host_id: str | None = Field(default=None, description="InfrastructureHost node ID")


class DeploymentBlueprint(RegistryNode):
    """A reusable Docker Swarm / Compose blueprint for deploying company software.

    Maps to OWL class :DeploymentBlueprint in ontology_company_infra.ttl.
    CONCEPT:ECO-4.04 — Infrastructure Blueprint Library.
    """

    type: RegistryNodeType = RegistryNodeType.DEPLOYMENT_BLUEPRINT
    blueprint_name: str = Field(
        description="Blueprint identifier (e.g., 'erpnext', 'twenty-crm')"
    )
    blueprint_path: str = Field(
        description="Path to YAML definition in infrastructure-blueprints skill-graph"
    )
    deployment_mode: Literal[
        "docker_compose", "docker_swarm", "kubernetes", "native"
    ] = Field(default="docker_compose", description="Deployment orchestration mode")
    requires_memory_mb: int = Field(default=512, description="Minimum RAM in MB")
    requires_cpu: float = Field(default=1.0, description="Minimum CPU cores")
    requires_gpu: bool = Field(default=False, description="Whether GPU is required")
    deploys_software_id: str | None = Field(
        default=None, description="CompanySoftware node ID this blueprint creates"
    )


# ===================================================================
# Edge Type Constants for Company Operations
# ===================================================================

COMPANY_EDGE_TYPES = {
    "HAS_DEPARTMENT": RegistryEdgeType.HAS_DEPARTMENT,
    "HAS_GOAL": RegistryEdgeType.HAS_GOAL,
    "HAS_KPI": RegistryEdgeType.HAS_KPI,
    "MEASURES_GOAL": RegistryEdgeType.MEASURES_GOAL,
    "GOAL_CASCADES_TO": RegistryEdgeType.GOAL_CASCADES_TO,
    "RUNS_WORKFLOW": RegistryEdgeType.RUNS_WORKFLOW,
    "USES_SOFTWARE": RegistryEdgeType.USES_SOFTWARE,
    "ENROLLED_IN": RegistryEdgeType.ENROLLED_IN,
    "HOLDS_LICENSE": RegistryEdgeType.HOLDS_LICENSE,
    "GOVERNED_BY_DOC": RegistryEdgeType.GOVERNED_BY_DOC,
    "OWNS_IP": RegistryEdgeType.OWNS_IP,
    "FILED_BY": RegistryEdgeType.FILED_BY,
    "SERVES_FUNCTION": RegistryEdgeType.SERVES_FUNCTION,
    "DEPLOYED_ON": RegistryEdgeType.DEPLOYED_ON,
    "MANAGED_BY_AGENT": RegistryEdgeType.MANAGED_BY_AGENT,
    "DEPLOYS_SOFTWARE": RegistryEdgeType.DEPLOYS_SOFTWARE,
}
