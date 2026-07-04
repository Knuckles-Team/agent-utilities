"""Tests for Company Operations Pydantic Models.

CONCEPT:AU-KG.research.research-pipeline-runner — Company Operations Domain
CONCEPT:AU-KG.memory.tiered-memory-caching — Company Intelligence Graph

Validates all company Pydantic models, their field constraints,
and OWL ontology alignment.
"""

import pytest
from pydantic import ValidationError

from agent_utilities.models.company import (
    KPI,
    AgentDepartment,
    BenefitsPlan,
    CompanyLicense,
    CompanyProfile,
    CompanySoftware,
    CorporateGovernanceDoc,
    DeploymentBlueprint,
    IntellectualPropertyAsset,
    PayrollRecord,
    RegulatoryFiling,
    StrategicGoal,
)
from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNodeType,
)


class TestCompanyProfile:
    """Tests for CompanyProfile model."""

    def test_create_company(self):
        company = CompanyProfile(
            id="acme_corp",
            name="Acme Corporation",
            legal_name="Acme Corporation",
            entity_type="corp",
            state_of_incorporation="DE",
            ein="12-3456789",
        )
        assert company.type == RegistryNodeType.COMPANY
        assert company.legal_name == "Acme Corporation"
        assert company.entity_type == "corp"
        assert company.departments == []
        assert company.goals == []

    def test_company_with_dba(self):
        company = CompanyProfile(
            id="acme_llc",
            name="Acme Holdings LLC",
            legal_name="Acme Holdings LLC",
            dba_name="Acme Tech",
            entity_type="llc",
            state_of_incorporation="WY",
            ein="98-7654321",
        )
        assert company.dba_name == "Acme Tech"

    def test_company_with_departments(self):
        company = CompanyProfile(
            id="test_co",
            name="Test Co",
            legal_name="Test Co",
            entity_type="s_corp",
            state_of_incorporation="CA",
            ein="11-2233445",
            departments=["dept_legal", "dept_hr", "dept_finance"],
            goals=["goal_revenue", "goal_growth"],
            kpis=["kpi_mrr", "kpi_churn"],
        )
        assert len(company.departments) == 3
        assert len(company.goals) == 2
        assert len(company.kpis) == 2


class TestStrategicGoal:
    """Tests for StrategicGoal model."""

    def test_create_goal(self):
        goal = StrategicGoal(
            id="goal_revenue",
            name="Revenue Target",
            goal_type="profit",
            target_value=1000000.0,
            measurement_unit="USD",
        )
        assert goal.type == RegistryNodeType.STRATEGIC_GOAL
        assert goal.actual_value == 0.0
        assert goal.deadline is None

    def test_goal_with_cascade(self):
        goal = StrategicGoal(
            id="goal_dept",
            name="Department Efficiency",
            goal_type="efficiency",
            target_value=0.95,
            measurement_unit="percent",
            cascades_to=["goal_team_a", "goal_team_b"],
        )
        assert len(goal.cascades_to) == 2

    def test_goal_progress(self):
        goal = StrategicGoal(
            id="goal_growth",
            name="Growth Target",
            goal_type="growth",
            target_value=100.0,
            actual_value=75.0,
            measurement_unit="count",
            deadline="2026-12-31",
        )
        assert goal.actual_value == 75.0
        assert goal.deadline == "2026-12-31"


class TestKPI:
    """Tests for KPI model."""

    def test_create_kpi(self):
        kpi = KPI(
            id="kpi_mrr",
            name="Monthly Recurring Revenue",
            kpi_type="revenue",
            current_value=50000.0,
            target_value=100000.0,
            trend="improving",
            measurement_frequency="monthly",
        )
        assert kpi.type == RegistryNodeType.KPI
        assert kpi.trend == "improving"

    def test_kpi_linked_to_goal(self):
        kpi = KPI(
            id="kpi_churn",
            name="Churn Rate",
            kpi_type="efficiency",
            current_value=5.0,
            target_value=3.0,
            trend="stable",
            measurement_frequency="weekly",
            goal_id="goal_retention",
        )
        assert kpi.goal_id == "goal_retention"


class TestAgentDepartment:
    """Tests for AgentDepartment model."""

    def test_create_department(self):
        dept = AgentDepartment(
            id="dept_legal",
            name="Legal Department",
            department_type="legal",
            team_config_id="tc_legal",
            automation_level=0.8,
            human_oversight_required=True,
        )
        assert dept.type == RegistryNodeType.AGENT_DEPARTMENT
        assert dept.automation_level == 0.8
        assert dept.human_oversight_required is True

    def test_department_with_workflows(self):
        dept = AgentDepartment(
            id="dept_hr",
            name="HR Department",
            department_type="hr",
            team_config_id="tc_hr",
            automation_level=0.9,
            workflow_ids=["onboarding", "payroll", "performance_review"],
            software_stack=["erpnext", "orangehrm"],
            prompt_agent="hr_operations_coordinator",
        )
        assert len(dept.workflow_ids) == 3
        assert dept.prompt_agent == "hr_operations_coordinator"

    def test_automation_level_bounds(self):
        with pytest.raises(ValidationError):
            AgentDepartment(
                id="bad",
                name="Bad Dept",
                department_type="finance",
                team_config_id="tc",
                automation_level=1.5,  # > 1.0
            )


class TestBenefitsPlan:
    """Tests for BenefitsPlan model."""

    def test_create_plan(self):
        plan = BenefitsPlan(
            id="plan_medical",
            name="Medical Plan",
            plan_type="medical",
            provider_name="Blue Cross Blue Shield",
            coverage_level="family",
            monthly_cost=850.0,
        )
        assert plan.type == RegistryNodeType.BENEFITS_PLAN
        assert plan.monthly_cost == 850.0


class TestPayrollRecord:
    """Tests for PayrollRecord model."""

    def test_create_record(self):
        record = PayrollRecord(
            id="pr_2026_01",
            name="Payroll 2026-01",
            employee_id="emp_001",
            pay_period_start="2026-01-01",
            pay_period_end="2026-01-15",
            gross_pay=5000.0,
            deductions=800.0,
            net_pay=4200.0,
            tax_withholding=600.0,
        )
        assert record.type == RegistryNodeType.PAYROLL_RECORD
        assert record.gross_pay - record.deductions == record.net_pay


class TestCompanyLicense:
    """Tests for CompanyLicense model."""

    def test_create_license(self):
        lic = CompanyLicense(
            id="lic_business",
            name="Business License",
            license_type="business",
            jurisdiction="State of Delaware",
            expiration_date="2027-12-31",
            renewal_required=True,
        )
        assert lic.type == RegistryNodeType.COMPANY_LICENSE
        assert lic.renewal_required is True


class TestCorporateGovernanceDoc:
    """Tests for CorporateGovernanceDoc model."""

    def test_create_doc(self):
        doc = CorporateGovernanceDoc(
            id="gov_articles",
            name="Articles of Incorporation",
            doc_type="articles_of_incorporation",
            effective_date="2026-01-01",
        )
        assert doc.type == RegistryNodeType.CORPORATE_GOVERNANCE_DOC
        assert doc.last_amended is None


class TestRegulatoryFiling:
    """Tests for RegulatoryFiling model."""

    def test_create_filing(self):
        filing = RegulatoryFiling(
            id="filing_eeo1",
            name="EEO-1 Filing",
            filing_agency="EEOC",
            filing_type="eeo1",
            filing_deadline="2026-03-31",
        )
        assert filing.type == RegistryNodeType.REGULATORY_FILING
        assert filing.filing_status == "pending"


class TestIntellectualPropertyAsset:
    """Tests for IntellectualPropertyAsset model."""

    def test_create_ip(self):
        ip = IntellectualPropertyAsset(
            id="ip_patent_001",
            name="Patent US11234567",
            ip_type="patent",
            registration_number="US11,234,567",
            ip_status="active",
        )
        assert ip.type == RegistryNodeType.INTELLECTUAL_PROPERTY
        assert ip.ip_status == "active"


class TestCompanySoftware:
    """Tests for CompanySoftware model."""

    def test_create_software(self):
        sw = CompanySoftware(
            id="sw_erpnext",
            name="ERPNext Instance",
            software_name="ERPNext",
            version="15.0",
            function_category="accounting",
            api_endpoint="https://erp.example.com/api",
            mcp_server_id="erpnext-mcp",
        )
        assert sw.type == RegistryNodeType.COMPANY_SOFTWARE
        assert sw.function_category == "accounting"


class TestDeploymentBlueprint:
    """Tests for DeploymentBlueprint model."""

    def test_create_blueprint(self):
        bp = DeploymentBlueprint(
            id="bp_erpnext",
            name="ERPNext Blueprint",
            blueprint_name="erpnext",
            blueprint_path="skill_graphs/infrastructure-blueprints/company-software/erpnext.yaml",
            deployment_mode="docker_compose",
            requires_memory_mb=2048,
            requires_cpu=2.0,
        )
        assert bp.type == RegistryNodeType.DEPLOYMENT_BLUEPRINT
        assert bp.requires_gpu is False


class TestRegistryNodeTypes:
    """Verify all company node types are registered."""

    def test_company_node_types_exist(self):
        assert RegistryNodeType.COMPANY == "company"
        assert RegistryNodeType.STRATEGIC_GOAL == "strategic_goal"
        assert RegistryNodeType.KPI == "kpi"
        assert RegistryNodeType.AGENT_DEPARTMENT == "agent_department"
        assert RegistryNodeType.BENEFITS_PLAN == "benefits_plan"
        assert RegistryNodeType.PAYROLL_RECORD == "payroll_record"
        assert RegistryNodeType.COMPANY_LICENSE == "company_license"
        assert RegistryNodeType.COMPANY_SOFTWARE == "company_software"
        assert RegistryNodeType.DEPLOYMENT_BLUEPRINT == "deployment_blueprint"


class TestRegistryEdgeTypes:
    """Verify all company edge types are registered."""

    def test_company_edge_types_exist(self):
        assert RegistryEdgeType.HAS_DEPARTMENT == "has_department"
        assert RegistryEdgeType.HAS_GOAL == "has_goal"
        assert RegistryEdgeType.HAS_KPI == "has_kpi"
        assert RegistryEdgeType.MEASURES_GOAL == "measures_goal"
        assert RegistryEdgeType.GOAL_CASCADES_TO == "goal_cascades_to"
        assert RegistryEdgeType.RUNS_WORKFLOW == "runs_workflow"
        assert RegistryEdgeType.USES_SOFTWARE == "uses_software"
        assert RegistryEdgeType.DEPLOYED_ON == "deployed_on"
        assert RegistryEdgeType.DEPLOYS_SOFTWARE == "deploys_software"
