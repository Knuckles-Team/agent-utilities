"""Tests for HR/Workforce Management Domain (CONCEPT:KG-2.80)."""

from agent_utilities.domains.hr.models import (
    CompensationBandNode,
    CompetencyNode,
    CredentialNode,
    DepartmentNode,
    EmployeeNode,
    EmploymentStatus,
    HiringPipelineNode,
    OKRNode,
    PerformanceReviewNode,
    PipelineStage,
    ProficiencyLevel,
)
from agent_utilities.domains.hr.workforce_manager import WorkforceManager


class TestHRModels:
    """Test HR Pydantic model construction."""

    def test_employee_node(self):
        emp = EmployeeNode(
            id="emp:001",
            name="Jane Doe",
            employee_id="EMP001",
            department_id="dept:eng",
            manager_id="emp:000",
            title="Senior Engineer",
            status=EmploymentStatus.ACTIVE,
        )
        assert emp.id == "emp:001"
        assert emp.status == EmploymentStatus.ACTIVE

    def test_department_node(self):
        dept = DepartmentNode(
            id="dept:eng",
            name="Engineering",
            parent_department_id="dept:tech",
            cost_center="CC-100",
        )
        assert dept.name == "Engineering"
        assert dept.parent_department_id == "dept:tech"

    def test_competency_node(self):
        comp = CompetencyNode(
            id="comp:python",
            name="Python",
            category="technical",
            proficiency_level=ProficiencyLevel.ENABLE,
        )
        assert comp.proficiency_level == ProficiencyLevel.ENABLE

    def test_credential_node(self):
        cred = CredentialNode(
            id="cred:aws",
            name="AWS Solutions Architect",
            issuer="Amazon",
            is_active=True,
        )
        assert cred.is_active is True

    def test_performance_review(self):
        review = PerformanceReviewNode(
            id="rev:001",
            employee_id="emp:001",
            reviewer_id="emp:000",
            rating=4.5,
        )
        assert review.rating == 4.5

    def test_compensation_band(self):
        band = CompensationBandNode(
            id="band:l5",
            name="L5-Engineering",
            min_salary=120000,
            max_salary=180000,
            mid_salary=150000,
        )
        assert band.mid_salary == 150000

    def test_okr_node(self):
        okr = OKRNode(
            id="okr:001",
            objective="Ship v2.0",
            key_results=["95% test coverage", "Zero P0 bugs"],
            progress=0.7,
        )
        assert okr.progress == 0.7

    def test_hiring_pipeline(self):
        pipe = HiringPipelineNode(
            id="pipe:001",
            position_title="Staff Engineer",
            stage=PipelineStage.INTERVIEW,
            candidates_count=5,
        )
        assert pipe.stage == PipelineStage.INTERVIEW


class TestWorkforceManager:
    """Test WorkforceManager service layer."""

    def _build_manager(self) -> WorkforceManager:
        wm = WorkforceManager()
        wm.add_department(DepartmentNode(id="dept:eng", name="Engineering"))
        wm.add_department(
            DepartmentNode(
                id="dept:fe", name="Frontend", parent_department_id="dept:eng"
            )
        )
        wm.add_employee(
            EmployeeNode(
                id="emp:ceo",
                name="CEO",
                department_id="dept:eng",
            )
        )
        wm.add_employee(
            EmployeeNode(
                id="emp:vp",
                name="VP Eng",
                department_id="dept:eng",
                manager_id="emp:ceo",
            )
        )
        wm.add_employee(
            EmployeeNode(
                id="emp:lead",
                name="Tech Lead",
                department_id="dept:fe",
                manager_id="emp:vp",
                competencies=["comp:python", "comp:react"],
            )
        )
        wm.add_employee(
            EmployeeNode(
                id="emp:dev",
                name="Developer",
                department_id="dept:fe",
                manager_id="emp:lead",
                competencies=["comp:python"],
            )
        )
        wm.add_competency(CompetencyNode(id="comp:python", name="Python"))
        wm.add_competency(CompetencyNode(id="comp:react", name="React"))
        return wm

    def test_direct_reports(self):
        wm = self._build_manager()
        reports = wm.get_direct_reports("emp:vp")
        assert len(reports) == 1
        assert reports[0].id == "emp:lead"

    def test_report_chain(self):
        wm = self._build_manager()
        chain = wm.get_report_chain("emp:ceo")
        ids = {e.id for e in chain}
        assert "emp:vp" in ids
        assert "emp:lead" in ids
        assert "emp:dev" in ids

    def test_management_chain(self):
        wm = self._build_manager()
        chain = wm.get_management_chain("emp:dev")
        names = [e.name for e in chain]
        assert "Tech Lead" in names
        assert "VP Eng" in names
        assert "CEO" in names

    def test_department_headcount(self):
        wm = self._build_manager()
        count = wm.get_department_headcount("dept:fe")
        assert count == 2

    def test_department_hierarchy(self):
        wm = self._build_manager()
        subs = wm.get_department_hierarchy("dept:eng")
        assert len(subs) == 1
        assert subs[0].id == "dept:fe"

    def test_competency_matrix(self):
        wm = self._build_manager()
        matrix = wm.get_competency_matrix("dept:fe")
        assert "emp:lead" in matrix
        assert len(matrix["emp:lead"]) == 2

    def test_find_employees_with_competency(self):
        wm = self._build_manager()
        result = wm.find_employees_with_competency("comp:python")
        assert len(result) == 2

    def test_succession_candidates(self):
        wm = self._build_manager()
        candidates = wm.find_succession_candidates(
            ["comp:python", "comp:react"], department_id="dept:fe"
        )
        assert len(candidates) >= 1
        assert candidates[0][0].id == "emp:lead"  # Best match
        assert candidates[0][1] == 1.0

    def test_okr_cascading(self):
        wm = self._build_manager()
        wm.add_okr(OKRNode(id="okr:org", objective="Ship v2.0"))
        wm.add_okr(
            OKRNode(
                id="okr:dept",
                objective="Frontend rewrite",
                parent_okr_id="okr:org",
            )
        )
        wm.add_okr(
            OKRNode(
                id="okr:ind",
                objective="Component library",
                parent_okr_id="okr:dept",
            )
        )
        cascaded = wm.get_cascaded_okrs("okr:org")
        assert len(cascaded) == 2

    def test_hiring_pipeline_advance(self):
        wm = self._build_manager()
        wm.add_pipeline(
            HiringPipelineNode(
                id="pipe:001",
                position_title="Staff Engineer",
            )
        )
        result = wm.advance_pipeline("pipe:001", PipelineStage.OFFER)
        assert result is not None
        assert result.stage == PipelineStage.OFFER

    def test_workforce_summary(self):
        wm = self._build_manager()
        summary = wm.get_workforce_summary()
        assert summary["total_employees"] == 4
        assert summary["active_employees"] == 4
        assert summary["departments"] == 2
