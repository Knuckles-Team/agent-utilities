"""Tests for the AHE (Agentic Harness Engineering) harness package.

Tests cover:
    - Manifest models (ComponentType, ComponentEdit, ChangeManifest, VerificationResult)
    - Evidence corpus models (EvidenceLayer, EvidenceEntry, FailureCluster, EvidenceCorpus)
    - Component registry (register, history, rollback)
    - Constraint engine (levels, escalation, tool call checking)
    - Trace distiller (classification, clustering, overview generation)
    - Manifest verifier (prediction comparison, auto-revert)
"""

import json
import os
import tempfile

import pytest

from agent_utilities.harness.component_registry import HarnessComponentRegistry
from agent_utilities.harness.constraint_engine import (
    ConstraintEngine,
    ConstraintLevel,
    HierarchicalConstraint,
)
from agent_utilities.harness.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    FailureCluster,
)
from agent_utilities.harness.manifest import (
    ChangeManifest,
    ComponentEdit,
    ComponentType,
    VerificationResult,
)
from agent_utilities.harness.trace_backend import (
    FileTraceBackend,
    create_trace_backend,
)
from agent_utilities.harness.trace_distiller import DistillationConfig, TraceDistiller
from agent_utilities.harness.verifier import ManifestVerifier

# ===========================================================================
# Phase 1: Manifest Models
# ===========================================================================


class TestComponentType:
    """Test ComponentType enum."""

    def test_all_seven_types_exist(self):
        """AHE defines exactly 7 component types."""
        assert len(ComponentType) == 7
        assert ComponentType.SYSTEM_PROMPT == "system_prompt"
        assert ComponentType.TOOL_DESCRIPTION == "tool_description"
        assert ComponentType.TOOL_IMPLEMENTATION == "tool_implementation"
        assert ComponentType.MIDDLEWARE == "middleware"
        assert ComponentType.SKILL == "skill"
        assert ComponentType.SUB_AGENT == "sub_agent"
        assert ComponentType.LONG_TERM_MEMORY == "long_term_memory"


class TestComponentEdit:
    """Test ComponentEdit model."""

    def test_creation_with_defaults(self):
        edit = ComponentEdit(
            component_type=ComponentType.SYSTEM_PROMPT,
            file_path="prompts/main_agent.json",
            edit_summary="Updated system prompt for better routing",
        )
        assert edit.component_type == ComponentType.SYSTEM_PROMPT
        assert edit.file_path == "prompts/main_agent.json"
        assert edit.id.startswith("edit:")
        assert edit.predicted_fixes == []
        assert edit.predicted_regressions == []
        assert edit.git_commit_sha is None

    def test_creation_with_predictions(self):
        edit = ComponentEdit(
            component_type=ComponentType.TOOL_IMPLEMENTATION,
            file_path="tools/dev_tools.py",
            edit_summary="Fixed file reading tool",
            predicted_fixes=["task_042", "task_089"],
            predicted_regressions=["task_017"],
            evidence_references=["cluster:abc123"],
        )
        assert len(edit.predicted_fixes) == 2
        assert "task_042" in edit.predicted_fixes
        assert len(edit.predicted_regressions) == 1

    def test_json_serialization(self):
        edit = ComponentEdit(
            component_type=ComponentType.MIDDLEWARE,
            file_path="guardrails.py",
            edit_summary="Added rate limiting",
        )
        data = json.loads(edit.model_dump_json())
        assert data["component_type"] == "middleware"
        assert "timestamp" in data


class TestChangeManifest:
    """Test ChangeManifest model."""

    def test_creation(self):
        manifest = ChangeManifest()
        assert manifest.round_id.startswith("round:")
        assert manifest.edits == []
        assert manifest.verification_status == "pending"

    def test_add_edit(self):
        manifest = ChangeManifest()
        edit = ComponentEdit(
            component_type=ComponentType.SYSTEM_PROMPT,
            file_path="prompts/main.json",
            edit_summary="Test edit",
            predicted_fixes=["task_001"],
        )
        manifest.add_edit(edit)
        assert len(manifest.edits) == 1

    def test_get_all_predicted_fixes(self):
        manifest = ChangeManifest()
        manifest.add_edit(
            ComponentEdit(
                component_type=ComponentType.SYSTEM_PROMPT,
                file_path="a.json",
                edit_summary="Fix A",
                predicted_fixes=["task_001", "task_002"],
            )
        )
        manifest.add_edit(
            ComponentEdit(
                component_type=ComponentType.TOOL_IMPLEMENTATION,
                file_path="b.py",
                edit_summary="Fix B",
                predicted_fixes=["task_002", "task_003"],
            )
        )
        fixes = manifest.get_all_predicted_fixes()
        assert set(fixes) == {"task_001", "task_002", "task_003"}

    def test_get_edits_by_type(self):
        manifest = ChangeManifest()
        manifest.add_edit(
            ComponentEdit(
                component_type=ComponentType.SYSTEM_PROMPT,
                file_path="a.json",
                edit_summary="A",
            )
        )
        manifest.add_edit(
            ComponentEdit(
                component_type=ComponentType.MIDDLEWARE,
                file_path="b.py",
                edit_summary="B",
            )
        )
        prompt_edits = manifest.get_edits_by_type(ComponentType.SYSTEM_PROMPT)
        assert len(prompt_edits) == 1

    def test_sdd_path(self):
        manifest = ChangeManifest(round_id="round:test123")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = manifest.to_sdd_path(tmpdir)
            assert "manifests" in path
            assert "round:test123.json" in path


class TestVerificationResult:
    """Test VerificationResult model."""

    def test_defaults(self):
        result = VerificationResult()
        assert result.fix_precision == 0.0
        assert result.unexpected_regressions == []
        assert result.recommendation == ""

    def test_with_data(self):
        result = VerificationResult(
            fix_precision=0.8,
            fix_recall=0.6,
            confirmed_fixes=["task_001"],
            unexpected_regressions=["task_099"],
            recommendation="partial_revert",
        )
        assert result.fix_precision == 0.8
        assert "task_099" in result.unexpected_regressions


# ===========================================================================
# Phase 1: Evidence Corpus
# ===========================================================================


class TestEvidenceCorpus:
    """Test EvidenceCorpus and related models."""

    def test_evidence_entry_creation(self):
        entry = EvidenceEntry(
            task_id="task_001",
            pass_fail=False,
            root_cause="Tool returned empty result",
            component_attribution=ComponentType.TOOL_IMPLEMENTATION,
            score=0.2,
        )
        assert entry.task_id == "task_001"
        assert not entry.pass_fail
        assert entry.component_attribution == ComponentType.TOOL_IMPLEMENTATION

    def test_corpus_filtering(self):
        corpus = EvidenceCorpus(
            entries=[
                EvidenceEntry(task_id="t1", pass_fail=True, score=0.9),
                EvidenceEntry(task_id="t2", pass_fail=False, score=0.1),
                EvidenceEntry(task_id="t3", pass_fail=True, score=0.8),
                EvidenceEntry(task_id="t4", pass_fail=False, score=0.3),
            ]
        )
        assert len(corpus.get_failures()) == 2
        assert len(corpus.get_successes()) == 2

    def test_component_filtering(self):
        corpus = EvidenceCorpus(
            entries=[
                EvidenceEntry(
                    task_id="t1",
                    pass_fail=False,
                    component_attribution=ComponentType.SYSTEM_PROMPT,
                ),
                EvidenceEntry(
                    task_id="t2",
                    pass_fail=False,
                    component_attribution=ComponentType.MIDDLEWARE,
                ),
            ]
        )
        prompt_entries = corpus.get_entries_by_component(ComponentType.SYSTEM_PROMPT)
        assert len(prompt_entries) == 1

    def test_overview_generation(self):
        corpus = EvidenceCorpus(
            total_tasks=100,
            pass_rate=0.75,
            benchmark_score=0.82,
            failure_clusters=[
                FailureCluster(
                    label="Tool timeout",
                    root_cause_summary="Tools time out on large inputs",
                    task_ids=["t1", "t2", "t3"],
                    frequency=3,
                    severity=0.8,
                ),
            ],
        )
        overview = corpus.generate_overview_text()
        assert "75.0%" in overview
        assert "Tool timeout" in overview

    def test_top_failure_clusters(self):
        corpus = EvidenceCorpus(
            failure_clusters=[
                FailureCluster(label="A", root_cause_summary="A", severity=0.3, frequency=1),
                FailureCluster(label="B", root_cause_summary="B", severity=0.9, frequency=5),
                FailureCluster(label="C", root_cause_summary="C", severity=0.5, frequency=3),
            ]
        )
        top = corpus.get_top_failure_clusters(n=2)
        assert len(top) == 2
        assert top[0].label == "B"  # Highest severity


# ===========================================================================
# Phase 1: Component Registry
# ===========================================================================


class TestHarnessComponentRegistry:
    """Test HarnessComponentRegistry."""

    def test_register_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = HarnessComponentRegistry(tmpdir)
            registry.register_component(
                "prompts/main.json",
                ComponentType.SYSTEM_PROMPT,
                "Main prompt",
            )
            comp = registry.get_component("prompts/main.json")
            assert comp is not None
            assert comp.component_type == ComponentType.SYSTEM_PROMPT

    def test_get_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = HarnessComponentRegistry(tmpdir)
            registry.register_component("a.json", ComponentType.SYSTEM_PROMPT)
            registry.register_component("b.py", ComponentType.MIDDLEWARE)
            registry.register_component("c.json", ComponentType.SYSTEM_PROMPT)
            prompts = registry.get_components_by_type(ComponentType.SYSTEM_PROMPT)
            assert len(prompts) == 2

    def test_record_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = HarnessComponentRegistry(tmpdir)
            registry.register_component("a.py", ComponentType.TOOL_IMPLEMENTATION)
            registry.record_edit("a.py", "edit:abc123")
            history = registry.get_component_history("a.py")
            assert "edit:abc123" in history

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry1 = HarnessComponentRegistry(tmpdir)
            registry1.register_component("x.py", ComponentType.MIDDLEWARE)

            # Create a new registry from the same path
            registry2 = HarnessComponentRegistry(tmpdir)
            comp = registry2.get_component("x.py")
            assert comp is not None
            assert comp.component_type == ComponentType.MIDDLEWARE

    def test_register_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = HarnessComponentRegistry(tmpdir)
            registry.register_defaults()
            all_comps = registry.get_all_components()
            assert len(all_comps) > 0
            # Should have at least one of each common type
            types = {c.component_type for c in all_comps.values()}
            assert ComponentType.SYSTEM_PROMPT in types
            assert ComponentType.MIDDLEWARE in types


# ===========================================================================
# Phase 2: Trace Backend & Distiller
# ===========================================================================


class TestTraceBackend:
    """Test trace backend creation and file backend."""

    def test_create_file_backend(self):
        backend = create_trace_backend(backend_type="file", trace_dir="/tmp")
        assert isinstance(backend, FileTraceBackend)

    @pytest.mark.asyncio
    async def test_file_backend_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileTraceBackend(trace_dir=tmpdir)
            traces = await backend.get_traces("nonexistent")
            assert traces == []

    @pytest.mark.asyncio
    async def test_file_backend_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a test trace file
            trace_data = [
                {
                    "id": "trace_001",
                    "name": "test_task",
                    "status": "success",
                    "score": 0.9,
                }
            ]
            with open(os.path.join(tmpdir, "round_test.json"), "w") as f:
                json.dump(trace_data, f)

            backend = FileTraceBackend(trace_dir=tmpdir)
            traces = await backend.get_traces("round_test")
            assert len(traces) == 1
            assert traces[0]["name"] == "test_task"


class TestTraceDistiller:
    """Test the trace distillation pipeline."""

    @pytest.mark.asyncio
    async def test_distill_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileTraceBackend(trace_dir=tmpdir)
            config = DistillationConfig(evidence_output_dir=os.path.join(tmpdir, "evidence"))
            distiller = TraceDistiller(backend=backend, config=config)
            corpus = await distiller.distill("nonexistent")
            assert corpus.total_tasks == 0

    @pytest.mark.asyncio
    async def test_distill_with_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test traces
            traces = [
                {"id": "t1", "name": "task_pass", "status": "success", "score": 0.9},
                {"id": "t2", "name": "task_fail", "status": "error", "score": 0.1, "error": "timeout"},
                {"id": "t3", "name": "task_pass2", "status": "success", "score": 0.85},
            ]
            with open(os.path.join(tmpdir, "round_001.json"), "w") as f:
                json.dump(traces, f)

            backend = FileTraceBackend(trace_dir=tmpdir)
            config = DistillationConfig(evidence_output_dir=os.path.join(tmpdir, "evidence"))
            distiller = TraceDistiller(backend=backend, config=config)
            corpus = await distiller.distill("round_001")

            assert corpus.total_tasks == 3
            assert len(corpus.get_successes()) == 2
            assert len(corpus.get_failures()) == 1
            assert corpus.pass_rate == pytest.approx(2 / 3, abs=0.01)


# ===========================================================================
# Phase 3: Manifest Verifier
# ===========================================================================


class TestManifestVerifier:
    """Test the manifest verification loop."""

    @pytest.mark.asyncio
    async def test_verify_perfect_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = HarnessComponentRegistry(tmpdir)
            verifier = ManifestVerifier(registry=registry)

            manifest = ChangeManifest(baseline_score=0.5)
            manifest.add_edit(
                ComponentEdit(
                    component_type=ComponentType.SYSTEM_PROMPT,
                    file_path="a.json",
                    edit_summary="Fix prompt",
                    predicted_fixes=["t1", "t2"],
                )
            )

            baseline = EvidenceCorpus(
                entries=[
                    EvidenceEntry(task_id="t1", pass_fail=False, score=0.0),
                    EvidenceEntry(task_id="t2", pass_fail=False, score=0.0),
                    EvidenceEntry(task_id="t3", pass_fail=True, score=1.0),
                ],
                benchmark_score=0.33,
            )
            new = EvidenceCorpus(
                entries=[
                    EvidenceEntry(task_id="t1", pass_fail=True, score=1.0),
                    EvidenceEntry(task_id="t2", pass_fail=True, score=1.0),
                    EvidenceEntry(task_id="t3", pass_fail=True, score=1.0),
                ],
                benchmark_score=1.0,
            )

            result = await verifier.verify(manifest, baseline, new)
            assert result.fix_precision == 1.0
            assert result.fix_recall == 1.0
            assert len(result.unexpected_regressions) == 0
            assert result.recommendation == "confirm"

    @pytest.mark.asyncio
    async def test_verify_with_regressions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = HarnessComponentRegistry(tmpdir)
            verifier = ManifestVerifier(registry=registry)

            manifest = ChangeManifest(baseline_score=0.5)
            manifest.add_edit(
                ComponentEdit(
                    component_type=ComponentType.MIDDLEWARE,
                    file_path="guard.py",
                    edit_summary="Added rate limiting",
                    predicted_fixes=["t1"],
                )
            )

            baseline = EvidenceCorpus(
                entries=[
                    EvidenceEntry(task_id="t1", pass_fail=False, score=0.0),
                    EvidenceEntry(task_id="t2", pass_fail=True, score=1.0),
                ],
                benchmark_score=0.5,
            )
            new = EvidenceCorpus(
                entries=[
                    EvidenceEntry(task_id="t1", pass_fail=True, score=1.0),
                    EvidenceEntry(task_id="t2", pass_fail=False, score=0.0),  # Regressed!
                ],
                benchmark_score=0.5,
            )

            result = await verifier.verify(manifest, baseline, new)
            assert "t2" in result.unexpected_regressions
            assert result.recommendation == "partial_revert"


# ===========================================================================
# Phase 4: Constraint Engine
# ===========================================================================


class TestConstraintEngine:
    """Test the hierarchical constraint engine."""

    def test_add_constraint(self):
        engine = ConstraintEngine()
        engine.add_constraint(
            HierarchicalConstraint(
                id="no_delete",
                description="Never delete production files",
                current_level=ConstraintLevel.MIDDLEWARE,
                applies_to=["delete_file", "remove_file"],
                condition="not_allowed",
                action="block",
            )
        )
        assert len(engine.get_all_constraints()) == 1

    def test_tool_call_blocking(self):
        engine = ConstraintEngine()
        engine.add_constraint(
            HierarchicalConstraint(
                id="no_delete",
                description="Block deletions",
                current_level=ConstraintLevel.MIDDLEWARE,
                applies_to=["delete_file"],
                condition="not_allowed",
                action="block",
            )
        )
        allowed, violations = engine.check_tool_call("delete_file")
        assert not allowed
        assert len(violations) == 1
        assert violations[0].auto_blocked

    def test_prompt_level_no_block(self):
        engine = ConstraintEngine()
        engine.add_constraint(
            HierarchicalConstraint(
                id="soft_constraint",
                description="Prefer async operations",
                current_level=ConstraintLevel.PROMPT,
                applies_to=["sync_operation"],
                condition="not_allowed",
                action="warn",
            )
        )
        # Prompt-level constraints don't block
        allowed, violations = engine.check_tool_call("sync_operation")
        assert allowed
        assert len(violations) == 0

    def test_escalation(self):
        engine = ConstraintEngine()
        engine.add_constraint(
            HierarchicalConstraint(
                id="test_constraint",
                description="Test",
                current_level=ConstraintLevel.PROMPT,
            )
        )
        new_level = engine.escalate_constraint("test_constraint", "Testing")
        assert new_level == ConstraintLevel.TOOL_DESCRIPTION

        new_level = engine.escalate_constraint("test_constraint", "Testing again")
        assert new_level == ConstraintLevel.MIDDLEWARE

    def test_auto_escalation(self):
        engine = ConstraintEngine()
        engine.add_constraint(
            HierarchicalConstraint(
                id="auto_test",
                description="Auto-escalation test",
                current_level=ConstraintLevel.PROMPT,
                escalation_threshold=3,
            )
        )
        # Simulate violations
        constraint = engine.get_constraint("auto_test")
        assert constraint is not None
        constraint.violation_count = 5

        escalated = engine.auto_escalate_all()
        assert "auto_test" in escalated
        assert constraint.current_level == ConstraintLevel.TOOL_DESCRIPTION

    def test_prompt_constraints_text(self):
        engine = ConstraintEngine()
        engine.add_constraint(
            HierarchicalConstraint(
                id="safety_rule",
                description="Always validate inputs before processing",
                current_level=ConstraintLevel.PROMPT,
                applies_to=["process_data"],
            )
        )
        text = engine.get_prompt_constraints()
        assert "safety_rule" in text
        assert "validate inputs" in text

    def test_constraint_level_ordering(self):
        """Constraint levels should have proper ordering."""
        assert ConstraintLevel.PROMPT < ConstraintLevel.TOOL_DESCRIPTION
        assert ConstraintLevel.TOOL_DESCRIPTION < ConstraintLevel.MIDDLEWARE
        assert ConstraintLevel.MIDDLEWARE < ConstraintLevel.TOOL_IMPLEMENTATION
