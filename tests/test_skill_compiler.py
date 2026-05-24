"""Tests for SkillCompiler & Distillation Hook.

CONCEPT:ORCH-1.25 — Skill-to-Workflow Compilation

Tests cover:
- SkillCompiler SKILL.md parsing
- Team.yaml metadata extraction
- WorkflowDistillationHook threshold and scaffold logic
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent_utilities.models.graph import ExecutionStep, GraphPlan
from agent_utilities.workflows.distillation_hook import (
    WorkflowDistillationHook,
    _compute_pattern_key,
)
from agent_utilities.workflows.skill_compiler import SkillCompiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_plan() -> GraphPlan:
    """Create a multi-step GraphPlan for testing."""
    return GraphPlan(
        steps=[
            ExecutionStep(
                node_id="agent-a",
                refined_subtask="Step 1",
                is_parallel=False,
                depends_on=[],
            ),
            ExecutionStep(
                node_id="agent-b",
                refined_subtask="Step 2",
                is_parallel=False,
                depends_on=["agent-a"],
            ),
        ],
        metadata={"scenario_name": "test"},
    )


@pytest.fixture
def trivial_plan() -> GraphPlan:
    """Create a single-step GraphPlan (should be rejected as trivial)."""
    return GraphPlan(
        steps=[
            ExecutionStep(
                node_id="agent-a",
                refined_subtask="Only step",
                is_parallel=False,
                depends_on=[],
            ),
        ],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Test Skill Compiler
# ---------------------------------------------------------------------------


class TestSkillCompiler:
    """Test SkillCompiler parsing."""

    def test_compile_from_text_with_headers(self) -> None:
        """Parse standard ### Step N: headers."""
        markdown = """---
name: test-skill
---
# Test Skill

### Step 1: Agent A
Do step 1 task.

### Step 2: Agent B
Do step 2 task.
"""
        plan = SkillCompiler.compile_from_text("test-skill", markdown)
        assert len(plan.steps) == 2
        assert plan.steps[0].node_id == "agent-a"
        assert plan.steps[0].refined_subtask is not None
        assert "Do step 1 task" in plan.steps[0].refined_subtask
        assert plan.steps[1].node_id == "agent-b"
        assert plan.steps[1].refined_subtask is not None
        assert "Do step 2 task" in plan.steps[1].refined_subtask
        assert plan.steps[1].depends_on == ["agent-a"]

    def test_compile_from_text_fallback(self) -> None:
        """Parse text without standard headers as a single step."""
        markdown = "Just do this task."
        plan = SkillCompiler.compile_from_text("test-skill", markdown)
        assert len(plan.steps) == 1
        assert plan.steps[0].node_id == "executor"
        assert plan.steps[0].refined_subtask == "Just do this task."

    def test_load_team_config_missing(self) -> None:
        """Returns None if team.yaml is missing."""
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            assert SkillCompiler.load_team_config(skill_dir) is None

    def test_load_team_config_present(self) -> None:
        """Returns parsed dict if team.yaml is present."""
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            refs_dir = skill_dir / "references"
            refs_dir.mkdir()
            team_data = {"name": "test_team", "execution_mode": "sequential"}
            with open(refs_dir / "team.yaml", "w") as f:
                yaml.dump(team_data, f)

            config = SkillCompiler.load_team_config(skill_dir)
            assert config is not None
            assert config["name"] == "test_team"

    def test_register_in_kg(self) -> None:
        """Simulates registering workflow and team config."""
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            with open(skill_dir / "SKILL.md", "w") as f:
                f.write("### Step 1: test\ndo")

            refs_dir = skill_dir / "references"
            refs_dir.mkdir()
            with open(refs_dir / "team.yaml", "w") as f:
                yaml.dump({"name": "test_team"}, f)

            outcome = SkillCompiler.register_in_kg(MagicMock(), skill_dir)
            assert outcome["registered"] is True
            assert outcome["workflow_id"] is not None
            assert outcome["team_config_id"] == "test_team"

    def test_lossless_roundtrip_update(self) -> None:
        """Verify that SkillCompiler.update_markdown retains formatting and updates steps correctly."""
        original_markdown = """---
name: test-lossless-skill
custom_field: preserve-this
---
# Main Header
This is some introductory prose that must be preserved.
> [!NOTE]
> An alert callout.

### Step 1: agent-a
Gather baseline details.

### Step 2: agent-b [depends_on: agent-a]
Process the data.

## Concluding Section
Some post-execution steps.
"""
        # Parse it into a GraphPlan
        plan = SkillCompiler.compile_from_text("test-lossless-skill", original_markdown)
        assert len(plan.steps) == 2
        assert plan.steps[0].node_id == "agent-a"
        assert plan.steps[1].node_id == "agent-b"

        # Programmatically mutate the steps
        plan.steps[0].refined_subtask = "Gather updated details."
        # Update depends_on for step 2
        plan.steps[1].depends_on = ["agent-a", "agent-extra"]

        # Call update_markdown
        updated_markdown = SkillCompiler.update_markdown(original_markdown, plan)

        # Verify YAML frontmatter is completely preserved
        assert "custom_field: preserve-this" in updated_markdown
        # Verify intro prose and alert are preserved
        assert "This is some introductory prose that must be preserved." in updated_markdown
        assert "> [!NOTE]" in updated_markdown
        # Verify concluding section is preserved
        assert "## Concluding Section" in updated_markdown
        assert "Some post-execution steps." in updated_markdown

        # Verify step updates
        assert "Gather updated details." in updated_markdown
        assert "### Step 2: agent-b [depends_on: agent-a, agent-extra]" in updated_markdown

    def test_lossless_roundtrip_save_new(self) -> None:
        """Verify that SkillCompiler.save creates a new file if it doesn't exist."""
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            plan = GraphPlan(
                steps=[
                    ExecutionStep(node_id="new-agent", refined_subtask="New step task", depends_on=[])
                ]
            )
            SkillCompiler.save(skill_dir, plan)

            skill_path = skill_dir / "SKILL.md"
            assert skill_path.exists()
            content = skill_path.read_text(encoding="utf-8")
            assert "New step task" in content
            assert "### Step 1: new-agent" in content



# ---------------------------------------------------------------------------
# Test Distillation Hook
# ---------------------------------------------------------------------------


class TestDistillationHook:
    """Test WorkflowDistillationHook threshold and scaffolding logic."""

    @pytest.mark.asyncio
    async def test_no_engine_returns_no_engine(self, sample_plan: GraphPlan) -> None:
        """Hook without engine should return 'no_engine' reason."""
        hook = WorkflowDistillationHook(engine=None, promotion_threshold=3)
        outcome = await hook.on_execution_complete(
            run_id="test-1", plan=sample_plan, result="success", quality_score=1.0
        )
        assert not outcome["promoted"]
        assert outcome["reason"] == "no_engine"

    @pytest.mark.asyncio
    async def test_quality_below_minimum(self, sample_plan: GraphPlan) -> None:
        """Low quality score should prevent distillation."""
        hook = WorkflowDistillationHook(
            engine=MagicMock(), promotion_threshold=3, quality_minimum=0.6
        )
        outcome = await hook.on_execution_complete(
            run_id="test-1", plan=sample_plan, result="low quality", quality_score=0.3
        )
        assert not outcome["promoted"]
        assert "quality_below_minimum" in outcome["reason"]

    @pytest.mark.asyncio
    async def test_trivial_workflow_rejected(self, trivial_plan: GraphPlan) -> None:
        """Single-step workflows should not be promoted."""
        hook = WorkflowDistillationHook(
            engine=MagicMock(), promotion_threshold=1, quality_minimum=0.0
        )
        outcome = await hook.on_execution_complete(
            run_id="test-1", plan=trivial_plan, result="done", quality_score=1.0
        )
        assert not outcome["promoted"]
        assert outcome["reason"] == "trivial_workflow"

    @pytest.mark.asyncio
    async def test_below_threshold(self, sample_plan: GraphPlan) -> None:
        """First success should not trigger promotion with threshold=3."""
        mock_engine = MagicMock()
        mock_engine.backend = None  # No backend → _record_success returns 1

        hook = WorkflowDistillationHook(
            engine=mock_engine, promotion_threshold=3, quality_minimum=0.0
        )
        outcome = await hook.on_execution_complete(
            run_id="test-1", plan=sample_plan, result="done", quality_score=1.0
        )
        assert not outcome["promoted"]
        assert "below_threshold" in outcome["reason"]

    def test_compute_pattern_key(self, sample_plan: GraphPlan) -> None:
        """Pattern key should be deterministic and topology-based."""
        key = _compute_pattern_key(sample_plan)
        assert "agent-a" in key
        assert "agent-b" in key
        # Same plan → same key
        assert _compute_pattern_key(sample_plan) == key

    def test_compute_pattern_key_different_plans(self) -> None:
        """Different topologies should produce different keys."""
        plan_a = GraphPlan(
            steps=[
                ExecutionStep(node_id="x", refined_subtask="t", depends_on=[]),
                ExecutionStep(node_id="y", refined_subtask="t", depends_on=["x"]),
            ]
        )
        plan_b = GraphPlan(
            steps=[
                ExecutionStep(node_id="x", refined_subtask="t", depends_on=[]),
                ExecutionStep(node_id="z", refined_subtask="t", depends_on=["x"]),
            ]
        )
        assert _compute_pattern_key(plan_a) != _compute_pattern_key(plan_b)

    @pytest.mark.asyncio
    async def test_scaffold_skill_success(self, sample_plan: GraphPlan) -> None:
        """Test _scaffold_skill generates correct directory and files."""
        hook = WorkflowDistillationHook(engine=MagicMock(), promotion_threshold=1)

        with patch.object(Path, "mkdir"):
            with patch("builtins.open", new_callable=MagicMock) as mock_open:
                # Assuming scaffold_skill works even if Path is patched
                skill_dir = await hook._scaffold_skill(
                    "test-pattern", sample_plan, None
                )  # type: ignore
                assert "distilled-" in skill_dir.name

                # Verify that it tried to write SKILL.md
                mock_open.assert_called()
