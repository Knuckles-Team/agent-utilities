"""Tests for GoalSpec model and KG integration (ORCH-5.0)."""

from __future__ import annotations

import pytest

from agent_utilities.models.goal import (
    GoalCheckpoint,
    GoalIteration,
    GoalKGIntegration,
    GoalResult,
    GoalSpec,
    GoalStatus,
)


class TestGoalSpecParsing:
    """Test natural-language parsing of /goal inputs."""

    def test_simple_objective(self):
        spec = GoalSpec.parse_goal_input("fix the README formatting")
        assert spec.objective == "fix the README formatting"
        assert spec.end_state == ""
        assert spec.constraints == []
        assert spec.validation_cmd == ""

    def test_with_goal_prefix(self):
        spec = GoalSpec.parse_goal_input("/goal fix the README formatting")
        assert spec.objective == "fix the README formatting"

    def test_objective_with_end_state(self):
        spec = GoalSpec.parse_goal_input("fix failing tests until pytest passes")
        assert spec.objective == "fix failing tests"
        assert spec.end_state == "pytest passes"

    def test_objective_with_end_state_and_constraints(self):
        spec = GoalSpec.parse_goal_input(
            "fix every failing test until npm test exits 0 without modifying /auth"
        )
        assert spec.objective == "fix every failing test"
        assert spec.end_state == "npm test exits 0"
        assert spec.constraints == ["modifying /auth"]
        assert spec.validation_cmd == "npm test"

    def test_multiple_constraints(self):
        spec = GoalSpec.parse_goal_input(
            "refactor auth until all tests pass without changing the API, removing indexes"
        )
        assert spec.objective == "refactor auth"
        assert spec.end_state == "all tests pass"
        assert len(spec.constraints) == 2
        assert "changing the API" in spec.constraints
        assert "removing indexes" in spec.constraints

    def test_validation_cmd_extraction_exits(self):
        spec = GoalSpec.parse_goal_input(
            "optimize code until npm test exits 0"
        )
        assert spec.validation_cmd == "npm test"

    def test_validation_cmd_extraction_passes(self):
        spec = GoalSpec.parse_goal_input(
            "fix bugs until pytest passes"
        )
        assert spec.validation_cmd == "pytest"

    def test_raw_input_preserved(self):
        raw = "/goal fix tests until pytest passes"
        spec = GoalSpec.parse_goal_input(raw)
        assert spec.raw_input == raw

    def test_default_values(self):
        spec = GoalSpec.parse_goal_input("simple task")
        assert spec.max_iterations == 20
        assert spec.auto_approve is True
        assert spec.session_id == ""
        assert spec.kg_node_type == "GoalNode"

    def test_case_insensitive_until(self):
        spec = GoalSpec.parse_goal_input("fix tests UNTIL all pass")
        assert spec.objective == "fix tests"
        assert spec.end_state == "all pass"

    def test_empty_input(self):
        spec = GoalSpec.parse_goal_input("")
        assert spec.objective == ""


class TestGoalSpecSystemPrompt:
    """Test system prompt generation."""

    def test_basic_prompt(self):
        spec = GoalSpec(objective="fix tests", max_iterations=10)
        prompt = spec.to_system_prompt()
        assert "fix tests" in prompt
        assert "Autonomous Goal Mode" in prompt
        assert "10 iterations" in prompt

    def test_prompt_with_end_state(self):
        spec = GoalSpec(objective="fix tests", end_state="pytest passes")
        prompt = spec.to_system_prompt()
        assert "Success Criteria" in prompt
        assert "pytest passes" in prompt

    def test_prompt_with_constraints(self):
        spec = GoalSpec(
            objective="fix tests",
            constraints=["modifying /auth", "removing code"],
        )
        prompt = spec.to_system_prompt()
        assert "Do NOT modifying /auth" in prompt
        assert "Do NOT removing code" in prompt

    def test_prompt_with_validation_cmd(self):
        spec = GoalSpec(objective="fix tests", validation_cmd="npm test")
        prompt = spec.to_system_prompt()
        assert "`npm test`" in prompt


class TestGoalSpecKGFields:
    """Test KG-native fields on GoalSpec."""

    def test_kg_context_default(self):
        spec = GoalSpec(objective="test")
        assert spec.kg_context == []

    def test_kg_rules_default(self):
        spec = GoalSpec(objective="test")
        assert spec.kg_rules == []

    def test_related_goals_default(self):
        spec = GoalSpec(objective="test")
        assert spec.related_goals == []

    def test_kg_node_type(self):
        spec = GoalSpec(objective="test")
        assert spec.kg_node_type == "GoalNode"

    def test_kg_fields_populated(self):
        spec = GoalSpec(
            objective="test",
            kg_context=["[File] app.py", "[Symbol] main"],
            kg_rules=["Never delete production data"],
            related_goals=["goal-123", "goal-456"],
        )
        assert len(spec.kg_context) == 2
        assert len(spec.kg_rules) == 1
        assert len(spec.related_goals) == 2


class TestGoalIteration:
    """Test GoalIteration model."""

    def test_defaults(self):
        it = GoalIteration(iteration=1, action="fixed test")
        assert it.iteration == 1
        assert it.action == "fixed test"
        assert it.is_complete is False
        assert it.tool_calls == 0

    def test_complete_iteration(self):
        it = GoalIteration(iteration=3, action="ran tests", is_complete=True)
        assert it.is_complete is True


class TestGoalResult:
    """Test GoalResult model and reporting."""

    def test_success_property(self):
        result = GoalResult(status=GoalStatus.COMPLETED)
        assert result.success is True

    def test_failure_property(self):
        result = GoalResult(status=GoalStatus.FAILED)
        assert result.success is False

    def test_report_generation(self):
        result = GoalResult(
            goal_id="test-123",
            status=GoalStatus.COMPLETED,
            total_iterations=5,
            total_duration_ms=30000,
            total_tool_calls=15,
            summary="All tests fixed",
            iterations=[
                GoalIteration(iteration=1, action="identified failures"),
                GoalIteration(iteration=2, action="fixed test_auth", is_complete=True),
            ],
        )
        report = result.to_report()
        assert "Goal Result" in report
        assert "completed" in report
        assert "5" in report
        assert "30.0s" in report
        assert "15" in report
        assert "All tests fixed" in report

    def test_report_with_error(self):
        result = GoalResult(
            status=GoalStatus.FAILED,
            error="Max iterations reached",
        )
        report = result.to_report()
        assert "Max iterations reached" in report

    def test_report_duration_minutes(self):
        result = GoalResult(total_duration_ms=120000)
        report = result.to_report()
        assert "2.0m" in report


class TestGoalCheckpoint:
    """Test GoalCheckpoint model."""

    def test_checkpoint_creation(self):
        spec = GoalSpec.parse_goal_input("fix tests until pytest passes")
        checkpoint = GoalCheckpoint(
            goal_spec=spec,
            current_iteration=3,
            status=GoalStatus.RUNNING,
            session_id="session-abc",
        )
        assert checkpoint.goal_spec.objective == "fix tests"
        assert checkpoint.current_iteration == 3
        assert checkpoint.status == GoalStatus.RUNNING

    def test_checkpoint_with_iterations(self):
        spec = GoalSpec(objective="test")
        iterations = [
            GoalIteration(iteration=1, action="step 1"),
            GoalIteration(iteration=2, action="step 2"),
        ]
        checkpoint = GoalCheckpoint(
            goal_spec=spec, iterations=iterations, current_iteration=2
        )
        assert len(checkpoint.iterations) == 2


class TestGoalKGIntegration:
    """Test KG integration (graceful degradation when engine is None)."""

    def test_enrich_no_engine(self):
        kg = GoalKGIntegration(engine=None)
        spec = GoalSpec(objective="test")
        result = kg.enrich_from_kg(spec)
        assert result.kg_context == []

    def test_validate_no_engine(self):
        kg = GoalKGIntegration(engine=None)
        spec = GoalSpec(objective="test")
        result = kg.validate_against_rules(spec)
        assert result.kg_rules == []

    def test_find_related_no_engine(self):
        kg = GoalKGIntegration(engine=None)
        spec = GoalSpec(objective="test")
        result = kg.find_related_goals(spec)
        assert result.related_goals == []

    def test_persist_no_engine(self):
        kg = GoalKGIntegration(engine=None)
        spec = GoalSpec(objective="test")
        node_id = kg.persist_goal(spec)
        assert node_id == spec.id

    def test_update_status_no_engine(self):
        kg = GoalKGIntegration(engine=None)
        # Should not raise
        kg.update_goal_status("test-id", "completed", "done")


class TestGoalStatus:
    """Test GoalStatus enum."""

    def test_all_statuses(self):
        assert GoalStatus.PENDING == "pending"
        assert GoalStatus.RUNNING == "running"
        assert GoalStatus.VALIDATING == "validating"
        assert GoalStatus.COMPLETED == "completed"
        assert GoalStatus.FAILED == "failed"
        assert GoalStatus.CANCELLED == "cancelled"
        assert GoalStatus.PAUSED == "paused"
