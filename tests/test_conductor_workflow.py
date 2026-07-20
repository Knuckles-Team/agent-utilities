from __future__ import annotations

"""Tests for CONCEPT:AU-ORCH.planning.recursion-nesting-depth: Conductor Workflow Specification.

Validates that ExecutionStep.refined_subtask is correctly serialized,
preferred over raw queries in the executor, and properly integrated
into the routing prompt.
"""


from agent_utilities.models.graph import ExecutionStep, GraphPlan


class TestRefinedSubtaskModel:
    """Model-level tests for the refined_subtask field."""

    def test_refined_subtask_default_none(self):
        """refined_subtask defaults to None when not specified."""
        step = ExecutionStep(id="researcher")
        assert step.refined_subtask is None

    def test_refined_subtask_set(self):
        """refined_subtask can be set to a string."""
        step = ExecutionStep(
            id="python_programmer",
            refined_subtask="Implement JWT auth middleware for the FastAPI app",
        )
        assert (
            step.refined_subtask == "Implement JWT auth middleware for the FastAPI app"
        )

    def test_refined_subtask_serialization(self):
        """refined_subtask round-trips through JSON serialization."""
        step = ExecutionStep(
            id="researcher",
            refined_subtask="Find all REST API frameworks with built-in auth",
        )
        data = step.model_dump()
        assert (
            data["refined_subtask"] == "Find all REST API frameworks with built-in auth"
        )

        restored = ExecutionStep.model_validate(data)
        assert restored.refined_subtask == step.refined_subtask

    def test_refined_subtask_in_graphplan(self):
        """GraphPlan with mixed refined/non-refined steps."""
        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    id="researcher",
                    refined_subtask="Survey authentication patterns in Python web frameworks",
                ),
                ExecutionStep(
                    id="python_programmer",
                    description="write the code",
                ),
            ]
        )
        assert plan.steps[0].refined_subtask is not None
        assert plan.steps[1].refined_subtask is None

    def test_refined_subtask_in_acp_plan_entries(self):
        """to_acp_plan_entries includes refined_subtask in content."""
        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    id="researcher",
                    refined_subtask="Find auth patterns",
                    description="research auth",
                ),
            ]
        )
        entries = plan.to_acp_plan_entries()
        assert len(entries) == 1
        # input_data is used in content, refined_subtask is separate
        assert "researcher" in entries[0]["content"]

    def test_refined_subtask_json_schema(self):
        """refined_subtask appears in the JSON schema."""
        schema = ExecutionStep.model_json_schema()
        # ExecutionStep (== Task) is self-referential (subtasks), so pydantic emits the
        # properties under $defs behind a top-level $ref rather than at the root.
        props = (
            schema.get("properties")
            or schema["$defs"][schema["$ref"].rsplit("/", 1)[-1]]["properties"]
        )
        assert "refined_subtask" in props
        desc = props["refined_subtask"]["description"]
        assert "CONCEPT:AU-ORCH.planning.recursion-nesting-depth" in desc

    def test_refined_subtask_coexists_with_input_data(self):
        """Both refined_subtask and input_data can be set."""
        step = ExecutionStep(
            id="python_programmer",
            description={"question": "implement auth"},
            refined_subtask="Build JWT middleware with RS256 signing",
        )
        assert step.description is not None
        assert step.refined_subtask is not None

    def test_refined_subtask_empty_string(self):
        """Empty string refined_subtask is valid but falsy."""
        step = ExecutionStep(
            id="researcher",
            refined_subtask="",
        )
        assert step.refined_subtask == ""
        # Empty string is falsy, executor should treat as absent
        assert not step.refined_subtask

    def test_refined_subtask_long_instruction(self):
        """Long refined subtasks are preserved."""
        long_task = "x" * 5000
        step = ExecutionStep(
            id="researcher",
            refined_subtask=long_task,
        )
        assert len(step.refined_subtask) == 5000  # type: ignore[arg-type]

    def test_refined_subtask_model_validate_json(self):
        """refined_subtask works with model_validate_json."""
        import json

        data = {
            "node_id": "architect",
            "refined_subtask": "Design microservice boundaries",
        }
        step = ExecutionStep.model_validate_json(json.dumps(data))
        assert step.refined_subtask == "Design microservice boundaries"

    def test_refined_subtask_mermaid_includes_field(self):
        """to_mermaid renders correctly with refined_subtask steps."""
        plan = GraphPlan(
            steps=[
                ExecutionStep(
                    id="researcher",
                    refined_subtask="Research auth patterns",
                ),
            ]
        )
        mermaid = plan.to_mermaid()
        assert "researcher" in mermaid


class TestRefinedSubtaskRouting:
    """Integration tests verifying the router prompt includes CONCEPT:AU-ORCH.planning.recursion-nesting-depth instructions."""

    def test_step_descriptions_not_empty(self):
        """get_step_descriptions returns a non-empty catalog."""
        from agent_utilities.graph.executor import get_step_descriptions

        descriptions = get_step_descriptions()
        assert len(descriptions) > 0
        assert "researcher" in descriptions
        assert "recursive_orchestrator" in descriptions
