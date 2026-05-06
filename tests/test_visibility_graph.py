"""Tests for CONCEPT:ORCH-1.1: Execution Visibility Graph.

Validates the access_list field on ExecutionStep and the
_resolve_access_context() helper that filters results_registry
for precise per-specialist context injection.
"""

from __future__ import annotations

from agent_utilities.graph.executor import _resolve_access_context
from agent_utilities.models.graph import ExecutionStep, GraphPlan


class TestAccessListModel:
    """Model-level tests for the access_list field."""

    def test_access_list_default_empty(self):
        """access_list defaults to an empty list."""
        step = ExecutionStep(node_id="researcher")
        assert step.access_list == []

    def test_access_list_set_all(self):
        """access_list can be set to ['all']."""
        step = ExecutionStep(node_id="python_programmer", access_list=["all"])
        assert step.access_list == ["all"]

    def test_access_list_specific_nodes(self):
        """access_list with specific node IDs."""
        step = ExecutionStep(
            node_id="synthesizer",
            access_list=["researcher", "architect"],
        )
        assert "researcher" in step.access_list
        assert "architect" in step.access_list
        assert len(step.access_list) == 2

    def test_access_list_serialization(self):
        """access_list round-trips through JSON."""
        step = ExecutionStep(
            node_id="verifier",
            access_list=["researcher", "python_programmer"],
        )
        data = step.model_dump()
        assert data["access_list"] == ["researcher", "python_programmer"]

        restored = ExecutionStep.model_validate(data)
        assert restored.access_list == step.access_list

    def test_access_list_json_schema(self):
        """access_list appears in the JSON schema with CONCEPT:ORCH-1.1 reference."""
        schema = ExecutionStep.model_json_schema()
        assert "access_list" in schema["properties"]
        desc = schema["properties"]["access_list"]["description"]
        assert "CONCEPT:ORCH-1.1" in desc

    def test_access_list_in_graphplan(self):
        """GraphPlan with mixed access_list configurations."""
        plan = GraphPlan(
            steps=[
                ExecutionStep(node_id="researcher"),  # No access list
                ExecutionStep(
                    node_id="python_programmer",
                    access_list=["researcher"],
                ),
                ExecutionStep(
                    node_id="verifier",
                    access_list=["all"],
                ),
            ]
        )
        assert plan.steps[0].access_list == []
        assert plan.steps[1].access_list == ["researcher"]
        assert plan.steps[2].access_list == ["all"]


class TestResolveAccessContext:
    """Tests for the _resolve_access_context helper."""

    def test_empty_access_list_returns_empty(self):
        """Empty access_list → no context."""
        step = ExecutionStep(node_id="test", access_list=[])
        result = _resolve_access_context(step, {"researcher": "some data"})
        assert result == ""

    def test_all_access_returns_full_registry(self):
        """['all'] → full registry contents."""
        step = ExecutionStep(node_id="test", access_list=["all"])
        registry = {
            "researcher": "Found 3 APIs",
            "architect": "Designed microservice layout",
        }
        result = _resolve_access_context(step, registry)
        assert "researcher" in result
        assert "architect" in result
        assert "Found 3 APIs" in result
        assert "Designed microservice layout" in result

    def test_all_access_empty_registry(self):
        """['all'] with empty registry → empty string."""
        step = ExecutionStep(node_id="test", access_list=["all"])
        result = _resolve_access_context(step, {})
        assert result == ""

    def test_specific_node_filters_correctly(self):
        """Specific node IDs filter to only matching results."""
        step = ExecutionStep(
            node_id="test",
            access_list=["researcher"],
        )
        registry = {
            "researcher_0": "Found 3 APIs",
            "architect_1": "Designed layout",
            "python_programmer_2": "Wrote code",
        }
        result = _resolve_access_context(step, registry)
        assert "Found 3 APIs" in result
        assert "Designed layout" not in result
        assert "Wrote code" not in result

    def test_multiple_specific_nodes(self):
        """Multiple specific node IDs include all matching results."""
        step = ExecutionStep(
            node_id="test",
            access_list=["researcher", "architect"],
        )
        registry = {
            "researcher_0": "Research findings",
            "architect_1": "Architecture plan",
            "python_programmer_2": "Implementation",
        }
        result = _resolve_access_context(step, registry)
        assert "Research findings" in result
        assert "Architecture plan" in result
        assert "Implementation" not in result

    def test_non_matching_node_returns_empty(self):
        """Node IDs with no match → empty string."""
        step = ExecutionStep(
            node_id="test",
            access_list=["nonexistent_specialist"],
        )
        registry = {"researcher_0": "data"}
        result = _resolve_access_context(step, registry)
        assert result == ""

    def test_case_insensitive_matching(self):
        """Access list matching is case-insensitive."""
        step = ExecutionStep(
            node_id="test",
            access_list=["Researcher"],
        )
        registry = {"researcher_0": "findings"}
        result = _resolve_access_context(step, registry)
        assert "findings" in result

    def test_partial_key_matching(self):
        """Access list matches partial keys (e.g., 'researcher' matches 'researcher_0')."""
        step = ExecutionStep(
            node_id="test",
            access_list=["researcher"],
        )
        registry = {
            "researcher_0": "first research",
            "researcher_1": "second research",
        }
        result = _resolve_access_context(step, registry)
        assert "first research" in result
        assert "second research" in result

    def test_context_format(self):
        """Verify the output format includes proper headers."""
        step = ExecutionStep(node_id="test", access_list=["all"])
        registry = {"researcher_0": "data"}
        result = _resolve_access_context(step, registry)
        assert "### Prior result from" in result
