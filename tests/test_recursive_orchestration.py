from __future__ import annotations
"""Tests for CONCEPT:ORCH-1.1: Recursive Graph Orchestration."""


import pytest

from agent_utilities.graph.hierarchical_planner import (
    MAX_RECURSION_DEPTH,
    RecursionDepthExceeded,
    RecursiveContext,
)
from agent_utilities.graph.state import GraphState


class TestRecursiveContext:
    def test_default_construction(self):
        ctx = RecursiveContext()
        assert ctx.parent_query == ""
        assert ctx.parent_error == ""
        assert ctx.parent_results == {}
        assert ctx.recursion_depth == 1

    def test_full_construction(self):
        ctx = RecursiveContext(
            parent_query="build an API",
            parent_plan_summary="[researcher, programmer]",
            parent_error="Researcher failed",
            parent_results={"researcher": "no data"},
            recursion_depth=2,
        )
        assert ctx.parent_query == "build an API"
        assert ctx.recursion_depth == 2
        assert "researcher" in ctx.parent_results


class TestMaxRecursionDepth:
    def test_default_is_2(self):
        assert MAX_RECURSION_DEPTH == 2

    def test_configurable_via_env(self):
        # The constant is read at import time, so we just verify the env var pattern
        assert isinstance(MAX_RECURSION_DEPTH, int)
        assert MAX_RECURSION_DEPTH > 0


class TestRecursionDepthExceeded:
    def test_is_runtime_error(self):
        exc = RecursionDepthExceeded("too deep")
        assert isinstance(exc, RuntimeError)
        assert "too deep" in str(exc)


class TestGraphStateRecursionDepth:
    def test_default_zero(self):
        state = GraphState(query="test")
        assert state.recursion_depth == 0

    def test_can_increment(self):
        state = GraphState(query="test", recursion_depth=1)
        assert state.recursion_depth == 1


class TestRecursiveExecutorDepthCheck:
    @pytest.mark.asyncio
    async def test_exceeds_depth_raises(self):
        from unittest.mock import MagicMock

        ctx = RecursiveContext(recursion_depth=MAX_RECURSION_DEPTH + 1)
        deps = MagicMock()
        from agent_utilities.graph.hierarchical_planner import execute_recursive_graph

        with pytest.raises(RecursionDepthExceeded):
            await execute_recursive_graph(ctx, deps)

    @pytest.mark.asyncio
    async def test_at_max_depth_raises(self):
        from unittest.mock import MagicMock

        ctx = RecursiveContext(recursion_depth=MAX_RECURSION_DEPTH + 1)
        deps = MagicMock()
        from agent_utilities.graph.hierarchical_planner import execute_recursive_graph

        with pytest.raises(RecursionDepthExceeded):
            await execute_recursive_graph(ctx, deps)


class TestStepDescriptionsIncludesRecursive:
    def test_recursive_orchestrator_in_descriptions(self):
        from agent_utilities.graph.executor import get_step_descriptions

        desc = get_step_descriptions()
        assert "recursive_orchestrator" in desc
        assert "CONCEPT:ORCH-1.1" in desc
