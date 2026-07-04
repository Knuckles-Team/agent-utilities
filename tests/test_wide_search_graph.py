"""CONCEPT:AU-ORCH.planning.recursion-nesting-depth"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_graph import StepContext

from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.graph.verification import join_step, wide_search_joiner_step
from agent_utilities.models.graph import WideSearchWorkboard


@pytest.fixture
def base_state():
    state = GraphState(query="Test wide search", session_id="test-session")
    return state


@pytest.fixture
def base_deps():
    return GraphDeps(
        event_queue=AsyncMock(), tag_prompts={}, tag_env_vars={}, mcp_toolsets=[]
    )


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_join_step_fast_path_pass(base_state, base_deps):
    # Setup workboard that passes validation
    base_state.pending_parallel_count = 1
    base_state.workboard = WideSearchWorkboard(
        schema_definition={"name": "str", "age": "int"},
        expected_row_count=2,
        row_slots={
            "entity_1": {"name": "Alice", "age": "30"},
            "entity_2": {"name": "Bob", "age": "40"},
        },
    )

    ctx = StepContext(state=base_state, deps=base_deps, inputs=None)

    next_node = await join_step(ctx)
    assert next_node == "dispatcher"
    assert base_state.validation_feedback is None


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_join_step_fast_path_fail_missing_rows(base_state, base_deps):
    # Setup workboard that fails missing rows validation
    base_state.pending_parallel_count = 1
    base_state.workboard = WideSearchWorkboard(
        schema_definition={"name": "str"},
        expected_row_count=3,
        row_slots={"entity_1": {"name": "Alice"}},
    )

    ctx = StepContext(state=base_state, deps=base_deps, inputs=None)

    next_node = await join_step(ctx)
    assert next_node == "wide_search_joiner"
    assert "Missing rows: Expected 3, got 1" in base_state.validation_feedback


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_join_step_fast_path_fail_schema(base_state, base_deps):
    # Setup workboard that fails schema validation
    base_state.pending_parallel_count = 1
    base_state.workboard = WideSearchWorkboard(
        schema_definition={"name": "str", "age": "int"},
        expected_row_count=1,
        row_slots={
            "entity_1": {"name": "Alice"}  # missing age
        },
    )

    ctx = StepContext(state=base_state, deps=base_deps, inputs=None)

    next_node = await join_step(ctx)
    assert next_node == "wide_search_joiner"
    assert "missing required column: 'age'" in base_state.validation_feedback


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_wide_search_joiner_replan(base_state, base_deps):
    base_state.workboard = WideSearchWorkboard(
        schema_definition={"name": "str", "age": "int"},
        expected_row_count=1,
        row_slots={"entity_1": {"name": "Alice"}},
    )
    base_state.validation_feedback = "missing required column: 'age'"

    ctx = StepContext(state=base_state, deps=base_deps, inputs=None)

    # Mock repair agent to return "research" (needs replan)
    with patch("agent_utilities.graph.verification.Agent") as mock_agent:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = AsyncMock(
            output="I need more research to find age"
        )
        mock_agent.return_value = mock_instance

        next_node = await wide_search_joiner_step(ctx)

        assert next_node == "planner"
        assert base_state.needs_replan is True


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_wide_search_joiner_success(base_state, base_deps):
    base_state.workboard = WideSearchWorkboard()
    base_state.validation_feedback = "minor formatting issue"

    ctx = StepContext(state=base_state, deps=base_deps, inputs=None)

    # Mock repair agent to return a normal fix message
    with patch("agent_utilities.graph.verification.Agent") as mock_agent:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = AsyncMock(output="I have fixed the schema")
        mock_agent.return_value = mock_instance

        next_node = await wide_search_joiner_step(ctx)

        assert next_node == "dispatcher"
        assert base_state.validation_feedback is None
