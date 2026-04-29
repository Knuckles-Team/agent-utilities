import pytest
from unittest.mock import MagicMock, AsyncMock
from agent_utilities.graph.hsm import (
    register_on_enter_hook,
    register_on_exit_hook,
    on_enter_specialist,
    on_exit_specialist,
    run_orthogonal_regions,
    assert_state_valid,
    static_route_query,
    StateInvariantError
)
from agent_utilities.graph.state import GraphState

@pytest.mark.asyncio
async def test_hsm_hooks():
    enter_hook = AsyncMock()
    exit_hook = AsyncMock()

    register_on_enter_hook(enter_hook)
    register_on_exit_hook(exit_hook)

    deps = MagicMock()
    # Initialize _entry_times to avoid MagicMock math errors
    deps._entry_times = {}

    state = GraphState(query="test")
    state.node_history = []

    await on_enter_specialist(deps, state, "test_agent", "test_server")
    assert "test_agent" in state.node_history
    enter_hook.assert_called()

    await on_exit_specialist(deps, state, "test_agent", True, "test_server")
    exit_hook.assert_called()

@pytest.mark.asyncio
async def test_run_orthogonal_regions():
    agent = MagicMock()
    # Mock the return value of agent.run()
    mock_run_res = MagicMock()
    mock_run_res.output = "Result"
    agent.run = AsyncMock(return_value=mock_run_res)

    queries = ["query1", "query2"]
    results = await run_orthogonal_regions(agent, queries, agent_name="test")

    assert len(results) == 2
    assert results["query1"] == "Result"
    assert results["query2"] == "Result"

def test_assert_state_valid():
    state = GraphState(query="")
    with pytest.raises(StateInvariantError, match="Empty query"):
        assert_state_valid(state, "test")

    state.query = "valid"
    state.step_cursor = -1
    with pytest.raises(StateInvariantError, match="Negative step cursor"):
        assert_state_valid(state, "test")

    state.step_cursor = 0
    state.global_research_loops = 6
    with pytest.raises(StateInvariantError, match="Infinite re-plan loop"):
        assert_state_valid(state, "test")

def test_static_route_query():
    specialists = {"researcher": "Search for info", "git_expert": "Git operations"}

    assert static_route_query("Please run researcher", specialists) == "researcher"
    assert static_route_query("Use git expert here", specialists) == "git_expert"
    assert static_route_query("Unknown task", specialists) is None
