import pytest
from unittest.mock import MagicMock
from pydantic_ai.models.test import TestModel

from agent_utilities.graph.state import GraphState, GraphDeps
from pydantic_graph import End
from pydantic_graph.beta import StepContext
from agent_utilities.graph.steps import (
    usage_guard_step,
    verifier_step,
    dispatcher_step,
    router_step,
    planner_step,
)
from agent_utilities.graph.graph_models import ValidationResult
from agent_utilities.models import (
    GraphPlan,
    ExecutionStep,
)
from transformers.conversion_mapping import VLMS


@pytest.fixture
def mock_deps():
    return GraphDeps(
        tag_prompts={"test": "Test domain"},
        tag_env_vars={"test": "TESTTOOL"},
        mcp_toolsets=[],
        agent_model=TestModel(),
        router_model=TestModel(),
    )


def test_graph_deps_tool_guard_mode_regression():
    """Ensure GraphDeps has tool_guard_mode to prevent regressions."""
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[],
    )
    assert hasattr(deps, "tool_guard_mode")
    assert deps.tool_guard_mode in ["on", "off", "strict"]


@pytest.mark.asyncio
async def test_usage_guard_passes(mock_deps):
    """Test usage guard passes for a normal query."""
    state = GraphState(query="hello")
    # Mock StepContext
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    # We'll mock the Agent.run inside the usage_guard_step if possible,
    # or just let the TestModel return something that contains "PASS".
    # Since TestModel usually returns the prompt by default, we can set the model's call_index or similar.
    # Actually, simpler: patch the Agent.run

    class MockRes:
        output = "PASS"

    async def mock_run(*args, **kwargs):
        return MockRes()

    # We need to be careful with patching.
    # Let's try to set the router_model to return PASS if we can.
    # If not, patching is fine for unit tests.

    mock_deps.router_model = TestModel()

    import unittest.mock

    with unittest.mock.patch("pydantic_ai.Agent.run", new=mock_run):
        res = await usage_guard_step(ctx)
        assert res == "router"


@pytest.mark.asyncio
async def test_verifier_step_success(mock_deps):
    """Test verifier_step succeeds when validation score is high."""
    state = GraphState(query="test query")
    state.results_registry["node1"] = "execution result"
    state.plan = GraphPlan(steps=[ExecutionStep(node_id="node1", is_parallel=False)])

    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    from unittest.mock import AsyncMock, patch

    # Mock the validation result
    validation_result = ValidationResult(is_valid=True, score=0.9, feedback="Good")

    # Helper to create a mock stream
    def create_mock_stream(output):
        stream = AsyncMock()
        stream.__aenter__.return_value = stream
        async def mock_stream_text(*args, **kwargs):
            yield "chunk1"
        stream.stream_text = mock_stream_text
        stream.get_output = AsyncMock(return_value=output)
        return stream

    with patch("pydantic_ai.Agent.run_stream") as mock_run_stream:
        mock_run_stream.side_effect = [
            create_mock_stream(validation_result)
        ]
        res = await verifier_step(ctx)
        assert res == "synthesizer"
        assert "execution result" in state.results_registry["node1"]


@pytest.mark.asyncio
async def test_verifier_step_retry(mock_deps):
    """Test verifier_step triggers a re-plan when validation score is low."""
    state = GraphState(query="test query")
    state.results_registry["node1"] = "poor result"
    state.plan = GraphPlan(steps=[ExecutionStep(node_id="node1", is_parallel=False)])

    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    from unittest.mock import AsyncMock, patch

    # Mock the validation result
    validation_result = ValidationResult(is_valid=False, score=0.2, feedback="Too short")

    # Helper to create a mock stream
    def create_mock_stream(output):
        stream = AsyncMock()
        stream.__aenter__.return_value = stream
        async def mock_stream_text(*args, **kwargs):
            yield "chunk1"
        stream.stream_text = mock_stream_text
        stream.get_output = AsyncMock(return_value=output)
        return stream

    with patch("pydantic_ai.Agent.run_stream") as mock_run_stream:
        mock_run_stream.return_value = create_mock_stream(validation_result)
        res = await verifier_step(ctx)
        assert res == "planner"
        assert state.verification_attempts == 1
        assert state.validation_feedback == "Too short"


@pytest.mark.asyncio
async def test_dispatcher_step_sequential(mock_deps):
    """Test dispatcher_step correctly routes sequential steps."""
    state = GraphState(query="test")
    state.plan = GraphPlan(
        steps=[
            ExecutionStep(node_id="expert1", is_parallel=False),
            ExecutionStep(node_id="verifier", is_parallel=False),
        ]
    )
    state.step_cursor = 0
    state.deferred_events = []
    state.exploration_notes = "pre-loaded"

    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    res = await dispatcher_step(ctx)
    assert res == "parallel_batch_processor"
    assert state.pending_batch.tasks[0].node_id == "expert1"
    assert state.step_cursor == 1


@pytest.mark.asyncio
async def test_dispatcher_step_parallel(mock_deps):
    """Test dispatcher_step correctly groups parallel steps."""
    state = GraphState(query="test")
    state.plan = GraphPlan(
        steps=[
            ExecutionStep(node_id="expert1", is_parallel=True),
            ExecutionStep(node_id="expert2", is_parallel=True),
            ExecutionStep(node_id="verifier", is_parallel=False),
        ]
    )
    state.step_cursor = 0
    state.deferred_events = []
    state.exploration_notes = "pre-loaded"

    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    res = await dispatcher_step(ctx)
    assert res == "parallel_batch_processor"
    assert len(state.pending_batch.tasks) == 2
    assert state.pending_batch.tasks[0].node_id == "expert1"
    assert state.pending_batch.tasks[1].node_id == "expert2"

@pytest.mark.asyncio
async def test_router_step_fast_path(mock_deps):
    """Test router_step fast-path for trivial queries."""
    state = GraphState(query="hello")
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    from unittest.mock import AsyncMock, patch
    mock_res = MagicMock()
    mock_res.output = "Hi there!"

    with patch("pydantic_ai.Agent.run", new=AsyncMock(return_value=mock_res)):
        res = await router_step(ctx)
        assert isinstance(res, End)
        assert res.data.results["output"] == "Hi there!"
        assert res.data.metadata["fast_path"] is True


@pytest.mark.asyncio
async def test_router_step_planning(mock_deps):
    """Test router_step generates a plan for complex queries."""
    state = GraphState(query="Build a web app with react")
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    from unittest.mock import AsyncMock, patch
    mock_plan = GraphPlan(steps=[ExecutionStep(node_id="researcher")])

    # Mock stream for run_stream
    mock_stream = AsyncMock()
    mock_stream.__aenter__.return_value = mock_stream
    mock_stream.get_output.return_value = mock_plan
    mock_stream.usage.return_value = MagicMock()

    with patch("pydantic_ai.Agent.run_stream", return_value=mock_stream):
        with patch("agent_utilities.graph.steps.fetch_unified_context", return_value="Context"):
            from agent_utilities.graph.steps import router_step
            res = await router_step(ctx)
            assert res == "dispatcher"
            assert len(state.plan.steps) == 1
            assert state.plan.steps[0].node_id == "researcher"


@pytest.mark.asyncio
async def test_planner_step_retry(mock_deps):
    """Test planner_step generates a corrected plan after failure."""
    state = GraphState(query="test query")
    state.validation_feedback = "Previous attempt failed"
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    from unittest.mock import AsyncMock, patch
    mock_plan = GraphPlan(steps=[ExecutionStep(node_id="github_expert")])

    with patch("pydantic_ai.Agent.run", new=AsyncMock(return_value=MagicMock(output=mock_plan))):
        with patch("agent_utilities.graph.steps.fetch_unified_context", return_value="Context"):
            from agent_utilities.graph.steps import planner_step
            res = await planner_step(ctx)
            assert res == "dispatcher"
            assert state.plan.steps[0].node_id == "github_expert"
            assert state.step_cursor == 0
