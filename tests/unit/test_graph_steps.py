import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_utilities.graph import steps
from agent_utilities.graph.state import GraphState

@pytest.mark.asyncio
async def test_usage_guard_step():
    ctx = MagicMock()
    ctx.state = GraphState(query="test query")
    ctx.deps = MagicMock()
    ctx.deps.event_queue = None

    result = await steps.usage_guard_step(ctx)
    assert result == "router"

@pytest.mark.asyncio
async def test_router_step_fast_path():
    ctx = MagicMock()
    ctx.state = GraphState(query="hello")
    ctx.deps = MagicMock()
    ctx.deps.event_queue = None

    # Mock fast_agent.run
    mock_resp = MagicMock()
    mock_resp.output = "hi there"

    # Mock Agent class
    with patch("agent_utilities.graph.steps.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock(return_value=mock_resp)

        result = await steps.router_step(ctx)

    assert type(result).__name__ == "End"
    assert result.data.results["output"] == "hi there"  # type: ignore[union-attr]

@pytest.mark.asyncio
async def test_synthesizer_step():
    ctx = MagicMock()
    ctx.state = GraphState(query="test query")
    ctx.state.results_registry = {"node1": "result1"}
    ctx.deps = MagicMock()
    ctx.deps.verifier_timeout = 60

    # Mock synthesizer agent run_stream
    mock_stream = MagicMock()

    async def mock_stream_text(delta: bool = False):
        # delta=True → incremental chunk; delta=False → accumulated text so far.
        # Single-chunk mock: both modes yield the same content.
        if delta:
            yield "composition"
        else:
            yield "composition"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="final answer")

    class MockStreamContext:
        async def __aenter__(self):
            return mock_stream
        async def __aexit__(self, *args: object) -> bool:
            return False

    ctx.deps.agent_model = MagicMock()
    with patch("agent_utilities.graph.steps.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run_stream.return_value = MockStreamContext()

        result = await steps.synthesizer_step(ctx)

    assert type(result).__name__ == "End"
    assert result.data.results["output"] == "final answer"
