from unittest.mock import AsyncMock, MagicMock, patch

# CONCEPT:AU-ORCH.execution.inject-signal-board-observations — Swarm Orchestration & Specialist Factory
# CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort — Task Prioritization
# CONCEPT:AU-ORCH.planning.recursion-nesting-depth — Guardrails & Safety Patterns
# CONCEPT:AU-AHE.harness.harness-evolution — Exception Handling & Recovery
import pytest

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
async def test_router_step_never_returns_end():
    """CONCEPT:AU-ORCH.execution.direct-completion-shape — the router must NEVER terminate the graph with ``End``: a second
    router→end edge made pydantic-graph BROADCAST-FORK the router to {__end__, dispatcher},
    killing every full-graph turn at __end__. Trivial turns are now answered OUTSIDE the graph
    (``_run_direct_completion``), so they never reach router_step; when it does run it always
    routes ONWARD (a node-id str or a GraphPlan), never End."""
    from pydantic_graph import End

    ctx = MagicMock()
    ctx.state = GraphState(query="hello")
    ctx.deps = MagicMock()
    ctx.deps.event_queue = None

    mock_resp = MagicMock()
    mock_resp.output = "hi there"

    with patch("agent_utilities.graph._router_impl.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock(return_value=mock_resp)

        result = await steps.router_step(ctx)

    assert not isinstance(result, End)  # routes onward, never terminates the graph


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
    with patch("agent_utilities.graph.verification.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run_stream.return_value = MockStreamContext()

        result = await steps.synthesizer_step(ctx)

    assert type(result).__name__ == "End"
    assert result.data.results["output"] == "final answer"
