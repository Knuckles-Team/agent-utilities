from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.graph import runner
from agent_utilities.models import GraphResponse


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.run = AsyncMock()
    return graph

@pytest.mark.asyncio
async def test_run_graph_basic(mock_graph):
    # Mock graph.run result - it should return a GraphResponse
    mock_result = GraphResponse(
        status="completed",
        results={"output": "final answer"}
    )
    mock_graph.run.return_value = mock_result

    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None
    config = {"deps": deps}

    with patch("agent_utilities.graph.runner.load_node_agents_registry") as mock_reg:
        mock_reg.return_value.agents = []
        response = await runner.run_graph(
            mock_graph,
            config,
            query="hello",
            streamdown=False
        )

    assert response["status"] == "completed"
    assert response["results"]["output"] == "final answer"
    mock_graph.run.assert_called_once()

@pytest.mark.asyncio
async def test_run_graph_exception(mock_graph):
    mock_graph.run.side_effect = Exception("test error")

    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    config = {"deps": deps}

    with patch("agent_utilities.graph.runner.load_node_agents_registry") as mock_reg:
        mock_reg.return_value.agents = []
        response = await runner.run_graph(
            mock_graph,
            config,
            query="hello",
            streamdown=False
        )

    assert response["status"] == "error"
    assert "test error" in response["error"]

@pytest.mark.asyncio
async def test_run_graph_stream_basic(mock_graph):
    # Mock graph.run_stream
    mock_deps = MagicMock()
    mock_deps.mcp_toolsets = []
    mock_deps.tag_prompts = {}
    mock_deps.event_queue = None
    mock_deps.request_id = "test-run"

    config = {
        "deps": mock_deps,
        "router_model": "test",
        "agent_model": "test"
    }

    # Mock graph.run for the background task
    mock_graph.run = AsyncMock(return_value=GraphResponse(status="completed", results={"output": "bg done"}))

    class MockStreamedRun:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "_yielded"):
                raise StopAsyncIteration
            self._yielded = True
            # Return a serializable object
            mock_chunk = MagicMock()
            mock_chunk.timestamp = 123456789
            # pydantic-graph chunks often have a .data or similar.
            # In runner.py it seems it just yields the chunk serialized if it's an event.
            return {"type": "chunk", "content": "hello"}

        async def result(self):
            return GraphResponse(status="completed", results={"output": "stream done"})

    mock_graph.run_stream.return_value = MockStreamedRun()

    with patch("agent_utilities.graph.runner.load_node_agents_registry") as mock_reg, \
         patch("agent_utilities.graph.runner.create_model") as mock_model:
        mock_reg.return_value.agents = []
        mock_model.return_value = MagicMock()

        # We need to test run_graph_stream generator
        stream = runner.run_graph_stream(
            mock_graph,
            config,
            query="hello"
        )

        results = []
        async for chunk in stream:
            results.append(chunk)

    assert len(results) > 0
    # The runner might not include "stream done" in the SSE stream directly
    # if it's only in the final holder, but let's check what it yields.
    # It yields events from the queue.
    assert any("graph_start" in str(r) for r in results)
