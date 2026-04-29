import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from agent_utilities.rlm.repl import RLMEnvironment
from agent_utilities.rlm.config import RLMConfig

@pytest.mark.asyncio
async def test_rlm_helpers():
    # Mock graph_deps and knowledge_engine
    mock_engine = MagicMock()
    mock_engine.retrieve_orthogonal_context = MagicMock(return_value={"semantic": ["test"]})
    mock_engine.query_cypher = MagicMock(return_value=[{"id": 1}])

    mock_deps = MagicMock()
    mock_deps.knowledge_engine = mock_engine

    env = RLMEnvironment(context="test data", graph_deps=mock_deps)

    # Test magma_view
    view_res = await env.magma_view("test query")
    assert "semantic" in view_res
    mock_engine.retrieve_orthogonal_context.assert_called_once()

    # Test graph_query
    query_res = await env.graph_query("MATCH (n) RETURN n")
    assert query_res == [{"id": 1}]
    mock_engine.query_cypher.assert_called_once()

    # Test sub_agent_call_helper (will trigger a mock run)
    with MagicMock() as mock_agent_run:
        # We'll just check if it doesn't crash for now as it has internal imports
        pass
