from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.pipeline import IntelligencePipeline
from agent_utilities.models.knowledge_graph import (
    PipelineConfig,
    RegistryEdgeType,
    RegistryNodeType,
)


@pytest.fixture
def mock_graph():
    graph = nx.MultiDiGraph()
    # Add an agent
    graph.add_node(
        "TestBot",
        type=RegistryNodeType.AGENT,
        name="TestBot",
        description="A test bot",
        agent_type="prompt",
    )
    # Add a tool
    graph.add_node(
        "tool:search",
        type=RegistryNodeType.TOOL,
        name="search",
        description="Search tool",
        mcp_server="TestBot",
    )
    # Link them
    graph.add_edge("TestBot", "tool:search", type=RegistryEdgeType.PROVIDES)
    return graph


@pytest.mark.asyncio
async def test_intelligence_pipeline_mock(tmp_path):
    config = PipelineConfig(
        workspace_path=str(tmp_path),
        persist_to_ladybug=False,
        enable_embeddings=False,
    )

    mock_agent = MagicMock()
    mock_agent.name = "TestBot"
    mock_agent.description = "desc"
    mock_agent.agent_type = "prompt"
    mock_agent.system_prompt = "prompt"
    mock_agent.endpoint_url = None
    mock_agent.tool_count = 0

    mock_registry = MagicMock()
    mock_registry.agents = [mock_agent]
    mock_registry.tools = []

    with patch(
        "agent_utilities.graph.config_helpers.get_discovery_registry",
        return_value=mock_registry,
    ):
        pipeline = IntelligencePipeline(config)
        metadata = await pipeline.run()
        assert metadata.node_count > 0
        assert pipeline.graph.number_of_nodes() == metadata.node_count


@pytest.mark.asyncio
async def test_intelligence_engine_queries(mock_graph):
    engine = IntelligenceGraphEngine(mock_graph)

    # Test tool to agent mapping
    agents = engine.find_agent_for_tool("search")
    assert "TestBot" in agents

    # Test agent to tools mapping
    tools = engine.get_agent_tools("TestBot")
    assert "search" in tools


def test_intelligence_shortest_path(mock_graph):
    mock_graph.add_node(
        "T2", type=RegistryNodeType.TOOL, name="T2", mcp_server="TestBot"
    )
    mock_graph.add_edge("tool:search", "T2", type=RegistryEdgeType.DEPENDS_ON)

    engine = IntelligenceGraphEngine(mock_graph)
    path = engine.get_shortest_path("TestBot", "T2")
    assert path == ["TestBot", "tool:search", "T2"]


@pytest.mark.asyncio
async def test_memory_operations():
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph)

    # Add
    content = "User prefers dark mode"
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "testuuid"
        mem_id = engine.add_memory(content, name="PrefTest", category="preference")
        assert mem_id == "mem:testuuid"
        assert mem_id in graph
        assert graph.nodes[mem_id]["description"] == content

    # Search
    results = engine.search_memories("dark mode")
    assert len(results) == 1
    assert results[0]["id"] == mem_id

    # Update
    engine.update_memory(mem_id, importance=0.9)
    assert graph.nodes[mem_id]["importance"] == 0.9

    # Delete
    engine.delete_memory(mem_id)
    assert mem_id not in graph
