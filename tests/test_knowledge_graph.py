import pytest
import asyncio
from unittest.mock import patch
from pathlib import Path
from agent_utilities.models.knowledge_graph import PipelineConfig, RegistryNodeType, RegistryEdgeType
from agent_utilities.knowledge_graph.pipeline import RegistryPipeline
from agent_utilities.knowledge_graph.engine import RegistryGraphEngine
from agent_utilities.workspace import get_agent_workspace

from agent_utilities.workspace import initialize_workspace, write_workspace_file

@pytest.fixture
def registry_config(tmp_path):
    # Initialize a dummy workspace in tmp_path
    with patch("agent_utilities.workspace.get_agent_workspace", return_value=tmp_path):
        initialize_workspace()
        # Add a dummy agent to NODE_AGENTS.md to satisfy the parser
        write_workspace_file(
            "NODE_AGENTS.md",
            "## Agent Mapping Table\n\n"
            "| Name | Description | Prompt File | Endpoint URL | Type | Capabilities | MCP Tools | Extra Config | Score |\n"
            "|------|-------------|-------------|--------------|------|--------------|-----------|--------------|-------|\n"
            "| TestBot | A test bot | main_agent.md | | prompt | search, coding | | | 0 |\n"
            "\n## Tool Inventory Table\n\n"
            "| Tool Name | Description | Tag | Source | Score | Approval |\n"
            "|-----------|-------------|-----|--------|-------|----------|\n"
            "| search | Search tool | tag | TestBot | 1 | none |\n",
        )

    return PipelineConfig(
        workspace_path=str(tmp_path),
        persist_to_ladybug=False,  # Skip DB for unit tests
        enable_embeddings=False,
    )

@pytest.mark.asyncio
async def test_registry_pipeline_run(registry_config):
    pipeline = RegistryPipeline(registry_config)
    metadata = await pipeline.run()

    assert metadata.node_count > 0
    assert metadata.agent_count > 0
    assert metadata.tool_count > 0
    assert pipeline.graph.number_of_nodes() == metadata.node_count

@pytest.mark.asyncio
async def test_registry_engine_queries(registry_config):
    pipeline = RegistryPipeline(registry_config)
    await pipeline.run()

    engine = RegistryGraphEngine(pipeline.graph)

    # Test tool to agent mapping
    # We'll find a tool from the graph first to ensure it exists
    tool_nodes = [n for n, d in pipeline.graph.nodes(data=True) if d.get("type") == RegistryNodeType.TOOL]
    if tool_nodes:
        tool_id = tool_nodes[0]
        tool_name = pipeline.graph.nodes[tool_id]["name"]
        agents = engine.find_agent_for_tool(tool_name)
        assert len(agents) > 0

    # Test agent to tools mapping
    agent_nodes = [n for n, d in pipeline.graph.nodes(data=True) if d.get("type") == RegistryNodeType.AGENT]
    if agent_nodes:
        agent_name = agent_nodes[0]
        tools = engine.get_agent_tools(agent_name)
        # Some agents might have 0 tools if they are prompt-only
        assert isinstance(tools, list)

def test_registry_shortest_path(registry_config):
    # This needs a synchronous wrapper or a separate graph
    pipeline = RegistryPipeline(registry_config)
    # Use a mock graph for path testing
    pipeline.graph.add_node("A", type=RegistryNodeType.AGENT)
    pipeline.graph.add_node("T1", type=RegistryNodeType.TOOL)
    pipeline.graph.add_node("T2", type=RegistryNodeType.TOOL)
    pipeline.graph.add_edge("A", "T1", type=RegistryEdgeType.PROVIDES)
    pipeline.graph.add_edge("T1", "T2", type=RegistryEdgeType.DEPENDS_ON)

    engine = RegistryGraphEngine(pipeline.graph)
    path = engine.get_shortest_path("A", "T2")
    assert path == ["A", "T1", "T2"]

@pytest.mark.asyncio
async def test_memory_operations():
    import networkx as nx
    from agent_utilities.knowledge_graph.engine import RegistryGraphEngine

    graph = nx.MultiDiGraph()
    # Mock ladybug if needed or just use memory
    engine = RegistryGraphEngine(graph)

    # Add
    content = "User prefers dark mode"
    mem_id = engine.add_memory(content, name="PrefTest", category="preference")
    assert mem_id.startswith("mem:")
    assert mem_id in graph
    assert graph.nodes[mem_id]['description'] == content

    # Search
    results = engine.search_memories("dark mode")
    assert len(results) == 1
    assert results[0]['id'] == mem_id

    # Update
    engine.update_memory(mem_id, importance=0.9)
    assert graph.nodes[mem_id]['importance'] == 0.9

    # Delete
    engine.delete_memory(mem_id)
    assert mem_id not in graph
