import pytest
import networkx as nx
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend

@pytest.fixture
def graph_engine():
    backend = LadybugBackend(db_path=":memory:")
    backend.create_schema()
    engine = IntelligenceGraphEngine(graph=nx.MultiDiGraph(), backend=backend)
    return engine

def test_mcp_server_ingestion_and_discovery(graph_engine):
    """Test full cycle of MCP server ingestion, discovery, and spawning."""
    tools = [
        {"name": "get_weather", "description": "Get weather for a city", "capabilities": ["weather", "search"]},
        {"name": "post_tweet", "description": "Post a message to Twitter", "capabilities": ["social", "write"]}
    ]

    # 1. Ingest MCP Server
    graph_engine.ingest_mcp_server(
        name="WeatherTwitter",
        url="http://localhost:8000/mcp",
        tools=tools
    )

    # 2. Verify CallableResources exist
    res = graph_engine.query_cypher("MATCH (r:CallableResource) RETURN r.name as name, r.resource_type as type")
    assert len(res) == 2
    names = [r["name"] for r in res]
    assert "get_weather" in names
    assert "post_tweet" in names

    # 3. Discovery based on task
    discovered = graph_engine.find_relevant_callable_resources("What is the weather in London?")
    assert len(discovered) > 0
    assert "get_weather" in [d["name"] for d in discovered]

    # 4. Spawn agent with discovered tools
    tool_ids = [d["id"] for d in discovered]
    agent_id = graph_engine.spawn_specialized_agent(
        task_description="Check London weather and tweet it",
        tool_ids=tool_ids
    )

    # 5. Verify agent is linked to tools
    links = graph_engine.query_cypher("MATCH (a:SpawnedAgent {id: $aid})-[:USES]->(r:CallableResource) RETURN r.name as name", {"aid": agent_id})
    assert len(links) > 0
    assert "get_weather" in [l["name"] for l in links]

def test_mcp_metadata_linkage(graph_engine):
    """Test that ToolMetadata is correctly linked and queryable."""
    tools = [{"name": "search_docs", "description": "Search documentation", "tags": ["docs", "internal"]}]
    graph_engine.ingest_mcp_server("DocServer", "http://docs.local", tools)

    # Query via metadata tags
    query = """
    MATCH (r:CallableResource)-[:HAS_METADATA]->(m:ToolMetadata)
    WHERE 'docs' IN m.tags
    RETURN r.name as name
    """
    res = graph_engine.query_cypher(query)
    assert len(res) > 0
    assert res[0]["name"] == "search_docs"
