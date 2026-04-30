import networkx as nx
import pytest

from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.models.codemap import CodemapArtifact


@pytest.fixture
def sample_graph():
    g = nx.MultiDiGraph()
    # Add nodes with metadata
    g.add_node("file1.py", type="file", name="file1.py", centrality=0.9)
    g.add_node("func1", type="function", name="func1", file="file1.py", centrality=0.8)
    g.add_node("file2.py", type="file", name="file2.py", centrality=0.3)
    g.add_node("func2", type="function", name="func2", file="file2.py", centrality=0.2)

    # Add edges
    g.add_edge("file1.py", "func1", type="contains")
    g.add_edge("func1", "func2", type="calls")
    return g

@pytest.mark.asyncio
async def test_extract_focused_subgraph(sample_graph):
    engine = IntelligenceGraphEngine(sample_graph)

    # Search for "func1"
    subgraph = await engine.extract_focused_subgraph(query="func1", max_nodes=10)

    # Should include func1 and its related nodes
    node_ids = [n["id"] for n in subgraph.nodes]
    assert "func1" in node_ids
    assert "file1.py" in node_ids
    assert "func2" in node_ids # func1 calls func2

@pytest.mark.asyncio
async def test_codemap_persistence(sample_graph):
    engine = IntelligenceGraphEngine(sample_graph)

    artifact = CodemapArtifact(
        id="test-codemap",
        prompt="test prompt",
        mode="fast",
        hierarchy=[]
    )

    # Store
    await engine.store_codemap(artifact)

    # Retrieve
    retrieved = await engine.get_codemap_by_id("test-codemap")
    assert retrieved is not None
    assert retrieved.id == "test-codemap"
    assert retrieved.prompt == "test prompt"

@pytest.mark.asyncio
async def test_hybrid_search(sample_graph):
    engine = IntelligenceGraphEngine(sample_graph)
    results = engine.search_hybrid("file1")
    assert len(results) > 0
    assert results[0]["id"] == "file1.py"
