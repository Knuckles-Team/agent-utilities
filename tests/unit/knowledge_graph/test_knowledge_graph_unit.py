"""CONCEPT:AU-KG.query.object-graph-mapper"""

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.models.codemap import CodemapArtifact


@pytest.fixture
def sample_graph(engine_graph):
    g = engine_graph
    # Add nodes with metadata
    g.add_node("file1.py", node_type="file", name="file1.py", centrality=0.9)
    g.add_node(
        "func1", node_type="function", name="func1", file="file1.py", centrality=0.8
    )
    g.add_node("file2.py", node_type="file", name="file2.py", centrality=0.3)
    g.add_node(
        "func2", node_type="function", name="func2", file="file2.py", centrality=0.2
    )

    # Add edges
    g.add_edge("file1.py", "func1", relationship="contains")
    g.add_edge("func1", "func2", relationship="calls")
    return g


@pytest.fixture
def engine(sample_graph):
    """An IntelligenceGraphEngine bound to the SAME isolated tenant ``sample_graph``
    seeded -- a bare ``IntelligenceGraphEngine(db_path=":memory:")`` resolves its
    own backend independently and can land on a different/session-mismatched
    graph (PermissionError: "A graph-scoped view cannot retarget the verified
    GraphSession")."""
    eng = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=sample_graph.graph_name)
    )
    yield eng
    IntelligenceGraphEngine._ACTIVE_ENGINE = None


@pytest.mark.asyncio
async def test_extract_focused_subgraph(engine):
    # Search for "func1". No embedder in the hermetic unit suite (see
    # tests/unit/conftest.py's autouse _hermetic_embeddings), so retrieval
    # degrades to the keyword-only path, which stamps no per-result score --
    # skip_quality_gate opts out of the relevance gate that would otherwise
    # always reject it (composite=0.0) regardless of match quality.
    subgraph = await engine.extract_focused_subgraph(
        query="func1", max_nodes=10, skip_quality_gate=True
    )

    # Should include func1 and its related nodes
    node_ids = [n["id"] for n in subgraph.nodes]
    assert "func1" in node_ids
    assert "file1.py" in node_ids
    assert "func2" in node_ids  # func1 calls func2


@pytest.mark.asyncio
async def test_codemap_persistence(engine):
    artifact = CodemapArtifact(
        id="test-codemap",
        prompt_ref="pref:test",
        mode="fast",
        hierarchy=[],
    )

    # Store
    await engine.store_codemap(artifact)

    # Retrieve
    retrieved = await engine.get_codemap_by_id("test-codemap")
    assert retrieved is not None
    assert retrieved.id == "test-codemap"
    assert retrieved.prompt_ref == "pref:test"


@pytest.mark.asyncio
async def test_hybrid_search(engine):
    # No embedder in the hermetic unit suite -> keyword-only path, which
    # stamps no per-result score; skip the relevance gate that would
    # otherwise always reject a scoreless result (see
    # test_extract_focused_subgraph above).
    results = engine.search_hybrid("file1", skip_quality_gate=True)
    assert len(results) > 0
    assert results[0]["id"] == "file1.py"
