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
