from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from agent_utilities.graph.models import (
    Concept,
    Evidence,
    Source,
)
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.maintenance import GraphMaintainer


@pytest.mark.asyncio
async def test_kb_model_validation():
    """Test validation of Research KB models."""
    # Source
    src = Source(
        id="src:1",
        source_id="src:1",
        title="Impact of Climate on Health",
        doi="10.1234/nature.2026",
        authors=["Alice", "Bob"],
    )
    assert src.node_id == "src:1"
    assert "Source" in src.labels

    # Concept
    concept = Concept(
        id="con:p53",
        concept_id="con:p53",
        name="p53 Gene",
        definition="A tumor suppressor gene.",
        is_permanent=True,
    )
    assert concept.is_permanent is True

    # Evidence
    ev = Evidence(
        id="ev:1",
        evidence_id="ev:1",
        claim="p53 mutations lead to cancer.",
        confidence_score=0.95,
    )
    assert ev.confidence_score == 0.95


@pytest.mark.asyncio
async def test_pruning_with_permanent_flag():
    """Test that is_permanent flag protects nodes from pruning."""
    mock_backend = MagicMock()
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    # Mock execute for pruning query
    # We just verify the query includes the condition
    maintainer.prune_low_importance_nodes(threshold=0.2)

    # Check the call arguments
    args, kwargs = mock_backend.execute.call_args
    query = args[0]
    assert "n.is_permanent IS NULL OR n.is_permanent = False" in query
    assert "$threshold" in kwargs or "0.2" in str(args)


@pytest.mark.asyncio
async def test_concept_merging():
    """Test merging of similar concepts based on embeddings."""
    mock_backend = MagicMock()
    # Return two similar concepts
    mock_backend.execute.side_effect = [
        [
            {"id": "c1", "name": "Global Warming", "embedding": [0.1, 0.2, 0.3]},
            {"id": "c2", "name": "Climate Change", "embedding": [0.11, 0.21, 0.31]},
        ],
        None,  # For the delete query
    ]

    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    with patch(
        "agent_utilities.knowledge_graph.engine.cosine_similarity", return_value=0.99
    ):
        merged = maintainer.merge_similar_concepts(similarity_threshold=0.9)
        assert merged == 1
        # Verify delete was called for c2
        mock_backend.execute.assert_any_call(
            "MATCH (old:Concept {id: $old_id}) DETACH DELETE old", {"old_id": "c2"}
        )


@pytest.mark.asyncio
async def test_cross_domain_emergence():
    """Test that topics are linked via shared concepts or similarity."""
    mock_backend = MagicMock()
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    # This tests the link_topics_to_policies_and_processes logic (extended for general topics)
    # Since we use vector.similarity in Cypher, we just check if it runs
    maintainer.link_topics_to_policies_and_processes()

    # Should have executed a query with vector.similarity
    mock_backend.execute.assert_called()
    last_query = mock_backend.execute.call_args[0][0]
    assert "vector.similarity" in last_query
