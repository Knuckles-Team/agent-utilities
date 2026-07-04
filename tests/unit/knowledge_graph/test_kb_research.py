"""CONCEPT:AU-KG.research.research-pipeline-runner"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.graph.models import (
    Concept,
    Evidence,
    Source,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer


@pytest.fixture(autouse=True)
def mock_epistemic_graph_client():
    with patch("epistemic_graph.client.SyncEpistemicGraphClient") as mock_client:
        yield mock_client


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
    assert src.id == "src:1"
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
    GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(backend=mock_backend)
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

    # Query-aware mock: the consolidated merge re-points typed edges, merges
    # node properties, records provenance, then deletes — so a fixed-length
    # side_effect list no longer suffices. Return data by query shape instead.
    def _execute(query, params=None):
        q = " ".join(query.split())
        if "c.embedding IS NOT NULL" in q:
            return [
                {"id": "c1", "name": "Global Warming", "embedding": [0.1, 0.2, 0.3]},
                {"id": "c2", "name": "Climate Change", "embedding": [0.11, 0.21, 0.31]},
            ]
        if "RETURN properties(old) AS old_props" in q:
            return [
                {
                    "old_props": {"id": "c2", "name": "Climate Change"},
                    "new_props": {"id": "c1", "name": "Global Warming"},
                }
            ]
        # Edge-enumeration queries, provenance, delete, init checks -> no rows.
        return []

    mock_backend.execute.side_effect = _execute

    GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    with patch(
        "agent_utilities.knowledge_graph.core.engine.cosine_similarity",
        return_value=0.99,
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
    GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    # This tests the link_topics_to_policies_and_processes logic (extended for general topics)
    # Since we use vector.similarity in Cypher, we just check if it runs
    maintainer.link_topics_to_policies_and_processes()

    # Should have executed a query with vector.similarity
    mock_backend.execute.assert_called()
    last_query = mock_backend.execute.call_args[0][0]
    assert "vector.similarity" in last_query
