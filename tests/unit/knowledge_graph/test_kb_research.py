"""CONCEPT:AU-KG.research.research-pipeline-runner"""

from unittest.mock import MagicMock, NonCallableMagicMock, patch

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
        # ``GraphComputeEngine.__init__`` does NOT use this top-level mocked
        # client directly -- it rewraps it via ``_sync_client_view()``
        # (graph_compute.py), which reconstructs each client namespace via
        # ``type(namespace)(self)`` where ``namespace = getattr(base, name)``
        # and ``base = sync_client._client`` -- a DIFFERENT, separately
        # auto-vivified ``MagicMock`` attribute chain (``connected._client
        # .tenants``, not ``connected.tenants``) that a shallow pin never
        # reaches. A bare ``MagicMock`` auto-vivifies ``.tenants`` as itself
        # callable, so the reconstructed namespace mis-resolves as a leaf and
        # ``.tenants.create`` breaks with "'function' object has no attribute
        # 'create'" (D-GS1-3). Pin BOTH the shallow and the ``._client``-nested
        # chain to a non-callable mock so both legs round-trip as the
        # namespace the real client exposes.
        connected = mock_client.connect.return_value
        connected.tenants = NonCallableMagicMock()
        connected.tenants.list.return_value = []
        connected._client.tenants = NonCallableMagicMock()
        connected._client.tenants.list.return_value = []
        # ``epistemic_graph.client.SyncEpistemicGraphClient`` is patched as a
        # whole CLASS here, so ``_sync_client_view()``'s own
        # ``SyncEpistemicGraphClient(async_view, loop, thread)`` construction
        # call (graph_compute.py) does not build a real instance -- it returns
        # ``mock_client.return_value``, a THIRD auto-vivifying mock object
        # distinct from both ``connected`` and ``connected._client`` above.
        # This is the object every downstream ``self._client.tenants...``
        # access on the reconstructed view actually resolves against; pin it
        # too.
        mock_client.return_value.tenants = NonCallableMagicMock()
        mock_client.return_value.tenants.list.return_value = []
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
    assert src.node_id == "src:1"  # GraphNode.node_id has alias="id" (input-only)
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
    """Test that topics are linked to Policies/ProcessFlows via similarity.

    The engine's native Cypher write subset supports only one leading MATCH
    and MERGE on a single bare node
    (epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184), so an inline
    ``vector.similarity(...)`` WHERE clause paired with a cross-node MERGE
    never executes -- ``link_topics_to_policies_and_processes`` computes
    similarity in Python (like ``_similar_concept_pairs``' numpy fallback)
    and links through the typed engine API instead.
    """
    mock_backend = MagicMock()

    def _execute(query, params=None):
        q = " ".join(query.split())
        if "MATCH (t:KnowledgeBaseTopic) WHERE t.embedding" in q:
            return [{"id": "topic:1", "embedding": [0.1, 0.2, 0.3]}]
        if "MATCH (p:Policy) WHERE p.embedding" in q:
            return [{"id": "policy:1", "embedding": [0.1, 0.2, 0.31]}]
        # GROUNDED_IN/REFERENCES existing-link lookups, ProcessFlow read,
        # and the downstream typed-edge label lookups/MERGE -> no rows.
        return []

    mock_backend.execute.side_effect = _execute

    GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    with patch(
        "agent_utilities.knowledge_graph.core.engine.cosine_similarity",
        return_value=0.99,
    ):
        linked = maintainer.link_topics_to_policies_and_processes()

    assert linked == 1
    # The typed edge write reaches the backend as a GROUNDED_IN MERGE (the
    # portable fallback for a non-native mock backend).
    mock_backend.execute.assert_called()
    assert any(
        "GROUNDED_IN" in str(call.args[0])
        for call in mock_backend.execute.call_args_list
    )
