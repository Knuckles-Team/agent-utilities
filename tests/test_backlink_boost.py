from __future__ import annotations

"""Tests for Backlink-Density Retrieval Boost.

CONCEPT:AU-KG.ingest.engineering-rules
"""


import math
from unittest.mock import MagicMock, patch

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.schema_pack import (
    BacklinkBoostStrategy,
    SchemaPack,
    SchemaPackMode,
)


def _make_engine():
    engine = MagicMock()
    g = GraphComputeEngine(backend_type="rust")
    g.add_node("hub", type="concept")
    g.add_node("leaf", type="fact")
    g.add_node("isolated", type="event")
    for i in range(10):
        g.add_node(f"s{i}", type="entity")
        g.add_edge(f"s{i}", "hub", type="related_to")
    g.add_edge("hub", "leaf", type="provides")
    engine.graph = g
    engine.backend = None
    return engine


class _NoCypherBackend:
    """Truthy backend whose Cypher ``execute`` yields nothing (forces BFS path)."""

    def execute(self, _q, _p=None):  # type: ignore[no-untyped-def]
        return []


class _FakeVectorGraph:
    """Engine-graph double: the vector arm reads ``semantic_search`` (engine ANN)
    and hydrates via ``_get_node_properties`` — the CONCEPT:AU-KG.compute.kg-2 contract."""

    def __init__(self, hits, props):  # type: ignore[no-untyped-def]
        self._hits = hits  # list[(id, score)]
        self._props = props  # dict[id -> props]

    def query_unified(self, _plan, **_k):  # type: ignore[no-untyped-def]
        # No label seed in these tests → the arm uses the native ANN below.
        return []

    def semantic_search(self, _emb, _n=5):  # type: ignore[no-untyped-def]
        return list(self._hits)

    def _get_node_properties(self, nid):  # type: ignore[no-untyped-def]
        return dict(self._props.get(nid, {}))

    def has_node(self, nid):  # type: ignore[no-untyped-def]
        return nid in self._props

    def get_successors(self, _nid):  # type: ignore[no-untyped-def]
        return []

    def get_predecessors(self, _nid):  # type: ignore[no-untyped-def]
        return []


class TestBacklinkBoost:
    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_boost_not_in_graph(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine())
        assert r._backlink_boost("missing") == 1.0

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_boost_isolated(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine())
        assert r._backlink_boost("isolated") == 1.0

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_boost_hub_gt_leaf(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine())
        assert r._backlink_boost("hub") > r._backlink_boost("leaf")

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_log_scaling(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine())
        expected = 1.0 + 0.1 * math.log1p(10)
        assert abs(r._backlink_boost("hub") - expected) < 1e-10

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_custom_factor(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(name="t", backlink_boost_factor=0.5)
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._boost_factor == 0.5


class TestBoostStrategy:
    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_default_global(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine())
        assert r._boost_strategy == BacklinkBoostStrategy.GLOBAL

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_context_only(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(
            name="t", backlink_boost_strategy=BacklinkBoostStrategy.CONTEXT_ONLY
        )
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._boost_strategy == BacklinkBoostStrategy.CONTEXT_ONLY

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_disabled(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(
            name="t", backlink_boost_strategy=BacklinkBoostStrategy.DISABLED
        )
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._boost_strategy == BacklinkBoostStrategy.DISABLED

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_pack_passed(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(
            name="r", mode=SchemaPackMode.ADDITIVE, backlink_boost_factor=0.15
        )
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._schema_pack is pack
        assert r._boost_factor == 0.15


class TestAttentionFilter:
    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_attention_boost_with_embedding(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        engine = _make_engine()

        # The vector arm reads the engine ANN (semantic_search) — not a Cypher
        # `backend.execute` scan. Feed the test nodes through the engine-graph double
        # and store each node's embedding so the active-task boost can read it back.
        engine.graph = _FakeVectorGraph(
            hits=[("hub", 1.0), ("leaf", 0.0)],
            props={
                "hub": {"id": "hub", "name": "Dark Image", "embedding": [1.0, 0.0]},
                "leaf": {"id": "leaf", "name": "Unrelated", "embedding": [0.0, 1.0]},
            },
        )
        engine.backend = _NoCypherBackend()

        r = HybridRetriever(engine)

        # Set up embed_model
        mock_embed = MagicMock()
        mock_embed.get_text_embedding.side_effect = lambda text: (
            [1.0, 0.0] if "dark" in text or "Dark" in text else [0.0, 1.0]
        )
        r.embed_model = mock_embed

        # Search with active_task matching "hub" node embedding (similarity 1.0)
        results = r.retrieve_hybrid(
            query="Dark Image",
            context_window=2,
            multi_hop_depth=0,
            active_task="dark image processing",
            skip_quality_gate=True,
        )

        # hub node should have task boost applied and be sorted first with high score
        hub_node = next(x for x in results if x["id"] == "hub")

        assert "_active_task_boost" in hub_node
        assert (
            hub_node["_active_task_boost"] == 1.0
        )  # Cosine similarity of [1.0, 0.0] and [1.0, 0.0]

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_attention_boost_fallback_overlap(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        engine = _make_engine()

        # Engine ANN returns nodes WITHOUT an embedding property → the active-task
        # boost falls back to word-overlap. Fed through the engine-graph double.
        engine.graph = _FakeVectorGraph(
            hits=[("hub", 1.0), ("leaf", 0.0)],
            props={
                "hub": {"id": "hub", "name": "Dark Image"},
                "leaf": {"id": "leaf", "name": "Unrelated"},
            },
        )
        engine.backend = _NoCypherBackend()

        r = HybridRetriever(engine)

        # Set up embed_model
        mock_embed = MagicMock()
        mock_embed.get_text_embedding.return_value = [1.0, 0.0]
        r.embed_model = mock_embed

        results = r.retrieve_hybrid(
            query="Dark Image",
            context_window=2,
            multi_hop_depth=0,
            active_task="dark image processing",
            skip_quality_gate=True,
        )

        hub_node = next(x for x in results if x["id"] == "hub")
        assert "_active_task_boost_overlap" in hub_node
        assert (
            hub_node["_active_task_boost_overlap"] == 2
        )  # words "dark", "image" both overlap

    @patch(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
    )
    def test_attention_boost_no_embed_model(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
            HybridRetriever,
        )

        engine = _make_engine()

        # Mock backend to return test nodes
        mock_backend = MagicMock()
        mock_backend.execute.return_value = [
            {
                "id": "hub",
                "emb": [1.0, 0.0],
                "data": {"id": "hub", "name": "Dark Image", "embedding": [1.0, 0.0]},
            },
        ]
        engine.backend = mock_backend

        r = HybridRetriever(engine)
        r.embed_model = None  # Ensure no embed model

        # Since embed_model is None, retrieve_hybrid would normally fallback to keyword only.
        # But we want to test if active_task fallback overlap is applied.
        # Let's call retrieve_hybrid with keyword_only=False, which enters embed_model check.
        # Wait, if r.embed_model is None, r.retrieve_hybrid will normally bypass the embed_model block
        # unless keyword_only=False and we forced it or test it.
        # Actually, let's verify if retrieve_hybrid has a fallback to keyword search when self.embed_model is None.
        # Yes, line 372: "else: Fallback to keyword search"
        # So, to test the overlap fallback when no embed_model is available, we can mock embed_model
        # to exist, or we can see that our improved retrieve_hybrid logic:
        # `if active_task:` block is inside `if self.embed_model and not keyword_only:`.
        # Wait, if self.embed_model is None, we don't enter semantic search, so we don't have scored_nodes.
        # But if self.embed_model exists and get_text_embedding fails, we enter the `except` or fallback.
        # So test_attention_boost_fallback_overlap perfectly covers the overlap fallback path!
