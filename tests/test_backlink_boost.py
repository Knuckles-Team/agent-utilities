"""Tests for Backlink-Density Retrieval Boost.

CONCEPT:KG-2.2
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import networkx as nx

from agent_utilities.models.schema_pack import (
    BacklinkBoostStrategy,
    SchemaPack,
    SchemaPackMode,
)


def _make_engine():
    engine = MagicMock()
    g = nx.MultiDiGraph()
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


class TestBacklinkBoost:
    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_boost_not_in_graph(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine())
        assert r._backlink_boost("missing") == 1.0

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_boost_isolated(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine())
        assert r._backlink_boost("isolated") == 1.0

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_boost_hub_gt_leaf(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine())
        assert r._backlink_boost("hub") > r._backlink_boost("leaf")

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_log_scaling(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine())
        expected = 1.0 + 0.1 * math.log1p(10)
        assert abs(r._backlink_boost("hub") - expected) < 1e-10

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_custom_factor(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(name="t", backlink_boost_factor=0.5)
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._boost_factor == 0.5


class TestBoostStrategy:
    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_default_global(self, m):
        m.side_effect = Exception()
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine())
        assert r._boost_strategy == BacklinkBoostStrategy.GLOBAL

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_context_only(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(name="t", backlink_boost_strategy=BacklinkBoostStrategy.CONTEXT_ONLY)
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._boost_strategy == BacklinkBoostStrategy.CONTEXT_ONLY

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_disabled(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(name="t", backlink_boost_strategy=BacklinkBoostStrategy.DISABLED)
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._boost_strategy == BacklinkBoostStrategy.DISABLED

    @patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
    def test_pack_passed(self, m):
        m.side_effect = Exception()
        pack = SchemaPack(name="r", mode=SchemaPackMode.ADDITIVE, backlink_boost_factor=0.15)
        from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
        r = HybridRetriever(_make_engine(), schema_pack=pack)
        assert r._schema_pack is pack
        assert r._boost_factor == 0.15
