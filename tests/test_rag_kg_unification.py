#!/usr/bin/python
from __future__ import annotations

"""Tests for RAG-KG Unification, Research Orchestration, and Graph Distillation.

Covers:
- CONCEPT:KG-2.3 — KGNativeRetrievalRetriever (unified retrieval pipeline)
- CONCEPT:KG-2.6 — ResearchOrchestrator (orchestration integration)
- CONCEPT:KG-2.6 — GraphDistillationMigrator (similarity shortcut migration)
"""


import asyncio
import time
import uuid

import numpy as np

from agent_utilities.models.knowledge_graph import (
    DistillationIndexNode,
    OrchestrationCycleNode,
    RegistryEdgeType,
    RegistryNode,
    RegistryNodeType,
    SimilarityEdgeNode,
    UnifiedRAGConfigNode,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_node(name: str, embedding: list[float] | None = None) -> RegistryNode:
    """Create a test RegistryNode with optional embedding."""
    return RegistryNode(
        id=f"test_{uuid.uuid4().hex[:6]}",
        type=RegistryNodeType.MEMORY,
        name=name,
        description=f"Test node: {name}",
        embedding=embedding,
    )


def _random_embedding(dim: int = 64, seed: int | None = None) -> list[float]:
    """Generate a random unit-norm embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _similar_embedding(
    base: list[float], noise: float = 0.1, seed: int = 42
) -> list[float]:
    """Create a vector similar to base with controlled noise."""
    rng = np.random.default_rng(seed)
    arr = np.array(base) + rng.standard_normal(len(base)) * noise
    arr /= np.linalg.norm(arr)
    return arr.tolist()


# =====================================================================
# CONCEPT:KG-2.3 — KGNativeRetrievalRetriever Tests
# =====================================================================


class TestKGNativeRetrievalRetriever:
    """Tests for the KG-native unified retrieval pipeline."""

    def test_import(self):
        """Module imports cleanly."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()
        assert retriever is not None
        assert retriever.config.enable_similarity_shortcuts is True

    def test_config_defaults(self):
        """Default configuration values are sensible."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalConfig,
        )

        config = KGNativeRetrievalConfig()
        assert config.similarity_weight == 0.72
        assert config.keyword_weight == 0.28
        assert config.shortcut_boost == 1.15
        assert config.cluster_scope_threshold == 0.55

    def test_config_custom(self):
        """Custom configuration is applied."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalConfig,
            KGNativeRetrievalRetriever,
        )

        config = KGNativeRetrievalConfig(
            enable_cluster_scoping=False,
            similarity_weight=0.6,
            keyword_weight=0.4,
        )
        retriever = KGNativeRetrievalRetriever(config=config)
        assert retriever.config.enable_cluster_scoping is False
        assert retriever.config.similarity_weight == 0.6

    def test_build_cluster_index(self):
        """Cluster index builds from nodes with embeddings."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()

        # Create clusterable nodes (two groups)
        nodes = []
        for i in range(5):
            emb = _random_embedding(dim=16, seed=i)
            nodes.append(_make_node(f"group_a_{i}", embedding=emb))
        for i in range(5):
            emb = _random_embedding(dim=16, seed=100 + i)
            nodes.append(_make_node(f"group_b_{i}", embedding=emb))

        n_clusters = retriever.build_cluster_index(nodes)
        assert n_clusters >= 1

    def test_build_cluster_index_insufficient_nodes(self):
        """Cluster index handles <2 nodes gracefully."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()
        node = _make_node("solo", embedding=_random_embedding(dim=16))
        n_clusters = retriever.build_cluster_index([node])
        assert n_clusters == 0

    def test_build_similarity_index(self):
        """Similarity edge index builds from existing edges."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()

        edges = [
            SimilarityEdgeNode(
                id=f"sim_{i}",
                name=f"Edge {i}",
                type=RegistryNodeType.SIMILARITY_EDGE,
                source_node_id=f"node_{i}",
                target_node_id=f"node_{i + 1}",
                cosine_similarity=0.85,
                decay_lambda=0.01,
                current_weight=0.85,
                creation_epoch=time.time(),
                last_accessed_epoch=time.time(),
                access_count=0,
            )
            for i in range(5)
        ]

        count = retriever.build_similarity_index(edges)
        assert count == 5

    def test_retrieve_unified_keyword_only(self):
        """Unified retrieval works without embeddings (keyword fallback)."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()
        nodes = [
            _make_node("spectral clustering algorithm"),
            _make_node("neural network training"),
            _make_node("graph database indexing"),
        ]

        results = retriever.retrieve_unified(
            query="spectral clustering",
            nodes=nodes,
            top_k=3,
        )

        assert len(results) >= 1
        # Keyword match should rank spectral clustering first
        assert "spectral" in results[0][0].name.lower()

    def test_retrieve_unified_with_embeddings(self):
        """Unified retrieval with embeddings uses semantic scoring."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()

        base_emb = _random_embedding(dim=32, seed=42)
        nodes = [
            _make_node("target node", embedding=base_emb),
            _make_node("distant node", embedding=_random_embedding(dim=32, seed=99)),
        ]

        query_emb = _similar_embedding(base_emb, noise=0.05, seed=1)

        results = retriever.retrieve_unified(
            query="target",
            nodes=nodes,
            query_embedding=query_emb,
            top_k=2,
        )

        assert len(results) == 2
        assert results[0][1] > results[1][1]  # Target should score higher

    def test_stats_tracking(self):
        """Retrieval statistics are tracked."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            KGNativeRetrievalRetriever,
        )

        retriever = KGNativeRetrievalRetriever()
        nodes = [_make_node("test")]

        retriever.retrieve_unified("test", nodes, top_k=1)

        stats = retriever.stats
        assert "shortcut_hits" in stats
        assert "full_scan" in stats
        assert stats["full_scan"] >= 1

        retriever.reset_stats()
        assert retriever.stats["full_scan"] == 0


# =====================================================================
# CONCEPT:KG-2.6 — ResearchOrchestrator Tests
# =====================================================================


class TestResearchOrchestrator:
    """Tests for the research orchestration loop."""

    def test_import(self):
        """Module imports cleanly."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            ResearchOrchestrator,
        )

        orch = ResearchOrchestrator()
        assert orch is not None

    def test_config_defaults(self):
        """Default orchestration configuration is sensible."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            OrchestrationConfig,
        )

        config = OrchestrationConfig()
        assert config.max_papers_per_cycle == 50
        assert config.citation_depth == 2
        assert config.enable_similarity_linking is True
        assert config.cycle_interval_hours == 24

    def test_can_run_cycle_first_time(self):
        """First cycle is always allowed."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            ResearchOrchestrator,
        )

        orch = ResearchOrchestrator()
        assert orch.can_run_cycle() is True

    def test_can_run_cycle_rate_limit(self):
        """Rate limiting prevents immediate re-run."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            OrchestrationConfig,
            ResearchOrchestrator,
        )

        config = OrchestrationConfig(cycle_interval_hours=24)
        orch = ResearchOrchestrator(config=config)
        orch._last_cycle_time = time.time()  # Just ran
        assert orch.can_run_cycle() is False

    def test_orchestration_report_structure(self):
        """OrchestrationReport has all expected fields."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            OrchestrationReport,
        )

        report = OrchestrationReport(
            cycle_id="test_cycle",
            papers_discovered=10,
            papers_ingested=5,
            citations_traversed=20,
            similarity_edges_created=8,
            clusters_built=3,
        )

        assert report.cycle_id == "test_cycle"
        assert report.papers_discovered == 10
        assert report.similarity_edges_created == 8

    def test_run_research_cycle_no_engine(self):
        """Cycle runs gracefully without an engine (no-op pipeline)."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            ResearchOrchestrator,
        )

        orch = ResearchOrchestrator(engine=None)

        report = asyncio.get_event_loop().run_until_complete(
            orch.run_research_cycle(
                query="test",
                papers=[],  # Empty papers list = no-op
            )
        )

        assert report.cycle_id != ""
        assert report.duration_seconds >= 0

    def test_create_pipeline_runner(self):
        """Pipeline runner creation works."""
        from agent_utilities.knowledge_graph.orchestration.research_orchestrator import (
            ResearchOrchestrator,
        )

        orch = ResearchOrchestrator()
        runner = orch._create_pipeline_runner()
        assert runner is not None
        assert runner.config.max_papers_per_run == 50


# =====================================================================
# CONCEPT:KG-2.6 — GraphDistillationMigrator Tests
# =====================================================================


class TestGraphDistillationMigrator:
    """Tests for the graph distillation migration module."""

    def test_import(self):
        """Module imports cleanly."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()
        assert migrator is not None

    def test_distill_batch_empty(self):
        """Distilling empty batch returns zero stats."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()
        stats = migrator.distill_batch([])
        assert stats.nodes_processed == 0
        assert stats.edges_created == 0

    def test_distill_batch_creates_edges(self):
        """Distilling similar nodes creates similarity edges."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        base_emb = _random_embedding(dim=32, seed=42)
        nodes = [
            _make_node("node_a", embedding=base_emb),
            _make_node(
                "node_b", embedding=_similar_embedding(base_emb, noise=0.05, seed=1)
            ),
            _make_node(
                "node_c", embedding=_similar_embedding(base_emb, noise=0.05, seed=2)
            ),
        ]

        stats = migrator.distill_batch(nodes)
        assert stats.nodes_processed >= 2
        assert stats.edges_created >= 1

    def test_distill_batch_incremental(self):
        """Incremental distillation skips already-distilled nodes."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        base_emb = _random_embedding(dim=32, seed=42)
        nodes = [
            _make_node("node_a", embedding=base_emb),
            _make_node("node_b", embedding=_similar_embedding(base_emb, noise=0.05)),
        ]

        stats1 = migrator.distill_batch(nodes, incremental=True)
        stats2 = migrator.distill_batch(nodes, incremental=True)

        # Second batch should process 0 new nodes
        assert stats2.nodes_processed == 0

    def test_distilled_retrieve(self):
        """Distilled retrieval returns scored results."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        base_emb = _random_embedding(dim=32, seed=42)
        nodes = [
            _make_node("target node", embedding=base_emb),
            _make_node("other node", embedding=_random_embedding(dim=32, seed=99)),
        ]

        # Distill first
        migrator.distill_batch(nodes)

        # Retrieve
        results = migrator.distilled_retrieve(
            query="target",
            nodes=nodes,
            query_embedding=_similar_embedding(base_emb, noise=0.05),
            top_k=2,
        )

        assert len(results) >= 1
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], RegistryNode)
        assert isinstance(results[0][1], float)

    def test_coverage_report(self):
        """Coverage report reflects distillation state."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        # Empty state
        report = migrator.coverage_report()
        assert report.total_edges == 0
        assert "LOW COVERAGE" in report.recommendation or report.total_nodes == 0

    def test_coverage_report_with_data(self):
        """Coverage report reflects actual edge data."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        base_emb = _random_embedding(dim=32, seed=42)
        nodes = [
            _make_node(
                f"node_{i}", embedding=_similar_embedding(base_emb, noise=0.05, seed=i)
            )
            for i in range(10)
        ]

        migrator.distill_batch(nodes)
        report = migrator.coverage_report(nodes)

        assert report.total_nodes == 10
        assert report.total_edges >= 1
        assert report.recommendation != ""

    def test_migrate_existing(self):
        """Batch migration processes all nodes."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        nodes = [
            _make_node(f"node_{i}", embedding=_random_embedding(dim=16, seed=i))
            for i in range(15)
        ]

        all_stats = migrator.migrate_existing(nodes, batch_size=5)
        assert len(all_stats) == 3  # 15 nodes / 5 batch size

    def test_get_all_edges(self):
        """Edge accessor returns current edges."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
        )

        migrator = GraphDistillationMigrator()

        base_emb = _random_embedding(dim=32, seed=42)
        nodes = [
            _make_node("a", embedding=base_emb),
            _make_node("b", embedding=_similar_embedding(base_emb, noise=0.05)),
        ]

        migrator.distill_batch(nodes)
        edges = migrator.get_all_edges()
        assert isinstance(edges, list)

    def test_retriever_access(self):
        """Underlying retriever is accessible."""
        from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
            GraphDistillationMigrator,
            KGNativeRetrievalRetriever,
        )

        migrator = GraphDistillationMigrator()
        assert isinstance(migrator.retriever, KGNativeRetrievalRetriever)


# =====================================================================
# Pydantic Model Tests
# =====================================================================


class TestNewPydanticModels:
    """Tests for KG-2.38, KG-2.39, KG-2.40 Pydantic models."""

    def test_unified_rag_config_node(self):
        """UnifiedRAGConfigNode creates with defaults."""
        node = UnifiedRAGConfigNode(
            id="urag_1",
            name="Unified RAG Config",
        )
        assert node.type == RegistryNodeType.UNIFIED_RAG_CONFIG
        assert node.enable_similarity_shortcuts is True
        assert node.shortcut_hits == 0

    def test_orchestration_cycle_node(self):
        """OrchestrationCycleNode creates with all fields."""
        node = OrchestrationCycleNode(
            id="orch_1",
            name="Research Cycle 1",
            cycle_id="orch_abc",
            papers_discovered=10,
            papers_ingested=5,
            citations_traversed=20,
            similarity_edges_created=8,
            clusters_built=3,
            duration_seconds=12.5,
            query="spectral clustering",
        )
        assert node.type == RegistryNodeType.ORCHESTRATION_CYCLE
        assert node.papers_discovered == 10
        assert node.query == "spectral clustering"

    def test_distillation_index_node(self):
        """DistillationIndexNode creates with all fields."""
        node = DistillationIndexNode(
            id="dist_1",
            name="Distillation Index Snapshot",
            total_nodes=100,
            nodes_with_shortcuts=70,
            total_edges=150,
            coverage_ratio=0.7,
            avg_edge_weight=0.65,
            stale_edge_count=5,
            recommendation="HEALTHY: Good coverage.",
        )
        assert node.type == RegistryNodeType.DISTILLATION_INDEX
        assert node.coverage_ratio == 0.7
        assert "HEALTHY" in node.recommendation

    def test_new_edge_types(self):
        """New edge types are registered."""
        assert RegistryEdgeType.SHORTCUT_RETRIEVAL == "shortcut_retrieval"
        assert RegistryEdgeType.ORCHESTRATED_BY == "orchestrated_by"

    def test_new_node_types(self):
        """New node types are registered."""
        assert RegistryNodeType.UNIFIED_RAG_CONFIG == "unified_rag_config"
        assert RegistryNodeType.ORCHESTRATION_CYCLE == "orchestration_cycle"
        assert RegistryNodeType.DISTILLATION_INDEX == "distillation_index"
