#!/usr/bin/python
"""Graph Distillation Migration.

CONCEPT:KG-2.40 — Graph Distillation Migration

Migrates standard RAG retrieval pathways to use SimilarityEdgeNode
shortcuts for improved latency. Instead of performing O(N) cosine
similarity computation against all nodes at query time, this module:

1. Pre-computes similarity edges during ingestion (via AutoSimilarityLinker).
2. At retrieval time, walks the pre-computed edge graph for O(degree)
   shortcut retrieval.
3. Falls back to full-scan retrieval only when shortcut coverage is
   insufficient.

Architecture::

    ┌──────────────────────────────────────────────┐
    │           GraphDistillationMigrator           │
    ├──────────────────────────────────────────────┤
    │ distill_batch()      → Pre-compute edges     │
    │ distilled_retrieve() → Shortcut retrieval    │
    │ coverage_report()    → Index health check    │
    │ migrate_existing()   → Batch migration       │
    └──────────────────────────────────────────────┘

Integrates with:
- KG-2.36 (AutoSimilarityLinker): Edge creation
- KG-2.38 (UnifiedRAGKGRetriever): As the primary retrieval backend
- KG-2.34 (SpectralClusterNavigator): Cluster-scoped distillation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from agent_utilities.knowledge_graph.memory.auto_similarity import (
    AutoSimilarityLinker,
)
from agent_utilities.knowledge_graph.retrieval.unified_rag_kg import (
    UnifiedRAGKGConfig,
    UnifiedRAGKGRetriever,
)
from agent_utilities.models.knowledge_graph import (
    MemoryDecayConfig,
    RegistryNode,
    SimilarityEdgeNode,
)

logger = logging.getLogger(__name__)


@dataclass
class DistillationStats:
    """Statistics from a distillation batch.

    Attributes:
        nodes_processed: Total nodes evaluated.
        edges_created: New similarity edges created.
        edges_pruned: Stale edges removed.
        coverage_ratio: Fraction of nodes with at least one shortcut edge.
        avg_edges_per_node: Mean number of shortcut edges per node.
        duration_seconds: Processing time.
    """

    nodes_processed: int = 0
    edges_created: int = 0
    edges_pruned: int = 0
    coverage_ratio: float = 0.0
    avg_edges_per_node: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class CoverageReport:
    """Report on the health of the distillation index.

    Attributes:
        total_nodes: Total KG nodes with embeddings.
        nodes_with_shortcuts: Nodes that have at least one shortcut edge.
        total_edges: Total similarity edges in the index.
        coverage_ratio: nodes_with_shortcuts / total_nodes.
        avg_edge_weight: Mean decayed weight of all edges.
        stale_edge_count: Edges below prune threshold.
        recommendation: Suggested action based on coverage.
    """

    total_nodes: int = 0
    nodes_with_shortcuts: int = 0
    total_edges: int = 0
    coverage_ratio: float = 0.0
    avg_edge_weight: float = 0.0
    stale_edge_count: int = 0
    recommendation: str = ""


class GraphDistillationMigrator:
    """Migrates RAG retrieval to use pre-computed similarity shortcuts.

    CONCEPT:KG-2.40 — Graph Distillation Migration

    Pre-computes similarity edges across KG nodes to enable O(degree)
    retrieval instead of O(N) full-index scans. Manages the lifecycle
    of the distillation index including creation, pruning, and health
    monitoring.

    Example::

        migrator = GraphDistillationMigrator()

        # Pre-compute shortcuts for a batch of nodes
        stats = migrator.distill_batch(nodes)
        print(f"Created {stats.edges_created} shortcut edges")

        # Use shortcuts for retrieval
        results = migrator.distilled_retrieve(
            query="spectral clustering",
            nodes=nodes,
            query_embedding=emb,
        )

        # Check index health
        report = migrator.coverage_report()
        print(f"Coverage: {report.coverage_ratio:.0%}")
    """

    def __init__(
        self,
        decay_config: MemoryDecayConfig | None = None,
        retriever_config: UnifiedRAGKGConfig | None = None,
    ):
        """Initialize the graph distillation migrator.

        Args:
            decay_config: Configuration for similarity edge decay.
            retriever_config: Configuration for the unified retriever.
        """
        self._linker = AutoSimilarityLinker(config=decay_config)
        self._retriever = UnifiedRAGKGRetriever(config=retriever_config)

        # Edge storage
        self._all_edges: list[SimilarityEdgeNode] = []
        self._edge_index: dict[str, list[SimilarityEdgeNode]] = {}
        self._distilled_node_ids: set[str] = set()

    def distill_batch(
        self,
        nodes: list[RegistryNode],
        incremental: bool = True,
    ) -> DistillationStats:
        """Pre-compute similarity edges for a batch of nodes.

        This is the core distillation step. For each node, it computes
        cosine similarity against all other nodes and creates edges
        above the threshold.

        Args:
            nodes: KG nodes with embeddings to distill.
            incremental: If True, skip nodes already distilled.

        Returns:
            DistillationStats with creation and pruning counts.
        """
        start_time = time.time()
        stats = DistillationStats()

        # Filter to nodes with embeddings
        embeddable = [n for n in nodes if n.embedding]
        if not embeddable:
            return stats

        for i, node in enumerate(embeddable):
            if incremental and node.id in self._distilled_node_ids:
                continue

            # Compare against all prior nodes
            predecessors = embeddable[:i]
            if not incremental:
                predecessors = [n for n in embeddable if n.id != node.id]

            new_edges = self._linker.link_new_node(
                new_node=node,
                existing_nodes=predecessors,
            )

            # Store edges
            for edge in new_edges:
                self._all_edges.append(edge)
                self._index_edge(edge)

            stats.edges_created += len(new_edges)
            stats.nodes_processed += 1
            self._distilled_node_ids.add(node.id)

        # Prune stale edges
        if self._all_edges:
            kept, pruned = self._linker.prune_stale_edges(self._all_edges)
            stats.edges_pruned = len(pruned)
            self._all_edges = kept
            self._rebuild_index()

        # Compute coverage
        if embeddable:
            stats.coverage_ratio = len(self._distilled_node_ids) / len(embeddable)
        if self._distilled_node_ids:
            stats.avg_edges_per_node = len(self._all_edges) / len(
                self._distilled_node_ids
            )

        stats.duration_seconds = time.time() - start_time

        # Update unified retriever with new edge index
        self._retriever.build_similarity_index(self._all_edges)

        logger.info(
            "[CONCEPT:KG-2.40] Distillation batch: %d nodes, %d edges created, "
            "%d pruned, coverage=%.0f%%, %.1fs",
            stats.nodes_processed,
            stats.edges_created,
            stats.edges_pruned,
            stats.coverage_ratio * 100,
            stats.duration_seconds,
        )

        return stats

    def distilled_retrieve(
        self,
        query: str,
        nodes: list[RegistryNode],
        query_embedding: list[float] | None = None,
        top_k: int = 10,
    ) -> list[tuple[RegistryNode, float]]:
        """Retrieve using pre-computed similarity shortcuts.

        Falls back to the unified retriever's full pipeline, which
        leverages shortcut edges for acceleration.

        Args:
            query: Search query string.
            nodes: Candidate KG nodes.
            query_embedding: Pre-computed query embedding.
            top_k: Number of results to return.

        Returns:
            List of (node, score) tuples, sorted by descending score.
        """
        return self._retriever.retrieve_unified(
            query=query,
            nodes=nodes,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def coverage_report(
        self,
        nodes: list[RegistryNode] | None = None,
    ) -> CoverageReport:
        """Generate a health report for the distillation index.

        Args:
            nodes: Optional full node set for coverage calculation.

        Returns:
            CoverageReport with index health metrics.
        """
        report = CoverageReport()

        if nodes:
            embeddable = [n for n in nodes if n.embedding]
            report.total_nodes = len(embeddable)
            nodes_with_edges = set()
            for edge in self._all_edges:
                nodes_with_edges.add(edge.source_node_id)
                nodes_with_edges.add(edge.target_node_id)
            report.nodes_with_shortcuts = len(
                nodes_with_edges.intersection(n.id for n in embeddable)
            )
        else:
            report.total_nodes = len(self._distilled_node_ids)
            report.nodes_with_shortcuts = len(
                {e.source_node_id for e in self._all_edges}
                | {e.target_node_id for e in self._all_edges}
            )

        report.total_edges = len(self._all_edges)

        if report.total_nodes > 0:
            report.coverage_ratio = report.nodes_with_shortcuts / report.total_nodes

        # Compute average weight and stale count
        if self._all_edges:
            weights = [self._linker.decay_weight(e) for e in self._all_edges]
            report.avg_edge_weight = float(np.mean(weights))
            report.stale_edge_count = sum(
                1 for w in weights if w < self._linker.config.prune_threshold
            )

        # Generate recommendation
        if report.coverage_ratio < 0.3:
            report.recommendation = (
                "LOW COVERAGE: Run distill_batch() on more nodes to improve "
                "shortcut coverage. Current shortcut retrieval may fall back "
                "to full-scan frequently."
            )
        elif report.stale_edge_count > report.total_edges * 0.3:
            report.recommendation = (
                "STALE EDGES: >30% of edges are below prune threshold. "
                "Run distill_batch() to prune stale edges and refresh weights."
            )
        elif report.coverage_ratio >= 0.7:
            report.recommendation = (
                "HEALTHY: Good shortcut coverage. Most retrievals will use "
                "O(degree) shortcuts instead of O(N) full-scan."
            )
        else:
            report.recommendation = (
                "MODERATE: Shortcut coverage is adequate but could be improved. "
                "Consider running distill_batch() on recent uncovered nodes."
            )

        return report

    def migrate_existing(
        self,
        nodes: list[RegistryNode],
        batch_size: int = 100,
    ) -> list[DistillationStats]:
        """Batch-migrate all existing nodes to the distillation index.

        Processes nodes in batches to manage memory usage on large graphs.

        Args:
            nodes: All KG nodes to migrate.
            batch_size: Number of nodes per batch.

        Returns:
            List of DistillationStats, one per batch.
        """
        all_stats: list[DistillationStats] = []

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            stats = self.distill_batch(batch, incremental=True)
            all_stats.append(stats)
            logger.info(
                "Migrated batch %d/%d: %d edges created",
                i // batch_size + 1,
                (len(nodes) + batch_size - 1) // batch_size,
                stats.edges_created,
            )

        return all_stats

    def get_all_edges(self) -> list[SimilarityEdgeNode]:
        """Get all current similarity edges in the index.

        Returns:
            List of all active (non-pruned) SimilarityEdgeNode instances.
        """
        return list(self._all_edges)

    @property
    def retriever(self) -> UnifiedRAGKGRetriever:
        """Access the underlying unified retriever."""
        return self._retriever

    # ── Internal Helpers ─────────────────────────────────────────────

    def _index_edge(self, edge: SimilarityEdgeNode) -> None:
        """Add an edge to the internal lookup index."""
        src = edge.source_node_id
        tgt = edge.target_node_id
        if src not in self._edge_index:
            self._edge_index[src] = []
        self._edge_index[src].append(edge)
        if tgt not in self._edge_index:
            self._edge_index[tgt] = []
        self._edge_index[tgt].append(edge)

    def _rebuild_index(self) -> None:
        """Rebuild the edge lookup index from the edge list."""
        self._edge_index.clear()
        for edge in self._all_edges:
            self._index_edge(edge)
