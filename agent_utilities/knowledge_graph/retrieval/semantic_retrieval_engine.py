from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from agent_utilities.knowledge_graph.core.spectral_navigator import (
    SpectralClusterNavigator,
)
from agent_utilities.knowledge_graph.memory.auto_similarity import (
    AutoSimilarityLinker,
)
from agent_utilities.models.knowledge_graph import (
    HybridSearchConfig,
    MemoryDecayConfig,
    RegistryNode,
    SimilarityEdgeNode,
    SpectralClusterNode,
)

# --- Merged from semantic_retrieval_engine.py ---

#!/usr/bin/python
"""Hybrid Search Scorer.

CONCEPT:KG-2.3 — Hybrid Search Index

Provides weighted semantic+keyword search scoring adapted from contextplus's
hybrid search. Uses existing embedding infrastructure.
"""


logger = logging.getLogger(__name__)


def _split_compound_name(text: str) -> set[str]:
    """Split camelCase, PascalCase, and snake_case into tokens."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", spaced)
    tokens = re.split(r"[\s_\-]+", spaced.lower())
    return {t for t in tokens if len(t) > 1}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class HybridSearchScorer:
    """Weighted hybrid semantic+keyword scoring engine.

    CONCEPT:KG-2.3 — Hybrid Search Index

    Scores documents via configurable blend of semantic similarity
    and keyword matching with phrase boost and symbol-specific scoring.

    Example::

        scorer = HybridSearchScorer()
        results = scorer.score_documents(
            query="spectral clustering",
            query_embedding=[0.1, 0.2, ...],
            documents=[{"id": "d1", "text": "...", "embedding": [...]}],
        )
    """

    def __init__(self, config: HybridSearchConfig | None = None):
        self.config = config or HybridSearchConfig()

    def _keyword_score(
        self,
        query: str,
        query_terms: set[str],
        doc_text: str,
        symbols: list[str] | None = None,
    ) -> tuple[float, list[str]]:
        """Compute keyword score with phrase boost."""
        if not query_terms:
            return 0.0, []

        doc_terms = _split_compound_name(doc_text)
        term_coverage = sum(1 for t in query_terms if t in doc_terms) / len(query_terms)

        matched_symbols: list[str] = []
        symbol_coverage = 0.0
        if symbols:
            all_sym_terms: set[str] = set()
            for sym in symbols:
                sym_terms = _split_compound_name(sym)
                if sym_terms & query_terms:
                    matched_symbols.append(sym)
                all_sym_terms.update(sym_terms)
            if query_terms:
                symbol_coverage = sum(
                    1 for t in query_terms if t in all_sym_terms
                ) / len(query_terms)

        phrase_boost = (
            self.config.phrase_boost
            if query.strip().lower() in doc_text.lower()
            else 0.0
        )
        score = min(1.0, term_coverage * 0.65 + symbol_coverage * 0.2 + phrase_boost)
        return score, matched_symbols

    def _combined_score(self, semantic: float, keyword: float) -> float:
        """Compute weighted combined score."""
        total = self.config.semantic_weight + self.config.keyword_weight
        if total <= 0:
            return max(semantic, 0.0)
        return min(
            1.0,
            (
                self.config.semantic_weight * max(semantic, 0)
                + self.config.keyword_weight * keyword
            )
            / total,
        )

    def score_documents(
        self,
        query: str,
        query_embedding: list[float],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and rank documents using hybrid scoring.

        Each doc dict should have: id, text, embedding, symbols (optional).

        Returns:
            Sorted list with added: semantic_score, keyword_score,
            combined_score, matched_symbols.
        """
        query_terms = _split_compound_name(query)
        results: list[dict[str, Any]] = []

        for doc in documents:
            doc_emb = doc.get("embedding")
            sem_score = (
                _cosine_similarity(query_embedding, doc_emb)
                if doc_emb and query_embedding
                else 0.0
            )
            kw_score, matched = self._keyword_score(
                query, query_terms, doc.get("text", ""), doc.get("symbols", [])
            )
            combined = self._combined_score(sem_score, kw_score)

            if max(sem_score, 0) < self.config.min_semantic_score:
                continue
            if kw_score < self.config.min_keyword_score:
                continue
            if combined < self.config.min_combined_score:
                continue

            result = dict(doc)
            result.update(
                {
                    "semantic_score": round(sem_score, 4),
                    "keyword_score": round(kw_score, 4),
                    "combined_score": round(combined, 4),
                    "matched_symbols": matched,
                }
            )
            results.append(result)

        results.sort(
            key=lambda x: (
                x["combined_score"],
                x["keyword_score"],
                x["semantic_score"],
            ),
            reverse=True,
        )
        return results[: self.config.top_k]


# --- Merged from semantic_retrieval_engine.py ---

#!/usr/bin/python
"""RAG-KG Unified Retriever.

CONCEPT:KG-2.3 — RAG-KG Unification

Collapses the separate RAG vector index into KG-native retrieval by
leveraging three acceleration primitives:

1. **Similarity-edge shortcuts** (KG-2.36): Pre-computed ``SIMILAR_TO``
   edges provide O(1) retrieval paths for known-similar nodes, bypassing
   runtime cosine computation.
2. **Spectral cluster scoping** (KG-2.34): Queries are first classified
   into a spectral cluster, reducing the search space by 10-100x before
   fine-grained similarity matching.
3. **Hybrid scoring** (KG-2.37): Weighted semantic+keyword scoring with
   CamelCase splitting and phrase boost.

Integrates with the existing ``HybridRetriever`` as a drop-in enhancement
via ``retrieve_unified()``.
"""


if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class KGNativeRetrievalConfig:
    """Configuration for the RAG-KG unified retriever.

    Attributes:
        enable_similarity_shortcuts: Use pre-computed SIMILAR_TO edges.
        enable_cluster_scoping: Scope queries to spectral clusters first.
        enable_hybrid_scoring: Use weighted semantic+keyword scoring.
        shortcut_max_hops: Maximum edge hops for similarity shortcuts.
        cluster_scope_threshold: Minimum similarity to cluster centroid
            for scope inclusion.
        similarity_weight: Weight for semantic similarity in final scoring.
        keyword_weight: Weight for keyword matching in final scoring.
        shortcut_boost: Multiplicative boost for nodes found via shortcuts.
    """

    def __init__(
        self,
        enable_similarity_shortcuts: bool = True,
        enable_cluster_scoping: bool = True,
        enable_hybrid_scoring: bool = True,
        shortcut_max_hops: int = 2,
        cluster_scope_threshold: float = 0.55,
        similarity_weight: float = 0.72,
        keyword_weight: float = 0.28,
        shortcut_boost: float = 1.15,
    ):
        self.enable_similarity_shortcuts = enable_similarity_shortcuts
        self.enable_cluster_scoping = enable_cluster_scoping
        self.enable_hybrid_scoring = enable_hybrid_scoring
        self.shortcut_max_hops = shortcut_max_hops
        self.cluster_scope_threshold = cluster_scope_threshold
        self.similarity_weight = similarity_weight
        self.keyword_weight = keyword_weight
        self.shortcut_boost = shortcut_boost


class KGNativeRetrievalRetriever:
    """KG-native retrieval combining similarity shortcuts, spectral scoping, and hybrid scoring.

    CONCEPT:KG-2.3 — RAG-KG Unification

    This replaces the pure-vector RAG pipeline with a graph-aware retrieval
    path that leverages pre-computed structure (similarity edges, spectral
    clusters) for faster and more precise context assembly.

    Architecture::

        Query → Cluster Scope → Similarity Shortcut Walk → Hybrid Score → Results
                     ↓                    ↓                     ↓
              SpectralCluster       SIMILAR_TO edges      Semantic+Keyword

    Example::

        retriever = KGNativeRetrievalRetriever(engine)
        results = retriever.retrieve_unified("spectral clustering for agents", top_k=10)
        for node, score in results:
            print(f"{node.name}: {score:.3f}")
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        config: KGNativeRetrievalConfig | None = None,
    ):
        """Initialize the unified RAG-KG retriever.

        Args:
            engine: The KG engine for graph access. Optional for standalone use.
            config: Retrieval configuration. Uses defaults if None.
        """
        self.engine = engine
        self.config = config or KGNativeRetrievalConfig()

        # Initialize sub-components
        self._spectral_nav = SpectralClusterNavigator()
        self._similarity_linker = AutoSimilarityLinker()
        self._hybrid_scorer = HybridSearchScorer()

        # Cached cluster index (built lazily)
        self._cluster_index: list[SpectralClusterNode] = []
        self._cluster_centroids: np.ndarray | None = None
        self._cluster_members: dict[str, list[str]] = {}

        # Similarity edge index (built lazily)
        self._similarity_index: dict[str, list[SimilarityEdgeNode]] = {}

        # Stats
        self._stats: dict[str, int] = {
            "shortcut_hits": 0,
            "cluster_scoped": 0,
            "full_scan": 0,
        }

    # ── Cluster Index ────────────────────────────────────────────────

    def build_cluster_index(
        self,
        nodes: list[RegistryNode],
        domain: str = "general",
    ) -> int:
        """Build the spectral cluster index from KG nodes.

        Clusters all nodes with embeddings using the SpectralClusterNavigator
        and caches centroids for O(k) query-time cluster matching.

        Args:
            nodes: KG nodes with embeddings to cluster.
            domain: Domain label for the clusters.

        Returns:
            Number of clusters created.
        """
        nodes_with_embeddings = [n for n in nodes if n.embedding is not None]
        if len(nodes_with_embeddings) < 2:
            logger.debug("Not enough nodes with embeddings for clustering")
            return 0

        vectors = [
            n.embedding for n in nodes_with_embeddings if n.embedding is not None
        ]
        clusters = self._spectral_nav.cluster(vectors, domain=domain)

        # Convert to KG nodes
        self._cluster_index = self._spectral_nav.cluster_to_kg_nodes(
            clusters, domain=domain
        )

        # Build centroid matrix for fast matching
        centroids = []
        self._cluster_members = {}
        for cluster, result in zip(self._cluster_index, clusters, strict=False):
            centroids.append(cluster.centroid_embedding)
            member_ids = [nodes_with_embeddings[i].id for i in result.indices]
            self._cluster_members[cluster.id] = member_ids

        self._cluster_centroids = np.array(centroids) if centroids else None

        logger.info(
            "Built cluster index: %d clusters from %d nodes",
            len(self._cluster_index),
            len(nodes_with_embeddings),
        )
        return len(self._cluster_index)

    def _scope_to_cluster(
        self,
        query_embedding: list[float],
    ) -> list[str] | None:
        """Find the best-matching cluster for a query and return member IDs.

        Args:
            query_embedding: The query's embedding vector.

        Returns:
            List of node IDs in the matching cluster, or None if no match.
        """
        if self._cluster_centroids is None or len(self._cluster_index) == 0:
            return None

        # Find closest cluster centroid
        query_vec = np.array(query_embedding)
        best_sim = -1.0
        best_cluster_id = None

        for i, centroid in enumerate(self._cluster_centroids):
            sim = _cosine_similarity(query_vec.tolist(), centroid.tolist())
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = self._cluster_index[i].id

        if best_sim >= self.config.cluster_scope_threshold and best_cluster_id:
            self._stats["cluster_scoped"] += 1
            return self._cluster_members.get(best_cluster_id, [])

        return None

    # ── Similarity Shortcut Index ────────────────────────────────────

    def build_similarity_index(
        self,
        edges: list[SimilarityEdgeNode],
    ) -> int:
        """Build the similarity shortcut index from existing edges.

        Args:
            edges: Pre-computed SIMILAR_TO edges from the KG.

        Returns:
            Number of edges indexed.
        """
        self._similarity_index.clear()
        valid_count = 0

        for edge in edges:
            # Only index non-stale edges
            current_weight = self._similarity_linker.decay_weight(edge)
            if current_weight >= self._similarity_linker.config.prune_threshold:
                src = edge.source_node_id
                if src not in self._similarity_index:
                    self._similarity_index[src] = []
                self._similarity_index[src].append(edge)

                tgt = edge.target_node_id
                if tgt not in self._similarity_index:
                    self._similarity_index[tgt] = []
                self._similarity_index[tgt].append(edge)

                valid_count += 1

        logger.info("Built similarity index: %d edges", valid_count)
        return valid_count

    def _walk_similarity_shortcuts(
        self,
        seed_node_ids: list[str],
        max_hops: int | None = None,
    ) -> set[str]:
        """Walk SIMILAR_TO edges from seed nodes to discover related nodes.

        This provides O(degree) retrieval for known-similar nodes instead
        of O(N) full-index scan.

        Args:
            seed_node_ids: Starting node IDs.
            max_hops: Maximum hops to traverse.

        Returns:
            Set of discovered node IDs (including seeds).
        """
        if not self._similarity_index:
            return set(seed_node_ids)

        max_hops = max_hops or self.config.shortcut_max_hops
        discovered: set[str] = set(seed_node_ids)
        frontier = set(seed_node_ids)

        for _hop in range(max_hops):
            next_frontier: set[str] = set()
            for node_id in frontier:
                edges = self._similarity_index.get(node_id, [])
                for edge in edges:
                    neighbor = (
                        edge.target_node_id
                        if edge.source_node_id == node_id
                        else edge.source_node_id
                    )
                    if neighbor not in discovered:
                        discovered.add(neighbor)
                        next_frontier.add(neighbor)

            if not next_frontier:
                break
            frontier = next_frontier
            self._stats["shortcut_hits"] += len(next_frontier)

        return discovered

    # ── Unified Retrieval ────────────────────────────────────────────

    def retrieve_unified(
        self,
        query: str,
        nodes: list[RegistryNode],
        query_embedding: list[float] | None = None,
        top_k: int = 10,
    ) -> list[tuple[RegistryNode, float]]:
        """Perform KG-native unified retrieval with all acceleration layers.

        Pipeline:
        1. **Cluster scoping** (optional): Narrow to the relevant cluster.
        2. **Similarity shortcuts** (optional): Walk pre-computed edges.
        3. **Hybrid scoring**: Weighted semantic+keyword final ranking.

        Args:
            query: The search query string.
            nodes: Candidate KG nodes to search over.
            query_embedding: Pre-computed query embedding. If None, falls
                back to keyword-only scoring.
            top_k: Number of results to return.

        Returns:
            List of (node, score) tuples, sorted by descending score.
        """
        start_time = time.time()

        # Build node lookup
        node_map: dict[str, RegistryNode] = {n.id: n for n in nodes}
        candidate_ids: set[str] | None = None

        # Phase 1: Cluster scoping
        if (
            self.config.enable_cluster_scoping
            and query_embedding
            and self._cluster_centroids is not None
        ):
            scoped_ids = self._scope_to_cluster(query_embedding)
            if scoped_ids:
                candidate_ids = set(scoped_ids)
                logger.debug("Cluster scoped to %d candidates", len(candidate_ids))

        # Phase 2: Similarity shortcut walk
        if self.config.enable_similarity_shortcuts and self._similarity_index:
            # Start from top-scoring candidates (or all if no scoping)
            if candidate_ids:
                seed_ids = list(candidate_ids)[:5]  # Top seeds from cluster
            elif query_embedding:
                # Quick pre-filter: find top-5 by embedding
                scored = []
                for n in nodes:
                    if n.embedding:
                        sim = _cosine_similarity(query_embedding, n.embedding)
                        scored.append((n.id, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                seed_ids = [s[0] for s in scored[:5]]
            else:
                seed_ids = []

            if seed_ids:
                expanded = self._walk_similarity_shortcuts(seed_ids)
                if candidate_ids:
                    candidate_ids = candidate_ids.union(expanded)
                else:
                    candidate_ids = expanded

        # Determine final candidate set
        if candidate_ids:
            candidates = [node_map[nid] for nid in candidate_ids if nid in node_map]
        else:
            candidates = nodes
            self._stats["full_scan"] += 1

        # Phase 3: Hybrid scoring
        if self.config.enable_hybrid_scoring:
            # Build document dicts for HybridSearchScorer API
            doc_dicts = [
                {
                    "id": c.id,
                    "text": f"{c.name} {c.description or ''}",
                    "embedding": c.embedding,
                }
                for c in candidates
            ]

            scored_results = self._hybrid_scorer.score_documents(
                query=query,
                query_embedding=query_embedding or [],
                documents=doc_dicts,
            )

            # Map back to nodes
            results: list[tuple[RegistryNode, float]] = []
            id_to_node = {c.id: c for c in candidates}
            for doc_result in scored_results:
                node_id = doc_result.get("id", "")
                node = id_to_node.get(node_id)
                if node:
                    score = doc_result.get("combined_score", 0.0)
                    # Apply shortcut boost if node was found via shortcuts
                    if (
                        self.config.enable_similarity_shortcuts
                        and candidate_ids
                        and node.id in candidate_ids
                    ):
                        score *= self.config.shortcut_boost
                    results.append((node, score))
        else:
            # Fallback: pure semantic scoring
            results = []
            for c in candidates:
                if c.embedding and query_embedding:
                    sim = _cosine_similarity(query_embedding, c.embedding)
                    results.append((c, sim))
                else:
                    results.append((c, 0.0))

        # Sort and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            "Unified retrieval: %d results in %.1fms (clusters=%s, shortcuts=%s)",
            len(results),
            elapsed_ms,
            self.config.enable_cluster_scoping,
            self.config.enable_similarity_shortcuts,
        )

        return results

    @property
    def stats(self) -> dict[str, int]:
        """Retrieval statistics for monitoring.

        Returns:
            Dict with shortcut_hits, cluster_scoped, full_scan counts.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset retrieval statistics."""
        self._stats = {"shortcut_hits": 0, "cluster_scoped": 0, "full_scan": 0}


# --- Merged from semantic_retrieval_engine.py ---

#!/usr/bin/python
"""Graph Distillation Migration.

CONCEPT:KG-2.6 — Graph Distillation Migration

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
- KG-2.38 (KGNativeRetrievalRetriever): As the primary retrieval backend
- KG-2.34 (SpectralClusterNavigator): Cluster-scoped distillation
"""


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

    CONCEPT:KG-2.6 — Graph Distillation Migration

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
        retriever_config: KGNativeRetrievalConfig | None = None,
    ):
        """Initialize the graph distillation migrator.

        Args:
            decay_config: Configuration for similarity edge decay.
            retriever_config: Configuration for the unified retriever.
        """
        self._linker = AutoSimilarityLinker(config=decay_config)
        self._retriever = KGNativeRetrievalRetriever(config=retriever_config)

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
            "[CONCEPT:KG-2.6] Distillation batch: %d nodes, %d edges created, "
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
    def retriever(self) -> KGNativeRetrievalRetriever:
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
