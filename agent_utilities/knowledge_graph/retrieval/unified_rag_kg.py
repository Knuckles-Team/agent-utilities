#!/usr/bin/python
"""RAG-KG Unified Retriever.

CONCEPT:KG-2.38 — RAG-KG Unification

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

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from agent_utilities.knowledge_graph.core.spectral_navigator import (
    SpectralClusterNavigator,
)
from agent_utilities.knowledge_graph.memory.auto_similarity import (
    AutoSimilarityLinker,
    _cosine_similarity,
)
from agent_utilities.knowledge_graph.retrieval.hybrid_search_scorer import (
    HybridSearchScorer,
)
from agent_utilities.models.knowledge_graph import (
    RegistryNode,
    SimilarityEdgeNode,
    SpectralClusterNode,
)

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class UnifiedRAGKGConfig:
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


class UnifiedRAGKGRetriever:
    """KG-native retrieval combining similarity shortcuts, spectral scoping, and hybrid scoring.

    CONCEPT:KG-2.38 — RAG-KG Unification

    This replaces the pure-vector RAG pipeline with a graph-aware retrieval
    path that leverages pre-computed structure (similarity edges, spectral
    clusters) for faster and more precise context assembly.

    Architecture::

        Query → Cluster Scope → Similarity Shortcut Walk → Hybrid Score → Results
                     ↓                    ↓                     ↓
              SpectralCluster       SIMILAR_TO edges      Semantic+Keyword

    Example::

        retriever = UnifiedRAGKGRetriever(engine)
        results = retriever.retrieve_unified("spectral clustering for agents", top_k=10)
        for node, score in results:
            print(f"{node.name}: {score:.3f}")
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        config: UnifiedRAGKGConfig | None = None,
    ):
        """Initialize the unified RAG-KG retriever.

        Args:
            engine: The KG engine for graph access. Optional for standalone use.
            config: Retrieval configuration. Uses defaults if None.
        """
        self.engine = engine
        self.config = config or UnifiedRAGKGConfig()

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
        nodes_with_embeddings = [n for n in nodes if n.embedding]
        if len(nodes_with_embeddings) < 2:
            logger.debug("Not enough nodes with embeddings for clustering")
            return 0

        vectors = [n.embedding for n in nodes_with_embeddings]
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
