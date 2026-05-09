#!/usr/bin/python
"""Auto-Similarity Memory Graph.

CONCEPT:KG-2.36 — Auto-Similarity Memory Graph

Provides automatic similarity edge creation and exponential decay scoring
for memory nodes in the Knowledge Graph. Adapted from contextplus's
memory-graph.ts with the following enhancements:

* **Auto-linking** — On node insertion, computes cosine similarity against
  recent nodes and creates ``SIMILAR_TO`` edges above threshold (default 0.72).
* **Decay scoring** — Edge weights decay exponentially: ``w * e^(-λ * Δt)``.
* **Stale pruning** — Edges below the prune threshold are removed.
* **Hub control** — Maximum edges per node prevents explosion.

Integrates with the existing ``MemoryRetriever`` and ``IntelligenceGraphEngine``
for graph-augmented RAG retrieval shortcuts.
"""

from __future__ import annotations

import logging
import math
import time
import uuid

import numpy as np

from agent_utilities.models.knowledge_graph import (
    MemoryDecayConfig,
    RegistryEdge,
    RegistryEdgeType,
    RegistryNode,
    SimilarityEdgeNode,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class AutoSimilarityLinker:
    """Creates and manages auto-similarity edges between KG memory nodes.

    CONCEPT:KG-2.36 — Auto-Similarity Memory Graph

    On node insertion, finds similar existing nodes via cosine similarity
    and creates weighted edges with exponential decay.

    Example::

        linker = AutoSimilarityLinker()
        new_edges = linker.link_new_node(
            new_node=node,
            existing_nodes=recent_nodes,
        )
        for edge in new_edges:
            print(f"Similar: {edge.source_node_id} → {edge.target_node_id} "
                  f"({edge.cosine_similarity:.2f})")
    """

    def __init__(self, config: MemoryDecayConfig | None = None):
        """Initialize the auto-similarity linker.

        Args:
            config: Decay and threshold configuration. Uses defaults if None.
        """
        self.config = config or MemoryDecayConfig()

    def link_new_node(
        self,
        new_node: RegistryNode,
        existing_nodes: list[RegistryNode],
    ) -> list[SimilarityEdgeNode]:
        """Find similar nodes and create similarity edges.

        Args:
            new_node: The newly inserted node with an embedding.
            existing_nodes: Recent nodes to compare against (bounded by
                ``config.batch_window``).

        Returns:
            List of SimilarityEdgeNode instances for edges above threshold.
        """
        if not new_node.embedding:
            return []

        # Limit comparison window
        candidates = existing_nodes[-self.config.batch_window :]
        candidates_with_embeddings = [n for n in candidates if n.embedding]

        edges: list[SimilarityEdgeNode] = []
        now = time.time()

        for candidate in candidates_with_embeddings:
            if (
                candidate.id == new_node.id
                or not candidate.embedding
                or not new_node.embedding
            ):
                continue

            sim = _cosine_similarity(new_node.embedding, candidate.embedding)

            if sim >= self.config.similarity_threshold:
                edge = SimilarityEdgeNode(
                    id=f"sim_{uuid.uuid4().hex[:8]}",
                    name=f"Similarity: {new_node.name} ↔ {candidate.name}",
                    description=(
                        f"Auto-created similarity edge (cosine={sim:.3f}) "
                        f"between {new_node.id} and {candidate.id}"
                    ),
                    source_node_id=new_node.id,
                    target_node_id=candidate.id,
                    cosine_similarity=sim,
                    decay_lambda=self.config.decay_lambda,
                    current_weight=sim,  # Initial weight = similarity score
                    creation_epoch=now,
                    last_accessed_epoch=now,
                    access_count=0,
                )
                edges.append(edge)

        # Hub control: keep only top-N by similarity
        if len(edges) > self.config.max_edges_per_node:
            edges.sort(key=lambda e: e.cosine_similarity, reverse=True)
            edges = edges[: self.config.max_edges_per_node]

        return edges

    def decay_weight(self, edge: SimilarityEdgeNode) -> float:
        """Compute the current decayed weight of an edge.

        Uses exponential decay: ``original_sim * e^(-λ * days_elapsed)``

        Args:
            edge: The similarity edge to compute decay for.

        Returns:
            Current weight after applying time decay.
        """
        now = time.time()
        days_elapsed = (now - edge.creation_epoch) / 86400.0
        decayed = edge.cosine_similarity * math.exp(-edge.decay_lambda * days_elapsed)
        return max(0.0, decayed)

    def prune_stale_edges(
        self,
        edges: list[SimilarityEdgeNode],
    ) -> tuple[list[SimilarityEdgeNode], list[SimilarityEdgeNode]]:
        """Prune edges that have decayed below the threshold.

        Args:
            edges: All similarity edges to evaluate.

        Returns:
            Tuple of (kept_edges, pruned_edges).
        """
        kept: list[SimilarityEdgeNode] = []
        pruned: list[SimilarityEdgeNode] = []

        for edge in edges:
            current_weight = self.decay_weight(edge)
            if current_weight >= self.config.prune_threshold:
                edge.current_weight = current_weight
                kept.append(edge)
            else:
                pruned.append(edge)

        if pruned:
            logger.info(
                "Pruned %d stale similarity edges (below threshold %.3f)",
                len(pruned),
                self.config.prune_threshold,
            )

        return kept, pruned

    def to_registry_edges(
        self,
        similarity_edges: list[SimilarityEdgeNode],
    ) -> list[RegistryEdge]:
        """Convert SimilarityEdgeNodes to standard RegistryEdge format.

        Useful for bulk persistence via the IntelligenceGraphEngine.

        Args:
            similarity_edges: Similarity edge nodes to convert.

        Returns:
            List of RegistryEdge instances.
        """
        return [
            RegistryEdge(
                source=edge.source_node_id,
                target=edge.target_node_id,
                type=RegistryEdgeType.SIMILAR_TO,
                weight=edge.current_weight,
                metadata={
                    "cosine_similarity": edge.cosine_similarity,
                    "decay_lambda": edge.decay_lambda,
                    "creation_epoch": edge.creation_epoch,
                },
            )
            for edge in similarity_edges
        ]
