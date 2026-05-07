#!/usr/bin/python
"""Topological Analogy Engine.

CONCEPT:KG-2.15 — Topological Analogy Engine
Leverages `networkx` and vectorized embeddings (`EncPI`) to find analogous
subgraphs across different domains. This enables cross-domain innovation extraction
and structural pattern matching within the Knowledge Graph.
"""

import networkx as nx
import numpy as np

from agent_utilities.models.knowledge_graph import (
    AnalogyMatchNode,
    RegistryNode,
)


class TopologicalAnalogyEngine:
    """Finds analogous subgraphs across different domains using topological similarities."""

    def __init__(self, graph: nx.MultiDiGraph):
        """Initializes the analogy engine.

        Args:
            graph: The in-memory NetworkX intelligence graph.
        """
        self.graph = graph

    def _compute_cosine_similarity(
        self, vec_a: list[float] | None, vec_b: list[float] | None
    ) -> float:
        """Computes cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_analogous_subgraphs(
        self, target_subgraph: nx.MultiDiGraph, threshold: float = 0.8
    ) -> list[AnalogyMatchNode]:
        """Finds subgraphs in the main graph that are analogous to the target_subgraph.

        This uses a combination of Graph Edit Distance heuristics and vector embedding
        (EncPI) similarities to determine topological analogies.

        Args:
            target_subgraph: The query subgraph to find analogies for.
            threshold: The similarity threshold (0.0 to 1.0) required to consider a match.

        Returns:
            A list of AnalogyMatchNode objects representing the discovered analogies.
        """
        matches: list[AnalogyMatchNode] = []

        # Simplified analogy heuristic: Find nodes in the main graph that have similar
        # degree profiles and embedding semantics to the root nodes of the target subgraph.
        # In a full implementation, this would use deep Graph Isomorphism Network (GIN)
        # or exact subgraph isomorphism algorithms.

        target_roots = [n for n, d in target_subgraph.in_degree() if d == 0]
        if not target_roots:
            target_roots = list(target_subgraph.nodes())

        if not target_roots:
            return matches

        # Sample a root node from the target to find structural anchors
        anchor_id = target_roots[0]
        anchor_data = target_subgraph.nodes[anchor_id].get("data")

        if not anchor_data or not hasattr(anchor_data, "embedding"):
            return matches

        anchor_emb = anchor_data.embedding
        anchor_out_degree = target_subgraph.out_degree(anchor_id)

        for node_id in self.graph.nodes:
            # Skip if comparing to itself
            if node_id in target_subgraph.nodes:
                continue

            node_data = self.graph.nodes[node_id].get("data")
            if not isinstance(node_data, RegistryNode):
                continue

            out_degree = self.graph.out_degree(node_id)

            # Topological heuristic: Out-degree must be somewhat similar
            if abs(out_degree - anchor_out_degree) > 2:
                continue

            # Semantic heuristic: Vectorized EncPI topology embedding similarity
            similarity = self._compute_cosine_similarity(
                anchor_emb, node_data.embedding
            )

            if similarity >= threshold:
                match_node = AnalogyMatchNode(
                    id=f"analogy_{anchor_id}_to_{node_id}",
                    name=f"Analogy: {anchor_data.name} ≈ {node_data.name}",
                    target_domain=node_data.type.value,
                    similarity_score=similarity,
                    matched_nodes=min(len(target_subgraph), out_degree + 1),
                    analogy_rationale=f"High structural and semantic similarity ({similarity:.2f}) found between {anchor_data.name} and {node_data.name}.",
                )
                matches.append(match_node)

        # Sort by similarity score descending
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches
