#!/usr/bin/python
"""Topological Analogy Engine.

CONCEPT:KG-2.15 — Topological Analogy Engine
Leverages `networkx` and vectorized embeddings (`EncPI`) to find analogous
subgraphs across different domains. This enables cross-domain innovation extraction
and structural pattern matching within the Knowledge Graph.
"""

from typing import Any

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

        This uses exact subgraph isomorphism (VF2) combined with vectorized
        EncPI embeddings to discover deep structural and semantic analogies.

        Args:
            target_subgraph: The query subgraph to find analogies for.
            threshold: The semantic similarity threshold (0.0 to 1.0).

        Returns:
            A list of AnalogyMatchNode objects representing the discovered analogies.
        """
        matches: list[AnalogyMatchNode] = []

        if len(target_subgraph) == 0 or len(self.graph) == 0:
            return matches

        def node_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
            """Custom match function for VF2 isomorphism: semantic embedding similarity."""
            data1 = n1.get("data")
            data2 = n2.get("data")

            # Must be RegistryNodes with embeddings
            if not isinstance(data1, RegistryNode) or not isinstance(
                data2, RegistryNode
            ):
                return False
            if not data1.embedding or not data2.embedding:
                return False

            sim = self._compute_cosine_similarity(data1.embedding, data2.embedding)
            return sim >= threshold

        # Initialize the VF2 graph matcher
        from networkx.algorithms.isomorphism import DiGraphMatcher

        matcher = DiGraphMatcher(self.graph, target_subgraph, node_match=node_match)

        # Store matched target nodes to avoid duplicate reports for the same analogy
        discovered_matches = set()

        for subgraph_mapping in matcher.subgraph_isomorphisms_iter():
            # subgraph_mapping maps node_in_G -> node_in_target
            # Let's find the 'root' of the analogy in the main graph

            # Identify the root in the target subgraph
            target_roots = [n for n, d in target_subgraph.in_degree() if d == 0]
            if not target_roots:
                target_roots = list(target_subgraph.nodes())

            target_root = target_roots[0]

            # Find the corresponding root in the main graph
            main_graph_root = None
            for g_node, t_node in subgraph_mapping.items():
                if t_node == target_root:
                    main_graph_root = g_node
                    break

            if not main_graph_root or main_graph_root in discovered_matches:
                continue

            discovered_matches.add(main_graph_root)

            anchor_data = target_subgraph.nodes[target_root].get("data")
            node_data = self.graph.nodes[main_graph_root].get("data")

            if not isinstance(node_data, RegistryNode) or not isinstance(
                anchor_data, RegistryNode
            ):
                continue

            # Calculate the semantic similarity of the root nodes to report
            similarity = self._compute_cosine_similarity(
                anchor_data.embedding, node_data.embedding
            )

            match_node = AnalogyMatchNode(
                id=f"analogy_{target_root}_to_{main_graph_root}",
                name=f"Analogy: {anchor_data.name} ≈ {node_data.name}",
                target_domain=node_data.type.value,
                similarity_score=similarity,
                matched_nodes=len(subgraph_mapping),
                analogy_rationale=(
                    f"Exact structural isomorphism found with {len(subgraph_mapping)} nodes. "
                    f"Root semantic similarity ({similarity:.2f}) between {anchor_data.name} "
                    f"and {node_data.name}."
                ),
            )
            matches.append(match_node)

        # Sort by similarity score descending
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches
