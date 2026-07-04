#!/usr/bin/python
"""Topological Analogy Engine.

CONCEPT:AU-KG.compute.topological-analogy — Topological Analogy Engine
Leverages optimized `epistemic-graph` backend primitives and vectorized embeddings (`EncPI`) to find analogous
subgraphs across different domains. This enables cross-domain innovation extraction
and structural pattern matching within the Knowledge Graph.
"""

from typing import Any

from agent_utilities.knowledge_graph.core import graph_primitives as rx
from agent_utilities.models.knowledge_graph import (
    AnalogyMatchNode,
)
from agent_utilities.numeric import xp as np


class TopologicalAnalogyEngine:
    """Finds analogous subgraphs across different domains using topological similarities on the epistemic-graph service."""

    def __init__(self, graph: Any):
        """Initializes the analogy engine.

        Args:
            graph: The GraphComputeEngine utilizing the Tokio-based epistemic-graph service.
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
        self, target_subgraph: Any, threshold: float = 0.8
    ) -> list[AnalogyMatchNode]:
        """Finds subgraphs in the main graph that are analogous to the target_subgraph.

        Uses rustworkx subgraph isomorphism combined with vectorized
        EncPI embeddings to discover deep structural and semantic analogies.

        Args:
            target_subgraph: The query subgraph (rx.PyDiGraph or similar).
            threshold: The semantic similarity threshold (0.0 to 1.0).

        Returns:
            A list of AnalogyMatchNode objects representing the discovered analogies.
        """

        def get_embedding(node_payload: Any) -> list[float] | None:
            if isinstance(node_payload, dict):
                if "embedding" in node_payload:
                    return node_payload["embedding"]
                data = node_payload.get("data")
                if data:
                    if isinstance(data, dict):
                        return data.get("embedding")
                    if hasattr(data, "embedding"):
                        return data.embedding
            return None

        def get_name(node_payload: Any) -> str:
            if isinstance(node_payload, dict):
                if "name" in node_payload and node_payload["name"]:
                    return node_payload["name"]
                data = node_payload.get("data")
                if data:
                    if isinstance(data, dict):
                        return data.get("name") or ""
                    if hasattr(data, "name"):
                        return data.name or ""
            return ""

        matches: list[AnalogyMatchNode] = []

        # Build rustworkx graph from GCE for isomorphism search
        main_rx: rx.PyDiGraph = rx.PyDiGraph()
        gce_node_map: dict[str, int] = {}

        if hasattr(self.graph, "node_ids"):
            for node_id in self.graph.node_ids():
                props = self.graph._get_node_properties(node_id)
                idx = main_rx.add_node({"id": node_id, **props})
                gce_node_map[node_id] = idx
            for src, tgt in self.graph._get_all_edges():
                if src in gce_node_map and tgt in gce_node_map:
                    main_rx.add_edge(gce_node_map[src], gce_node_map[tgt], {})

        if main_rx.num_nodes() == 0:
            return matches

        # Build target rx.PyDiGraph
        target_rx = rx.PyDiGraph()
        target_node_map: dict[str, int] = {}
        if hasattr(target_subgraph, "node_ids"):
            for node_id in target_subgraph.node_ids():
                props = target_subgraph._get_node_properties(node_id)
                idx = target_rx.add_node({"id": node_id, **props})
                target_node_map[node_id] = idx
            for src, tgt in target_subgraph._get_all_edges():
                if src in target_node_map and tgt in target_node_map:
                    target_rx.add_edge(target_node_map[src], target_node_map[tgt], {})
        elif isinstance(target_subgraph, rx.PyDiGraph):
            target_rx = target_subgraph

        if target_rx.num_nodes() == 0:
            return matches

        # Define a node matcher that checks semantic similarity
        def node_matcher(n1, n2):
            emb1 = get_embedding(n1)
            emb2 = get_embedding(n2)
            if emb1 is None or emb2 is None:
                return False
            sim = self._compute_cosine_similarity(emb1, emb2)
            return sim >= threshold

        mappings = list(
            rx.vf2_mapping(
                main_rx,
                target_rx,
                node_matcher=node_matcher,
                subgraph=True,
                induced=False,
            )
        )

        seen_names = set()
        for mapping in mappings:
            # mapping is a dict mapping target_idx -> main_idx
            total_sim = 0.0
            matched_nodes_count = len(mapping)
            if matched_nodes_count == 0:
                continue

            pairs = []
            for target_idx, main_idx in mapping.items():
                n_main = main_rx[main_idx]
                n_target = target_rx[target_idx]
                emb_main = get_embedding(n_main)
                emb_target = get_embedding(n_target)
                sim = self._compute_cosine_similarity(emb_main, emb_target)
                total_sim += sim
                pairs.append((n_target, n_main))

            avg_sim = total_sim / matched_nodes_count
            if avg_sim < threshold:
                continue

            # Sort to keep naming deterministic
            pairs_sorted = sorted(pairs, key=lambda x: get_name(x[0]))
            t_name = get_name(pairs_sorted[0][0]) if pairs_sorted else "Target"
            m_name = get_name(pairs_sorted[0][1]) if pairs_sorted else "Base"

            name = f"Analogy: {t_name} ≈ {m_name}"
            if name in seen_names:
                continue
            seen_names.add(name)

            match_node = AnalogyMatchNode(
                id=f"analogy_{t_name}_{m_name}".lower().replace(" ", "_"),
                name=name,
                target_domain="unknown",
                similarity_score=avg_sim,
                matched_nodes=matched_nodes_count,
                analogy_rationale="Vectorized topological isomorphism discovered via rustworkx.",
            )
            matches.append(match_node)

        return matches
