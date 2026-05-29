#!/usr/bin/python
from __future__ import annotations

"""Iterative Knowledge Deduplication Engine.

CONCEPT:KG-2.2 — Knowledge Distillation Engine

Provides semantic deduplication of IdeaBlock nodes using:
1. LSH-accelerated candidate pair generation (for large sets)
2. Dense cosine similarity (for small sets)
3. Community-based clustering (BFS + Louvain for large graphs)
4. LLM-powered merge producing canonical IdeaBlocks

Defaults are tuned for agent-utilities' KG-integrated environment:
- 3 iterations (higher precision per round than Blockify's 4)
- 0.65 base threshold (tighter than Blockify's 0.55 to reduce false merges)
- 0.02 threshold increment per round

All parameters are configurable via constructor.
"""


import json
import logging
import uuid
from collections import defaultdict
from typing import Any

import numpy as np

from .lsh_index import LSHIndex

logger = logging.getLogger(__name__)

# Threshold for switching from dense to LSH candidate generation
_LSH_THRESHOLD = 100


class KnowledgeDeduplicator:
    """Iterative deduplication engine for IdeaBlock knowledge units.

    Args:
        iterations: Number of deduplication rounds.
        base_threshold: Starting cosine similarity threshold for merging.
        threshold_increment: How much to increase threshold each round.
        max_cluster_size: Maximum blocks per LLM merge call.
        lsh_num_tables: LSH tables for candidate generation.
        lsh_hash_size: Bits per LSH hash.
        embedding_dim: Dimensionality of embeddings.
        louvain_node_threshold: Use Louvain clustering when graph exceeds this.
    """

    def __init__(
        self,
        iterations: int = 3,
        base_threshold: float = 0.65,
        threshold_increment: float = 0.02,
        max_cluster_size: int = 15,
        lsh_num_tables: int = 8,
        lsh_hash_size: int = 12,
        embedding_dim: int = 768,
        louvain_node_threshold: int = 500,
    ) -> None:
        self.iterations = iterations
        self.base_threshold = base_threshold
        self.threshold_increment = threshold_increment
        self.max_cluster_size = max_cluster_size
        self.lsh_num_tables = lsh_num_tables
        self.lsh_hash_size = lsh_hash_size
        self.embedding_dim = embedding_dim
        self.louvain_node_threshold = louvain_node_threshold

    def find_similar_pairs(
        self,
        blocks: list[dict[str, Any]],
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """Find pairs of blocks with cosine similarity above threshold.

        Uses LSH for candidate generation when block count exceeds
        ``_LSH_THRESHOLD``, otherwise computes dense pairwise similarity.

        Args:
            blocks: List of block dicts with 'id' and 'embedding' keys.
            threshold: Minimum cosine similarity for a pair to be returned.

        Returns:
            List of ``(id_a, id_b, similarity)`` tuples.
        """
        pairs: list[tuple[str, str, float]] = []
        valid_blocks = [b for b in blocks if b.get("embedding") and b.get("id")]

        if not valid_blocks:
            return pairs

        if len(valid_blocks) > _LSH_THRESHOLD:
            return self._find_pairs_lsh(valid_blocks, threshold)
        return self._find_pairs_dense(valid_blocks, threshold)

    def _find_pairs_dense(
        self,
        blocks: list[dict[str, Any]],
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """Dense O(n²) pairwise cosine similarity computation."""
        pairs: list[tuple[str, str, float]] = []

        # Build matrix for vectorized computation
        ids = [b["id"] for b in blocks]
        matrix = np.array([b["embedding"] for b in blocks], dtype=np.float32)

        # Normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized = matrix / norms

        # Compute similarity matrix
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle pairs above threshold
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    pairs.append((ids[i], ids[j], sim))

        return pairs

    def _find_pairs_lsh(
        self,
        blocks: list[dict[str, Any]],
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """LSH-accelerated candidate generation with cosine re-ranking."""
        pairs: list[tuple[str, str, float]] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Detect embedding dimension from first block
        dim = len(blocks[0]["embedding"])

        lsh = LSHIndex(
            num_tables=self.lsh_num_tables,
            hash_size=self.lsh_hash_size,
            input_dim=dim,
        )

        # Index all blocks
        for block in blocks:
            lsh.index(block["id"], block["embedding"])

        # Query each block for candidates
        for block in blocks:
            candidates = lsh.query(
                block["embedding"],
                k=50,
                exclude_id=block["id"],
            )

            for cand_id, sim in candidates:
                if sim < threshold:
                    continue

                pair_key = tuple(sorted([block["id"], cand_id]))
                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)
                pairs.append((block["id"], cand_id, sim))

        return pairs

    def cluster_similar_blocks(
        self,
        pairs: list[tuple[str, str, float]],
    ) -> list[list[str]]:
        """Group similar block pairs into merge clusters.

        Uses BFS-based connected component discovery for small graphs.
        Falls back to Louvain community detection for large graphs.

        Args:
            pairs: List of ``(id_a, id_b, similarity)`` tuples.

        Returns:
            List of clusters, where each cluster is a list of block IDs.
        """
        if not pairs:
            return []

        # Build adjacency graph
        adjacency: dict[str, set[str]] = defaultdict(set)
        all_nodes: set[str] = set()

        for id_a, id_b, _sim in pairs:
            adjacency[id_a].add(id_b)
            adjacency[id_b].add(id_a)
            all_nodes.add(id_a)
            all_nodes.add(id_b)

        # Use Louvain for large graphs, BFS for small
        if len(all_nodes) > self.louvain_node_threshold:
            return self._cluster_louvain(adjacency, all_nodes, pairs)
        return self._cluster_bfs(adjacency, all_nodes)

    def _cluster_bfs(
        self,
        adjacency: dict[str, set[str]],
        all_nodes: set[str],
    ) -> list[list[str]]:
        """BFS-based connected component discovery."""
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for node in all_nodes:
            if node in visited:
                continue

            # BFS from this node
            cluster: list[str] = []
            queue = [node]
            visited.add(node)

            while queue:
                current = queue.pop(0)
                cluster.append(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Split oversized clusters
            if len(cluster) > self.max_cluster_size:
                for i in range(0, len(cluster), self.max_cluster_size):
                    sub = cluster[i : i + self.max_cluster_size]
                    if len(sub) > 1:
                        clusters.append(sub)
            elif len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _cluster_louvain(
        self,
        adjacency: dict[str, set[str]],
        all_nodes: set[str],
        pairs: list[tuple[str, str, float]],
    ) -> list[list[str]]:
        """Community detection for large similarity graphs via graph primitives."""
        try:
            from agent_utilities.knowledge_graph.core import graph_primitives as rx

            G = rx.PyGraph()
            node_map: dict[str, int] = {}
            for node in all_nodes:
                idx = G.add_node(node)
                node_map[node] = idx
            for id_a, id_b, sim in pairs:
                G.add_edge(node_map[id_a], node_map[id_b], sim)

            # Use connected components as community proxy
            visited: set[int] = set()
            components: list[list[int]] = []
            for start in G.node_indices():
                if start in visited:
                    continue
                comp: list[int] = []
                stack = [start]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.append(n)
                    stack.extend(nb for nb in G.neighbors(n) if nb not in visited)
                components.append(comp)

            clusters: list[list[str]] = []
            for component in components:
                members = [G[idx] for idx in component]
                if len(members) > self.max_cluster_size:
                    for i in range(0, len(members), self.max_cluster_size):
                        sub = members[i : i + self.max_cluster_size]
                        if len(sub) > 1:
                            clusters.append(sub)
                elif len(members) > 1:
                    clusters.append(members)

            return clusters

        except Exception:
            logger.warning("Graph primitives clustering failed, falling back to BFS")
            return self._cluster_bfs(adjacency, all_nodes)

    def merge_cluster(
        self,
        cluster_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge a cluster of similar blocks into a canonical IdeaBlock.

        Uses LLM to synthesize the best question and answer from overlapping
        knowledge units, preserving all unique information.

        Args:
            cluster_blocks: List of block dicts to merge.

        Returns:
            A merged block dict with ``critical_question``, ``trusted_answer``,
            ``name``, ``tags``, ``keywords``, and ``merged_from`` fields.
        """
        if len(cluster_blocks) == 1:
            return cluster_blocks[0]

        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            model = create_model()

            # Build merge prompt from cluster blocks
            block_summaries = []
            for i, block in enumerate(cluster_blocks):
                q = block.get("critical_question", block.get("description", ""))
                a = block.get("trusted_answer", block.get("name", ""))
                block_summaries.append(
                    f"Block {i + 1}:\n  Question: {q}\n  Answer: {a}"
                )

            prompt = (
                "You are merging overlapping knowledge blocks into a single canonical block. "
                "Preserve all unique information while eliminating redundancy. "
                "Return a JSON object with exactly these keys:\n"
                '  "name": a concise descriptive title,\n'
                '  "critical_question": the unified question this knowledge answers,\n'
                '  "trusted_answer": the validated, comprehensive answer (2-4 sentences),\n'
                '  "tags": a list of classification tags,\n'
                '  "keywords": a list of retrieval keywords\n\n'
                "Blocks to merge:\n" + "\n".join(block_summaries)
            )

            agent = Agent(model=model, system_prompt="Return only valid JSON.")
            result = agent.run_sync(prompt)

            # Parse the LLM response
            import re

            json_match = re.search(r"\{.*\}", result.data, re.DOTALL)
            if json_match:
                merged = json.loads(json_match.group())
                merged["id"] = f"ideablock:{uuid.uuid4().hex[:8]}"
                merged["merged_from"] = [b["id"] for b in cluster_blocks]
                return merged

        except Exception as e:
            logger.warning("LLM merge failed, using first block: %s", e)

        # Fallback: use the first block as canonical, annotate as merged
        fallback = dict(cluster_blocks[0])
        fallback["merged_from"] = [b["id"] for b in cluster_blocks]
        return fallback

    def deduplicate(
        self,
        blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the full iterative deduplication pipeline.

        Args:
            blocks: List of block dicts with 'id' and 'embedding' keys.

        Returns:
            Dict with 'blocks' (deduplicated list), 'stats' (processing
            statistics), and 'rounds' (per-round metrics).
        """
        current_blocks = list(blocks)
        starting_count = len(current_blocks)
        rounds: list[dict[str, Any]] = []

        for iteration in range(self.iterations):
            threshold = self.base_threshold + (iteration * self.threshold_increment)
            threshold = min(threshold, 0.98)

            logger.info(
                "Distillation round %d/%d (threshold=%.3f, blocks=%d)",
                iteration + 1,
                self.iterations,
                threshold,
                len(current_blocks),
            )

            # Find similar pairs
            pairs = self.find_similar_pairs(current_blocks, threshold)
            if not pairs:
                logger.info("No similar pairs found, stopping early")
                rounds.append(
                    {
                        "iteration": iteration + 1,
                        "threshold": threshold,
                        "pairs_found": 0,
                        "clusters": 0,
                        "blocks_before": len(current_blocks),
                        "blocks_after": len(current_blocks),
                    }
                )
                break

            # Cluster similar blocks
            clusters = self.cluster_similar_blocks(pairs)
            if not clusters:
                rounds.append(
                    {
                        "iteration": iteration + 1,
                        "threshold": threshold,
                        "pairs_found": len(pairs),
                        "clusters": 0,
                        "blocks_before": len(current_blocks),
                        "blocks_after": len(current_blocks),
                    }
                )
                break

            # Merge each cluster
            blocks_before = len(current_blocks)
            merged_ids: set[str] = set()
            new_blocks: list[dict[str, Any]] = []

            for cluster_ids in clusters:
                cluster_blocks = [
                    b for b in current_blocks if b["id"] in set(cluster_ids)
                ]
                if len(cluster_blocks) < 2:
                    continue

                merged = self.merge_cluster(cluster_blocks)
                new_blocks.append(merged)
                for b in cluster_blocks:
                    merged_ids.add(b["id"])

            # Replace merged blocks with their canonical versions
            surviving = [b for b in current_blocks if b["id"] not in merged_ids]
            current_blocks = surviving + new_blocks

            rounds.append(
                {
                    "iteration": iteration + 1,
                    "threshold": threshold,
                    "pairs_found": len(pairs),
                    "clusters": len(clusters),
                    "blocks_before": blocks_before,
                    "blocks_after": len(current_blocks),
                }
            )

            logger.info(
                "Round %d: %d → %d blocks (%d merged)",
                iteration + 1,
                blocks_before,
                len(current_blocks),
                len(merged_ids),
            )

        final_count = len(current_blocks)
        reduction = (
            ((starting_count - final_count) / starting_count * 100)
            if starting_count > 0
            else 0
        )

        return {
            "blocks": current_blocks,
            "stats": {
                "starting_count": starting_count,
                "final_count": final_count,
                "blocks_removed": starting_count - final_count,
                "reduction_percent": round(reduction, 2),
                "iterations_run": len(rounds),
            },
            "rounds": rounds,
        }
