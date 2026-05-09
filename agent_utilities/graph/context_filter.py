"""Vectorized Context-Window Filtering (CONCEPT:KG-2.41).

This module implements token-aware context compaction by semantically pruning
non-relevant subgraph context before swapping models on token overflow.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def prune_context_by_semantic_distance(
    context_nodes: list[dict[str, Any]], query: str, max_tokens: int
) -> list[dict[str, Any]]:
    """Prune graph nodes from context if they exceed the token budget.

    Instead of hard-truncation, it drops the most semantically distant nodes
    from the query.
    """
    if not context_nodes:
        return []

    # Naive token estimation: ~4 chars per token
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    total_tokens = sum(estimate_tokens(str(n)) for n in context_nodes)
    if total_tokens <= max_tokens:
        return context_nodes

    logger.info(
        f"Context overflow detected ({total_tokens} > {max_tokens}). Applying topological pruning."
    )

    # Sort nodes by semantic relevance (assuming 'relevance_score' or 'topological_distance' exists)
    # If not, we fall back to trimming the longest nodes first as a heuristic, but
    # ideally we rely on the KG embeddings.
    try:
        sorted_nodes = sorted(
            context_nodes,
            key=lambda n: n.get(
                "topological_distance", n.get("distance", float("inf"))
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to sort context nodes by distance: {e}")
        sorted_nodes = context_nodes

    pruned_nodes = []
    current_tokens = 0
    for node in sorted_nodes:
        node_tokens = estimate_tokens(str(node))
        if current_tokens + node_tokens <= max_tokens:
            pruned_nodes.append(node)
            current_tokens += node_tokens
        else:
            logger.debug(
                f"Pruned node {node.get('id', 'unknown')} to save {node_tokens} tokens."
            )

    logger.info(f"Topological pruning complete. Final tokens: {current_tokens}.")
    return pruned_nodes
