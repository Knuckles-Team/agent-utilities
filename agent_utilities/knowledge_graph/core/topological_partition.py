"""Topological Mincut Partitioning and Community Detection.

CONCEPT:KG-2.5 — Mincut Partitioning
This module uses NetworkX Louvain community detection to dynamically
partition the Knowledge Graph into emergent topological communities.
Stable communities are persisted back to the backend.
"""

import logging
from typing import Any

import networkx as nx

from agent_utilities.models.knowledge_graph import (
    CommunityNode,
    RegistryEdge,
    RegistryEdgeType,
)

logger = logging.getLogger(__name__)


def detect_communities(graph: nx.Graph) -> list[set[str]]:
    """Detect emergent communities in a NetworkX graph using Louvain method.

    CONCEPT:KG-2.5

    Args:
        graph: The undirected NetworkX graph to partition.

    Returns:
        A list of sets, where each set contains the node IDs belonging
        to a specific community.
    """
    if len(graph) == 0:
        return []

    try:
        # Use louvain communities to detect emergent clusters
        communities = nx.community.louvain_communities(graph, weight="weight")
        return [set(c) for c in communities if len(c) > 1]
    except Exception as e:
        logger.warning(
            f"Louvain community detection failed: {e}. Falling back to Label Propagation."
        )
        try:
            communities = nx.community.label_propagation_communities(graph)
            return [set(c) for c in communities if len(c) > 1]
        except Exception as fallback_e:
            logger.error(f"Label Propagation fallback failed: {fallback_e}")
            return []


def persist_stable_communities(engine: Any) -> int:
    """Detect and persist stable communities into the Cypher backend.

    CONCEPT:KG-2.5

    Called by the maintenance cron to permanently register topological
    waypoints in the graph.

    Args:
        engine: The KnowledgeGraphEngine instance.

    Returns:
        The number of communities persisted.
    """
    logger.info("Starting topological partitioning of knowledge base...")

    # 1. Fetch current networkx graph (assume engine.get_networkx_graph() or similar exists)
    try:
        if hasattr(engine, "get_networkx_snapshot"):
            nx_graph = engine.get_networkx_snapshot()
        elif hasattr(engine, "_nx_graph"):
            nx_graph = engine._nx_graph
        else:
            logger.warning("Engine does not support NetworkX snapshotting.")
            return 0
    except Exception as e:
        logger.warning(f"Could not retrieve NetworkX snapshot: {e}")
        return 0

    if not isinstance(nx_graph, nx.Graph):
        # Ensure it's undirected for Louvain
        undirected_graph = nx_graph.to_undirected()
    else:
        undirected_graph = nx_graph

    communities = detect_communities(undirected_graph)
    persisted_count = 0

    for i, comm in enumerate(communities):
        # We consider a community "stable" if it has > 3 members
        if len(comm) < 3:
            continue

        comm_id = f"community_cluster_{i}"

        # Calculate naive coherence (just ratio of internal edges to possible internal edges)
        subgraph = undirected_graph.subgraph(comm)
        actual_edges = subgraph.number_of_edges()
        possible_edges = len(comm) * (len(comm) - 1) / 2
        coherence = (actual_edges / possible_edges) if possible_edges > 0 else 1.0

        community_node = CommunityNode(
            id=comm_id,
            name=f"Emergent Cluster {i}",
            description=f"Auto-detected topological community with {len(comm)} members.",
            coherence_score=coherence,
            member_count=len(comm),
            is_permanent=True,
        )

        try:
            # Upsert the node
            engine.upsert_node(community_node)

            # Upsert the edges connecting members to the community
            for node_id in comm:
                # Ensure node exists before linking
                edge = RegistryEdge(
                    source=str(node_id),
                    target=comm_id,
                    type=RegistryEdgeType.PART_OF_COMMUNITY,
                    weight=coherence,
                )
                engine.upsert_edge(edge)

            persisted_count += 1
        except Exception as e:
            logger.error(f"Failed to persist community {comm_id}: {e}")

    logger.info(f"Persisted {persisted_count} emergent communities.")
    return persisted_count
