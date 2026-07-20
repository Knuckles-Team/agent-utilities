"""Topological Mincut Partitioning and Community Detection.

CONCEPT:AU-KG.compute.topological-mincut-partitioning — Mincut Partitioning
This module uses GraphComputeEngine community detection to dynamically
partition the Knowledge Graph into emergent topological communities.
Stable communities are persisted back to the backend.
"""

import logging
from typing import Any

from agent_utilities.models.knowledge_graph import (
    CommunityNode,
    RegistryEdge,
    RegistryEdgeType,
)

logger = logging.getLogger(__name__)


def detect_communities(graph: Any) -> list[set[str]]:
    """Detect emergent communities using GraphComputeEngine.

    CONCEPT:AU-KG.compute.topological-mincut-partitioning

    Args:
        graph: The GraphComputeEngine or compatible graph object.

    Returns:
        A list of sets, where each set contains the node IDs belonging
        to a specific community.
    """
    # Try GCE native community detection first
    if hasattr(graph, "community_detection"):
        try:
            communities_raw = graph.community_detection()
            # Group by community label
            clusters: dict[int, set[str]] = {}
            for node_id, label in communities_raw:
                clusters.setdefault(label, set()).add(node_id)
            return [c for c in clusters.values() if len(c) > 1]
        except Exception as e:
            logger.warning(f"GCE community detection failed: {e}")

    # Graph primitives fallback
    try:
        from agent_utilities.knowledge_graph.core import graph_primitives as rx

        # If graph is already a PyGraph, use it directly
        if isinstance(graph, rx.PyGraph):
            G = graph
        else:
            G = rx.PyGraph()
            node_map: dict[str, int] = {}

            if hasattr(graph, "node_ids"):
                for node_id in graph.node_ids():
                    idx = G.add_node(node_id)
                    node_map[node_id] = idx
                for src, tgt in graph._get_all_edges():
                    if src in node_map and tgt in node_map:
                        G.add_edge(node_map[src], node_map[tgt], 1.0)

        # Community detection via bridge-edge removal
        # Bridge edges are edges whose removal increases the number of
        # connected components.  Removing them naturally splits the graph
        # into dense clusters (cliques / near-cliques).
        if G.num_nodes() < 2:
            return []

        def _connected_components(graph_obj) -> list[set]:
            """Find connected components using BFS."""
            vis: set[int] = set()
            comps: list[set] = []
            for start_idx in graph_obj.node_indices():
                if start_idx in vis:
                    continue
                comp: set = set()
                stack = [start_idx]
                while stack:
                    node = stack.pop()
                    if node in vis:
                        continue
                    vis.add(node)
                    comp.add(graph_obj[node])
                    stack.extend(
                        nb for nb in graph_obj.neighbors(node) if nb not in vis
                    )
                if len(comp) > 1:
                    comps.append(comp)
            return comps

        # Find bridge edges using naive method: try removing each edge
        # and check if the graph becomes disconnected.
        bridges = []
        for u_idx in G.node_indices():
            for v_idx in G.neighbors(u_idx):
                if u_idx >= v_idx:
                    continue  # avoid duplicate pairs
                # Temporarily check if removing this edge disconnects
                # by seeing if u can reach v without this edge
                visited_check: set[int] = set()
                queue = [u_idx]
                found = False
                while queue and not found:
                    current = queue.pop(0)
                    if current in visited_check:
                        continue
                    visited_check.add(current)
                    for nb in G.neighbors(current):
                        if nb == v_idx and current == u_idx:
                            continue  # skip this edge
                        if nb == u_idx and current == v_idx:
                            continue  # skip this edge (reverse direction)
                        if nb == v_idx:
                            found = True
                            break
                        if nb not in visited_check:
                            queue.append(nb)
                if not found:
                    bridges.append((u_idx, v_idx))

        if bridges:
            # Build new graph without bridge edges
            G2 = rx.PyGraph()
            idx_map: dict[int, int] = {}
            for idx in G.node_indices():
                idx_map[idx] = G2.add_node(G[idx])
            bridge_set = {(u, v) for u, v in bridges}
            bridge_set |= {(v, u) for u, v in bridges}
            for u_idx in G.node_indices():
                for v_idx in G.neighbors(u_idx):
                    if u_idx >= v_idx:
                        continue
                    if (u_idx, v_idx) not in bridge_set:
                        G2.add_edge(idx_map[u_idx], idx_map[v_idx], 1.0)

            comps = _connected_components(G2)
            if len(comps) > 1:
                return comps

        # Fallback: connected components on original graph
        return _connected_components(G)
    except Exception as e:
        logger.error(f"Community detection fallback failed: {e}")
        return []


def persist_stable_communities(engine: Any) -> int:
    """Detect and persist stable communities into the Cypher backend.

    CONCEPT:AU-KG.compute.topological-mincut-partitioning

    Called by the maintenance cron to permanently register topological
    waypoints in the graph.

    Args:
        engine: The KnowledgeGraphEngine instance.

    Returns:
        The number of communities persisted.
    """
    logger.info("Starting topological partitioning of knowledge base...")

    # Use the engine's graph (GraphComputeEngine) directly
    if not hasattr(engine, "graph"):
        logger.warning("Engine does not expose a graph attribute.")
        return 0

    graph = engine.graph
    communities = detect_communities(graph)
    persisted_count = 0

    for i, comm in enumerate(communities):
        # We consider a community "stable" if it has > 3 members
        if len(comm) < 3:
            continue

        comm_id = f"community_cluster_{i}"

        # Calculate naive coherence from edge density
        internal_edges = 0
        if hasattr(graph, "_get_all_edges"):
            for src, tgt in graph._get_all_edges():
                if src in comm and tgt in comm:
                    internal_edges += 1
        possible_edges = len(comm) * (len(comm) - 1) / 2
        coherence = (internal_edges / possible_edges) if possible_edges > 0 else 1.0

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
