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


def _native_communities(graph: Any) -> list[set[Any]] | None:
    """Return native communities, or ``None`` when fallback is required."""
    if not hasattr(graph, "community_detection"):
        return None

    try:
        clusters: dict[int, set[Any]] = {}
        for node_id, label in graph.community_detection():
            clusters.setdefault(label, set()).add(node_id)
        return [community for community in clusters.values() if len(community) > 1]
    except Exception as exc:
        logger.warning(f"GCE community detection failed: {exc}")
        return None


def _build_graph(graph: Any, rx: Any) -> Any:
    """Adapt a graph-like object to the fallback ``PyGraph`` representation."""
    if isinstance(graph, rx.PyGraph):
        return graph

    result = rx.PyGraph()
    node_map: dict[Any, int] = {}
    if hasattr(graph, "node_ids"):
        for node_id in graph.node_ids():
            node_map[node_id] = result.add_node(node_id)
        for source, target in graph._get_all_edges():
            if source in node_map and target in node_map:
                result.add_edge(node_map[source], node_map[target], 1.0)
    return result


def _component_from(graph_obj: Any, start_idx: int, visited: set[int]) -> set[Any]:
    """Collect one connected component with a stack-based traversal."""
    component: set[Any] = set()
    stack = [start_idx]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        component.add(graph_obj[node])
        stack.extend(nb for nb in graph_obj.neighbors(node) if nb not in visited)
    return component


def _connected_components(graph_obj: Any) -> list[set[Any]]:
    """Find connected components using the fallback graph's BFS-compatible API."""
    visited: set[int] = set()
    components: list[set[Any]] = []
    for start_idx in graph_obj.node_indices():
        if start_idx in visited:
            continue
        component = _component_from(graph_obj, start_idx, visited)
        if len(component) > 1:
            components.append(component)
    return components


def _skip_removed_edge(current: int, neighbor: int, source: int, target: int) -> bool:
    """Identify either direction of the edge being tested for removal."""
    return (current == source and neighbor == target) or (
        current == target and neighbor == source
    )


def _has_alternate_path(graph_obj: Any, source: int, target: int) -> bool:
    """Return whether ``source`` can reach ``target`` without their direct edge."""
    visited: set[int] = set()
    queue = [source]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in graph_obj.neighbors(current):
            if _skip_removed_edge(current, neighbor, source, target):
                continue
            if neighbor == target:
                return True
            if neighbor not in visited:
                queue.append(neighbor)
    return False


def _find_bridges(graph_obj: Any) -> list[tuple[int, int]]:
    """Find undirected edges without an alternate path between their endpoints."""
    bridges: list[tuple[int, int]] = []
    for source in graph_obj.node_indices():
        for target in graph_obj.neighbors(source):
            if source >= target:
                continue
            if not _has_alternate_path(graph_obj, source, target):
                bridges.append((source, target))
    return bridges


def _without_bridges(graph_obj: Any, bridges: list[tuple[int, int]], rx: Any) -> Any:
    """Copy ``graph_obj`` while omitting bridge edges."""
    result = rx.PyGraph()
    index_map = {
        index: result.add_node(graph_obj[index]) for index in graph_obj.node_indices()
    }
    bridge_set = set(bridges)
    bridge_set.update((target, source) for source, target in bridges)
    for source in graph_obj.node_indices():
        for target in graph_obj.neighbors(source):
            if source >= target or (source, target) in bridge_set:
                continue
            result.add_edge(index_map[source], index_map[target], 1.0)
    return result


def _fallback_communities(graph: Any, rx: Any) -> list[set[Any]]:
    """Run bridge-removal community detection on a graph-like object."""
    graph_obj = _build_graph(graph, rx)
    if graph_obj.num_nodes() < 2:
        return []

    bridges = _find_bridges(graph_obj)
    if bridges:
        components = _connected_components(_without_bridges(graph_obj, bridges, rx))
        if len(components) > 1:
            return components
    return _connected_components(graph_obj)


def detect_communities(graph: Any) -> list[set[str]]:
    """Detect emergent communities using GraphComputeEngine.

    CONCEPT:AU-KG.compute.topological-mincut-partitioning

    Args:
        graph: The GraphComputeEngine or compatible graph object.

    Returns:
        A list of sets, where each set contains the node IDs belonging
        to a specific community.
    """
    native = _native_communities(graph)
    if native is not None:
        return native

    try:
        from agent_utilities.knowledge_graph.core import graph_primitives as rx

        return _fallback_communities(graph, rx)
    except Exception as exc:
        logger.error(f"Community detection fallback failed: {exc}")
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
