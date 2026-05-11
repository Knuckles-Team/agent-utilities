"""CONCEPT:KG-2.0"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def get_owl_context(node_id: str, graph: nx.MultiDiGraph) -> str:
    """Extracts OWL inferred facts for a node."""

    if node_id not in graph:
        return ""

    inferred_facts = []
    # Look at outgoing edges
    for _, tgt, data in graph.out_edges(node_id, data=True):
        if data.get("inferred", False):
            edge_type = data.get("type", "related_to")
            inferred_facts.append(f"{edge_type}: {tgt}")

    if inferred_facts:
        return "OWL Inferred Facts: " + " | ".join(inferred_facts)
    return ""


def get_hierarchical_context(
    node_id: str,
    graph: nx.MultiDiGraph,
    max_depth: int = 2,
    max_relations_per_level: int = 5,
) -> str:
    """Extracts multi-hop parents and children up to max_depth."""
    if node_id not in graph:
        return ""

    context_parts = []

    # Track nodes to avoid cycles
    visited_parents = {node_id}
    visited_children = {node_id}

    current_parents = [node_id]
    current_children = [node_id]

    for level in range(1, max_depth + 1):
        # Gather next level parents (incoming edges)
        next_parents = []
        for n in current_parents:
            for src, _ in graph.in_edges(n):
                if src not in visited_parents:
                    visited_parents.add(src)
                    next_parents.append(src)

        # Gather next level children (outgoing edges)
        next_children = []
        for n in current_children:
            for _, tgt in graph.out_edges(n):
                if tgt not in visited_children:
                    visited_children.add(tgt)
                    next_children.append(tgt)

        if next_parents:
            label = (
                "Parents"
                if level == 1
                else "Grandparents"
                if level == 2
                else f"L{level} Ancestors"
            )
            limited_parents = next_parents[:max_relations_per_level]
            context_parts.append(f"{label}: [{', '.join(limited_parents)}]")

        if next_children:
            label = (
                "Children"
                if level == 1
                else "Grandchildren"
                if level == 2
                else f"L{level} Descendants"
            )
            limited_children = next_children[:max_relations_per_level]
            context_parts.append(f"{label}: [{', '.join(limited_children)}]")

        current_parents = next_parents
        current_children = next_children

    if context_parts:
        return "Topology: " + " | ".join(context_parts)
    return ""


def build_contextual_description(
    node_id: str, graph: nx.MultiDiGraph, base_description: str
) -> str:
    """
    Builds a rich contextual description by appending topological and ontological data.

    Args:
        node_id: The ID of the node.
        graph: The in-memory MultiDiGraph.
        base_description: The original text description of the node.

    Returns:
        A concatenated string containing the base description and graph context.
    """
    parts = [base_description] if base_description else [f"Node {node_id}"]

    owl_ctx = get_owl_context(node_id, graph)
    if owl_ctx:
        parts.append(owl_ctx)

    topo_ctx = get_hierarchical_context(node_id, graph, max_depth=2)
    if topo_ctx:
        parts.append(topo_ctx)

    return "\n\n".join(parts)
