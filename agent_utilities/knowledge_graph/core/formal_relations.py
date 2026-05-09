#!/usr/bin/env python3
"""Formal Relations and Equivalence Classes.

CONCEPT:KG-2.47 — Formal Relations Engine

Implements mathematical relation properties (Reflexive, Symmetric, Transitive)
and Equivalence Classes from *Mathematics for Computer Science* (MCS Ch 4).
Provides zero-shot entity resolution by formally defining equivalence sets
across the Knowledge Graph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx

logger = logging.getLogger(__name__)


def is_reflexive(graph: nx.DiGraph, nodes: Iterable[str] | None = None) -> bool:
    """Check if the relation is reflexive over the given nodes.

    A relation R on A is reflexive if for all a in A, aRa.
    """
    node_set = set(nodes) if nodes is not None else set(graph.nodes())
    return all(graph.has_edge(n, n) for n in node_set)


def is_symmetric(graph: nx.DiGraph) -> bool:
    """Check if the relation is symmetric.

    A relation R is symmetric if aRb implies bRa.
    """
    return all(graph.has_edge(v, u) for u, v in graph.edges())


def is_transitive(graph: nx.DiGraph) -> bool:
    """Check if the relation is transitive.

    A relation R is transitive if aRb and bRc implies aRc.
    """
    for u, v in graph.edges():
        for _, w in graph.out_edges(v):
            if not graph.has_edge(u, w):
                return False
    return True


def is_equivalence_relation(
    graph: nx.DiGraph, nodes: Iterable[str] | None = None
) -> bool:
    """Check if a directed graph represents an equivalence relation."""
    return is_reflexive(graph, nodes) and is_symmetric(graph) and is_transitive(graph)


def equivalence_classes(graph: nx.DiGraph) -> list[set[str]]:
    """Compute equivalence classes for a symmetric and transitive relation.

    Returns a list of disjoint sets of nodes that are equivalent.
    If the graph is symmetric and transitive, its connected components
    form the equivalence classes.
    """
    if not is_symmetric(graph):
        logger.warning(
            "Graph is not symmetric. Treating edges as undirected for equivalence classes."
        )

    undirected = graph.to_undirected()
    classes = list(nx.connected_components(undirected))
    return [set(c) for c in classes]


def transitive_closure(graph: nx.DiGraph) -> nx.DiGraph:
    """Compute the transitive closure of a relation."""
    return nx.transitive_closure(graph)


def hasse_diagram(graph: nx.DiGraph) -> nx.DiGraph:
    """Compute the Hasse diagram (transitive reduction) of a DAG.

    Useful for partial orders (posets).
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph is not a DAG. Cannot compute Hasse diagram.")
    return nx.transitive_reduction(graph)


def resolve_entities(equivalences: list[tuple[str, str]]) -> dict[str, str]:
    """Zero-shot entity resolution using equivalence classes.

    Given a list of equivalence pairs (u, v), computes the equivalence
    classes and maps every entity to a canonical representative
    (the lexicographically smallest ID in its class).

    Args:
        equivalences: List of (entity1, entity2) tuples.

    Returns:
        Mapping from entity_id to canonical_entity_id.
    """
    G = nx.DiGraph()
    G.add_edges_from(equivalences)

    classes = equivalence_classes(G)
    resolution_map = {}

    for eq_class in classes:
        canonical = min(eq_class)
        for node in eq_class:
            resolution_map[node] = canonical

    return resolution_map
