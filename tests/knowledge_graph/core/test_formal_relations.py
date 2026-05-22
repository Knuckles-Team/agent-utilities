"""CONCEPT:KG-2.4"""

import networkx as nx

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    equivalence_classes,
    hasse_diagram,
    is_equivalence_relation,
    is_reflexive,
    is_symmetric,
    is_transitive,
    resolve_entities,
)


def test_is_reflexive():
    G = nx.DiGraph()
    G.add_nodes_from([1, 2])
    G.add_edge(1, 1)
    assert not is_reflexive(G)
    G.add_edge(2, 2)
    assert is_reflexive(G)


def test_is_symmetric():
    G = nx.DiGraph()
    G.add_edge(1, 2)
    assert not is_symmetric(G)
    G.add_edge(2, 1)
    assert is_symmetric(G)


def test_is_transitive():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3)])
    assert not is_transitive(G)
    G.add_edge(1, 3)
    assert is_transitive(G)


def test_is_equivalence_relation():
    G = nx.DiGraph()
    G.add_edges_from([(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)])
    assert is_equivalence_relation(G)
    G.add_edge(2, 3)
    # Not symmetric or transitive
    assert not is_equivalence_relation(G)


def test_equivalence_classes():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 1), (3, 4), (4, 3)])
    classes = equivalence_classes(G)
    assert len(classes) == 2
    assert {1, 2} in classes
    assert {3, 4} in classes


def test_resolve_entities():
    equivalences = [("A", "B"), ("B", "C"), ("X", "Y")]
    resolution = resolve_entities(equivalences)
    assert resolution["A"] == "A"
    assert resolution["B"] == "A"
    assert resolution["C"] == "A"
    assert resolution["X"] == "X"
    assert resolution["Y"] == "X"


def test_hasse_diagram():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    hasse = hasse_diagram(G)
    assert hasse.has_edge(1, 2)
    assert hasse.has_edge(2, 3)
    assert not hasse.has_edge(1, 3)  # Redundant edge removed
