"""CONCEPT:AU-KG.compute.cross-pillar-synergy"""

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    equivalence_classes,
    hasse_diagram,
    is_equivalence_relation,
    is_reflexive,
    is_symmetric,
    is_transitive,
    resolve_entities,
)
from agent_utilities.knowledge_graph.core.graph_primitives import PyDiGraph


def test_is_reflexive():
    G = PyDiGraph()
    n1 = G.add_node(1)
    n2 = G.add_node(2)
    G.add_edge(n1, n1, {})
    assert not is_reflexive(G)
    G.add_edge(n2, n2, {})
    assert is_reflexive(G)


def test_is_symmetric():
    G = PyDiGraph()
    n1 = G.add_node(1)
    n2 = G.add_node(2)
    G.add_edge(n1, n2, {})
    assert not is_symmetric(G)
    G.add_edge(n2, n1, {})
    assert is_symmetric(G)


def test_is_transitive():
    G = PyDiGraph()
    n1 = G.add_node(1)
    n2 = G.add_node(2)
    n3 = G.add_node(3)
    for u, v in [(n1, n2), (n2, n3)]:
        G.add_edge(u, v, {})
    assert not is_transitive(G)
    G.add_edge(n1, n3, {})
    assert is_transitive(G)


def test_is_equivalence_relation():
    G = PyDiGraph()
    n1 = G.add_node(1)
    n2 = G.add_node(2)
    n3 = G.add_node(3)
    for u, v in [(n1, n1), (n2, n2), (n3, n3), (n1, n2), (n2, n1)]:
        G.add_edge(u, v, {})
    assert is_equivalence_relation(G)
    G.add_edge(n2, n3, {})
    # Not symmetric or transitive
    assert not is_equivalence_relation(G)


def test_equivalence_classes():
    G = PyDiGraph()
    n1 = G.add_node(1)
    n2 = G.add_node(2)
    n3 = G.add_node(3)
    n4 = G.add_node(4)
    for u, v in [(n1, n2), (n2, n1), (n3, n4), (n4, n3)]:
        G.add_edge(u, v, {})
    classes = equivalence_classes(G)
    assert len(classes) == 2
    assert {"1", "2"} in classes
    assert {"3", "4"} in classes


def test_resolve_entities():
    equivalences = [("A", "B"), ("B", "C"), ("X", "Y")]
    resolution = resolve_entities(equivalences)
    assert resolution["A"] == "A"
    assert resolution["B"] == "A"
    assert resolution["C"] == "A"
    assert resolution["X"] == "X"
    assert resolution["Y"] == "X"


def test_hasse_diagram():
    G = PyDiGraph()
    n1 = G.add_node(1)
    n2 = G.add_node(2)
    n3 = G.add_node(3)
    for u, v in [(n1, n2), (n2, n3), (n1, n3)]:
        G.add_edge(u, v, {})
    hasse = hasse_diagram(G)
    assert hasse.has_edge(n1, n2)
    assert hasse.has_edge(n2, n3)
    assert not hasse.has_edge(n1, n3)  # Redundant edge removed
