import pytest
import networkx as nx
from agent_utilities.knowledge_graph.core.context_builder import (
    get_owl_context,
    get_hierarchical_context,
    build_contextual_description,
)


@pytest.mark.timeout(5)
def test_owl_context():
    graph = nx.MultiDiGraph()
    graph.add_node("n1")
    graph.add_node("n2")
    graph.add_edge("n1", "n2", type="is_a", inferred=True)

    ctx = get_owl_context("n1", graph)
    assert "OWL Inferred Facts: is_a: n2" in ctx


@pytest.mark.timeout(5)
def test_hierarchical_context():
    graph = nx.MultiDiGraph()
    # A -> B -> C -> D
    #      B -> E -> F
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")
    graph.add_node("E")
    graph.add_node("F")

    graph.add_edge("A", "B", type="parent")
    graph.add_edge("B", "C", type="parent")
    graph.add_edge("B", "E", type="parent")
    graph.add_edge("C", "D", type="parent")
    graph.add_edge("E", "F", type="parent")

    ctx = get_hierarchical_context("B", graph, max_depth=2)
    # B has parent A, children C, E, grandchildren D, F
    assert "Parents: [A]" in ctx
    assert "Children: [C, E]" in ctx or "Children: [E, C]" in ctx
    assert "Grandchildren: [D, F]" in ctx or "Grandchildren: [F, D]" in ctx


@pytest.mark.timeout(5)
def test_build_contextual_description():
    graph = nx.MultiDiGraph()
    graph.add_node("root", description="The root node")
    graph.add_node("child1")
    graph.add_edge("root", "child1", type="has_child")

    desc = build_contextual_description("root", graph, "The root node")
    assert "The root node" in desc
    assert "Children: [child1]" in desc
