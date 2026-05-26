"""CONCEPT:KG-2.2 Rustworkx graph compute unit tests."""

import pytest
pytest.importorskip("rustworkx")
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

def test_rustworkx_graph_compute_nodes_and_edges():
    """Verify that the rustworkx backend can build and manage nodes and edges."""
    engine = GraphComputeEngine(backend_type="rustworkx")

    # Verify properties
    assert engine.backend_type == "rustworkx"

    # Add nodes
    engine.add_node("A", {"label": "Agent"})
    engine.add_node("B", {"label": "Tool"})
    engine.add_node("C", {"label": "Knowledge"})

    # Verify node count
    assert engine.node_count() == 3

    # Add edges
    engine.add_edge("A", "B", {"weight": 1.5})
    engine.add_edge("B", "C", {"weight": 2.0})

    # Verify edge count
    assert engine.edge_count() == 2

def test_rustworkx_topological_sort_and_cycles():
    """Verify topological sorting and cycle detection on rustworkx backend."""
    engine = GraphComputeEngine(backend_type="rustworkx")

    engine.add_node("A", {})
    engine.add_node("B", {})
    engine.add_node("C", {})

    engine.add_edge("A", "B", {})
    engine.add_edge("B", "C", {})

    # Verify topological order
    order = engine.topological_sort()
    assert order == ["A", "B", "C"]

    # Verify no cycle is found
    assert engine.find_cycle() is None

    # Introduce cycle B -> A
    engine.add_edge("C", "A", {})

    # Cycle detection
    cycle = engine.find_cycle()
    assert cycle is not None
    assert len(cycle) > 1

    # Topological sort should now raise an error due to the cycle
    with pytest.raises(ValueError, match="Graph contains cycles"):
        engine.topological_sort()

def test_rustworkx_shortest_path():
    """Verify shortest path computations on rustworkx backend."""
    engine = GraphComputeEngine(backend_type="rustworkx")

    engine.add_node("X", {})
    engine.add_node("Y", {})
    engine.add_node("Z", {})

    engine.add_edge("X", "Y", {})
    engine.add_edge("Y", "Z", {})

    # Get shortest path
    path = engine.get_shortest_path("X", "Z")
    assert path == ["X", "Y", "Z"]

    # Path for unconnected elements
    engine.add_node("W", {})
    assert engine.get_shortest_path("X", "W") is None
