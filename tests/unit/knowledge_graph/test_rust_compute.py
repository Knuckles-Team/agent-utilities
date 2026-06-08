"""CONCEPT:KG-2.2 Rust epistemic-graph compute unit tests.
Verifies the optimized graph functions using the Tokio-based epistemic-graph daemon.
"""

import uuid

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


@pytest.fixture
def engine():
    name = f"test_{uuid.uuid4().hex[:8]}"
    g = GraphComputeEngine(backend_type="rust", graph_name=name)
    if g._client:
        try:
            g._client.create_graph(name)
        except Exception:
            pass
        g._client.clear()
    return g


def test_rust_graph_compute_nodes_and_edges(engine):
    """Verify that the optimized Tokio-based epistemic_graph service can build and manage nodes and edges."""

    # Verify properties
    assert engine._mode in ["service", "embedded"]

    # Add nodes
    engine.add_node("A", {"label": "Agent"})
    engine.add_node("B", {"label": "Tool"})
    engine.add_node("C", {"label": "Knowledge"})

    # Verify node count
    assert engine.node_count() == 3

    # Add edges
    engine.add_edge("A", "B", {"weight": 1.5})
    engine.add_edge("B", "C", {"weight": 2.0})

    # Verify nodes and edges exist
    assert engine.has_node("A")
    assert engine.has_node("B")
    assert engine.has_node("C")
    assert engine.has_edge("A", "B")
    assert engine.has_edge("B", "C")
    assert not engine.has_edge("A", "C")

    # Remove edge
    engine.remove_edge("A", "B")
    assert not engine.has_edge("A", "B")

    # Remove node
    engine.remove_node("C")
    assert not engine.has_node("C")


def test_rust_topological_sort_and_cycles(engine):
    """Verify topological sorting and cycle detection on the optimized Tokio-based epistemic_graph service."""

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

    # Introduce cycle C -> A
    engine.add_edge("C", "A", {})

    # Cycle detection
    cycle = engine.find_cycle()
    assert cycle is not None
    assert len(cycle) > 1

    # Topological sort should now raise an error due to the cycle
    with pytest.raises(ValueError, match="Graph contains cycles"):
        engine.topological_sort()


def test_rust_shortest_path(engine):
    """Verify shortest path computations on the optimized Tokio-based epistemic_graph service."""

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


def test_rust_blast_radius(engine):
    """Verify blast radius computation on the optimized Tokio-based epistemic_graph service."""

    engine.add_node("A", {})
    engine.add_node("B", {})
    engine.add_node("C", {})
    engine.add_node("D", {})

    engine.add_edge("A", "B", {})
    engine.add_edge("B", "C", {})
    engine.add_edge("B", "D", {})

    blast = engine.get_blast_radius("A", max_depth=2)
    # Expected outgoing neighbors from A: B (depth 1), then C and D (depth 2)
    assert len(blast) == 3
    node_ids = {node["id"] for node in blast}
    assert node_ids == {"B", "C", "D"}


def test_rust_repository_ast_parsing(engine, tmp_path):
    """Verify repository AST ingestion natively on the optimized Tokio-based epistemic_graph service."""
    p1 = tmp_path / "mod.py"
    p1.write_text("class MyAgent:\n    pass\ndef my_tool():\n    pass\n")

    engine.parse_repository(str(tmp_path))

    # Verify classes and functions found

    nodes = engine._get_all_nodes()
    assert any("MyAgent" in n for n in nodes)
    assert any("my_tool" in n for n in nodes)


def test_rust_vf2_subgraph_matching(engine):
    """Verify subgraph isomorphism matching natively on the optimized Tokio-based epistemic_graph service."""
    import uuid

    engine.add_node("A", {"type": "class"})
    engine.add_node("B", {"type": "function"})
    engine.add_node("C", {"type": "function"})
    engine.add_edge("A", "B", {})
    engine.add_edge("A", "C", {})

    pattern_name = f"test_pattern_{uuid.uuid4().hex[:8]}"
    pattern = GraphComputeEngine(backend_type="rust", graph_name=pattern_name)
    if pattern._client:
        try:
            pattern._client.create_graph(pattern_name)
        except Exception:
            pass
        pattern._client.clear()

    pattern.add_node("P1", {"type": "class"})
    pattern.add_node("P2", {"type": "function"})
    pattern.add_edge("P1", "P2", {})

    matches = engine.vf2_subgraph_match(pattern)
    assert len(matches) == 2
    # Verify that pattern P1 matches A, P2 matches B or C
    assert matches[0]["P1"] == "A"
    assert matches[1]["P1"] == "A"
    matched_p2 = {m["P2"] for m in matches}
    assert matched_p2 == {"B", "C"}


def test_rust_reactive_state_ledger(engine):
    """Verify reactive state ledger serialization/deserialization and apply/replay."""
    import uuid

    engine.add_node("A", {"label": "Alpha"})
    engine.add_node("B", {"label": "Beta"})
    engine.add_edge("A", "B", {"weight": 1.0})

    # Capture ledger
    txs = engine.get_ledger()
    assert len(txs) >= 3

    # Replay onto another engine
    engine2_name = f"test_{uuid.uuid4().hex[:8]}"
    engine2 = GraphComputeEngine(backend_type="rust", graph_name=engine2_name)
    if engine2._client:
        try:
            engine2._client.create_graph(engine2_name)
        except Exception:
            pass
        engine2._client.clear()

    engine2.apply_ledger(txs)
    assert engine2.node_count() == 2
    assert engine2.has_edge("A", "B")

    # Verify msgpack serialization
    js = engine.to_msgpack()
    engine3_name = f"test_{uuid.uuid4().hex[:8]}"
    engine3 = GraphComputeEngine(backend_type="rust", graph_name=engine3_name)
    if engine3._client:
        try:
            engine3._client.create_graph(engine3_name)
        except Exception:
            pass
        engine3._client.clear()

    engine3.from_msgpack(js)
    assert engine3.node_count() == 2
    assert engine3.has_edge("A", "B")
