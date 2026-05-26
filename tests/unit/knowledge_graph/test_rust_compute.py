"""CONCEPT:KG-2.2 Rust epistemic-graph compute unit tests."""

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def test_rust_graph_compute_nodes_and_edges():
    """Verify that the Rust epistemic_graph backend can build and manage nodes and edges."""
    engine = GraphComputeEngine(backend_type="rust")

    # Verify properties
    assert engine.backend_type == "rust"

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

def test_rust_topological_sort_and_cycles():
    """Verify topological sorting and cycle detection on Rust epistemic_graph backend."""
    engine = GraphComputeEngine(backend_type="rust")

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

def test_rust_shortest_path():
    """Verify shortest path computations on Rust epistemic_graph backend."""
    engine = GraphComputeEngine(backend_type="rust")

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

def test_rust_blast_radius():
    """Verify blast radius computation on Rust epistemic_graph backend."""
    engine = GraphComputeEngine(backend_type="rust")

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


def test_rust_repository_ast_parsing(tmp_path):
    """Verify repository AST ingestion natively or via fallback."""
    p1 = tmp_path / "mod.py"
    p1.write_text("class MyAgent:\n    pass\ndef my_tool():\n    pass\n")

    for backend in ["rust", "networkx"]:
        engine = GraphComputeEngine(backend_type=backend)
        engine.parse_repository(str(tmp_path))

        # Verify classes and functions found
        if backend == "rust":
            nodes = [nid for nid, _ in engine._rust_graph.get_nodes()]
        else:
            nodes = engine._get_all_nodes()
        assert any("MyAgent" in n for n in nodes)
        assert any("my_tool" in n for n in nodes)


def test_rust_vf2_subgraph_matching():
    """Verify subgraph isomorphism matching natively and via fallback."""
    for backend in ["rust", "networkx"]:
        target = GraphComputeEngine(backend_type=backend)
        target.add_node("A", {"type": "class"})
        target.add_node("B", {"type": "function"})
        target.add_node("C", {"type": "function"})
        target.add_edge("A", "B", {})
        target.add_edge("A", "C", {})

        pattern = GraphComputeEngine(backend_type=backend)
        pattern.add_node("P1", {"type": "class"})
        pattern.add_node("P2", {"type": "function"})
        pattern.add_edge("P1", "P2", {})

        matches = target.vf2_subgraph_match(pattern)
        assert len(matches) == 2
        # Verify that pattern P1 matches A, P2 matches B or C
        assert matches[0]["P1"] == "A"
        assert matches[1]["P1"] == "A"
        matched_p2 = {m["P2"] for m in matches}
        assert matched_p2 == {"B", "C"}


def test_rust_reactive_state_ledger():
    """Verify reactive state ledger serialization/deserialization and apply/replay."""
    for backend in ["rust", "networkx"]:
        engine = GraphComputeEngine(backend_type=backend)
        engine.add_node("A", {"label": "Alpha"})
        engine.add_node("B", {"label": "Beta"})
        engine.add_edge("A", "B", {"weight": 1.0})

        # Capture ledger
        txs = engine.get_ledger()
        assert len(txs) >= 3

        # Replay onto another engine
        engine2 = GraphComputeEngine(backend_type=backend)
        engine2.apply_ledger(txs)
        assert engine2.node_count() == 2
        assert engine2.has_edge("A", "B")

        # Verify json serialization
        js = engine.to_json()
        engine3 = GraphComputeEngine(backend_type=backend)
        engine3.from_json(js)
        assert engine3.node_count() == 2
        assert engine3.has_edge("A", "B")


def test_rust_datalog_reasoning():
    """Verify high-performance compiled Rust Datalog OWL reasoning."""
    from typing import cast
    from agent_utilities.knowledge_graph.backends.owl import create_owl_backend
    from agent_utilities.knowledge_graph.backends.owl.oxigraph_datalog_backend import OxigraphDatalogBackend

    # 1. Create oxigraph datalog backend
    backend = create_owl_backend("oxigraph")
    assert backend.__class__.__name__ == "OxigraphDatalogBackend"

    backend_ox = cast(OxigraphDatalogBackend, backend)

    # Define schema rules
    backend_ox.subclass_relations = [("Agent", "Resource")]
    backend_ox.subproperty_relations = [("executes", "interactsWith")]
    backend_ox.symmetric_properties = ["partnerOf"]
    backend_ox.transitive_properties = ["dependsOn"]
    backend_ox.inverse_properties = [("childOf", "parentOf")]

    # 2. Promote node ABox individuals
    backend_ox.promote([
        {"id": "agent1", "type": "agent", "name": "TaskAgent"},
        {"id": "tool1", "type": "tool", "name": "TerminalTool"},
        {"id": "node2", "type": "concept", "name": "TargetNode"},
    ])

    # 3. Promote edge relationships
    backend_ox.promote_edges([
        {"source": "agent1", "target": "tool1", "type": "executes"},
        {"source": "tool1", "target": "node2", "type": "dependsOn"},
        {"source": "agent1", "target": "tool1", "type": "partnerOf"},
        {"source": "tool1", "target": "agent1", "type": "childOf"},
    ])

    # 4. Run native Rust reasoning
    inferences = backend_ox.reason()
    assert len(inferences) > 0

    # Extract mapped inferences
    triples = {(inf["subject"], inf["predicate"], inf["object"]) for inf in inferences}

    # Verify rule matches:
    # Rule 1 (Subclass): agent1 is an Agent -> agent1 is a Resource
    assert ("agent1", "type", "Resource") in triples

    # Rule 2 (Subproperty): agent1 -executes-> tool1 -> agent1 -interactsWith-> tool1
    assert ("agent1", "interactsWith", "tool1") in triples

    # Rule 3 (Symmetric): agent1 -partnerOf-> tool1 -> tool1 -partnerOf-> agent1
    assert ("tool1", "partnerOf", "agent1") in triples

    # Rule 4 (Inverse): tool1 -childOf-> agent1 -> agent1 -parentOf-> tool1
    assert ("agent1", "parentOf", "tool1") in triples


def test_rust_quant_factor_math():
    """Verify high-performance compiled Rust vectorized quantitative math."""
    from epistemic_graph import EpistemicGraph

    eg = EpistemicGraph()

    # Define raw time-series values
    values = [10.0, 12.0, 11.0, 15.0, 14.0]

    # Test rolling zscore with window = 3
    zscores = eg.compute_rolling_zscore(values, 3)
    assert len(zscores) == 5
    # The last value (14.0) z-score within window [11.0, 15.0, 14.0] (mean = 13.33)
    assert zscores[4] > 0.0

    # Test rolling standard deviation
    stds = eg.compute_rolling_std(values, 3)
    assert len(stds) == 5
    assert stds[0] == 0.0

    # Test exponential decay alpha = 0.5
    decay = eg.compute_exponential_decay(values, 0.5)
    assert len(decay) == 5
    assert decay[0] == 10.0
    assert decay[1] == 11.0  # 0.5 * 12 + 0.5 * 10


def test_rust_orderbook_matching():
    """Verify high-performance compiled Rust order book tick simulation matching."""
    from epistemic_graph import EpistemicGraph

    eg = EpistemicGraph()

    bids = [(100.0, 10.0), (99.0, 20.0)]
    asks = [(101.0, 5.0), (102.0, 15.0)]

    # Buy order at 101.5 for 6 shares (should match 5 shares at 101.0)
    orders = [("ord_1", "buy", 101.5, 6.0)]

    matches = eg.simulate_order_matching(bids, asks, orders)
    assert len(matches) == 1
    assert matches[0]["order_id"] == "ord_1"
    assert float(matches[0]["match_price"]) == 101.0
    assert float(matches[0]["match_volume"]) == 5.0
