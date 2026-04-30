#!/usr/bin/python
"""Unit tests for OWLBridge."""

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import (
    Owlready2Backend,
)
from agent_utilities.knowledge_graph.owl_bridge import OWLBridge


@pytest.fixture
def ontology_path():
    return str(Path(__file__).parent.parent.parent.parent / "agent_utilities" / "knowledge_graph" / "ontology.ttl")

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    engine.backend = None
    return engine

def test_bridge_run_cycle(mock_engine, ontology_path, monkeypatch):
    # Setup graph with promotable nodes and edges
    mock_engine.graph.add_node("symbol:A", type="symbol", importance_score=0.9)
    mock_engine.graph.add_node("symbol:B", type="symbol", importance_score=0.9)
    mock_engine.graph.add_node("symbol:C", type="symbol", importance_score=0.9)

    mock_engine.graph.add_edge("symbol:A", "symbol:B", type="depends_on")
    mock_engine.graph.add_edge("symbol:B", "symbol:C", type="depends_on")

    backend = Owlready2Backend(ontology_path=ontology_path)

    # Mock reasoner to simulate transitive inference A -> C
    def mock_reasoner(*args, **kwargs):
        onto = backend._onto
        symbol_A = onto.search_one(iri="*symbol_A")
        symbol_C = onto.search_one(iri="*symbol_C")
        if symbol_A and symbol_C:
            symbol_A.dependsOn.append(symbol_C)

    import owlready2
    monkeypatch.setattr(owlready2, "sync_reasoner_hermit", mock_reasoner)

    bridge = OWLBridge(graph=mock_engine.graph, owl_backend=backend, backend=mock_engine.backend)
    stats = bridge.run_cycle()

    assert stats["promoted_nodes"] == 3
    assert stats["promoted_edges"] == 2
    assert stats["inferred"] == 1
    assert stats["downfed"] == 1

    # Verify inference back in NX graph
    assert mock_engine.graph.has_edge("symbol:A", "symbol:C")
    edge_data = mock_engine.graph.get_edge_data("symbol:A", "symbol:C")[0]
    assert edge_data["type"] == "dependsOn"
    assert edge_data["inferred"] is True

    backend.close()

def test_bridge_eligibility(mock_engine, ontology_path):
    backend = Owlready2Backend(ontology_path=ontology_path)
    bridge = OWLBridge(
        graph=mock_engine.graph,
        owl_backend=backend,
        backend=mock_engine.backend,
        importance_threshold=0.5
    )

    # Important node
    assert bridge._is_eligible_node("1", {"type": "agent", "importance_score": 0.8}) is True

    # Unimportant node
    assert bridge._is_eligible_node("2", {"type": "agent", "importance_score": 0.2}) is False

    # Non-promotable type
    assert bridge._is_eligible_node("3", {"type": "unknown", "importance_score": 0.9}) is False

    # Permanent node (always eligible)
    assert bridge._is_eligible_node("4", {"type": "agent", "importance_score": 0.1, "is_permanent": True}) is True

    backend.close()
