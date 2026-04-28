#!/usr/bin/python
"""Unit tests for Owlready2Backend."""

import os
import pytest
from pathlib import Path
from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import Owlready2Backend

@pytest.fixture
def ontology_path():
    """Path to the standard ontology file."""
    return str(Path(__file__).parent.parent.parent.parent / "agent_utilities" / "knowledge_graph" / "ontology.ttl")

def test_owlready2_init(ontology_path):
    """Test backend initialization and ontology loading."""
    backend = Owlready2Backend(ontology_path=ontology_path)
    assert backend._onto is not None
    assert backend._world is not None
    stats = backend.get_stats()
    assert stats["classes"] > 0
    assert stats["properties"] > 0
    backend.close()

def test_owlready2_promote(ontology_path):
    """Test promotion of nodes to OWL individuals."""
    backend = Owlready2Backend(ontology_path=ontology_path)

    nodes = [
        {"id": "agent:test-agent", "type": "agent", "importance_score": 0.9},
        {"id": "tool:test-tool", "type": "tool"},
    ]

    count = backend.promote(nodes)  # type: ignore[arg-type]
    assert count == 2

    stats = backend.get_stats()
    assert stats["individuals"] == 2

    # Check if individuals exist in world
    agent = backend._onto.search_one(iri="*agent_test-agent")
    assert agent is not None
    assert "Agent" in [c.name for c in agent.is_a]

    backend.close()

def test_owlready2_promote_edges(ontology_path):
    """Test promotion of edges to OWL property assertions."""
    backend = Owlready2Backend(ontology_path=ontology_path)

    nodes = [
        {"id": "agent:test-agent", "type": "agent"},
        {"id": "tool:test-tool", "type": "tool"},
    ]
    backend.promote(nodes)

    edges = [
        {"source": "agent:test-agent", "target": "tool:test-tool", "type": "provides"}
    ]

    count = backend.promote_edges(edges)
    assert count == 1

    agent = backend._onto.search_one(iri="*agent_test-agent")
    tool = backend._onto.search_one(iri="*tool_test-tool")

    print(f"\nDebug: agent={agent}, tool={tool}")
    print(f"Debug: agent.provides={agent.provides if hasattr(agent, 'provides') else 'N/A'}")
    print(f"Debug: properties={[p.python_name for p in agent.get_properties()]}")

    assert tool in agent.provides

    backend.close()

def test_owlready2_reasoning(ontology_path, monkeypatch):
    """Test OWL reasoning (simulated to avoid Java dependency in tests)."""
    backend = Owlready2Backend(ontology_path=ontology_path)

    # A depends on B, B depends on C -> A depends on C (Transitive)
    nodes = [
        {"id": "symbol:A", "type": "symbol"},
        {"id": "symbol:B", "type": "symbol"},
        {"id": "symbol:C", "type": "symbol"},
    ]
    backend.promote(nodes)

    edges = [
        {"source": "symbol:A", "target": "symbol:B", "type": "depends_on"},
        {"source": "symbol:B", "target": "symbol:C", "type": "depends_on"},
    ]
    backend.promote_edges(edges)

    # Mock reasoner to simulate transitive inference
    def mock_reasoner(*args, **kwargs):
        # Manually add the inference to the ontology
        onto = backend._onto
        symbol_A = onto.search_one(iri="*symbol_A")
        symbol_C = onto.search_one(iri="*symbol_C")
        if symbol_A and symbol_C:
            symbol_A.dependsOn.append(symbol_C)

    import owlready2
    monkeypatch.setattr(owlready2, "sync_reasoner_hermit", mock_reasoner)

    inferences = backend.reason()

    # We expect an inference: A depends_on C
    found = False
    for inf in inferences:
        if inf["subject"] == "symbol_A" and inf["predicate"] == "dependsOn" and inf["object"] == "symbol_C":
            found = True
            break

    assert found, f"Inference A -> C not found in {inferences}"
    backend.close()

def test_owlready2_clear(ontology_path):
    """Test clearing ABox individuals."""
    backend = Owlready2Backend(ontology_path=ontology_path)
    backend.promote([{"id": "agent:1", "type": "agent"}])
    assert backend.get_stats()["individuals"] == 1

    backend.clear()
    assert backend.get_stats()["individuals"] == 0
    assert backend.get_stats()["classes"] > 0  # TBox preserved

    backend.close()
