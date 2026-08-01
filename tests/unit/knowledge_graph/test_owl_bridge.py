#!/usr/bin/python
"""CONCEPT:AU-KG.ingest.engineering-rules"""

"""Unit tests for OWLBridge."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

pytest.importorskip("owlready2")

from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import (
    Owlready2Backend,
)
from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge


@pytest.fixture
def ontology_path():
    return str(
        Path(__file__).parent.parent.parent.parent
        / "agent_utilities"
        / "knowledge_graph"
        / "ontology.ttl"
    )


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.graph = GraphComputeEngine(backend_type="rust")
    engine.backend = None
    return engine


def test_bridge_run_cycle(mock_engine, ontology_path, monkeypatch):
    # Setup graph with promotable nodes and edges
    mock_engine.graph.add_node("symbol:A", node_type="symbol", importance_score=0.9)
    mock_engine.graph.add_node("symbol:B", node_type="symbol", importance_score=0.9)
    mock_engine.graph.add_node("symbol:C", node_type="symbol", importance_score=0.9)

    mock_engine.graph.add_edge("symbol:A", "symbol:B", relationship="depends_on")
    mock_engine.graph.add_edge("symbol:B", "symbol:C", relationship="depends_on")

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

    bridge = OWLBridge(
        graph=mock_engine.graph, owl_backend=backend, backend=mock_engine.backend
    )
    stats = bridge.run_cycle(lightweight=False)

    assert stats["promoted_nodes"] == 3
    assert stats["promoted_edges"] == 2
    assert stats["inferred"] == 2
    assert stats["downfed"] == 2

    # Verify inference back in NX graph
    assert mock_engine.graph.has_edge("symbol:A", "symbol:C")
    edge_data = mock_engine.graph.get_edge_data("symbol:A", "symbol:C")[0]
    assert edge_data["relationship"] == "dependsOn"
    assert edge_data["inferred"] is True

    backend.close()


def test_bridge_eligibility(mock_engine, ontology_path):
    backend = Owlready2Backend(ontology_path=ontology_path)
    bridge = OWLBridge(
        graph=mock_engine.graph,
        owl_backend=backend,
        backend=mock_engine.backend,
        importance_threshold=0.5,
    )

    # Important node
    assert (
        bridge._is_eligible_node("1", {"node_type": "agent", "importance_score": 0.8})
        is True
    )

    # Unimportant node
    assert (
        bridge._is_eligible_node("2", {"node_type": "agent", "importance_score": 0.2})
        is False
    )

    # Non-promotable type
    assert (
        bridge._is_eligible_node("3", {"node_type": "unknown", "importance_score": 0.9})
        is False
    )

    # Permanent node (always eligible)
    assert (
        bridge._is_eligible_node(
            "4",
            {"node_type": "agent", "importance_score": 0.1, "is_permanent": True},
        )
        is True
    )

    backend.close()


def test_bridge_eligibility_camelcase_node_type():
    """D-TC-5 follow-on: ``_is_eligible_node`` must fold a CamelCased
    ``node_type`` (how the native compute engine round-trips it -- see
    ``_node_type_to_snake``'s docstring) back to the lowercase snake_case
    ``PROMOTABLE_NODE_TYPES`` convention, mirroring ``_promote_stable_edges``'s
    existing case-fold for the ``relationship`` property (D-GS7-1). No live
    engine needed: ``_is_eligible_node`` never touches ``self.graph``.
    """
    bridge = OWLBridge(graph=None, owl_backend=None, backend=None)

    assert (
        bridge._is_eligible_node(
            "host:1", {"node_type": "Host", "importance_score": 0.9}
        )
        is True
    )
    assert (
        bridge._is_eligible_node(
            "gpu:1", {"node_type": "GPUAccelerator", "importance_score": 0.9}
        )
        is True
    )
    assert (
        bridge._is_eligible_node(
            "storage:1", {"node_type": "StorageArray", "importance_score": 0.9}
        )
        is True
    )
    # A genuinely non-promotable type stays ineligible regardless of case.
    assert (
        bridge._is_eligible_node(
            "x:1", {"node_type": "TotallyUnknownType", "importance_score": 0.9}
        )
        is False
    )
