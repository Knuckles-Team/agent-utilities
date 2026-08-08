#!/usr/bin/python
"""CONCEPT:AU-KG.ingest.engineering-rules"""

"""Unit tests for Owlready2Backend."""

import pytest

pytest.importorskip("owlready2")

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import (
    Owlready2Backend,
)


@pytest.fixture
def ontology_path(monkeypatch):
    """Path to the standard ontology file."""
    monkeypatch.setenv("OWL_ALLOW_REMOTE_IMPORTS", "true")
    return str(
        Path(__file__).parent.parent.parent.parent
        / "agent_utilities"
        / "knowledge_graph"
        / "ontology.ttl"
    )


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
        {"id": "agent:test-agent", "node_type": "agent", "importance_score": 0.9},
        {"id": "tool:test-tool", "node_type": "tool"},
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


def test_owlready2_promote_camelcase_node_type(monkeypatch):
    """D-TC-5 follow-on: a node read back from the native compute engine has
    its ``node_type`` CamelCased (``"host"`` -> ``"Host"``, ``"gpu_accelerator"``
    -> ``"GPUAccelerator"``) rather than the lowercase snake_case
    ``RegistryNodeType``/``_NODE_TYPE_TO_OWL_CLASS`` convention. Before the
    ``_node_type_to_snake`` fold in ``_get_owl_class``, this silently promoted
    ZERO individuals for every node sourced from a live engine -- no error,
    just a permanently empty OWL world for that node's class (the same
    masking-bug class D-GS7-1 fixed for edge relationships).

    Uses ``ontology_infrastructure.ttl`` (not the general ``ontology.ttl`` the
    other tests in this file load) -- the actual ontology
    ``generate_matchmaking_recommendations`` loads, and the one that declares
    the canonical, unambiguous ``:Host``/``:GPUAccelerator``/``:StorageArray``
    classes this test asserts against.
    """
    monkeypatch.setenv("OWL_ALLOW_REMOTE_IMPORTS", "true")
    infra_ontology_path = str(
        Path(__file__).parent.parent.parent.parent
        / "agent_utilities"
        / "knowledge_graph"
        / "ontology_infrastructure.ttl"
    )
    backend = Owlready2Backend(ontology_path=infra_ontology_path)

    nodes = [
        {"id": "host:test-host", "node_type": "Host", "importance_score": 0.9},
        {
            "id": "gpu:test-gpu",
            "node_type": "GPUAccelerator",
            "importance_score": 0.9,
        },
        {
            "id": "storage:test-storage",
            "node_type": "StorageArray",
            "importance_score": 0.9,
        },
    ]

    # count == 3 (not 0) is the load-bearing assertion: pre-fix, ``_get_owl_class``
    # looked up the *raw* CamelCased node_type ("Host") against
    # ``_NODE_TYPE_TO_OWL_CLASS``'s snake_case keys ("host"), always missed, so
    # every node here would have been silently skipped (count == 0). Which
    # exact OWL class the world's substring `iri="*Host"` search resolves to
    # (this ontology's imports include more than one class ending in "Host")
    # is a separate, pre-existing concern this test doesn't assert on.
    count = backend.promote(nodes)  # type: ignore[arg-type]
    assert count == 3

    assert backend._onto.search_one(iri="*host_test-host") is not None
    assert backend._onto.search_one(iri="*gpu_test-gpu") is not None
    assert backend._onto.search_one(iri="*storage_test-storage") is not None

    backend.close()


def test_owlready2_promote_edges(ontology_path):
    """Test promotion of edges to OWL property assertions."""
    backend = Owlready2Backend(ontology_path=ontology_path)

    nodes = [
        {"id": "agent:test-agent", "node_type": "agent"},
        {"id": "tool:test-tool", "node_type": "tool"},
    ]
    backend.promote(nodes)

    edges = [
        {
            "source": "agent:test-agent",
            "target": "tool:test-tool",
            "relationship": "provides",
        }
    ]

    count = backend.promote_edges(edges)
    assert count == 1

    agent = backend._onto.search_one(iri="*agent_test-agent")
    tool = backend._onto.search_one(iri="*tool_test-tool")

    print(f"\nDebug: agent={agent}, tool={tool}")
    print(
        f"Debug: agent.provides={agent.provides if hasattr(agent, 'provides') else 'N/A'}"
    )
    print(f"Debug: properties={[p.python_name for p in agent.get_properties()]}")

    assert tool in agent.provides

    backend.close()


def test_owlready2_reasoning(ontology_path, monkeypatch):
    """Test OWL reasoning (simulated to avoid Java dependency in tests)."""
    backend = Owlready2Backend(ontology_path=ontology_path)

    # A depends on B, B depends on C -> A depends on C (Transitive)
    nodes = [
        {"id": "symbol:A", "node_type": "symbol"},
        {"id": "symbol:B", "node_type": "symbol"},
        {"id": "symbol:C", "node_type": "symbol"},
    ]
    backend.promote(nodes)

    edges = [
        {"source": "symbol:A", "target": "symbol:B", "relationship": "depends_on"},
        {"source": "symbol:B", "target": "symbol:C", "relationship": "depends_on"},
    ]
    backend.promote_edges(edges)

    # Mock reasoner to simulate transitive inference
    def mock_reasoner(*args, **kwargs):
        # Manually add the inference to the ontology
        world = backend._world
        symbol_A = world.search_one(iri="*symbol_A")
        symbol_C = world.search_one(iri="*symbol_C")
        if symbol_A and symbol_C:
            prop = backend._get_owl_property("depends_on")
            if prop:
                prop[symbol_A].append(symbol_C)
            else:
                raise RuntimeError("'dependsOn' property is not defined.")

    import owlready2

    monkeypatch.setattr(owlready2, "sync_reasoner_hermit", mock_reasoner)

    inferences = backend.reason()

    # We expect an inference: A depends_on C
    found = False
    for inf in inferences:
        if (
            inf["subject"] == "symbol_A"
            and inf["predicate"] == "dependsOn"
            and inf["object"] == "symbol_C"
        ):
            found = True
            break

    assert found, f"Inference A -> C not found in {inferences}"
    backend.close()


def test_owlready2_clear(ontology_path):
    """Test clearing ABox individuals."""
    backend = Owlready2Backend(ontology_path=ontology_path)
    backend.promote([{"id": "agent:1", "node_type": "agent"}])
    assert backend.get_stats()["individuals"] == 1

    backend.clear()
    assert backend.get_stats()["individuals"] == 0
    assert backend.get_stats()["classes"] > 0  # TBox preserved

    backend.close()
