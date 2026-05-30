#!/usr/bin/python
"""Unit tests for External Graph Federation mixin.

CONCEPT:KG-2.1 — External Graph Federation
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


@pytest.fixture
def graph_engine():
    """Fixture to create an in-memory graph engine."""
    GraphComputeEngine(backend_type="rust")
    # In-memory engine without a persistent backend
    engine = IntelligenceGraphEngine(db_path=":memory:")
    return engine


def test_register_external_ontology(graph_engine):
    """Test registering external ontologies and verify nodes are added."""
    uri = "http://example.org/ontology#"
    endpoint = "http://example.org/sparql"

    graph_engine.register_external_ontology(uri, endpoint)

    # Check in-memory mapping
    ontologies = graph_engine.get_registered_ontologies()
    assert uri in ontologies
    assert ontologies[uri] == endpoint

    # Check reference node created in graph
    node_id = f"OntologyReference_{hash(uri)}"
    assert node_id in graph_engine.graph
    node_data = graph_engine.graph.nodes[node_id]
    assert node_data["externalUri"] == uri
    assert node_data["sourceUrl"] == endpoint
    assert node_data["platform"] == "sparql"


def test_ingest_external_entity_stub(graph_engine):
    """Test ingesting metadata stubs from external KGs."""
    internal_node = "InternalConcept_1"
    graph_engine.graph.add_node(internal_node, name="My Concept")

    external_id = "ext-999"
    external_uri = "https://leanix.example.com/factsheet/ext-999"
    platform = "leanix"
    name = "LeanIX FactSheet"

    stub_id = graph_engine.ingest_external_entity_stub(
        internal_node_id=internal_node,
        external_id=external_id,
        external_uri=external_uri,
        platform=platform,
        name=name,
    )

    # Verify ExternalEntity node exists
    assert stub_id in graph_engine.graph
    stub_data = graph_engine.graph.nodes[stub_id]
    assert stub_data["externalSystemId"] == external_id
    assert stub_data["externalUri"] == external_uri
    assert stub_data["platform"] == platform
    assert stub_data["name"] == name

    # Verify edge exists
    edge = graph_engine.graph.edges[internal_node, stub_id, 0]
    assert edge["type"] == "MAPPED_TO_EXTERNAL"


@patch("requests.post")
def test_execute_federated_sparql(mock_post, graph_engine):
    """Test executing SPARQL query against mock external HTTP endpoint."""
    endpoint = "http://example.org/sparql"
    query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"

    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": {
            "bindings": [
                {
                    "s": {"value": "http://example.org/s1"},
                    "p": {"value": "http://example.org/p1"},
                    "o": {"value": "http://example.org/o1"},
                }
            ]
        }
    }
    mock_post.return_value = mock_response

    results = graph_engine.execute_federated_sparql(endpoint, query)

    assert len(results) == 1
    assert results[0]["s"] == "http://example.org/s1"
    assert results[0]["p"] == "http://example.org/p1"
    assert results[0]["o"] == "http://example.org/o1"
    mock_post.assert_called_once()


@patch("agent_utilities.knowledge_graph.orchestration.engine_federation.create_backend")
def test_execute_federated_lpg(mock_create_backend, graph_engine):
    """Test executing federated Cypher query against LPG backend."""
    endpoint = "bolt://neo4j.example.com:7687"
    query = "MATCH (n) RETURN n LIMIT 5"

    mock_backend = MagicMock()
    mock_backend.execute.return_value = [{"n": {"id": "123", "name": "Test Node"}}]
    mock_create_backend.return_value = mock_backend

    results = graph_engine.execute_federated_lpg(endpoint, query)

    assert len(results) == 1
    assert results[0]["n"]["id"] == "123"
    mock_create_backend.assert_called_once_with(uri=endpoint)
    mock_backend.execute.assert_called_once_with(query, {})
