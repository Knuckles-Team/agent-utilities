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
    external_uri = "https://ear.example.com/factsheet/ext-999"
    platform = "ear"
    name = "EAR FactSheet"

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


class _CountingCamundaClient:
    """Duck-typed camunda client that counts calls to detect TTL caching."""

    def __init__(self):
        self.calls = 0

    def list_process_definitions(self):
        self.calls += 1
        return [{"id": "invoice:1:abc", "key": "invoice", "name": "Invoice"}]

    def list_tasks(self):
        return []

    def list_incidents(self):
        return [
            {
                "id": "inc-1",
                "incidentMessage": "boom",
                "processDefinitionId": "invoice:1:abc",
            }
        ]


def test_register_and_query_rest_source(graph_engine):
    client = _CountingCamundaClient()
    graph_engine.register_rest_source("rest:camunda", "camunda", client)

    # Reference node is discoverable, tagged platform=rest.
    assert "rest:camunda" in graph_engine.graph
    assert graph_engine.graph.nodes["rest:camunda"]["platform"] == "rest"

    records = graph_engine.query_rest_source("rest:camunda")
    types = {r["type"] for r in records}
    assert types == {"BusinessProcess", "Incident"}

    # Type filter narrows to one canonical concept.
    incidents = graph_engine.query_rest_source("rest:camunda", node_type="Incident")
    assert len(incidents) == 1
    assert incidents[0]["id"] == "incident:inc-1"


def test_rest_source_ttl_cache(graph_engine):
    client = _CountingCamundaClient()
    graph_engine.register_rest_source("rest:camunda", "camunda", client, ttl_seconds=60)
    graph_engine.query_rest_source("rest:camunda")
    graph_engine.query_rest_source("rest:camunda")
    assert client.calls == 1  # second query served from cache

    # ttl=0 forces a refetch every call.
    client2 = _CountingCamundaClient()
    graph_engine.register_rest_source("rest:c2", "camunda", client2, ttl_seconds=0)
    graph_engine.query_rest_source("rest:c2")
    graph_engine.query_rest_source("rest:c2")
    assert client2.calls == 2


def test_execute_federated_query_routes_rest(graph_engine):
    client = _CountingCamundaClient()
    graph_engine.register_rest_source("rest:camunda", "camunda", client)
    results = graph_engine.execute_federated_query(
        "rest:camunda", query="", parameters={"node_type": "BusinessProcess"}
    )
    assert len(results) == 1
    assert results[0]["type"] == "BusinessProcess"


def test_query_rest_union_dedups_local_wins(graph_engine):
    client = _CountingCamundaClient()
    graph_engine.register_rest_source("rest:camunda", "camunda", client)
    local = [{"id": "incident:inc-1", "type": "Incident", "state": "local-edit"}]
    merged = graph_engine.query_rest_union("rest:camunda", local, node_type="Incident")
    assert len(merged) == 1  # same id deduped
    assert merged[0]["state"] == "local-edit"  # local precedence


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
