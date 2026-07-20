#!/usr/bin/python
"""Unit tests for External Graph Federation mixin.

CONCEPT:AU-KG.memory.tiered-memory-caching — External Graph Federation
"""

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.orchestration.engine_federation import (
    FederationMixin,
)


class _FederationHarness(FederationMixin):
    def __init__(self) -> None:
        self.backend = None
        self.graph = nx.MultiDiGraph()

    def add_node(self, *, node_id, node_type, properties):
        self.graph.add_node(node_id, type=node_type, **properties)

    def link_nodes(self, *, source_id, target_id, rel_type):
        self.graph.add_edge(source_id, target_id, type=rel_type.upper())


@pytest.fixture
def graph_engine():
    """Dependency-free federation harness with no engine process or transport."""
    return _FederationHarness()


def test_register_external_ontology(graph_engine):
    """Registration persists refs and aliases, never concrete transport data."""
    ontology_ref = "env://EXTERNAL_ONTOLOGY_URI"

    node_id = graph_engine.register_external_ontology(
        reference_id="business-ontology",
        ontology_ref=ontology_ref,
        connection="ontology-source",
    )

    ontologies = graph_engine.get_registered_ontologies()
    assert ontologies[node_id] == {
        "reference_id": "business-ontology",
        "ontology_ref": ontology_ref,
        "connection": "ontology-source",
    }

    assert node_id in graph_engine.graph
    node_data = graph_engine.graph.nodes[node_id]
    assert node_data["ontologyRef"] == ontology_ref
    assert node_data["connection"] == "ontology-source"
    assert "externalUri" not in node_data
    assert "sourceUrl" not in node_data


def test_external_ontology_rejects_concrete_uri(graph_engine):
    with pytest.raises(ValueError, match="runtime secret reference"):
        graph_engine.register_external_ontology(
            reference_id="business-ontology",
            ontology_ref="https://ontology.invalid/schema",
            connection="ontology-source",
        )


def test_ingest_external_entity_stub(graph_engine):
    """Test ingesting metadata stubs from external KGs."""
    internal_node = "InternalConcept_1"
    graph_engine.graph.add_node(internal_node, name="My Concept")

    external_id = "ext-999"
    external_uri_ref = "secret://runtime/external-entity-uri"
    source_alias = "enterprise-graph"
    name = "EAR FactSheet"

    stub_id = graph_engine.ingest_external_entity_stub(
        internal_node_id=internal_node,
        external_id=external_id,
        external_uri_ref=external_uri_ref,
        source_alias=source_alias,
        name=name,
    )

    # Verify ExternalEntity node exists
    assert stub_id in graph_engine.graph
    stub_data = graph_engine.graph.nodes[stub_id]
    assert stub_data["externalUriRef"] == external_uri_ref
    assert stub_data["sourceAlias"] == source_alias
    assert "externalSystemId" not in stub_data
    assert "externalUri" not in stub_data
    assert external_id not in stub_id
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


@patch("agent_utilities.mcp.kg_server.get_connection_registry")
def test_execute_federated_sparql_uses_registered_read_contract(
    mock_registry, graph_engine
):
    query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
    mock_backend = MagicMock()
    mock_backend.execute_read.return_value = [{"s": "subject-1"}]
    mock_registry.return_value.get_engine.return_value.backend = mock_backend

    results = graph_engine._execute_federated_connection("ontology-source", query)

    assert results == [{"s": "subject-1"}]
    mock_registry.return_value.get_engine.assert_called_once_with("ontology-source")
    mock_backend.execute_read.assert_called_once_with(query, {})


@patch("agent_utilities.mcp.kg_server.get_connection_registry")
def test_federated_connection_never_uses_ambiguous_execute(
    mock_registry, graph_engine
):
    mock_backend = MagicMock(spec=["execute"])
    mock_registry.return_value.get_engine.return_value.backend = mock_backend

    with pytest.raises(RuntimeError, match="Federated graph execution failed"):
        graph_engine._execute_federated_connection(
            "external-source", "MATCH (n) RETURN n"
        )

    mock_backend.execute.assert_not_called()
