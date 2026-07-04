"""Unit tests for the Stardog SPARQL DATA backend (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

Mocks pystardog so we assert the Cypher→SPARQL write translation, named-graph
routing, and native passthrough without a live server. The complement to the live
parity test in ``tests/integration/backends/test_sparql_backend_live.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def fake_stardog(monkeypatch):
    """Inject a fake ``stardog`` module; return the shared connection mock so a test
    can read the SPARQL strings the backend emitted and stub query results."""
    conn = MagicMock(name="connection")
    conn.select.return_value = {"results": {"bindings": []}}

    mod = SimpleNamespace()
    mod.Connection = MagicMock(return_value=conn)
    admin = MagicMock()
    admin.__enter__ = MagicMock(return_value=admin)
    admin.__exit__ = MagicMock(return_value=False)
    admin.databases.return_value = []
    mod.Admin = MagicMock(return_value=admin)
    mod.content = SimpleNamespace(Raw=MagicMock())
    monkeypatch.setitem(sys.modules, "stardog", mod)
    return conn


@pytest.fixture()
def backend(fake_stardog):
    from agent_utilities.knowledge_graph.backends.sparql.stardog_backend import (
        StardogSparqlBackend,
    )

    be = StardogSparqlBackend(
        endpoint="http://sd:5820", database="agent_kg", username="u", password="p"
    )
    return be


def _updates(conn) -> list[str]:
    return [c.args[0] for c in conn.update.call_args_list]


# ── source-partition routing ────────────────────────────────────────────────
def test_source_partition_routing():
    from agent_utilities.knowledge_graph.backends.sparql.source_partition import (
        graph_uri_for,
        source_of,
    )

    assert source_of({"source_system": "leanix"}) == "leanix"
    assert source_of({"ingested_from": "ServiceNow"}) == "servicenow"
    assert source_of({"source": "system"}) is None  # generic provenance → default
    assert source_of({}) is None
    assert graph_uri_for({"source_system": "leanix"}) == "urn:source:leanix"
    assert graph_uri_for({"source": "system"}) is None


# ── Cypher→SPARQL translation ───────────────────────────────────────────────
def test_node_merge_translates_to_named_graph_insert(backend, fake_stardog):
    backend.execute(
        "MERGE (n:Application {id: $id}) SET n.`name` = $name, n.`source_system` = $source_system",
        {"id": "app:1", "name": "Billing", "source_system": "leanix"},
    )
    blob = "\n".join(_updates(fake_stardog))
    assert "GRAPH <urn:source:leanix>" in blob  # routed by provenance
    assert "a <http://agent-utilities.dev/kg#Application>" in blob
    assert "#name>" in blob and "Billing" in blob


def test_internal_node_goes_to_default_graph(backend, fake_stardog):
    backend.execute(
        "MERGE (n:Episode {id: $id}) SET n.`name` = $name",
        {"id": "ep:1", "name": "x"},
    )
    blob = "\n".join(_updates(fake_stardog))
    assert "GRAPH <urn:source:" not in blob  # no source → default graph
    assert "ep:1" in blob


def test_edge_merge_translates_to_triple(backend, fake_stardog):
    backend.execute(
        "MATCH (s:Application {id: $sid}), (t:Capability {id: $tid}) "
        "MERGE (s)-[r:SUPPORTS]->(t)",
        {"sid": "app:1", "tid": "cap:9", "source_system": "leanix"},
    )
    blob = "\n".join(_updates(fake_stardog))
    assert "#SUPPORTS>" in blob
    assert "app:1" in blob and "cap:9" in blob
    assert "GRAPH <urn:source:leanix>" in blob


def test_unwind_batch_node_translation(backend, fake_stardog):
    backend.execute_batch(
        "UNWIND $batch AS row MERGE (n:DomainEntity {id: row.id}) SET n.name = row.name",
        [
            {"id": "sn:1", "name": "TRM-1", "source_system": "servicenow"},
            {"id": "sn:2", "name": "TRM-2", "source_system": "servicenow"},
        ],
    )
    blob = "\n".join(_updates(fake_stardog))
    assert "GRAPH <urn:source:servicenow>" in blob
    assert "sn:1" in blob and "sn:2" in blob


def test_label_lookup_reads_from_store(backend, fake_stardog):
    fake_stardog.select.return_value = {
        "results": {
            "bindings": [{"t": {"value": "http://agent-utilities.dev/kg#Application"}}]
        }
    }
    rows = backend.execute(
        "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl", {"id": "app:1"}
    )
    assert rows == [{"lbl": "Application"}]


def test_native_sparql_passthrough(backend, fake_stardog):
    fake_stardog.select.return_value = {
        "results": {"bindings": [{"s": {"value": "app:1"}}]}
    }
    rows = backend.execute("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
    assert rows == [{"s": "app:1"}]


def test_unknown_cypher_is_noop(backend, fake_stardog):
    assert backend.execute("MATCH (n) DETACH DELETE n") == []


# ── add_node / add_edge full fidelity ───────────────────────────────────────
def test_add_node_writes_all_properties(backend, fake_stardog):
    backend.add_node(
        "app:1", {"type": "Application", "name": "Billing", "criticality": "high"}
    )
    blob = "\n".join(_updates(fake_stardog))
    assert "#name>" in blob and "#criticality>" in blob


# ── factory wiring ──────────────────────────────────────────────────────────
def test_factory_builds_stardog_backend(fake_stardog, monkeypatch):
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.backends.sparql.stardog_backend import (
        StardogSparqlBackend,
    )

    be = create_backend(backend_type="stardog", endpoint="http://sd:5820")
    assert isinstance(be, StardogSparqlBackend)
    assert be.supports_sparql is True
