"""GraphSchema owns ontology attachment on the selected graph.

The former ``_ensure_ontology_graph`` tests provisioned a separate Global
ontology graph. That registry lifecycle is retired: AU now forwards the source
to GraphSchema without listing or creating tenants, or caching attachment state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import OntologyLifecycle


@pytest.fixture
def graph_schema():
    schema = Mock(
        spec_set=["graph_schema_attach", "graph_schema_detach", "graph_schema_list"]
    )
    receipt = Mock(spec_set=["model_dump"])
    receipt.model_dump.return_value = {
        "schema_version": 2,
        "graph": "tenant__acme",
        "composed_digest": "digest:2",
        "graph_version": 4,
        "changed": True,
    }
    schema.graph_schema_attach.return_value = receipt
    return schema


@pytest.mark.parametrize("wrapped", [False, True], ids=["compute", "engine"])
@pytest.mark.parametrize("graph_name", [None, "tenant__acme"])
def test_load_attaches_to_current_or_selected_graph(graph_schema, wrapped, graph_name):
    compute = Mock(spec_set=["for_graph"])
    compute.for_graph.return_value = graph_schema
    target = compute if graph_name else graph_schema
    engine = SimpleNamespace(graph_compute=target) if wrapped else target
    lifecycle = OntologyLifecycle(engine, tenant="acme", graph_name=graph_name)
    body = "<urn:acme:Class> a <http://www.w3.org/2002/07/owl#Class> ."

    result = lifecycle.load(body, source_type="text", iri="urn:acme", version="1")

    if graph_name:
        compute.for_graph.assert_called_once_with(graph_name)
    else:
        compute.for_graph.assert_not_called()
    graph_schema.graph_schema_attach.assert_called_once_with(
        result["source_id"], ontology_ttl=body
    )
    assert result["source_id"].startswith("admin:ontology:")
    assert result == {
        "action": "attach",
        "iri": "urn:acme",
        "version": "1",
        "source_id": result["source_id"],
        **graph_schema.graph_schema_attach.return_value.model_dump.return_value,
    }
    graph_schema.graph_schema_attach.return_value.model_dump.assert_called_once_with(
        mode="json"
    )
    graph_schema.graph_schema_list.assert_not_called()
    graph_schema.graph_schema_detach.assert_not_called()


def test_repeated_load_uses_engine_receipt_instead_of_local_noop(graph_schema):
    lifecycle = OntologyLifecycle(graph_schema)
    body = "<urn:a> <urn:p> <urn:b> ."
    receipt = graph_schema.graph_schema_attach.return_value
    first = lifecycle.load(body, source_type="text", iri="urn:acme", version="1")
    receipt.model_dump.return_value = {
        **receipt.model_dump.return_value,
        "changed": False,
        "schema_version": 7,
        "composed_digest": "digest:7",
        "graph_version": 9,
    }

    second = lifecycle.load(body, source_type="text", iri="urn:acme", version="1")

    assert second["source_id"] == first["source_id"]
    assert first["changed"] is True
    assert second == {
        **first,
        "changed": False,
        "schema_version": 7,
        "composed_digest": "digest:7",
        "graph_version": 9,
    }
    assert graph_schema.graph_schema_attach.call_args_list == [
        call(first["source_id"], ontology_ttl=body),
        call(first["source_id"], ontology_ttl=body),
    ]
    assert receipt.model_dump.call_args_list == [call(mode="json"), call(mode="json")]
    graph_schema.graph_schema_list.assert_not_called()
    graph_schema.graph_schema_detach.assert_not_called()
