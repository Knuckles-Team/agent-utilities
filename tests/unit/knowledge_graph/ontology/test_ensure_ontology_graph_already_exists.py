"""GraphSchema failures stay visible instead of becoming local registry success.

The retired tenant-provisioning path adopted a matching "already exists" error.
GraphSchema attachment now owns reconciliation: even that message is an error
at the AU boundary, and no tenant listing or process-local fallback may hide it.
"""

from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import OntologyLifecycle


@pytest.mark.parametrize(
    "message",
    [
        "Graph 'tenant__acme' already exists",
        "connection reset by peer",
        "Graph 'some_other_tenant' already exists",
    ],
    ids=["same-graph-already-exists", "transport-error", "other-graph-already-exists"],
)
def test_attach_error_propagates_and_retry_reaches_engine(message):
    schema = Mock(
        spec_set=["graph_schema_attach", "graph_schema_detach", "graph_schema_list"]
    )
    error = RuntimeError(message)
    receipt = Mock(spec_set=["model_dump"])
    receipt.model_dump.return_value = {"graph": "tenant__acme", "changed": True}
    schema.graph_schema_attach.side_effect = [error, receipt]
    lifecycle = OntologyLifecycle(schema)
    body = "<urn:a> <urn:p> <urn:b> ."

    with pytest.raises(RuntimeError) as caught:
        lifecycle.load(body, source_type="text", iri="urn:acme", version="1")

    assert caught.value is error
    receipt.model_dump.assert_not_called()
    result = lifecycle.load(body, source_type="text", iri="urn:acme", version="1")
    assert result["action"] == "attach"
    assert result["graph"] == "tenant__acme"
    assert result["changed"] is True
    assert schema.graph_schema_attach.call_args_list == [
        call(result["source_id"], ontology_ttl=body),
        call(result["source_id"], ontology_ttl=body),
    ]
    receipt.model_dump.assert_called_once_with(mode="json")
    schema.graph_schema_list.assert_not_called()
    schema.graph_schema_detach.assert_not_called()
