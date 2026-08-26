"""Tests for the ``/api/graph/ingest/*`` and ``/api/graph/write/*`` request/
response models in ``agent_utilities.gateway.schemas.graph_ingest``.

Per model: a realistic currently-valid payload is accepted, an invalid one
(wrong type on a field, or a missing required field) is rejected, and every
declared field carries a non-empty ``description`` (checked programmatically
by looping ``model_fields`` so a future undocumented field trips a test
rather than rotting silently). A dedicated regression test pins the
``node_id`` (not ``id``) field name on ``GraphWriteNodeRequest``.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent_utilities.gateway.schemas.graph_ingest import (
    GraphIngestAgentToolkitRequest,
    GraphIngestCorpusRequest,
    GraphIngestNoOpRequest,
    GraphIngestObserveRequest,
    GraphIngestRequest,
    GraphIngestSubmitRequest,
    GraphToolResponse,
    GraphWriteBulkRequest,
    GraphWriteChatRequest,
    GraphWriteEdgeDeleteRequest,
    GraphWriteEdgeRequest,
    GraphWriteExecutionRequest,
    GraphWriteNodeRequest,
    GraphWriteRequest,
)

ALL_MODELS: list[type[BaseModel]] = [
    GraphToolResponse,
    GraphWriteRequest,
    GraphWriteNodeRequest,
    GraphWriteEdgeRequest,
    GraphWriteEdgeDeleteRequest,
    GraphWriteBulkRequest,
    GraphWriteChatRequest,
    GraphWriteExecutionRequest,
    GraphIngestRequest,
    GraphIngestSubmitRequest,
    GraphIngestCorpusRequest,
    GraphIngestObserveRequest,
    GraphIngestAgentToolkitRequest,
    GraphIngestNoOpRequest,
]


# ══════════════════════════════════════════════════════════════════════════
# Cross-cutting: every field on every model has a non-empty description
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=lambda m: m.__name__)
def test_every_field_has_a_non_empty_description(model_cls: type[BaseModel]) -> None:
    # Vacuously true for a zero-field model (GraphIngestNoOpRequest) —
    # test_graph_ingest_noop_request_declares_no_fields asserts that shape
    # explicitly below, so this loop still means something for it.
    fields = model_cls.model_fields
    for name, field in fields.items():
        assert field.description, (
            f"{model_cls.__name__}.{name} has no description "
            "(every field must carry a Field(description=...))"
        )
        assert field.description.strip(), (
            f"{model_cls.__name__}.{name} has a blank/whitespace-only description"
        )


# ══════════════════════════════════════════════════════════════════════════
# GraphToolResponse — shared success envelope
# ══════════════════════════════════════════════════════════════════════════


def test_graph_tool_response_accepts_a_realistic_payload() -> None:
    resp = GraphToolResponse.model_validate(
        {"status": "success", "result": "Node n1 added."}
    )
    assert resp.status == "success"
    assert resp.result == "Node n1 added."

    resp2 = GraphToolResponse.model_validate(
        {
            "status": "success",
            "result": {
                "action": "bulk_ingest",
                "mode": "batch_update",
                "nodes_ingested": 3,
                "edges_ingested": 1,
                "chunks": 1,
            },
        }
    )
    assert resp2.result["mode"] == "batch_update"


def test_graph_tool_response_rejects_a_non_success_status() -> None:
    with pytest.raises(ValidationError):
        GraphToolResponse.model_validate({"status": "error", "result": "boom"})


def test_graph_tool_response_rejects_unknown_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GraphToolResponse.model_validate(
            {"status": "success", "result": "ok", "unexpected": True}
        )


# ══════════════════════════════════════════════════════════════════════════
# GraphWriteRequest — base POST /graph/write
# ══════════════════════════════════════════════════════════════════════════


def test_graph_write_request_accepts_a_realistic_add_node_payload() -> None:
    req = GraphWriteRequest.model_validate(
        {
            "action": "add_node",
            "node_id": "n1",
            "node_type": "Concept",
            "properties": '{"label": "example"}',
        }
    )
    assert req.action == "add_node"
    assert req.node_id == "n1"
    assert req.properties == '{"label": "example"}'
    # unset fields carry the underlying tool's own defaults
    assert req.nodes == "[]"
    assert req.upsert is True


def test_graph_write_request_accepts_a_compare_and_set_payload() -> None:
    req = GraphWriteRequest.model_validate(
        {
            "action": "compare_and_set",
            "node_id": "n1",
            "conditions": {"status": "pending"},
            "updates": {"status": "claimed", "owner": "agent-7"},
        }
    )
    assert req.conditions == {"status": "pending"}
    assert req.updates == {"status": "claimed", "owner": "agent-7"}


def test_graph_write_request_requires_action() -> None:
    with pytest.raises(ValidationError):
        GraphWriteRequest.model_validate({"node_id": "n1"})


def test_graph_write_request_rejects_wrong_type_on_action() -> None:
    with pytest.raises(ValidationError):
        GraphWriteRequest.model_validate({"action": 123})


def test_graph_write_request_allows_unknown_fields_as_full_passthrough() -> None:
    # extra="allow": the base route forwards the ENTIRE body to
    # _execute_tool("graph_write", **body); the server (not this schema) is
    # the authority on whether a given action accepts a given field.
    req = GraphWriteRequest.model_validate(
        {"action": "store_memory", "some_future_field": "value"}
    )
    assert req.model_dump(exclude_unset=False).get("some_future_field") == "value"


# ══════════════════════════════════════════════════════════════════════════
# GraphWriteNodeRequest — POST /graph/write/node (+ node_id regression)
# ══════════════════════════════════════════════════════════════════════════


def test_graph_write_node_request_accepts_a_realistic_payload() -> None:
    req = GraphWriteNodeRequest.model_validate(
        {"node_id": "n1", "node_type": "Concept", "properties": {"label": "example"}}
    )
    assert req.node_id == "n1"
    assert req.node_type == "Concept"
    assert req.properties == {"label": "example"}


def test_graph_write_node_request_rejects_missing_node_id() -> None:
    with pytest.raises(ValidationError):
        GraphWriteNodeRequest.model_validate({"node_type": "Concept"})


def test_graph_write_node_request_rejects_wrong_type_on_node_id() -> None:
    with pytest.raises(ValidationError):
        GraphWriteNodeRequest.model_validate({"node_id": 123, "node_type": "Concept"})


def test_graph_write_node_request_uses_node_id_not_id() -> None:
    """Regression test (DEFECT C): the graph_write tool declares this
    identifier parameter ``node_id``, never ``id``. A prior mismatch made
    ``graph_write_node_endpoint`` call ``_execute_tool("graph_write", id=...)``
    against a tool with no ``id`` parameter, failing every call closed with
    ``UnsupportedToolFieldError``. Pin the field name here so a regression
    in this schema (renaming it back to ``id``) trips a test rather than
    only a live 400.
    """
    assert "node_id" in GraphWriteNodeRequest.model_fields
    assert "id" not in GraphWriteNodeRequest.model_fields

    # An 'id'-keyed payload must NOT populate node_id — proving the model
    # genuinely requires the 'node_id' wire name, not merely permits it.
    with pytest.raises(ValidationError):
        GraphWriteNodeRequest.model_validate({"id": "n1", "node_type": "Concept"})


# ══════════════════════════════════════════════════════════════════════════
# GraphWriteEdgeRequest / GraphWriteEdgeDeleteRequest — POST/DELETE /graph/write/edge
# ══════════════════════════════════════════════════════════════════════════


def test_graph_write_edge_request_accepts_a_realistic_payload() -> None:
    req = GraphWriteEdgeRequest.model_validate(
        {
            "source_id": "n1",
            "target_id": "n2",
            "rel_type": "RELATES_TO",
            "properties": {"weight": 1},
        }
    )
    assert req.rel_type == "RELATES_TO"
    assert req.properties == {"weight": 1}


def test_graph_write_edge_request_rejects_missing_rel_type() -> None:
    with pytest.raises(ValidationError):
        GraphWriteEdgeRequest.model_validate({"source_id": "n1", "target_id": "n2"})


def test_graph_write_edge_delete_request_accepts_a_realistic_payload_with_no_properties() -> None:
    req = GraphWriteEdgeDeleteRequest.model_validate(
        {"source_id": "n1", "target_id": "n2", "rel_type": "RELATES_TO"}
    )
    assert req.target_id == "n2"
    assert not hasattr(req, "properties")


def test_graph_write_edge_delete_request_rejects_wrong_type_on_source_id() -> None:
    with pytest.raises(ValidationError):
        GraphWriteEdgeDeleteRequest.model_validate(
            {"source_id": ["not", "a", "string"], "target_id": "n2", "rel_type": "R"}
        )


# ══════════════════════════════════════════════════════════════════════════
# GraphWriteBulkRequest — POST /graph/write/bulk
# ══════════════════════════════════════════════════════════════════════════


def test_graph_write_bulk_request_accepts_a_realistic_payload() -> None:
    req = GraphWriteBulkRequest.model_validate(
        {
            "nodes": [
                {"id": "n1", "type": "Concept", "properties": {"label": "a"}},
                {
                    "kind": "edge",
                    "source_id": "n1",
                    "target_id": "n2",
                    "rel_type": "RELATES_TO",
                },
            ]
        }
    )
    assert len(req.nodes) == 2


def test_graph_write_bulk_request_defaults_to_an_empty_noop_batch() -> None:
    req = GraphWriteBulkRequest.model_validate({})
    assert req.nodes == []


def test_graph_write_bulk_request_rejects_wrong_type_on_nodes() -> None:
    with pytest.raises(ValidationError):
        GraphWriteBulkRequest.model_validate({"nodes": "not-a-list"})


def test_graph_write_bulk_request_silently_ignores_bulk_ingest_only_fields() -> None:
    """Documents the known data-loss gap: idempotency_key/evidence/upsert are
    valid graph_write(action='bulk_ingest') tool fields but this granular
    route's handler never reads them, so they must not surface as
    attributes on this model even when supplied."""
    req = GraphWriteBulkRequest.model_validate(
        {"nodes": [], "idempotency_key": "batch-1", "upsert": False}
    )
    assert not hasattr(req, "idempotency_key")
    assert not hasattr(req, "upsert")


# ══════════════════════════════════════════════════════════════════════════
# GraphWriteChatRequest — POST /graph/write/chat
# ══════════════════════════════════════════════════════════════════════════


def test_graph_write_chat_request_accepts_a_realistic_payload() -> None:
    req = GraphWriteChatRequest.model_validate(
        {"agent_id": "agent-7", "content": "hello world"}
    )
    assert req.content == "hello world"


def test_graph_write_chat_request_rejects_wrong_type_on_content() -> None:
    with pytest.raises(ValidationError):
        GraphWriteChatRequest.model_validate({"content": {"not": "a string"}})


def test_graph_write_chat_request_uses_wire_name_content_not_properties() -> None:
    assert "content" in GraphWriteChatRequest.model_fields
    assert "properties" not in GraphWriteChatRequest.model_fields


# ══════════════════════════════════════════════════════════════════════════
# GraphWriteExecutionRequest — POST /graph/write/execution
# ══════════════════════════════════════════════════════════════════════════


def test_graph_write_execution_request_accepts_a_realistic_payload() -> None:
    req = GraphWriteExecutionRequest.model_validate({"agent_id": "agent-7"})
    assert req.agent_id == "agent-7"


def test_graph_write_execution_request_rejects_wrong_type_on_agent_id() -> None:
    with pytest.raises(ValidationError):
        GraphWriteExecutionRequest.model_validate({"agent_id": 42})


# ══════════════════════════════════════════════════════════════════════════
# GraphIngestRequest — base POST /graph/ingest
# ══════════════════════════════════════════════════════════════════════════


def test_graph_ingest_request_accepts_a_realistic_ingest_payload() -> None:
    req = GraphIngestRequest.model_validate(
        {
            "action": "ingest",
            "target_path": "/repos/agent-utilities",
            "max_depth": 3,
            "agent_id": "agent-7",
        }
    )
    assert req.action == "ingest"
    assert req.target_path == "/repos/agent-utilities"


def test_graph_ingest_request_defaults_action_to_ingest() -> None:
    req = GraphIngestRequest.model_validate({})
    assert req.action == "ingest"


def test_graph_ingest_request_rejects_wrong_type_on_max_depth() -> None:
    with pytest.raises(ValidationError):
        GraphIngestRequest.model_validate({"max_depth": "not-an-int"})


def test_graph_ingest_request_rejects_priority_bucket_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        GraphIngestRequest.model_validate({"priority_bucket": 4})


def test_graph_ingest_request_allows_unknown_fields_as_full_passthrough() -> None:
    req = GraphIngestRequest.model_validate(
        {"action": "distill", "some_future_field": "value"}
    )
    assert req.model_dump(exclude_unset=False).get("some_future_field") == "value"


# ══════════════════════════════════════════════════════════════════════════
# GraphIngestSubmitRequest — POST /graph/ingest/submit
# ══════════════════════════════════════════════════════════════════════════


def test_graph_ingest_submit_request_accepts_a_single_path() -> None:
    req = GraphIngestSubmitRequest.model_validate({"target_path": "/repos/au"})
    assert req.target_path == "/repos/au"


def test_graph_ingest_submit_request_accepts_a_list_of_paths() -> None:
    req = GraphIngestSubmitRequest.model_validate(
        {"target_path": ["/repos/au", "/repos/eg"]}
    )
    assert req.target_path == ["/repos/au", "/repos/eg"]


def test_graph_ingest_submit_request_rejects_missing_target_path() -> None:
    with pytest.raises(ValidationError):
        GraphIngestSubmitRequest.model_validate({"agent_id": "agent-7"})


def test_graph_ingest_submit_request_rejects_wrong_type_on_target_path() -> None:
    with pytest.raises(ValidationError):
        GraphIngestSubmitRequest.model_validate({"target_path": {"not": "valid"}})


# ══════════════════════════════════════════════════════════════════════════
# GraphIngestCorpusRequest — POST /graph/ingest/corpus
# ══════════════════════════════════════════════════════════════════════════


def test_graph_ingest_corpus_request_accepts_a_realistic_payload() -> None:
    req = GraphIngestCorpusRequest.model_validate(
        {
            "corpus_name": "homelab-docs",
            "base_path": "/data/docs",
            "description": "Homelab documentation corpus",
        }
    )
    assert req.corpus_name == "homelab-docs"


def test_graph_ingest_corpus_request_rejects_missing_corpus_name() -> None:
    with pytest.raises(ValidationError):
        GraphIngestCorpusRequest.model_validate({"base_path": "/data/docs"})


def test_graph_ingest_corpus_request_rejects_wrong_type_on_corpus_name() -> None:
    with pytest.raises(ValidationError):
        GraphIngestCorpusRequest.model_validate({"corpus_name": 123})


# ══════════════════════════════════════════════════════════════════════════
# GraphIngestObserveRequest — POST /graph/ingest/observe
# ══════════════════════════════════════════════════════════════════════════


def test_graph_ingest_observe_request_accepts_a_realistic_payload() -> None:
    req = GraphIngestObserveRequest.model_validate(
        {"target_path": "/logs/transcript.jsonl", "agent_id": "agent-7"}
    )
    assert req.target_path == "/logs/transcript.jsonl"


def test_graph_ingest_observe_request_rejects_missing_target_path() -> None:
    with pytest.raises(ValidationError):
        GraphIngestObserveRequest.model_validate({"agent_id": "agent-7"})


def test_graph_ingest_observe_request_rejects_wrong_type_on_target_path() -> None:
    with pytest.raises(ValidationError):
        GraphIngestObserveRequest.model_validate({"target_path": 42})


# ══════════════════════════════════════════════════════════════════════════
# GraphIngestAgentToolkitRequest — POST /graph/ingest/agent-toolkit
# ══════════════════════════════════════════════════════════════════════════


def test_graph_ingest_agent_toolkit_request_accepts_a_realistic_payload() -> None:
    req = GraphIngestAgentToolkitRequest.model_validate(
        {
            "sources": ["https://example.com/.well-known/agent.json"],
            "agent_card_path": "/.well-known/agent.json",
        }
    )
    assert req.sources == ["https://example.com/.well-known/agent.json"]


def test_graph_ingest_agent_toolkit_request_defaults_to_empty_sources() -> None:
    req = GraphIngestAgentToolkitRequest.model_validate({})
    assert req.sources == []
    assert req.agent_card_path == ""


def test_graph_ingest_agent_toolkit_request_rejects_wrong_type_on_sources() -> None:
    with pytest.raises(ValidationError):
        GraphIngestAgentToolkitRequest.model_validate({"sources": "not-a-list"})


def test_graph_ingest_agent_toolkit_request_uses_wire_names_not_tool_param_names() -> None:
    assert "sources" in GraphIngestAgentToolkitRequest.model_fields
    assert "agent_card_path" in GraphIngestAgentToolkitRequest.model_fields
    assert "target_path" not in GraphIngestAgentToolkitRequest.model_fields
    assert "description" not in GraphIngestAgentToolkitRequest.model_fields


# ══════════════════════════════════════════════════════════════════════════
# GraphIngestNoOpRequest — POST /graph/ingest/{materialize,reflect,sync}
# ══════════════════════════════════════════════════════════════════════════


def test_graph_ingest_noop_request_accepts_an_empty_payload() -> None:
    req = GraphIngestNoOpRequest.model_validate({})
    assert req is not None


def test_graph_ingest_noop_request_accepts_arbitrary_extra_fields() -> None:
    # The handlers for materialize/reflect/sync never call request.json() at
    # all, so literally any body is accepted (and ignored) server-side.
    req = GraphIngestNoOpRequest.model_validate({"anything": "goes", "n": 1})
    assert req is not None


def test_graph_ingest_noop_request_rejects_a_non_object_payload() -> None:
    with pytest.raises(ValidationError):
        GraphIngestNoOpRequest.model_validate(["not", "an", "object"])


def test_graph_ingest_noop_request_declares_no_fields() -> None:
    # This model is exempted from test_every_field_has_a_non_empty_description
    # implicitly by having zero fields to loop over; assert that explicitly
    # so a future field added here is required to also get a description
    # via the shared parametrized test above.
    assert GraphIngestNoOpRequest.model_fields == {}
