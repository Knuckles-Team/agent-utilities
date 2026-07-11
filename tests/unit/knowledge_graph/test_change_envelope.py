"""Tests for :class:`ChangeEnvelope` (AU-P1-6, CONCEPT:AU-KG.ingest.change-envelope).

Covers construction defaults, validation (operation/payload-exclusivity/
confidence), idempotency-key determinism, the ``from_connector_record``/
``to_entity_dict`` round-trip bridge, and the ``snapshot_complete`` marker.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.change_envelope import (
    OPERATIONS,
    ChangeEnvelope,
)
from agent_utilities.models.company_brain import DataClassification
from agent_utilities.protocols.source_connectors.base import ExternalAccess


def test_defaults_are_safe_and_minimal():
    env = ChangeEnvelope(connector="servicenow-api")
    assert env.operation == "upsert"
    assert env.tenant == ""
    assert env.classification == DataClassification.INTERNAL  # fail-closed default
    assert env.legal_hold is False
    assert env.confidence == 1.0
    assert env.envelope_id  # auto uuid
    assert env.idempotency_key  # auto-derived
    assert env.observed_time.endswith("Z")


def test_idempotency_key_is_deterministic_and_scoped():
    a = ChangeEnvelope(
        connector="gitlab-api",
        source_instance="gl.corp",
        source_object_id="42",
        source_version="v1",
    )
    b = ChangeEnvelope(
        connector="gitlab-api",
        source_instance="gl.corp",
        source_object_id="42",
        source_version="v1",
    )
    assert a.idempotency_key == b.idempotency_key  # same identity -> same key
    assert a.envelope_id != b.envelope_id  # different delivery attempts

    c = ChangeEnvelope(
        connector="gitlab-api",
        source_instance="gl.corp",
        source_object_id="42",
        source_version="v2",  # different version -> different key
    )
    assert c.idempotency_key != a.idempotency_key

    d = ChangeEnvelope(
        connector="gitlab-api",
        source_instance="gl.corp",
        source_object_id="42",
        source_version="v1",
        operation="delete",  # different operation -> different key
    )
    assert d.idempotency_key != a.idempotency_key


def test_explicit_idempotency_key_is_not_overridden():
    env = ChangeEnvelope(connector="leanix-agent", idempotency_key="explicit-key")
    assert env.idempotency_key == "explicit-key"


@pytest.mark.parametrize("op", sorted(OPERATIONS))
def test_all_declared_operations_are_valid(op):
    env = ChangeEnvelope(connector="leanix-agent", operation=op)
    assert env.operation == op


def test_invalid_operation_rejected():
    with pytest.raises(ValueError, match="operation"):
        ChangeEnvelope(connector="leanix-agent", operation="patch")  # type: ignore[arg-type]


def test_typed_payload_and_blob_ref_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ChangeEnvelope(
            connector="documentdb-mcp",
            typed_payload={"a": 1},
            blob_ref="s3://bucket/key",
        )


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_confidence_out_of_range_rejected(bad):
    with pytest.raises(ValueError, match="confidence"):
        ChangeEnvelope(connector="microsoft-agent", confidence=bad)


def test_from_connector_record_bridges_todays_shape():
    record = {
        "id": "PROJ-123",
        "type": "Issue",
        "fields.summary": "Something broke",
        "updatedAt": "2026-07-10T12:00:00Z",
        "external_access": {"is_public": False, "group_ids": ["eng"]},
    }
    env = ChangeEnvelope.from_connector_record(
        record,
        connector="atlassian-agent",
        tenant="acme",
        source_instance="jira-cloud",
        id_field="id",
        version_field="updatedAt",
        schema_version="1",
        ontology_mapping_version="atlassian-2026-07",
    )
    assert env.connector == "atlassian-agent"
    assert env.tenant == "acme"
    assert env.source_instance == "jira-cloud"
    assert env.source_object_id == "PROJ-123"
    assert env.source_version == "2026-07-10T12:00:00Z"
    assert env.event_time == "2026-07-10T12:00:00Z"
    assert env.ontology_mapping_version == "atlassian-2026-07"
    assert env.typed_payload == record
    assert isinstance(env.source_acl, ExternalAccess)
    assert env.source_acl.group_ids == ["eng"]


def test_from_connector_record_to_entity_dict_round_trips():
    record = {"id": "n1", "type": "Order", "total": 42}
    env = ChangeEnvelope.from_connector_record(
        record, connector="acme-api", id_field="id", version_field="updatedAt"
    )
    row = env.to_entity_dict()
    assert row["id"] == "n1"
    assert row["type"] == "Order"
    assert row["total"] == 42


def test_to_entity_dict_requires_typed_payload():
    marker = ChangeEnvelope.snapshot_complete(connector="leanix-agent")
    with pytest.raises(ValueError, match="typed_payload"):
        marker.to_entity_dict()


def test_snapshot_complete_marker_carries_no_payload():
    marker = ChangeEnvelope.snapshot_complete(
        connector="leanix-agent", checkpoint="2026-07-10T00:00:00Z"
    )
    assert marker.operation == "snapshot_complete"
    assert marker.typed_payload is None
    assert marker.blob_ref is None
    assert marker.checkpoint == "2026-07-10T00:00:00Z"


def test_with_checkpoint_and_with_classification_are_immutable_copies():
    env = ChangeEnvelope(connector="langfuse-agent")
    updated = env.with_checkpoint("cursor-1").with_classification(DataClassification.PUBLIC)
    assert env.checkpoint is None
    assert env.classification == DataClassification.INTERNAL
    assert updated.checkpoint == "cursor-1"
    assert updated.classification == DataClassification.PUBLIC
    # both share the same idempotency_key (identity didn't change)
    assert updated.idempotency_key == env.idempotency_key


def test_as_dict_and_to_json_render_json_safe_values():
    env = ChangeEnvelope(
        connector="vector-mcp",
        source_acl=ExternalAccess.public(),
        classification=DataClassification.PUBLIC,
        typed_payload={"id": "v1"},
    )
    d = env.as_dict()
    assert d["classification"] == "public"
    assert d["source_acl"] == {
        "is_public": True,
        "user_emails": [],
        "group_ids": [],
        "markings": [],
    }
    text = env.to_json()
    assert '"connector": "vector-mcp"' in text


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
