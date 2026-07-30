"""Official OCEL 2.0 JSON import/export and governed tEKG boundary proofs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent_utilities.knowledge_graph.ingestion.ocel_adapter import (
    export_ocel_json,
    import_ocel_json,
)


def _ocel() -> dict:
    """Official four-array OCEL 2.0 JSON with E2O, O2O, and typed values."""
    return {
        "eventTypes": [
            {
                "name": "create-order",
                "attributes": [
                    {"name": "approved", "type": "boolean"},
                    {"name": "created-at", "type": "time"},
                    {"name": "ratio", "type": "float"},
                    {"name": "total-items", "type": "integer"},
                    {"name": "channel", "type": "string"},
                ],
            }
        ],
        "objectTypes": [
            {"name": "item", "attributes": []},
            {
                "name": "order",
                "attributes": [{"name": "status", "type": "string"}],
            },
        ],
        "events": [
            {
                "id": "e1",
                "type": "create-order",
                "time": "2026-01-01T00:00:00Z",
                "attributes": [
                    {"name": "total-items", "value": 1},
                    {"name": "channel", "value": "web"},
                    {"name": "ratio", "value": 1.5},
                    {"name": "created-at", "value": "2026-01-01T00:00:00Z"},
                    {"name": "approved", "value": True},
                ],
                "relationships": [
                    {"objectId": "order-1", "qualifier": "order"},
                    {"objectId": "item-1", "qualifier": "line-item"},
                ],
            }
        ],
        "objects": [
            {
                "id": "order-1",
                "type": "order",
                "attributes": [
                    {
                        "name": "status",
                        "time": "2026-01-01T00:00:00Z",
                        "value": "new",
                    },
                    {
                        "name": "status",
                        "time": "2026-01-02T00:00:00Z",
                        "value": "approved",
                    },
                ],
                "relationships": [{"objectId": "item-1", "qualifier": "contains"}],
            },
            {
                "id": "item-1",
                "type": "item",
                "attributes": [],
                "relationships": [],
            },
        ],
    }


def _import(body: dict, *, tenant: str = "tenant-a"):
    return import_ocel_json(
        body,
        tenant=tenant,
        source_ref="fixture://ocel/orders",
        mapping_version="ocel-json-2.0",
        provenance={
            "structured": {"system": "erp", "batch": "42"},
            "unstructured_refs": ["fixture://notes/orders-42"],
        },
    )


def test_published_minimal_example_is_accepted_verbatim() -> None:
    """Fixture from https://www.ocel-standard.org/specification/formats/json/."""
    document = {
        "eventTypes": [
            {
                "name": "create-order",
                "attributes": [{"name": "total-items", "type": "integer"}],
            }
        ],
        "objectTypes": [
            {
                "name": "order",
                "attributes": [{"name": "item", "type": "integer"}],
            }
        ],
        "events": [
            {
                "id": "e1",
                "type": "create-order",
                "time": "1970-01-01T00:00:00Z",
                "attributes": [{"name": "total-items", "value": 1}],
                "relationships": [{"objectId": "o1", "qualifier": "item"}],
            }
        ],
        "objects": [
            {
                "id": "o1",
                "type": "order",
                "attributes": [
                    {
                        "name": "item",
                        "time": "1970-01-01T00:00:00Z",
                        "value": 1,
                    }
                ],
            }
        ],
    }

    source, _ = import_ocel_json(document, tenant="tenant-a")

    assert export_ocel_json(source) == {
        **document,
        "objects": [{**document["objects"][0], "relationships": []}],
    }


def test_official_ocel_round_trip_is_lossless_and_deterministic() -> None:
    source, provenance = _import(_ocel())
    exported = export_ocel_json(source)
    restored, restored_provenance = _import(exported)

    assert set(exported) == {"eventTypes", "objectTypes", "events", "objects"}
    assert export_ocel_json(restored) == exported
    assert restored.canonical_digest() == source.canonical_digest()
    assert restored_provenance == provenance
    assert [value.value for value in restored.objects[1].attributes] == [
        "new",
        "approved",
    ]
    assert restored.object_relationships[0].target_object_id == "item-1"
    assert provenance["structured"]["ocel_content_sha256"]


def test_collection_and_nested_order_do_not_change_canonical_exchange() -> None:
    reordered = deepcopy(_ocel())
    reordered["eventTypes"][0]["attributes"].reverse()
    reordered["objectTypes"].reverse()
    reordered["events"][0]["attributes"].reverse()
    reordered["events"][0]["relationships"].reverse()
    reordered["objects"].reverse()
    reordered["objects"][1]["attributes"].reverse()

    first, _ = _import(_ocel())
    second, _ = _import(reordered)

    assert first.source_ref == second.source_ref
    assert first.canonical_digest() == second.canonical_digest()
    assert export_ocel_json(first) == export_ocel_json(second)


def test_empty_official_arrays_are_valid_and_map_to_an_empty_tekg_log() -> None:
    document = {
        "eventTypes": [],
        "objectTypes": [],
        "events": [],
        "objects": [],
    }
    source, _ = import_ocel_json(document, tenant="tenant-a")

    assert export_ocel_json(source) == document
    entities, relationships = source.to_graph_slice()
    assert [item["node_type"] for item in entities] == ["ObjectCentricEventLog"]
    assert relationships == []


def test_type_declarations_and_instances_materialize_as_tekg_nodes() -> None:
    source, _ = _import(_ocel())
    entities, relationships = source.to_graph_slice()

    assert {item["node_type"] for item in entities} >= {
        "ProcessEventType",
        "BusinessObjectType",
        "ProcessEvent",
        "BusinessObject",
    }
    assert {item["relationship"] for item in relationships} >= {
        "HAS_EVENT_TYPE",
        "HAS_OBJECT_TYPE",
        "INSTANCE_OF_EVENT_TYPE",
        "INSTANCE_OF_OBJECT_TYPE",
    }


def test_tenant_and_provenance_remain_governed_transport_metadata() -> None:
    source, provenance = _import(_ocel())
    first = source.to_change_envelope(tenant="tenant-a", provenance=provenance)
    replay = source.to_change_envelope(tenant="tenant-a", provenance=provenance)
    other_tenant = source.to_change_envelope(
        tenant="tenant-b",
        provenance=provenance,
    )
    exported = export_ocel_json(source)

    assert first.idempotency_key == replay.idempotency_key
    assert first.idempotency_key != other_tenant.idempotency_key
    assert first.provenance["structured"]["system"] == "erp"
    assert "tenant" not in exported
    assert "provenance" not in exported
    assert all(
        node["tenant_id"] == "tenant-a" for node in first.typed_payload["entities"]
    )


def test_duplicate_qualified_e2o_and_o2o_relations_round_trip_losslessly() -> None:
    document = _ocel()
    e2o = {"objectId": "order-1", "qualifier": "duplicate"}
    o2o = {"objectId": "item-1", "qualifier": "duplicate"}
    document["events"][0]["relationships"].extend([e2o, deepcopy(e2o)])
    document["objects"][0]["relationships"].extend([o2o, deepcopy(o2o)])

    source, _ = _import(document)
    exported = export_ocel_json(source)
    entities, _ = source.to_graph_slice()

    assert exported["events"][0]["relationships"].count(e2o) == 2
    assert exported["objects"][1]["relationships"].count(o2o) == 2
    participation_ids = [
        item["id"]
        for item in entities
        if item["node_type"] == "EventObjectParticipation"
    ]
    assert len(participation_ids) == len(set(participation_ids))


def test_standard_string_values_are_not_trimmed_during_round_trip() -> None:
    document = _ocel()
    document["events"][0]["relationships"][0]["qualifier"] = " order role "
    document["objects"][0]["relationships"][0]["qualifier"] = " contains "

    source, _ = _import(document)
    exported = export_ocel_json(source)

    assert {"objectId": "order-1", "qualifier": " order role "} in exported["events"][
        0
    ]["relationships"]
    assert {
        "objectId": "item-1",
        "qualifier": " contains ",
    } in exported["objects"][1]["relationships"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda body: body.pop("eventTypes"), "top-level arrays"),
        (
            lambda body: body.__setitem__("events", {}),
            "events.*must be an array",
        ),
        (
            lambda body: body["events"][0].__setitem__("relationships", {}),
            "events\\[\\]\\.relationships.*must be an array",
        ),
        (
            lambda body: body["objects"][0].__setitem__("attributes", {}),
            "objects\\[\\]\\.attributes.*must be an array",
        ),
        (
            lambda body: body["events"][0].__setitem__("id", 1),
            "events\\[\\]\\.id.*must be a string",
        ),
        (
            lambda body: body["events"].append(deepcopy(body["events"][0])),
            "event identifiers.*unique",
        ),
        (
            lambda body: body["objectTypes"].append(deepcopy(body["objectTypes"][0])),
            "objectTypes names.*unique",
        ),
        (
            lambda body: body["events"][0]["relationships"][0].__setitem__(
                "objectId", "missing"
            ),
            "event relation references an undeclared object",
        ),
        (
            lambda body: body["objects"][0]["relationships"][0].__setitem__(
                "objectId", "missing"
            ),
            "object relation references an undeclared object",
        ),
        (
            lambda body: body["events"][0].__setitem__("type", "missing"),
            "undeclared event type",
        ),
        (
            lambda body: body["events"][0]["attributes"][0].__setitem__(
                "name", "missing"
            ),
            "not declared",
        ),
        (
            lambda body: body["events"][0]["attributes"][0].__setitem__("value", True),
            "declared type 'integer'",
        ),
        (
            lambda body: body["events"][0]["attributes"][2].__setitem__(
                "value", float("nan")
            ),
            "declared type 'float'",
        ),
        (
            lambda body: body["eventTypes"][0]["attributes"].append(
                deepcopy(body["eventTypes"][0]["attributes"][0])
            ),
            "attribute names.*unique",
        ),
    ],
)
def test_rejects_malformed_schema_types_duplicates_and_refs(
    mutate,
    message: str,
) -> None:
    body = _ocel()
    mutate(body)
    with pytest.raises(ValueError, match=message):
        _import(body)
