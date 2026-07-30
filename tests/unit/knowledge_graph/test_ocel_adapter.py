"""Governed JSON-OCEL 2.0 import/export and tEKG boundary proofs."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.ocel_adapter import (
    export_ocel_json,
    import_ocel_json,
)


def _ocel() -> dict:
    return {
        "ocel:version": "2.0",
        "ocel:meta": {
            "tenant": "tenant-a",
            "log_id": "orders",
            "source_ref": "fixture://ocel/orders",
            "mapping_version": "ocel-v1",
            "provenance": {
                "structured": {"system": "erp", "batch": "42"},
                "unstructured_refs": ["fixture://notes/orders-42"],
            },
        },
        "ocel:objects": {
            "item-1": {"ocel:type": "Item", "ocel:ovmap": []},
            "order-1": {
                "ocel:type": "Order",
                "ocel:ovmap": [
                    {
                        "ocel:name": "status",
                        "ocel:value": "new",
                        "ocel:time": "2026-01-01T00:00:00Z",
                    }
                ],
                "ocel:states": [
                    {
                        "ocel:id": "order-1-v1",
                        "ocel:validFrom": "2026-01-01T00:00:00Z",
                        "ocel:validTo": "2026-01-02T00:00:00Z",
                        "ocel:observedAt": "2026-01-01T00:00:01Z",
                        "ocel:vmap": {"status": "new"},
                    }
                ],
                "ocel:o2o": [
                    {
                        "ocel:id": "order-item",
                        "ocel:oid": "item-1",
                        "ocel:qualifier": "contains",
                    }
                ],
            },
        },
        "ocel:events": {
            "create": {
                "ocel:activity": "create order",
                "ocel:timestamp": "2026-01-01T00:00:00Z",
                "ocel:typedOmap": [
                    {"ocel:oid": "order-1", "ocel:qualifier": "order"},
                    {"ocel:oid": "item-1", "ocel:qualifier": "line-item"},
                ],
                "ocel:vmap": {"channel": "web"},
            }
        },
    }


def test_ocel_round_trip_preserves_versioned_truth_and_provenance() -> None:
    source, provenance = import_ocel_json(_ocel(), tenant="tenant-a")
    exported = export_ocel_json(source, tenant="tenant-a", provenance=provenance)
    restored, restored_provenance = import_ocel_json(exported, tenant="tenant-a")

    assert restored.canonical_digest() == source.canonical_digest()
    assert restored_provenance == provenance
    assert exported["ocel:meta"]["content_hash"] == source.canonical_digest()
    assert restored.object_states[0].valid_to is not None
    assert restored.object_relationships[0].target_object_id == "item-1"


def test_ocel_tenant_provenance_and_idempotency_are_materialization_boundaries() -> (
    None
):
    source, provenance = import_ocel_json(_ocel(), tenant="tenant-a")
    first = source.to_change_envelope(tenant="tenant-a", provenance=provenance)
    replay = source.to_change_envelope(tenant="tenant-a", provenance=provenance)
    other_tenant = source.to_change_envelope(tenant="tenant-b", provenance=provenance)

    assert first.idempotency_key == replay.idempotency_key
    assert first.idempotency_key != other_tenant.idempotency_key
    assert first.provenance["structured"]["system"] == "erp"
    assert all(
        node["tenant_id"] == "tenant-a" for node in first.typed_payload["entities"]
    )
    assert all(
        edge["tenant_id"] == "tenant-a" for edge in first.typed_payload["relationships"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda body: body.__setitem__("ocel:version", "1.0"), "version"),
        (
            lambda body: body["ocel:events"]["create"]["ocel:typedOmap"].__setitem__(
                0, {"ocel:oid": "missing", "ocel:qualifier": "order"}
            ),
            "undeclared object",
        ),
        (
            lambda body: body["ocel:objects"]["order-1"]["ocel:o2o"].__setitem__(
                0,
                {
                    "ocel:id": "bad-relation",
                    "ocel:oid": "missing",
                    "ocel:qualifier": "contains",
                },
            ),
            "object relation references an undeclared object",
        ),
    ],
)
def test_ocel_rejects_malformed_or_cross_object_input(mutate, message: str) -> None:
    body = _ocel()
    mutate(body)
    with pytest.raises(ValueError, match=message):
        import_ocel_json(body, tenant="tenant-a")


def test_ocel_rejects_cross_tenant_document() -> None:
    with pytest.raises(ValueError, match="authorized tenant"):
        import_ocel_json(_ocel(), tenant="tenant-b")
