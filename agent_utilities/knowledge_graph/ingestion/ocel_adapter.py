"""Governed JSON-OCEL 2.0 exchange at the semantic-event boundary.

CONCEPT:AU-KG.ingest.semantic-event-contract — JSON-OCEL is an interchange
format, not a second event store.  This adapter validates it into the canonical
semantic slice and exports that same slice deterministically for replay.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .semantic_event_model import (
    BusinessObject,
    EventAttributeValue,
    EventObjectParticipation,
    ObjectCentricGraphSlice,
    ObjectState,
    QualifiedObjectRelationship,
    TemporalAttributeValue,
)

OCEL_VERSION = "2.0"
_META = "ocel:meta"


def _text(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"OCEL field {field!r} is required")
    return result


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"OCEL field {field!r} must be an object")
    return value


def _table(value: object, field: str) -> Mapping[str, Any]:
    table = _mapping(value, field)
    if not table:
        raise ValueError(f"OCEL field {field!r} must not be empty")
    return table


def _scalar_attributes(value: object, field: str) -> tuple[EventAttributeValue, ...]:
    attrs = _mapping(value or {}, field)
    result: list[EventAttributeValue] = []
    for name, item in attrs.items():
        if not isinstance(item, str | int | float | bool) and item is not None:
            raise ValueError(f"OCEL attribute {field}.{name!s} must be scalar")
        result.append(EventAttributeValue(name=_text(name, field), value=item))
    return tuple(sorted(result, key=lambda item: item.name))


def _typed_omap(
    value: object, object_types: Mapping[str, str]
) -> tuple[EventObjectParticipation, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("OCEL field 'ocel:typedOmap' must be an array")
    result: list[EventObjectParticipation] = []
    for entry in value:
        relation = _mapping(entry, "ocel:typedOmap[]")
        object_id = _text(relation.get("ocel:oid"), "ocel:typedOmap[].ocel:oid")
        object_type = object_types.get(object_id)
        if object_type is None:
            raise ValueError("OCEL event relation references an undeclared object")
        result.append(
            EventObjectParticipation(
                object_id=object_id,
                object_type=object_type,
                qualifier=str(relation.get("ocel:qualifier") or "").strip(),
            )
        )
    if not result:
        raise ValueError("OCEL event must participate in at least one object")
    return tuple(
        sorted(
            result, key=lambda item: (item.object_type, item.object_id, item.qualifier)
        )
    )


def _provenance(meta: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _mapping(meta.get("provenance") or {}, "ocel:meta.provenance")
    structured = _mapping(
        provenance.get("structured") or {}, "ocel:meta.provenance.structured"
    )
    unstructured = provenance.get("unstructured_refs") or []
    if not isinstance(unstructured, Sequence) or isinstance(
        unstructured, str | bytes | bytearray
    ):
        raise ValueError("OCEL provenance unstructured_refs must be an array")
    return {
        "structured": dict(sorted(structured.items())),
        "unstructured_refs": sorted(
            _text(item, "ocel:meta.provenance.unstructured_refs[]")
            for item in unstructured
        ),
    }


def import_ocel_json(
    payload: str | Mapping[str, Any],
    *,
    tenant: str,
    source_ref: str = "",
    mapping_version: str = "",
) -> tuple[ObjectCentricGraphSlice, dict[str, Any]]:
    """Validate one governed JSON-OCEL 2.0 document into source truth.

    Tenant identity is an authority boundary supplied by the caller, never
    inferred from the OCEL body.  A body may repeat it only as a consistency
    assertion; disagreement fails closed.
    """
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("OCEL input must be valid JSON") from exc
    else:
        raw = payload
    document = _mapping(raw, "OCEL document")
    if document.get("ocel:version") != OCEL_VERSION:
        raise ValueError("OCEL document must declare ocel:version '2.0'")
    meta = _mapping(document.get(_META) or {}, _META)
    authoritative_tenant = _text(tenant, "tenant")
    asserted_tenant = str(meta.get("tenant") or "").strip()
    if asserted_tenant and asserted_tenant != authoritative_tenant:
        raise ValueError("OCEL tenant does not match the authorized tenant")
    effective_source = _text(source_ref or meta.get("source_ref"), "source_ref")
    effective_mapping = _text(
        mapping_version or meta.get("mapping_version"), "mapping_version"
    )
    provenance = _provenance(meta)

    raw_objects = _table(document.get("ocel:objects"), "ocel:objects")
    object_types: dict[str, str] = {}
    objects: list[BusinessObject] = []
    states: list[ObjectState] = []
    relationships: list[QualifiedObjectRelationship] = []
    for object_id, raw_object in sorted(
        raw_objects.items(), key=lambda item: str(item[0])
    ):
        oid = _text(object_id, "ocel:objects key")
        record = _mapping(raw_object, f"ocel:objects.{oid}")
        object_type = _text(record.get("ocel:type"), f"ocel:objects.{oid}.ocel:type")
        object_types[oid] = object_type
        attributes: list[TemporalAttributeValue] = []
        for value in record.get("ocel:ovmap") or []:
            item = _mapping(value, "ocel:ovmap[]")
            attributes.append(
                TemporalAttributeValue(
                    name=_text(item.get("ocel:name"), "ocel:ovmap[].ocel:name"),
                    value=item.get("ocel:value"),
                    valid_from=item.get("ocel:time"),
                )
            )
        objects.append(
            BusinessObject(
                object_id=oid,
                object_type=object_type,
                attributes=tuple(
                    sorted(
                        attributes,
                        key=lambda item: (item.name, item.valid_from, str(item.value)),
                    )
                ),
            )
        )
        for value in record.get("ocel:states") or []:
            item = _mapping(value, "ocel:states[]")
            states.append(
                ObjectState(
                    state_id=_text(item.get("ocel:id"), "ocel:states[].ocel:id"),
                    object_id=oid,
                    object_type=object_type,
                    valid_from=item.get("ocel:validFrom"),
                    valid_to=item.get("ocel:validTo"),
                    observed_at=item.get("ocel:observedAt"),
                    attributes=_scalar_attributes(
                        item.get("ocel:vmap"), "ocel:states[].ocel:vmap"
                    ),
                )
            )

    for object_id, raw_object in sorted(
        raw_objects.items(), key=lambda item: str(item[0])
    ):
        oid = _text(object_id, "ocel:objects key")
        record = _mapping(raw_object, f"ocel:objects.{oid}")
        for index, value in enumerate(record.get("ocel:o2o") or []):
            item = _mapping(value, "ocel:o2o[]")
            target = _text(item.get("ocel:oid"), "ocel:o2o[].ocel:oid")
            if target not in object_types:
                raise ValueError("OCEL object relation references an undeclared object")
            relationships.append(
                QualifiedObjectRelationship(
                    relationship_id=_text(
                        item.get("ocel:id") or f"{oid}:{target}:{index}",
                        "ocel:o2o[].ocel:id",
                    ),
                    source_object_id=oid,
                    target_object_id=target,
                    qualifier=_text(
                        item.get("ocel:qualifier"), "ocel:o2o[].ocel:qualifier"
                    ),
                    valid_from=item.get("ocel:validFrom"),
                    valid_to=item.get("ocel:validTo"),
                )
            )

    raw_events = _table(document.get("ocel:events"), "ocel:events")
    events = []
    for event_id, raw_event in sorted(
        raw_events.items(), key=lambda item: str(item[0])
    ):
        eid = _text(event_id, "ocel:events key")
        record = _mapping(raw_event, f"ocel:events.{eid}")
        events.append(
            {
                "event_id": eid,
                "activity": _text(
                    record.get("ocel:activity"), f"ocel:events.{eid}.ocel:activity"
                ),
                "occurred_at": record.get("ocel:timestamp"),
                "objects": _typed_omap(record.get("ocel:typedOmap"), object_types),
                "attributes": _scalar_attributes(record.get("ocel:vmap"), "ocel:vmap"),
                "source_ref": _text(
                    record.get("ocel:source") or effective_source, "ocel:source"
                ),
                "sequence_tiebreaker": str(record.get("ocel:sequence") or ""),
            }
        )
    try:
        slice_ = ObjectCentricGraphSlice(
            log_id=_text(meta.get("log_id") or effective_source, "ocel:meta.log_id"),
            source_ref=effective_source,
            mapping_version=effective_mapping,
            events=tuple(events),
            objects=tuple(objects),
            object_states=tuple(states),
            object_relationships=tuple(relationships),
        )
    except ValidationError as exc:
        raise ValueError(
            f"OCEL document violates the semantic event contract: {exc}"
        ) from exc
    return slice_, provenance


def export_ocel_json(
    slice_: ObjectCentricGraphSlice,
    *,
    tenant: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export canonical source truth to one stable governed JSON-OCEL document."""
    effective_tenant = _text(tenant, "tenant")
    base_provenance = provenance or {"structured": {}, "unstructured_refs": []}
    checked_provenance = _provenance({_META: None, "provenance": base_provenance})
    objects: dict[str, Any] = {}
    relationships_by_source: dict[str, list[QualifiedObjectRelationship]] = {}
    for relationship in slice_.object_relationships:
        relationships_by_source.setdefault(relationship.source_object_id, []).append(
            relationship
        )
    states_by_object: dict[str, list[ObjectState]] = {}
    for state in slice_.object_states:
        states_by_object.setdefault(state.object_id, []).append(state)
    for item in sorted(slice_.objects, key=lambda value: value.object_id):
        objects[item.object_id] = {
            "ocel:type": item.object_type,
            "ocel:ovmap": [
                {
                    "ocel:name": attr.name,
                    "ocel:value": attr.value,
                    "ocel:time": attr.valid_from.isoformat(),
                }
                for attr in sorted(
                    item.attributes,
                    key=lambda value: (value.name, value.valid_from, str(value.value)),
                )
            ],
            "ocel:states": [
                {
                    "ocel:id": state.state_id,
                    "ocel:validFrom": state.valid_from.isoformat(),
                    "ocel:validTo": state.valid_to.isoformat()
                    if state.valid_to
                    else None,
                    "ocel:observedAt": state.observed_at.isoformat(),
                    "ocel:vmap": {
                        attr.name: attr.value
                        for attr in sorted(
                            state.attributes, key=lambda value: value.name
                        )
                    },
                }
                for state in sorted(
                    states_by_object.get(item.object_id, []),
                    key=lambda value: value.state_id,
                )
            ],
            "ocel:o2o": [
                {
                    "ocel:id": relation.relationship_id,
                    "ocel:oid": relation.target_object_id,
                    "ocel:qualifier": relation.qualifier,
                    "ocel:validFrom": relation.valid_from.isoformat()
                    if relation.valid_from
                    else None,
                    "ocel:validTo": relation.valid_to.isoformat()
                    if relation.valid_to
                    else None,
                }
                for relation in sorted(
                    relationships_by_source.get(item.object_id, []),
                    key=lambda value: value.relationship_id,
                )
            ],
        }
    events = {
        item.event_id: {
            "ocel:activity": item.activity,
            "ocel:timestamp": item.occurred_at.isoformat(),
            "ocel:typedOmap": [
                {"ocel:oid": relation.object_id, "ocel:qualifier": relation.qualifier}
                for relation in sorted(
                    item.objects,
                    key=lambda value: (
                        value.object_type,
                        value.object_id,
                        value.qualifier,
                    ),
                )
            ],
            "ocel:vmap": {
                attr.name: attr.value
                for attr in sorted(item.attributes, key=lambda value: value.name)
            },
            "ocel:source": item.source_ref,
            "ocel:sequence": item.sequence_tiebreaker,
        }
        for item in sorted(slice_.events, key=lambda value: value.event_id)
    }
    return {
        "ocel:version": OCEL_VERSION,
        _META: {
            "tenant": effective_tenant,
            "log_id": slice_.log_id,
            "source_ref": slice_.source_ref,
            "mapping_version": slice_.mapping_version,
            "content_hash": slice_.canonical_digest(),
            "provenance": checked_provenance,
        },
        "ocel:events": events,
        "ocel:objects": objects,
    }
