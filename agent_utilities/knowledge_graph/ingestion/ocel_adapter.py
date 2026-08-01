"""Governed OCEL 2.0 JSON exchange at the semantic-event boundary.

CONCEPT:AU-KG.ingest.semantic-event-contract — OCEL is an interchange format,
not a second event store.  This adapter implements the four-array JSON format
defined by the OCEL 2.0 specification (``eventTypes``, ``objectTypes``,
``events``, and ``objects``), validates its cross-references and declared
attribute types without an LLM, and maps it to the canonical tEKG slice.

Specification and schema:
https://www.ocel-standard.org/specification/formats/json/
https://www.ocel-standard.org/2.0/ocel20-schema-json.json

The schema supplies the structural contract. Declared value validation follows
the format page's five scalar types and numeric minimal example; the published
schema currently describes attribute values as strings.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, TypeVar, cast

from pydantic import ValidationError

from .semantic_event_model import (
    BusinessObject,
    EventAttributeValue,
    EventObjectParticipation,
    NeuralRelationPrediction,
    ObjectCentricGraphSlice,
    OcelAttributeDefinition,
    OcelAttributeType,
    OcelEventType,
    OcelObjectType,
    ProcessEvent,
    QualifiedObjectRelationship,
    ScalarValue,
    SemanticEntityRef,
    TemporalAttributeValue,
)

OCEL_VERSION = "2.0"
OCEL_MAPPING_VERSION = "ocel-json-2.0"
_ATTRIBUTE_TYPES = frozenset({"string", "time", "integer", "float", "boolean"})
_Declaration = TypeVar("_Declaration", OcelEventType, OcelObjectType)

# -- structural NeuralRelationPrediction producer (CONCEPT:AU-KG.evolution.neural-relation-proposal-producer, D-OB-11) --
# A degenerate event that touches more than this many objects is skipped for
# PAIRWISE co-occurrence expansion (its objects still count toward each
# object's own participation set) — a defensive bound against one pathological
# event turning C(n, 2) into a blowup on an otherwise ordinary log; not a
# deployment-varying value, so a constant, not an env flag.
_MAX_EVENT_FANOUT_FOR_CO_OCCURRENCE = 64
_NEURAL_PREDICTION_TOP_K = 20
_NEURAL_PREDICTION_MIN_SCORE = 0.1


def predict_object_co_occurrence_relations(
    events: Sequence[ProcessEvent],
    object_relationships: Sequence[QualifiedObjectRelationship],
    *,
    source_ref: str,
    candidate_set_ref: str,
    top_k: int = _NEURAL_PREDICTION_TOP_K,
    min_score: float = _NEURAL_PREDICTION_MIN_SCORE,
) -> tuple[NeuralRelationPrediction, ...]:
    """Object-object relation CANDIDATES from real event co-participation.

    D-OB-11: ``NeuralRelationPrediction`` (the model class, ontology concept, and
    SHACL shape) had zero producers anywhere. A sibling lane (D-73-6) confirmed
    by exhaustive repo-wide grep that no reviewed accept/reject label set exists
    for relation types anywhere in the codebase, and correctly declined to build
    a *relation-aware* (ComplEx/RotatE-style, "is there a link with predicate P")
    scorer rather than fabricate one. This producer stays strictly on the honest
    side of that line: it does NOT guess a relation TYPE (no qualifier is
    predicted) — it only proposes THAT two objects are plausibly related, scored
    by Jaccard similarity of the sets of events they co-participate in. That is
    a real, deterministic structural computation over the slice's own event log
    (the same feature family — common-neighbors / Jaccard — the engine's own
    ``graph_learn`` link-predictor already uses), not a neural-network inference;
    it is named and ``model_ref``-tagged accordingly so nobody mistakes an
    uncalibrated structural score for a validated probability (mirrors the
    documented-placeholder precedent in the sibling lane's entity-resolution
    ``_calibrate()``).

    Object pairs already asserted as a ``QualifiedObjectRelationship`` in this
    same slice are excluded — this only proposes genuinely NEW candidates, never
    restates known ground truth as a "prediction". Every prediction cites the
    real co-occurring event ids as its ``evidence_refs`` (traceable, never
    fabricated). ``decision_status`` stays the type's own invariant, "proposed":
    nothing here is a fact.
    """
    participation: dict[str, set[str]] = defaultdict(set)
    pair_shared_events: dict[frozenset[str], set[str]] = defaultdict(set)
    for event in events:
        object_ids = sorted({item.object_id for item in event.objects})
        for object_id in object_ids:
            participation[object_id].add(event.event_id)
        if len(object_ids) > _MAX_EVENT_FANOUT_FOR_CO_OCCURRENCE:
            continue
        for a, b in itertools.combinations(object_ids, 2):
            pair_shared_events[frozenset((a, b))].add(event.event_id)

    known_pairs = {
        frozenset((r.source_object_id, r.target_object_id))
        for r in object_relationships
    }

    scored: list[tuple[float, str, str, tuple[str, ...]]] = []
    for pair, shared in pair_shared_events.items():
        if pair in known_pairs:
            continue
        a, b = sorted(pair)
        union = participation[a] | participation[b]
        score = len(shared) / len(union) if union else 0.0
        if score < min_score:
            continue
        scored.append((score, a, b, tuple(sorted(shared))))
    # Deterministic ranking: score desc, then the pair itself for stable ties.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    predictions: list[NeuralRelationPrediction] = []
    for score, a, b, shared_events in scored[:top_k]:
        predictions.append(
            NeuralRelationPrediction(
                prediction_id=f"co-occurs:{source_ref}:{a}:{b}",
                subject=SemanticEntityRef(kind="object", source_id=a),
                predicate="co_occurrence_candidate",
                object=SemanticEntityRef(kind="object", source_id=b),
                score=round(score, 6),
                uncertainty=round(1.0 - score, 6),
                model_ref="structural-event-co-participation-jaccard-uncalibrated-v1",
                candidate_set_ref=candidate_set_ref,
                evidence_refs=tuple(shared_events),
            )
        )
    return tuple(predictions)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"OCEL field {field!r} must be a string")
    if not value:
        raise ValueError(f"OCEL field {field!r} is required")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"OCEL field {field!r} must be an object")
    return value


def _array(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"OCEL field {field!r} must be an array")
    return value


def _optional_array(
    value: Mapping[str, Any],
    key: str,
    field: str,
) -> Sequence[Any]:
    if key not in value:
        return ()
    return _array(value[key], field)


def _unique(values: Sequence[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"OCEL {field} values must be unique")


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"OCEL field {field!r} must be an ISO 8601 date-time string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"OCEL field {field!r} must be an ISO 8601 date-time string"
        ) from exc
    if result.tzinfo is None:
        raise ValueError(f"OCEL field {field!r} must include a timezone")
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize unordered OCEL node collections for content identity."""
    document = dict(value)
    for collection, identity in (
        ("eventTypes", "name"),
        ("objectTypes", "name"),
        ("events", "id"),
        ("objects", "id"),
    ):
        rows: list[dict[str, Any]] = []
        for raw_row in _array(document.get(collection), collection):
            row = dict(_mapping(raw_row, f"{collection}[]"))
            for nested in ("attributes", "relationships"):
                if nested in row:
                    row[nested] = sorted(
                        (
                            dict(_mapping(item, f"{collection}[].{nested}[]"))
                            for item in _array(row[nested], f"{collection}[].{nested}")
                        ),
                        key=lambda item: _canonical_json(item),
                    )
            rows.append(row)
        document[collection] = sorted(
            rows,
            key=lambda item: (
                str(item.get(identity) or ""),
                _canonical_json(item),
            ),
        )
    return document


def _declarations(
    value: object,
    field: str,
    declaration_type: type[_Declaration],
) -> tuple[_Declaration, ...]:
    declarations: list[_Declaration] = []
    for raw_declaration in _array(value, field):
        declaration = _mapping(raw_declaration, f"{field}[]")
        name = _text(declaration.get("name"), f"{field}[].name")
        attributes: list[OcelAttributeDefinition] = []
        for raw_attribute in _array(
            declaration.get("attributes"), f"{field}[].attributes"
        ):
            attribute = _mapping(raw_attribute, f"{field}[].attributes[]")
            value_type = _text(attribute.get("type"), f"{field}[].attributes[].type")
            if value_type not in _ATTRIBUTE_TYPES:
                raise ValueError(
                    f"OCEL attribute type {value_type!r} is not one of "
                    "string, time, integer, float, or boolean"
                )
            attributes.append(
                OcelAttributeDefinition(
                    name=_text(attribute.get("name"), f"{field}[].attributes[].name"),
                    value_type=cast(OcelAttributeType, value_type),
                )
            )
        _unique([item.name for item in attributes], f"{field} attribute names")
        declarations.append(
            declaration_type(
                name=name,
                attributes=tuple(sorted(attributes, key=lambda item: item.name)),
            )
        )
    _unique([item.name for item in declarations], f"{field} names")
    return tuple(sorted(declarations, key=lambda item: item.name))


def _check_value(value: object, declared_type: OcelAttributeType, field: str) -> None:
    valid = False
    if declared_type == "string":
        valid = isinstance(value, str)
    elif declared_type == "time":
        try:
            _timestamp(value, field)
            valid = True
        except ValueError:
            valid = False
    elif declared_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif declared_type == "float":
        valid = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    elif declared_type == "boolean":
        valid = isinstance(value, bool)
    if not valid:
        raise ValueError(
            f"OCEL field {field!r} does not match declared type {declared_type!r}"
        )


def _attributes(
    value: object,
    *,
    field: str,
    declarations: Mapping[str, OcelAttributeType],
    temporal: bool,
) -> tuple[EventAttributeValue | TemporalAttributeValue, ...]:
    parsed: list[EventAttributeValue | TemporalAttributeValue] = []
    for raw_attribute in _array(value, field):
        attribute = _mapping(raw_attribute, f"{field}[]")
        name = _text(attribute.get("name"), f"{field}[].name")
        declared_type = declarations.get(name)
        if declared_type is None:
            raise ValueError(
                f"OCEL attribute {name!r} is not declared for its event/object type"
            )
        if "value" not in attribute:
            raise ValueError(f"OCEL field {field + '[].value'!r} is required")
        raw_value = attribute["value"]
        _check_value(raw_value, declared_type, f"{field}[].value")
        scalar_value: ScalarValue = raw_value
        if temporal:
            parsed.append(
                TemporalAttributeValue(
                    name=name,
                    value=scalar_value,
                    valid_from=_timestamp(attribute.get("time"), f"{field}[].time"),
                )
            )
        else:
            parsed.append(EventAttributeValue(name=name, value=scalar_value))
    return tuple(
        sorted(
            parsed,
            key=lambda item: (
                item.name,
                getattr(item, "valid_from", datetime.min).isoformat(),
                json.dumps(item.value, sort_keys=True, default=str),
            ),
        )
    )


def _provenance(
    provenance: Mapping[str, Any] | None,
    *,
    content_sha256: str,
) -> dict[str, Any]:
    value = _mapping({} if provenance is None else provenance, "provenance")
    structured = (
        _mapping(value["structured"], "provenance.structured")
        if "structured" in value
        else {}
    )
    unstructured = _optional_array(
        value,
        "unstructured_refs",
        "provenance.unstructured_refs",
    )
    return {
        "structured": {
            **dict(sorted(structured.items())),
            "ocel_content_sha256": content_sha256,
            "interchange": "OCEL 2.0 JSON",
        },
        "unstructured_refs": sorted(
            _text(item, "provenance.unstructured_refs[]") for item in unstructured
        ),
    }


def _declaration_map(
    declaration: OcelEventType | OcelObjectType,
) -> dict[str, OcelAttributeType]:
    return {item.name: item.value_type for item in declaration.attributes}


def import_ocel_json(
    payload: str | Mapping[str, Any],
    *,
    tenant: str,
    source_ref: str = "",
    mapping_version: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> tuple[ObjectCentricGraphSlice, dict[str, Any]]:
    """Validate official OCEL 2.0 JSON into canonical object/event source truth.

    Tenant, source identity, mapping version, and provenance are governed
    transport metadata, not proprietary fields injected into the OCEL document.
    When no source reference is supplied, a content-addressed reference keeps
    replay deterministic without trusting caller-controlled object identifiers.
    """
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("OCEL input must be valid JSON") from exc
    else:
        raw = payload
    document = _mapping(raw, "document")
    required = {"eventTypes", "objectTypes", "events", "objects"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(
            "OCEL 2.0 JSON requires top-level arrays: " + ", ".join(missing)
        )
    _text(tenant, "tenant")  # Authority is retained by ChangeEnvelope, not OCEL.

    content_sha256 = hashlib.sha256(
        _canonical_json(_canonical_document(document))
    ).hexdigest()
    effective_source = source_ref.strip() or f"ocel-json:sha256:{content_sha256}"
    effective_mapping = mapping_version.strip() or OCEL_MAPPING_VERSION
    checked_provenance = _provenance(
        provenance,
        content_sha256=content_sha256,
    )

    event_types = _declarations(document["eventTypes"], "eventTypes", OcelEventType)
    object_types = _declarations(document["objectTypes"], "objectTypes", OcelObjectType)
    event_type_map = {item.name: item for item in event_types}
    object_type_map = {item.name: item for item in object_types}

    objects: list[BusinessObject] = []
    raw_objects = _array(document["objects"], "objects")
    object_type_by_id: dict[str, str] = {}
    object_records: list[Mapping[str, Any]] = []
    for raw_object in raw_objects:
        record = _mapping(raw_object, "objects[]")
        object_id = _text(record.get("id"), "objects[].id")
        object_type = _text(record.get("type"), "objects[].type")
        declaration = object_type_map.get(object_type)
        if declaration is None:
            raise ValueError("OCEL object references an undeclared object type")
        if object_id in object_type_by_id:
            raise ValueError("OCEL object identifiers must be unique")
        object_type_by_id[object_id] = object_type
        object_records.append(record)
        objects.append(
            BusinessObject(
                object_id=object_id,
                object_type=object_type,
                attributes=cast(
                    tuple[TemporalAttributeValue, ...],
                    _attributes(
                        _optional_array(
                            record,
                            "attributes",
                            "objects[].attributes",
                        ),
                        field="objects[].attributes",
                        declarations=_declaration_map(declaration),
                        temporal=True,
                    ),
                ),
            )
        )

    object_relationships: list[QualifiedObjectRelationship] = []
    for record in object_records:
        source_id = _text(record.get("id"), "objects[].id")
        normalized: list[tuple[str, str]] = []
        for raw_relationship in _optional_array(
            record,
            "relationships",
            "objects[].relationships",
        ):
            relationship = _mapping(raw_relationship, "objects[].relationships[]")
            target_id = _text(
                relationship.get("objectId"),
                "objects[].relationships[].objectId",
            )
            if target_id not in object_type_by_id:
                raise ValueError("OCEL object relation references an undeclared object")
            normalized.append(
                (
                    target_id,
                    _text(
                        relationship.get("qualifier"),
                        "objects[].relationships[].qualifier",
                    ),
                )
            )
        for occurrence, (target_id, qualifier) in enumerate(sorted(normalized)):
            relationship_digest = hashlib.sha256(
                _canonical_json(
                    [effective_source, source_id, target_id, qualifier, occurrence]
                )
            ).hexdigest()[:32]
            object_relationships.append(
                QualifiedObjectRelationship(
                    relationship_id=f"ocel-o2o:{relationship_digest}",
                    source_object_id=source_id,
                    target_object_id=target_id,
                    qualifier=qualifier,
                )
            )

    events: list[ProcessEvent] = []
    event_ids: list[str] = []
    for raw_event in _array(document["events"], "events"):
        record = _mapping(raw_event, "events[]")
        event_id = _text(record.get("id"), "events[].id")
        event_type = _text(record.get("type"), "events[].type")
        event_declaration = event_type_map.get(event_type)
        if event_declaration is None:
            raise ValueError("OCEL event references an undeclared event type")
        event_ids.append(event_id)
        participations: list[EventObjectParticipation] = []
        for raw_relationship in _optional_array(
            record,
            "relationships",
            "events[].relationships",
        ):
            relationship = _mapping(raw_relationship, "events[].relationships[]")
            object_id = _text(
                relationship.get("objectId"),
                "events[].relationships[].objectId",
            )
            participation_type = object_type_by_id.get(object_id)
            if participation_type is None:
                raise ValueError("OCEL event relation references an undeclared object")
            participations.append(
                EventObjectParticipation(
                    object_id=object_id,
                    object_type=participation_type,
                    qualifier=_text(
                        relationship.get("qualifier"),
                        "events[].relationships[].qualifier",
                    ),
                )
            )
        events.append(
            ProcessEvent(
                event_id=event_id,
                activity=event_type,
                occurred_at=_timestamp(record.get("time"), "events[].time"),
                objects=tuple(
                    sorted(
                        participations,
                        key=lambda item: (
                            item.object_type,
                            item.object_id,
                            item.qualifier,
                        ),
                    )
                ),
                attributes=cast(
                    tuple[EventAttributeValue, ...],
                    _attributes(
                        _optional_array(
                            record,
                            "attributes",
                            "events[].attributes",
                        ),
                        field="events[].attributes",
                        declarations=_declaration_map(event_declaration),
                        temporal=False,
                    ),
                ),
                source_ref=effective_source,
                sequence_tiebreaker=event_id,
            )
        )
    _unique(event_ids, "event identifiers")

    # Structural NeuralRelationPrediction candidates (D-OB-11) — native-by-default,
    # computed from THIS import's own event log, never fabricated (see
    # predict_object_co_occurrence_relations' docstring for the honesty boundary).
    neural_predictions = predict_object_co_occurrence_relations(
        events,
        object_relationships,
        source_ref=effective_source,
        candidate_set_ref=f"ocel-co-occurrence:{content_sha256[:32]}",
    )

    try:
        slice_ = ObjectCentricGraphSlice(
            log_id=f"ocel-log:{content_sha256[:32]}",
            source_ref=effective_source,
            mapping_version=effective_mapping,
            event_types=event_types,
            object_types=object_types,
            events=tuple(events),
            objects=tuple(sorted(objects, key=lambda item: item.object_id)),
            object_relationships=tuple(
                sorted(
                    object_relationships,
                    key=lambda item: item.relationship_id,
                )
            ),
            neural_predictions=neural_predictions,
        )
    except ValidationError as exc:
        raise ValueError(
            f"OCEL document violates the semantic event contract: {exc}"
        ) from exc
    return slice_, checked_provenance


def export_ocel_json(slice_: ObjectCentricGraphSlice) -> dict[str, Any]:
    """Export canonical source truth as official deterministic OCEL 2.0 JSON."""
    relationships_by_source: dict[str, list[QualifiedObjectRelationship]] = {}
    for relationship in slice_.object_relationships:
        relationships_by_source.setdefault(relationship.source_object_id, []).append(
            relationship
        )

    return {
        "eventTypes": [
            {
                "name": declaration.name,
                "attributes": [
                    {"name": attribute.name, "type": attribute.value_type}
                    for attribute in sorted(
                        declaration.attributes, key=lambda item: item.name
                    )
                ],
            }
            for declaration in sorted(slice_.event_types, key=lambda item: item.name)
        ],
        "objectTypes": [
            {
                "name": declaration.name,
                "attributes": [
                    {"name": attribute.name, "type": attribute.value_type}
                    for attribute in sorted(
                        declaration.attributes, key=lambda item: item.name
                    )
                ],
            }
            for declaration in sorted(slice_.object_types, key=lambda item: item.name)
        ],
        "events": [
            {
                "id": event.event_id,
                "type": event.activity,
                "time": event.occurred_at.isoformat().replace("+00:00", "Z"),
                "attributes": [
                    {"name": attribute.name, "value": attribute.value}
                    for attribute in sorted(
                        event.attributes,
                        key=lambda item: (
                            item.name,
                            json.dumps(item.value, sort_keys=True, default=str),
                        ),
                    )
                ],
                "relationships": [
                    {
                        "objectId": participation.object_id,
                        "qualifier": participation.qualifier,
                    }
                    for participation in sorted(
                        event.objects,
                        key=lambda item: (
                            item.object_type,
                            item.object_id,
                            item.qualifier,
                        ),
                    )
                ],
            }
            for event in sorted(slice_.events, key=lambda item: item.event_id)
        ],
        "objects": [
            {
                "id": business_object.object_id,
                "type": business_object.object_type,
                "attributes": [
                    {
                        "name": attribute.name,
                        "time": attribute.valid_from.isoformat().replace("+00:00", "Z"),
                        "value": attribute.value,
                    }
                    for attribute in sorted(
                        business_object.attributes,
                        key=lambda item: (
                            item.name,
                            item.valid_from,
                            json.dumps(item.value, sort_keys=True, default=str),
                        ),
                    )
                ],
                "relationships": [
                    {
                        "objectId": relationship.target_object_id,
                        "qualifier": relationship.qualifier,
                    }
                    for relationship in sorted(
                        relationships_by_source.get(business_object.object_id, ()),
                        key=lambda item: (
                            item.target_object_id,
                            item.qualifier,
                            item.relationship_id,
                        ),
                    )
                ],
            }
            for business_object in sorted(
                slice_.objects, key=lambda item: item.object_id
            )
        ],
    }
