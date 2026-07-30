"""Canonical object-centric, temporal, and neural graph boundary models.

CONCEPT:AU-KG.ingest.semantic-event-contract — OCEL-shaped source truth,
temporal Event Knowledge Graph state, and neural graph proposals share one
validated boundary.  These models perform no LLM calls and do not persist by
themselves; callers commit their canonical graph slice through ChangeEnvelope.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScalarValue = str | int | float | bool | None
SemanticEntityKind = Literal["event", "object", "object_state"]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def _stable_id(kind: str, source_ref: str, *parts: str) -> str:
    digest = hashlib.sha256(
        json.dumps(
            [source_ref, *parts],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"{kind}:{digest}"


class SemanticBoundaryModel(BaseModel):
    """Strict immutable base for source and derived semantic contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TemporalAttributeValue(SemanticBoundaryModel):
    """One typed object-attribute revision in OCEL 2.0 form."""

    name: str = Field(min_length=1)
    value: ScalarValue
    valid_from: datetime

    _normalize_valid_from = field_validator("valid_from")(_utc)


class EventAttributeValue(SemanticBoundaryModel):
    """One scalar event attribute."""

    name: str = Field(min_length=1)
    value: ScalarValue


class BusinessObject(SemanticBoundaryModel):
    """A source-qualified object with time-varying attributes."""

    object_id: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    attributes: tuple[TemporalAttributeValue, ...] = ()


class EventObjectParticipation(SemanticBoundaryModel):
    """One qualified event-to-object participation."""

    object_id: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    qualifier: str = ""


class ProcessEvent(SemanticBoundaryModel):
    """A lossless event record retained before any case projection."""

    event_id: str = Field(min_length=1)
    activity: str = Field(min_length=1)
    occurred_at: datetime
    objects: tuple[EventObjectParticipation, ...] = Field(min_length=1)
    attributes: tuple[EventAttributeValue, ...] = ()
    source_ref: str = Field(min_length=1)
    sequence_tiebreaker: str = ""

    _normalize_occurred_at = field_validator("occurred_at")(_utc)


class ObjectState(SemanticBoundaryModel):
    """Immutable object state valid for a bounded process-time interval."""

    state_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None
    observed_at: datetime
    attributes: tuple[EventAttributeValue, ...] = ()

    _normalize_valid_from = field_validator("valid_from")(_utc)
    _normalize_valid_to = field_validator("valid_to")(
        lambda value: None if value is None else _utc(value)
    )
    _normalize_observed_at = field_validator("observed_at")(_utc)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("object-state valid_to must be later than valid_from")
        return self


class QualifiedObjectRelationship(SemanticBoundaryModel):
    """One qualified, optionally temporal OCEL object-to-object relation."""

    relationship_id: str = Field(min_length=1)
    source_object_id: str = Field(min_length=1)
    target_object_id: str = Field(min_length=1)
    qualifier: str = Field(min_length=1)
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    _normalize_valid_from = field_validator("valid_from")(
        lambda value: None if value is None else _utc(value)
    )
    _normalize_valid_to = field_validator("valid_to")(
        lambda value: None if value is None else _utc(value)
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValueError(
                "object-relationship valid_to must be later than valid_from"
            )
        return self


class ProcessPerspective(SemanticBoundaryModel):
    """Versioned object-centric analytical selection, never a source mutation."""

    perspective_id: str = Field(min_length=1)
    object_types: tuple[str, ...] = Field(min_length=1)
    qualifiers: tuple[str, ...] = ()
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    derivation_version: str = Field(min_length=1)

    _normalize_effective_from = field_validator("effective_from")(
        lambda value: None if value is None else _utc(value)
    )
    _normalize_effective_to = field_validator("effective_to")(
        lambda value: None if value is None else _utc(value)
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to <= self.effective_from
        ):
            raise ValueError(
                "process-perspective effective_to must be later than effective_from"
            )
        return self


class SemanticEntityRef(SemanticBoundaryModel):
    """Typed reference preventing event/object/state identifier ambiguity."""

    kind: SemanticEntityKind
    source_id: str = Field(min_length=1)


class NeuralRepresentation(SemanticBoundaryModel):
    """Versioned latent representation attached to symbolic source truth."""

    representation_id: str = Field(min_length=1)
    target: SemanticEntityRef
    encoder_id: str = Field(min_length=1)
    encoder_version: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    artifact_ref: str = Field(min_length=1)
    source_graph_epoch: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_ref: str = ""


class NeuralRelationPrediction(SemanticBoundaryModel):
    """Calibrated relation candidate that cannot masquerade as accepted truth."""

    prediction_id: str = Field(min_length=1)
    subject: SemanticEntityRef
    predicate: str = Field(min_length=1)
    object: SemanticEntityRef
    score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    model_ref: str = Field(min_length=1)
    candidate_set_ref: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    decision_status: Literal["proposed"] = "proposed"


class EntityResolutionProposal(SemanticBoundaryModel):
    """Ambiguous mention-to-entity resolution awaiting governed review."""

    proposal_id: str = Field(min_length=1)
    mention_ref: str = Field(min_length=1)
    candidate: SemanticEntityRef
    score: float = Field(ge=0.0, le=1.0)
    blocking_features: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    decision_status: Literal["proposed"] = "proposed"


class ObjectCentricGraphSlice(SemanticBoundaryModel):
    """One validated OCEL/tEKG slice plus governed neural proposals."""

    log_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    mapping_version: str = Field(min_length=1)
    events: tuple[ProcessEvent, ...] = Field(min_length=1)
    objects: tuple[BusinessObject, ...] = Field(min_length=1)
    object_states: tuple[ObjectState, ...] = ()
    object_relationships: tuple[QualifiedObjectRelationship, ...] = ()
    perspectives: tuple[ProcessPerspective, ...] = ()
    neural_representations: tuple[NeuralRepresentation, ...] = ()
    neural_predictions: tuple[NeuralRelationPrediction, ...] = ()
    entity_resolution_proposals: tuple[EntityResolutionProposal, ...] = ()

    @staticmethod
    def _unique(values: list[str], label: str) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{label} identifiers must be unique")

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        self._unique([item.event_id for item in self.events], "event")
        self._unique([item.object_id for item in self.objects], "object")
        self._unique([item.state_id for item in self.object_states], "object-state")
        self._unique(
            [item.relationship_id for item in self.object_relationships],
            "object-relationship",
        )
        self._unique(
            [item.perspective_id for item in self.perspectives],
            "process-perspective",
        )
        self._unique(
            [item.representation_id for item in self.neural_representations],
            "neural-representation",
        )
        self._unique(
            [item.prediction_id for item in self.neural_predictions],
            "neural-prediction",
        )

        object_types = {item.object_id: item.object_type for item in self.objects}
        for event in self.events:
            for participation in event.objects:
                if object_types.get(participation.object_id) != participation.object_type:
                    raise ValueError(
                        "event participation must reference a declared object "
                        "with the same object_type"
                    )
        for state in self.object_states:
            if object_types.get(state.object_id) != state.object_type:
                raise ValueError(
                    "object state must reference a declared object with the same "
                    "object_type"
                )
        for relationship in self.object_relationships:
            if (
                relationship.source_object_id not in object_types
                or relationship.target_object_id not in object_types
            ):
                raise ValueError(
                    "object relationship endpoints must reference declared objects"
                )

        valid_refs = {
            ("event", item.event_id) for item in self.events
        } | {
            ("object", item.object_id) for item in self.objects
        } | {
            ("object_state", item.state_id) for item in self.object_states
        }
        refs = [
            representation.target for representation in self.neural_representations
        ]
        refs.extend(
            ref
            for prediction in self.neural_predictions
            for ref in (prediction.subject, prediction.object)
        )
        refs.extend(
            proposal.candidate for proposal in self.entity_resolution_proposals
        )
        if any((ref.kind, ref.source_id) not in valid_refs for ref in refs):
            raise ValueError(
                "neural and resolution references must target symbolic entities "
                "in the same validated slice"
            )
        return self

    def canonical_digest(self) -> str:
        """Content digest used for replay and mapping-version evidence."""
        payload_data = self.model_dump(mode="json", exclude_none=True)
        sort_keys = {
            "events": "event_id",
            "objects": "object_id",
            "object_states": "state_id",
            "object_relationships": "relationship_id",
            "perspectives": "perspective_id",
            "neural_representations": "representation_id",
            "neural_predictions": "prediction_id",
            "entity_resolution_proposals": "proposal_id",
        }
        for collection, identity in sort_keys.items():
            payload_data[collection] = sorted(
                payload_data.get(collection, []),
                key=lambda item: str(item[identity]),
            )
        for event in payload_data["events"]:
            event["objects"] = sorted(
                event.get("objects", []),
                key=lambda item: (
                    str(item["object_type"]),
                    str(item["object_id"]),
                    str(item.get("qualifier", "")),
                ),
            )
            event["attributes"] = sorted(
                event.get("attributes", []),
                key=lambda item: (
                    str(item["name"]),
                    json.dumps(item.get("value"), sort_keys=True, default=str),
                ),
            )
        for business_object in payload_data["objects"]:
            business_object["attributes"] = sorted(
                business_object.get("attributes", []),
                key=lambda item: (
                    str(item["name"]),
                    str(item["valid_from"]),
                    json.dumps(item.get("value"), sort_keys=True, default=str),
                ),
            )
        for state in payload_data["object_states"]:
            state["attributes"] = sorted(
                state.get("attributes", []),
                key=lambda item: (
                    str(item["name"]),
                    json.dumps(item.get("value"), sort_keys=True, default=str),
                ),
            )
        for perspective in payload_data["perspectives"]:
            perspective["object_types"] = sorted(
                set(perspective.get("object_types", []))
            )
            perspective["qualifiers"] = sorted(
                set(perspective.get("qualifiers", []))
            )
        payload = json.dumps(
            payload_data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_graph_slice(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return canonical nodes/links for the existing ChangeEnvelope writer."""
        digest = self.canonical_digest()
        entities: list[dict[str, Any]] = [
            {
                "id": _stable_id("object-centric-log", self.source_ref, self.log_id),
                "node_type": "ObjectCentricEventLog",
                "source_record_id": self.log_id,
                "source_ref": self.source_ref,
                "mapping_version": self.mapping_version,
                "content_hash": digest,
            }
        ]
        links: list[dict[str, Any]] = []
        log_node_id = entities[0]["id"]

        object_ids: dict[str, str] = {}
        for item in sorted(self.objects, key=lambda value: value.object_id):
            node_id = _stable_id(
                "business-object",
                self.source_ref,
                item.object_type,
                item.object_id,
            )
            object_ids[item.object_id] = node_id
            entities.append(
                {
                    "id": node_id,
                    "node_type": "BusinessObject",
                    "source_record_id": item.object_id,
                    "object_type": item.object_type,
                    "attributes": [
                        value.model_dump(mode="json") for value in item.attributes
                    ],
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )
            links.append(
                {
                    "source": log_node_id,
                    "target": node_id,
                    "relationship": "HAS_OBJECT",
                }
            )

        event_ids: dict[str, str] = {}
        for item in sorted(self.events, key=lambda value: value.event_id):
            event_id = _stable_id(
                "process-event", self.source_ref, item.event_id
            )
            event_ids[item.event_id] = event_id
            entities.append(
                {
                    "id": event_id,
                    "node_type": "ProcessEvent",
                    "source_record_id": item.event_id,
                    "activity": item.activity,
                    "occurred_at": item.occurred_at.isoformat(),
                    "attributes": [
                        value.model_dump(mode="json") for value in item.attributes
                    ],
                    "source_ref": item.source_ref,
                    "mapping_version": self.mapping_version,
                    "sequence_tiebreaker": item.sequence_tiebreaker,
                }
            )
            links.append(
                {
                    "source": log_node_id,
                    "target": event_id,
                    "relationship": "HAS_EVENT",
                }
            )
            for participation in sorted(
                item.objects,
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.qualifier,
                ),
            ):
                participation_id = _stable_id(
                    "event-object-participation",
                    self.source_ref,
                    item.event_id,
                    participation.object_type,
                    participation.object_id,
                    participation.qualifier,
                )
                entities.append(
                    {
                        "id": participation_id,
                        "node_type": "EventObjectParticipation",
                        "qualifier": participation.qualifier,
                        "source_ref": item.source_ref,
                        "mapping_version": self.mapping_version,
                    }
                )
                links.extend(
                    [
                        {
                            "source": event_id,
                            "target": participation_id,
                            "relationship": "HAS_PARTICIPATION",
                        },
                        {
                            "source": participation_id,
                            "target": object_ids[participation.object_id],
                            "relationship": "PARTICIPATES_AS",
                        },
                    ]
                )

        state_ids: dict[str, str] = {}
        for item in sorted(self.object_states, key=lambda value: value.state_id):
            state_id = _stable_id("object-state", self.source_ref, item.state_id)
            state_ids[item.state_id] = state_id
            entities.append(
                {
                    "id": state_id,
                    "node_type": "ObjectState",
                    "source_record_id": item.state_id,
                    "object_type": item.object_type,
                    "valid_from": item.valid_from.isoformat(),
                    "valid_to": (
                        item.valid_to.isoformat() if item.valid_to is not None else None
                    ),
                    "observed_at": item.observed_at.isoformat(),
                    "attributes": [
                        value.model_dump(mode="json") for value in item.attributes
                    ],
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )
            links.append(
                {
                    "source": state_id,
                    "target": object_ids[item.object_id],
                    "relationship": "STATE_OF",
                }
            )

        for item in sorted(
            self.object_relationships,
            key=lambda value: value.relationship_id,
        ):
            relationship_id = _stable_id(
                "object-relationship", self.source_ref, item.relationship_id
            )
            entities.append(
                {
                    "id": relationship_id,
                    "node_type": "ObjectRelationship",
                    "source_record_id": item.relationship_id,
                    "qualifier": item.qualifier,
                    "valid_from": (
                        item.valid_from.isoformat()
                        if item.valid_from is not None
                        else None
                    ),
                    "valid_to": (
                        item.valid_to.isoformat()
                        if item.valid_to is not None
                        else None
                    ),
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )
            links.extend(
                [
                    {
                        "source": relationship_id,
                        "target": object_ids[item.source_object_id],
                        "relationship": "RELATION_SOURCE",
                    },
                    {
                        "source": relationship_id,
                        "target": object_ids[item.target_object_id],
                        "relationship": "RELATION_TARGET",
                    },
                ]
            )

        for item in sorted(
            self.perspectives, key=lambda value: value.perspective_id
        ):
            perspective_data = item.model_dump(mode="json", exclude_none=True)
            perspective_data["object_types"] = sorted(set(item.object_types))
            perspective_data["qualifiers"] = sorted(set(item.qualifiers))
            entities.append(
                {
                    "id": _stable_id(
                        "process-perspective", self.source_ref, item.perspective_id
                    ),
                    "node_type": "ProcessPerspective",
                    **perspective_data,
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )

        canonical_ids: dict[tuple[str, str], str] = {
            **{("event", key): value for key, value in event_ids.items()},
            **{("object", key): value for key, value in object_ids.items()},
            **{("object_state", key): value for key, value in state_ids.items()},
        }
        for item in sorted(
            self.neural_representations,
            key=lambda value: value.representation_id,
        ):
            representation_id = _stable_id(
                "neural-representation", self.source_ref, item.representation_id
            )
            entities.append(
                {
                    "id": representation_id,
                    "node_type": "NeuralRepresentation",
                    "source_record_id": item.representation_id,
                    "encoder_id": item.encoder_id,
                    "encoder_version": item.encoder_version,
                    "dimension": item.dimension,
                    "artifact_ref": item.artifact_ref,
                    "source_graph_epoch": item.source_graph_epoch,
                    "content_hash": item.content_hash,
                    "calibration_ref": item.calibration_ref,
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )
            links.append(
                {
                    "source": canonical_ids[(item.target.kind, item.target.source_id)],
                    "target": representation_id,
                    "relationship": "HAS_NEURAL_REPRESENTATION",
                }
            )

        for item in sorted(
            self.neural_predictions, key=lambda value: value.prediction_id
        ):
            prediction_id = _stable_id(
                "neural-relation-prediction", self.source_ref, item.prediction_id
            )
            entities.append(
                {
                    "id": prediction_id,
                    "node_type": "NeuralRelationPrediction",
                    "source_record_id": item.prediction_id,
                    "predicate": item.predicate,
                    "prediction_score": item.score,
                    "prediction_uncertainty": item.uncertainty,
                    "model_ref": item.model_ref,
                    "candidate_set_ref": item.candidate_set_ref,
                    "evidence_refs": list(item.evidence_refs),
                    "decision_status": item.decision_status,
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )
            links.extend(
                [
                    {
                        "source": prediction_id,
                        "target": canonical_ids[
                            (item.subject.kind, item.subject.source_id)
                        ],
                        "relationship": "PREDICTS_SUBJECT",
                    },
                    {
                        "source": prediction_id,
                        "target": canonical_ids[
                            (item.object.kind, item.object.source_id)
                        ],
                        "relationship": "PREDICTS_OBJECT",
                    },
                ]
            )

        for item in sorted(
            self.entity_resolution_proposals,
            key=lambda value: value.proposal_id,
        ):
            proposal_id = _stable_id(
                "entity-resolution-proposal", self.source_ref, item.proposal_id
            )
            entities.append(
                {
                    "id": proposal_id,
                    "node_type": "EntityResolutionProposal",
                    "source_record_id": item.proposal_id,
                    "mention_ref": item.mention_ref,
                    "resolution_score": item.score,
                    "blocking_features": list(item.blocking_features),
                    "evidence_refs": list(item.evidence_refs),
                    "decision_status": item.decision_status,
                    "source_ref": self.source_ref,
                    "mapping_version": self.mapping_version,
                }
            )
            links.append(
                {
                    "source": proposal_id,
                    "target": canonical_ids[
                        (item.candidate.kind, item.candidate.source_id)
                    ],
                    "relationship": "PROPOSES_RESOLUTION",
                }
            )

        return entities, links
