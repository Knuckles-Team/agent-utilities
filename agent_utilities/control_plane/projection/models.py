"""Bounded contracts for the control-plane authority outbox and projection.

The relational/control-plane repository owns desired and observed state.  This
module describes the *one* immutable event it may publish with an authoritative
change and the small, privacy-safe checkpoint/observation records consumed by
GraphOS.  It deliberately carries references and digests rather than payloads:
secrets, grants, arguments, results, private evidence and raw bodies never cross
this boundary.

The models are persistence-independent.  A durable adapter may store them in a
relational transaction and a GraphOS adapter may project them, but neither side
is selected here.  Content-addressed IDs make retries and rebuilds deterministic
without using process time, host identity or a random UUID.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from agent_utilities.protocols.epistemic_operations import ProtocolModel

__all__ = [
    "MAX_BATCH_SIZE",
    "MAX_SUMMARY_FIELDS",
    "MAX_SUMMARY_FLOAT",
    "MAX_SUMMARY_INTEGER",
    "MAX_SUMMARY_VALUE_LENGTH",
    "OUTBOX_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "AggregateType",
    "AuthoritativeMutation",
    "AtomicCommitReceipt",
    "Digest",
    "DriftRecord",
    "EventOperation",
    "GraphProjectionReceipt",
    "ObservationPromotion",
    "ObservationPromotionPolicy",
    "OutboxEnvelope",
    "ProjectionCheckpoint",
    "ProjectionLease",
    "ProjectionOutcome",
    "ProjectionScope",
    "SummaryValue",
    "Tombstone",
    "authoritative_mutation_id_for",
    "checkpoint_digest_for",
    "event_digest_for",
    "event_id_for",
    "observation_promotion_id_for",
    "sha256_digest",
]


OUTBOX_SCHEMA_VERSION: Final[Literal["control-plane-outbox.v1"]] = (
    "control-plane-outbox.v1"
)
PROJECTION_SCHEMA_VERSION: Final[Literal["control-plane-projection.v1"]] = (
    "control-plane-projection.v1"
)
MAX_BATCH_SIZE = 256
MAX_SUMMARY_FIELDS = 32
MAX_SUMMARY_KEY_LENGTH = 64
MAX_SUMMARY_VALUE_LENGTH = 256
MAX_SUMMARY_INTEGER = 10**12
MAX_SUMMARY_FLOAT = 10**15
MAX_IDENTIFIER_LENGTH = 192
MAX_EVENT_TYPE_LENGTH = 96

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/@+-]{0,191}$")
_EVENT_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/@+-]{0,95}$")
_DIGEST_RE = r"^sha256:[0-9a-f]{64}$"
_SUMMARY_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_TEXT_RE = re.compile(
    r"(?:-----begin|bearer\s+|authorization\s*:|password\s*[=:]|secret\s*[=:]|"
    r"token\s*[=:]|api[_-]?key\s*[=:]|private[_-]?key\s*[=:]|\b(?:raw|body|payload)\b)",
    re.IGNORECASE,
)
_FORBIDDEN_REF_PREFIXES = (
    "body:",
    "data:",
    "env:",
    "file:",
    "http:",
    "https:",
    "private:",
    "result:",
    "secret:",
    "vault:",
)
_ALLOWED_SUMMARY_KEYS = frozenset(
    {
        "active",
        "aggregate_ref",
        "change_ref",
        "count",
        "deleted",
        "kind",
        "operation",
        "outcome",
        "policy_ref",
        "reason_code",
        "revision",
        "schema_ref",
        "source_ref",
        "state",
        "status",
        "subject_ref",
        "tenant_ref",
        "version",
    }
)

Identifier: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
        pattern=_IDENTIFIER_RE.pattern,
    ),
]
AggregateType: TypeAlias = Identifier
EventType: TypeAlias = Annotated[
    str,
    Field(
        min_length=1, max_length=MAX_EVENT_TYPE_LENGTH, pattern=_EVENT_TYPE_RE.pattern
    ),
]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE)]
SummaryValue: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr
Timestamp = datetime
EventOperation = Literal["upsert", "tombstone"]
MutationOperation = Literal["upsert", "delete", "rollback"]
PromotionOrigin = Literal["authoritative", "graph_observation"]


def _canonical(value: object) -> object:
    """Return JSON-safe, order-independent material for a content digest."""

    if isinstance(value, ProtocolModel):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def sha256_digest(value: object) -> str:
    """Return the versioned digest spelling used by every projection record."""

    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_requires_timezone")
    return value


def _validate_reference(value: str, field_name: str) -> str:
    lowered = value.casefold()
    if lowered.startswith(_FORBIDDEN_REF_PREFIXES) or _FORBIDDEN_TEXT_RE.search(value):
        raise ValueError(f"{field_name}_must_be_opaque_reference")
    return value


def _normalize_summary_key(raw_key: object) -> str:
    if not isinstance(raw_key, str):
        raise ValueError("summary_key_must_be_text")
    key = raw_key.strip()
    if key != raw_key or not _SUMMARY_KEY_RE.fullmatch(key):
        raise ValueError("summary_key_not_canonical")
    if key not in _ALLOWED_SUMMARY_KEYS:
        raise ValueError("summary_key_not_allowlisted")
    return key


def _normalize_summary_integer(value: int) -> int:
    if abs(value) > MAX_SUMMARY_INTEGER:
        raise ValueError("summary_integer_bound_exceeded")
    return value


def _normalize_summary_float(value: float) -> float:
    if not math.isfinite(value) or abs(value) > MAX_SUMMARY_FLOAT:
        raise ValueError("summary_float_bound_invalid")
    return value


def _normalize_summary_text(key: str, value: str) -> str:
    if not value or len(value) > MAX_SUMMARY_VALUE_LENGTH:
        raise ValueError("summary_text_length_invalid")
    if any(ord(char) < 32 and char not in "\t" for char in value):
        raise ValueError("summary_text_contains_control_character")
    if _FORBIDDEN_TEXT_RE.search(value):
        raise ValueError("summary_text_contains_sensitive_marker")
    if key.endswith("_ref"):
        _validate_reference(value, key)
    return value


def _normalize_summary_value(key: str, value: object) -> SummaryValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _normalize_summary_integer(value)
    if isinstance(value, float):
        return _normalize_summary_float(value)
    if isinstance(value, str):
        return _normalize_summary_text(key, value)
    raise ValueError("summary_value_must_be_scalar")


def _normalize_summary(value: object) -> dict[str, SummaryValue]:
    if not isinstance(value, Mapping):
        raise ValueError("summary_must_be_a_mapping")
    if len(value) > MAX_SUMMARY_FIELDS:
        raise ValueError("summary_field_limit_exceeded")
    normalized: dict[str, SummaryValue] = {}
    for raw_key, raw_item in value.items():
        key = _normalize_summary_key(raw_key)
        normalized[key] = _normalize_summary_value(key, raw_item)
    # Dict order is part of the public projection and therefore normalized at
    # construction time, even when an adapter received an unordered mapping.
    return {key: normalized[key] for key in sorted(normalized)}


def _summary_field(value: object) -> dict[str, SummaryValue]:
    return _normalize_summary(value)


def event_id_for(
    aggregate_type: str,
    aggregate_id: str,
    sequence: int,
    event_type: str,
    operation: EventOperation,
    payload_digest: str,
    aggregate_digest: str,
) -> str:
    """Derive one stable event identity from the exact authority change."""

    return (
        "event:"
        + hashlib.sha256(
            _canonical_json(
                {
                    "aggregate_digest": aggregate_digest,
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type,
                    "event_type": event_type,
                    "operation": operation,
                    "payload_digest": payload_digest,
                    "sequence": sequence,
                }
            ).encode("utf-8")
        ).hexdigest()
    )


def event_digest_for(material: Mapping[str, object]) -> str:
    """Digest the complete public event body, excluding caller identity fields."""

    return sha256_digest(material)


def authoritative_mutation_id_for(
    aggregate_type: str,
    aggregate_id: str,
    sequence: int,
    operation: MutationOperation,
    change_digest: str,
) -> str:
    return (
        "mutation:"
        + hashlib.sha256(
            _canonical_json(
                {
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type,
                    "change_digest": change_digest,
                    "operation": operation,
                    "sequence": sequence,
                }
            ).encode("utf-8")
        ).hexdigest()
    )


def checkpoint_digest_for(material: Mapping[str, object]) -> str:
    return sha256_digest(material)


def observation_promotion_id_for(observation_id: str, policy_ref: str) -> str:
    return (
        "promotion:"
        + hashlib.sha256(
            _canonical_json(
                {"observation_id": observation_id, "policy_ref": policy_ref}
            ).encode("utf-8")
        ).hexdigest()
    )


class _ProjectionModel(ProtocolModel):
    """Strict immutable base for the control-plane projection contract."""


class ProjectionScope(_ProjectionModel):
    """One aggregate partition addressed by a keyset checkpoint."""

    aggregate_type: AggregateType
    aggregate_id: Identifier
    scope_ref: Identifier | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.aggregate_type, self.aggregate_id, self.scope_ref or ""


class Tombstone(_ProjectionModel):
    """Bounded deletion evidence; it is not a copy of the deleted record."""

    reason_code: Annotated[
        str, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9._-]*$")
    ]
    target_digest: Digest
    deleted_at: Timestamp

    @field_validator("deleted_at")
    @classmethod
    def _timestamp_is_aware(cls, value: Timestamp) -> Timestamp:
        return _require_utc(value, "deleted_at")


class OutboxEnvelope(_ProjectionModel):
    """One immutable authority event safe to publish to GraphOS."""

    schema_version: Literal["control-plane-outbox.v1"] = OUTBOX_SCHEMA_VERSION
    event_id: Identifier
    aggregate_type: AggregateType
    aggregate_id: Identifier
    sequence: int = Field(ge=1)
    event_type: EventType
    operation: EventOperation
    authoritative_revision: int = Field(ge=1)
    payload_digest: Digest
    aggregate_digest: Digest
    summary: dict[str, SummaryValue] = Field(default_factory=dict)
    occurred_at: Timestamp
    tombstone: Tombstone | None = None
    event_digest: Digest

    _normalize_summary = field_validator("summary", mode="before")(_summary_field)

    @field_validator("occurred_at")
    @classmethod
    def _timestamp_is_aware(cls, value: Timestamp) -> Timestamp:
        return _require_utc(value, "occurred_at")

    @model_validator(mode="after")
    def _identity_and_operation_are_exact(self) -> OutboxEnvelope:
        expected_id = event_id_for(
            self.aggregate_type,
            self.aggregate_id,
            self.sequence,
            self.event_type,
            self.operation,
            self.payload_digest,
            self.aggregate_digest,
        )
        if self.event_id != expected_id:
            raise ValueError("event_id_is_not_content_derived")
        if (self.operation == "tombstone") != (self.tombstone is not None):
            raise ValueError("tombstone_operation_mismatch")
        expected_digest = event_digest_for(self._digest_material())
        if self.event_digest != expected_digest:
            raise ValueError("event_digest_mismatch")
        return self

    def _digest_material(self) -> dict[str, object]:
        return {
            "aggregate_digest": self.aggregate_digest,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "authoritative_revision": self.authoritative_revision,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "operation": self.operation,
            "payload_digest": self.payload_digest,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "summary": self.summary,
            "tombstone": self.tombstone,
        }

    @classmethod
    def create(
        cls,
        *,
        aggregate_type: str,
        aggregate_id: str,
        sequence: int,
        event_type: str,
        operation: EventOperation,
        authoritative_revision: int,
        payload_digest: str,
        aggregate_digest: str,
        summary: Mapping[str, object] | None = None,
        occurred_at: Timestamp,
        tombstone: Tombstone | None = None,
    ) -> OutboxEnvelope:
        normalized_summary = _normalize_summary(summary or {})
        material = {
            "aggregate_digest": aggregate_digest,
            "aggregate_id": aggregate_id,
            "aggregate_type": aggregate_type,
            "authoritative_revision": authoritative_revision,
            "event_type": event_type,
            "occurred_at": occurred_at,
            "operation": operation,
            "payload_digest": payload_digest,
            "schema_version": OUTBOX_SCHEMA_VERSION,
            "sequence": sequence,
            "summary": normalized_summary,
            "tombstone": tombstone,
        }
        return cls(
            event_id=event_id_for(
                aggregate_type,
                aggregate_id,
                sequence,
                event_type,
                operation,
                payload_digest,
                aggregate_digest,
            ),
            event_digest=event_digest_for(material),
            aggregate_digest=aggregate_digest,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            authoritative_revision=authoritative_revision,
            event_type=event_type,
            occurred_at=occurred_at,
            operation=operation,
            payload_digest=payload_digest,
            sequence=sequence,
            summary=normalized_summary,
            tombstone=tombstone,
        )


class AuthoritativeMutation(_ProjectionModel):
    """The redacted identity of one authority transaction.

    ``direction`` is intentionally closed: GraphOS projections cannot use this
    model to write back.  A graph observation needs the separate, policy-gated
    promotion contract below.
    """

    schema_version: Literal["control-plane-authority-mutation.v1"] = (
        "control-plane-authority-mutation.v1"
    )
    mutation_id: Identifier
    aggregate_type: AggregateType
    aggregate_id: Identifier
    sequence: int = Field(ge=1)
    operation: MutationOperation
    authoritative_revision: int = Field(ge=1)
    expected_revision: int | None = Field(default=None, ge=0)
    change_digest: Digest
    aggregate_digest: Digest
    summary: dict[str, SummaryValue] = Field(default_factory=dict)
    direction: Literal["authority_to_projection"] = "authority_to_projection"
    origin: Literal["authoritative", "graph_observation_promotion"] = "authoritative"

    _normalize_summary = field_validator("summary", mode="before")(_summary_field)

    @model_validator(mode="after")
    def _mutation_identity_is_exact(self) -> AuthoritativeMutation:
        expected = authoritative_mutation_id_for(
            self.aggregate_type,
            self.aggregate_id,
            self.sequence,
            self.operation,
            self.change_digest,
        )
        if self.mutation_id != expected:
            raise ValueError("mutation_id_is_not_content_derived")
        if (
            self.expected_revision is not None
            and self.expected_revision >= self.authoritative_revision
        ):
            raise ValueError("expected_revision_must_precede_authoritative_revision")
        return self

    @classmethod
    def create(
        cls,
        *,
        aggregate_type: str,
        aggregate_id: str,
        sequence: int,
        operation: MutationOperation,
        authoritative_revision: int,
        change_digest: str,
        aggregate_digest: str,
        expected_revision: int | None = None,
        summary: Mapping[str, object] | None = None,
    ) -> AuthoritativeMutation:
        return cls(
            mutation_id=authoritative_mutation_id_for(
                aggregate_type,
                aggregate_id,
                sequence,
                operation,
                change_digest,
            ),
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            sequence=sequence,
            operation=operation,
            authoritative_revision=authoritative_revision,
            expected_revision=expected_revision,
            change_digest=change_digest,
            aggregate_digest=aggregate_digest,
            summary=_normalize_summary(summary or {}),
        )

    @classmethod
    def create_observation_promotion(
        cls,
        *,
        aggregate_type: str,
        aggregate_id: str,
        sequence: int,
        authoritative_revision: int,
        change_digest: str,
        aggregate_digest: str,
        expected_revision: int | None = None,
        summary: Mapping[str, object] | None = None,
    ) -> AuthoritativeMutation:
        """Create the distinct mutation shape used by the policy-gated path."""

        mutation = cls.create(
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            sequence=sequence,
            operation="upsert",
            authoritative_revision=authoritative_revision,
            change_digest=change_digest,
            aggregate_digest=aggregate_digest,
            expected_revision=expected_revision,
            summary=summary,
        )
        return mutation.model_copy(update={"origin": "graph_observation_promotion"})


class AtomicCommitReceipt(_ProjectionModel):
    """Privacy-safe result of one authority/outbox transaction."""

    schema_version: Literal["control-plane-commit-receipt.v1"] = (
        "control-plane-commit-receipt.v1"
    )
    mutation_id: Identifier
    aggregate_type: AggregateType
    aggregate_id: Identifier
    authoritative_revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    event_id: Identifier | None = None
    committed: StrictBool
    replayed: StrictBool = False


class GraphProjectionReceipt(_ProjectionModel):
    """Idempotent GraphOS application receipt with no graph row payload."""

    receipt_version: Literal["control-plane-graph-projection-receipt.v1"] = (
        "control-plane-graph-projection-receipt.v1"
    )
    event_id: Identifier
    applied: StrictBool
    replayed: StrictBool = False


class ProjectionLease(_ProjectionModel):
    """A caller-owned fencing term for projection writes."""

    lease_version: Literal["control-plane-projection-lease.v1"] = (
        "control-plane-projection-lease.v1"
    )
    projector_id: Identifier
    scope: ProjectionScope
    fence_token: int = Field(ge=1)
    owner_ref: Identifier
    expires_at: Timestamp

    @field_validator("expires_at")
    @classmethod
    def _timestamp_is_aware(cls, value: Timestamp) -> Timestamp:
        return _require_utc(value, "expires_at")


class ProjectionCheckpoint(_ProjectionModel):
    """Fenced, keyset cursor for one aggregate projection stream."""

    checkpoint_version: Literal["control-plane-projection-checkpoint.v1"] = (
        "control-plane-projection-checkpoint.v1"
    )
    checkpoint_id: Identifier
    projector_id: Identifier
    scope: ProjectionScope
    last_sequence: int = Field(ge=0)
    last_event_id: Identifier | None = None
    last_event_digest: Digest | None = None
    fence_token: int = Field(ge=1)
    checkpoint_digest: Digest

    @model_validator(mode="after")
    def _checkpoint_identity_is_exact(self) -> ProjectionCheckpoint:
        if self.last_sequence == 0 and (
            self.last_event_id is not None or self.last_event_digest is not None
        ):
            raise ValueError("initial_checkpoint_cannot_have_event_identity")
        if self.last_sequence > 0 and (
            self.last_event_id is None or self.last_event_digest is None
        ):
            raise ValueError("advanced_checkpoint_requires_event_identity")
        expected_id = (
            "checkpoint:"
            + hashlib.sha256(
                _canonical_json(
                    {
                        "projector_id": self.projector_id,
                        "scope": self.scope,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
        if self.checkpoint_id != expected_id:
            raise ValueError("checkpoint_id_is_not_scope_derived")
        expected_digest = checkpoint_digest_for(self._digest_material())
        if self.checkpoint_digest != expected_digest:
            raise ValueError("checkpoint_digest_mismatch")
        return self

    def _digest_material(self) -> dict[str, object]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "fence_token": self.fence_token,
            "last_event_digest": self.last_event_digest,
            "last_event_id": self.last_event_id,
            "last_sequence": self.last_sequence,
            "projector_id": self.projector_id,
            "scope": self.scope,
        }

    @classmethod
    def initial(
        cls, *, projector_id: str, scope: ProjectionScope, fence_token: int
    ) -> ProjectionCheckpoint:
        material = {
            "checkpoint_version": "control-plane-projection-checkpoint.v1",
            "fence_token": fence_token,
            "last_event_digest": None,
            "last_event_id": None,
            "last_sequence": 0,
            "projector_id": projector_id,
            "scope": scope,
        }
        return cls(
            checkpoint_id="checkpoint:"
            + hashlib.sha256(
                _canonical_json({"projector_id": projector_id, "scope": scope}).encode(
                    "utf-8"
                )
            ).hexdigest(),
            checkpoint_digest=checkpoint_digest_for(material),
            fence_token=fence_token,
            last_event_digest=None,
            last_event_id=None,
            last_sequence=0,
            projector_id=projector_id,
            scope=scope,
        )

    @classmethod
    def after_event(
        cls,
        *,
        prior: ProjectionCheckpoint,
        event: OutboxEnvelope,
        fence_token: int,
    ) -> ProjectionCheckpoint:
        if event.sequence != prior.last_sequence + 1:
            raise ValueError("checkpoint_event_must_be_next_sequence")
        if (
            event.aggregate_type != prior.scope.aggregate_type
            or event.aggregate_id != prior.scope.aggregate_id
        ):
            raise ValueError("checkpoint_event_scope_mismatch")
        material = {
            "checkpoint_version": "control-plane-projection-checkpoint.v1",
            "fence_token": fence_token,
            "last_event_digest": event.event_digest,
            "last_event_id": event.event_id,
            "last_sequence": event.sequence,
            "projector_id": prior.projector_id,
            "scope": prior.scope,
        }
        return cls(
            checkpoint_id=prior.checkpoint_id,
            checkpoint_digest=checkpoint_digest_for(material),
            fence_token=fence_token,
            last_event_digest=event.event_digest,
            last_event_id=event.event_id,
            last_sequence=event.sequence,
            projector_id=prior.projector_id,
            scope=prior.scope,
        )


class ProjectionOutcome(_ProjectionModel):
    """One bounded result, suitable for logs and reconciliation dashboards."""

    outcome_version: Literal["control-plane-projection-outcome.v1"] = (
        "control-plane-projection-outcome.v1"
    )
    status: Literal[
        "applied",
        "replayed",
        "rejected",
        "gap",
        "conflict",
        "failed",
        "drift",
        "cleaned",
    ]
    aggregate_type: AggregateType
    aggregate_id: Identifier
    sequence: int = Field(ge=1)
    event_id: Identifier
    reason_code: str | None = Field(
        default=None, max_length=64, pattern=r"^[a-z][a-z0-9._-]*$"
    )


class DriftRecord(_ProjectionModel):
    """A repairable projection discrepancy without exception/raw payload text."""

    drift_version: Literal["control-plane-projection-drift.v1"] = (
        "control-plane-projection-drift.v1"
    )
    reason_code: Literal[
        "graph_apply_failed",
        "checkpoint_conflict",
        "checkpoint_fence_lost",
        "sequence_gap",
        "identity_conflict",
        "scope_mismatch",
        "tombstone_cleanup_failed",
        "rebuild_failed",
        "reverse_sync_rejected",
    ]
    projector_id: Identifier
    aggregate_type: AggregateType
    aggregate_id: Identifier
    sequence: int = Field(ge=0)
    event_id: Identifier | None = None
    expected_digest: Digest | None = None
    observed_digest: Digest | None = None
    repairable: StrictBool
    recorded_at: Timestamp

    @field_validator("recorded_at")
    @classmethod
    def _timestamp_is_aware(cls, value: Timestamp) -> Timestamp:
        return _require_utc(value, "recorded_at")


class ObservationPromotionPolicy(_ProjectionModel):
    """Explicit allowlist governing a graph observation becoming authority."""

    policy_version: Literal["control-plane-observation-promotion-policy.v1"] = (
        "control-plane-observation-promotion-policy.v1"
    )
    policy_ref: Identifier
    enabled: StrictBool
    allowed_observation_types: tuple[Identifier, ...] = Field(default=(), max_length=32)
    require_evidence_ref: StrictBool = True
    max_age_seconds: int = Field(ge=1, le=31_536_000)

    @model_validator(mode="after")
    def _policy_is_normalized(self) -> ObservationPromotionPolicy:
        if len(set(self.allowed_observation_types)) != len(
            self.allowed_observation_types
        ):
            raise ValueError("promotion_policy_observation_types_must_be_unique")
        if (
            tuple(sorted(self.allowed_observation_types))
            != self.allowed_observation_types
        ):
            raise ValueError("promotion_policy_observation_types_must_be_sorted")
        return self


class ObservationPromotion(_ProjectionModel):
    """A privacy-safe observation eligible for an explicit promotion policy."""

    observation_version: Literal["control-plane-graph-observation.v1"] = (
        "control-plane-graph-observation.v1"
    )
    observation_id: Identifier
    observation_type: Identifier
    aggregate_type: AggregateType
    aggregate_id: Identifier
    observation_digest: Digest
    source_ref: Identifier
    evidence_ref: Identifier | None = None
    observed_at: Timestamp
    summary: dict[str, SummaryValue] = Field(default_factory=dict)

    _normalize_summary = field_validator("summary", mode="before")(_summary_field)

    @field_validator("source_ref", "evidence_ref")
    @classmethod
    def _opaque_reference(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _validate_reference(value, str(getattr(info, "field_name", "reference")))

    @field_validator("observed_at")
    @classmethod
    def _timestamp_is_aware(cls, value: Timestamp) -> Timestamp:
        return _require_utc(value, "observed_at")
