"""Durable Repository Manager WorkItems.

CONCEPT:AU-ORCH.org.repository-workitem-authority — Durable repository-development WorkItem authority

This module is the Agent Utilities side of the repository-development v1
boundary.  It deliberately does not import Repository Manager: the two
packages exchange the frozen JSON contract, while this adapter projects that
contract onto the one engine-native :class:`WorkItem` state machine.

The adapter stores only opaque repository/job correlations and content
digests.  Repository paths, command bodies, credentials, and log contents
remain in the Repository Manager domain or artifact store.  The WorkItem is
still the sole authority for state, dependencies, leases, retries, fences,
checkpoints, cancellation, and terminal effects.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from agent_utilities.orchestration import work_item as _work_item
from agent_utilities.orchestration.operation_payload import (
    MAX_OPERATION_PAYLOAD_BYTES,
    RepositoryBuildExecutionPayloadV1,
    RepositoryOperationPayload,
    operation_payload_from_mapping,
    payload_digest,
)
from agent_utilities.orchestration.work_item import (
    DEFAULT_LEASE_TTL_S,
    WorkItemBackendUnavailable,
    cancel_work_item,
    checkpoint_work_item,
    claim_next,
    claim_specific,
    commit_result,
    get_work_item,
    heartbeat,
    mark_running,
    submit_work_item_atomic,
)
from agent_utilities.protocols.epistemic_operations._generated import (
    DevelopmentLaneCleanupIntent,
    DevelopmentLaneIntent,
)
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

CONTRACT_VERSION: Literal["1"] = "1"
_METADATA_KEY = "repository_work_item"
_JOB_ID_RE = re.compile(
    r"^rmjob:(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_WORK_ITEM_ID_RE = re.compile(
    r"^workitem:repository_manager:(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY_NAMESPACE = uuid.UUID("f2e04c6d-71aa-4ae8-8a15-20a2b2fce94f")
_MAX_LIST_LIMIT = 1000
TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE = (
    "typed_execution_payload_authority_unavailable"
)


class RepositoryWorkItemError(ValueError):
    """Base error for invalid or unauthorizable repository job requests."""


class RepositoryWorkItemConflict(RepositoryWorkItemError):
    """The idempotency key already names different immutable input."""


class RepositoryOperation(StrEnum):
    """Stable repository operation values from the RMDD v1 contract."""

    LANE_ALLOCATE = "lane.allocate"
    LANE_CHECK = "lane.check"
    # RMDD-28: the native DevelopmentLaneHold lifecycle WorkItem family. Distinct
    # from LANE_ALLOCATE/LANE_CHECK (RMDD-06's generic job submissions) -- these
    # two kinds are the sole typed carrier for a DevelopmentLaneIntent/
    # DevelopmentLaneCleanupIntent and are what the native reserve/renew/
    # observe/finish transaction and the separate fenced cleanup transaction
    # bind to (see the RMDD-28 lane brief, "Authority and identity model").
    LANE_LIFECYCLE = "lane.lifecycle"
    LANE_CLEANUP = "lane.cleanup"
    REPOSITORY = "repository"
    VALIDATION = "validation"
    BUILD = "build"
    MERGE = "merge"
    RELEASE = "release"
    CANDIDATE_SUBMIT = "candidate.submit"
    GENERATION_CERTIFY = "generation.certify"
    BRANCH_LAND = "branch.land"
    WORKSPACE_VALIDATE = "workspace.validate"
    WORKSPACE_BUMP = "workspace.bump"
    WORKSPACE_PUSH = "workspace.push"
    REPAIR = "repair"


class RepositoryWorkItemKind(StrEnum):
    """Registered WorkItem kinds owned by Repository Manager."""

    LANE_ALLOCATE = "repository.lane.allocate"
    LANE_CHECK = "repository.lane.check"
    LANE_LIFECYCLE = "repository.lane.lifecycle"
    LANE_CLEANUP = "repository.lane.cleanup"
    OPERATION = "repository.operation"
    VALIDATION = "repository.validation"
    BUILD = "repository.build"
    MERGE = "repository.merge"
    RELEASE = "repository.release"
    CANDIDATE_SUBMIT = "repository.candidate.submit"
    GENERATION_CERTIFY = "repository.generation.certify"
    BRANCH_LAND = "repository.branch.land"
    WORKSPACE_VALIDATE = "repository.workspace.validate"
    WORKSPACE_BUMP = "repository.workspace.bump"
    WORKSPACE_PUSH = "repository.workspace.push"
    REPAIR = "repository.repair"


class RepositoryJobState(StrEnum):
    """Repository-facing rendering of the native WorkItem lifecycle."""

    SUBMITTED = "submitted"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead-letter"


_OPERATION_TO_KIND: dict[RepositoryOperation, RepositoryWorkItemKind] = {
    RepositoryOperation.LANE_ALLOCATE: RepositoryWorkItemKind.LANE_ALLOCATE,
    RepositoryOperation.LANE_CHECK: RepositoryWorkItemKind.LANE_CHECK,
    RepositoryOperation.LANE_LIFECYCLE: RepositoryWorkItemKind.LANE_LIFECYCLE,
    RepositoryOperation.LANE_CLEANUP: RepositoryWorkItemKind.LANE_CLEANUP,
    RepositoryOperation.REPOSITORY: RepositoryWorkItemKind.OPERATION,
    RepositoryOperation.VALIDATION: RepositoryWorkItemKind.VALIDATION,
    RepositoryOperation.BUILD: RepositoryWorkItemKind.BUILD,
    RepositoryOperation.MERGE: RepositoryWorkItemKind.MERGE,
    RepositoryOperation.RELEASE: RepositoryWorkItemKind.RELEASE,
    RepositoryOperation.CANDIDATE_SUBMIT: RepositoryWorkItemKind.CANDIDATE_SUBMIT,
    RepositoryOperation.GENERATION_CERTIFY: RepositoryWorkItemKind.GENERATION_CERTIFY,
    RepositoryOperation.BRANCH_LAND: RepositoryWorkItemKind.BRANCH_LAND,
    RepositoryOperation.WORKSPACE_VALIDATE: RepositoryWorkItemKind.WORKSPACE_VALIDATE,
    RepositoryOperation.WORKSPACE_BUMP: RepositoryWorkItemKind.WORKSPACE_BUMP,
    RepositoryOperation.WORKSPACE_PUSH: RepositoryWorkItemKind.WORKSPACE_PUSH,
    RepositoryOperation.REPAIR: RepositoryWorkItemKind.REPAIR,
}
_REPOSITORY_KINDS = tuple(kind.value for kind in RepositoryWorkItemKind)


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(ord(char) < 0x20 for char in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _opaque_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence")
    try:
        values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence") from exc
    return tuple(sorted({_nonblank(item, field_name) for item in values}))


def _sorted_opaque_sequence(value: object, field_name: str) -> tuple[str, ...]:
    """Validate and sort opaque labels without changing multiplicity."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence")
    try:
        values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence") from exc
    return tuple(sorted(_nonblank(item, field_name) for item in values))


def _digest(value: object) -> str:
    encoded = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _as_mapping(contract: object) -> dict[str, Any]:
    if isinstance(contract, BaseModel):
        value = contract.model_dump(mode="json", exclude_none=False)
    elif isinstance(contract, Mapping):
        value = dict(contract)
    else:
        raise TypeError("repository request must be a mapping or Pydantic model")
    return dict(value)


def _nested_mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    return _as_mapping(value)


def _validate_digest(value: str | None, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


_OPAQUE_PREFIX = "opaque:v1:"


def _encode_opaque(value: str | None) -> str | None:
    """Encode one non-secret opaque identifier before generic sanitization.

    The persistence privacy guard quite correctly redacts card-shaped strings,
    but Git refs and inventory aliases are not card data and must round-trip
    exactly. Chunk separators make the encoded projection immune to the
    guard's digit-pattern recognizers while retaining no plaintext content.
    """

    if value is None:
        return None
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return _OPAQUE_PREFIX + ".".join(
        encoded[index : index + 3] for index in range(0, len(encoded), 3)
    )


def _decode_opaque(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RepositoryWorkItemConflict(
            f"repository {field_name} projection is invalid"
        )
    if not value.startswith(_OPAQUE_PREFIX):
        # Rows written before RMDD-02's encoded projection remain readable.
        return value
    compact = value[len(_OPAQUE_PREFIX) :].replace(".", "")
    if not compact:
        raise RepositoryWorkItemConflict(f"repository {field_name} projection is empty")
    compact += "=" * (-len(compact) % 4)
    try:
        decoded = base64.b64decode(compact, altchars=b"-_", validate=True)
        return decoded.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RepositoryWorkItemConflict(
            f"repository {field_name} projection is invalid"
        ) from exc


def _encode_opaque_sequence(values: Sequence[str]) -> list[str]:
    return [encoded for value in values if (encoded := _encode_opaque(value))]


def _decode_opaque_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise RepositoryWorkItemConflict(
            f"repository {field_name} projection is not a sequence"
        )
    try:
        values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RepositoryWorkItemConflict(
            f"repository {field_name} projection is not a sequence"
        ) from exc
    return tuple(
        decoded
        for item in values
        if (decoded := _decode_opaque(item, field_name)) is not None
    )


def _target_policy_metadata(policy: RepositoryTargetPolicy) -> dict[str, Any]:
    value = policy.model_dump(mode="json", exclude_none=False)
    value["alias"] = _encode_opaque(policy.alias)
    value["capability_labels"] = _encode_opaque_sequence(policy.capability_labels)
    return value


def _target_policy_from_metadata(value: object) -> RepositoryTargetPolicy:
    raw = _nested_mapping(value)
    if raw.get("alias") is not None:
        raw["alias"] = _decode_opaque(raw["alias"], "target alias")
    raw["capability_labels"] = _decode_opaque_sequence(
        raw.get("capability_labels"), "capability labels"
    )
    return RepositoryTargetPolicy.model_validate(raw)


class RepositoryTargetPolicy(BaseModel):
    """Privacy-safe projection of one C-03 execution target policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    kind: Literal["local", "inventory_alias"] = "local"
    alias: str | None = None
    capability_labels: tuple[str, ...] = ()

    @field_validator("capability_labels", mode="before")
    @classmethod
    def normalize_labels(cls, value: object) -> tuple[str, ...]:
        return _sorted_opaque_sequence(value, "capability_labels")

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _nonblank(value, "alias")
        if any(token in value for token in ("/", "\\", "@", ":")):
            raise ValueError("target alias must not contain connection data")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> RepositoryTargetPolicy:
        if self.kind == "local" and self.alias is not None:
            raise ValueError("local target must not carry an alias")
        if self.kind == "inventory_alias" and self.alias is None:
            raise ValueError("inventory_alias target requires an alias")
        return self

    @classmethod
    def from_contract(cls, value: object) -> RepositoryTargetPolicy:
        raw = _nested_mapping(value)
        kind = str(raw.get("kind") or "local")
        if kind == "remote":
            kind = "inventory_alias"
        return cls(
            kind=kind,
            alias=raw.get("alias") or raw.get("target_alias"),
            capability_labels=raw.get("capability_labels") or (),
        )


class RepositoryConsentPolicy(BaseModel):
    """C-01 consent/risk projection retained with the durable WorkItem."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    allow_push: StrictBool = False
    allow_destructive_cleanup: StrictBool = False
    risk_acknowledged: StrictBool = False
    risk_marker: str | None = None

    @field_validator("risk_marker")
    @classmethod
    def validate_risk_marker(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "risk_marker")

    @model_validator(mode="after")
    def validate_risk(self) -> RepositoryConsentPolicy:
        if (self.allow_push or self.allow_destructive_cleanup) and not (
            self.risk_acknowledged and self.risk_marker
        ):
            raise ValueError(
                "push or destructive cleanup requires a risk acknowledgement and marker"
            )
        if self.risk_marker and not self.risk_acknowledged:
            raise ValueError("risk_marker cannot be supplied without acknowledgement")
        return self

    @classmethod
    def from_contract(cls, value: object) -> RepositoryConsentPolicy:
        return cls(**_nested_mapping(value))


def _consent_metadata(policy: RepositoryConsentPolicy) -> dict[str, Any]:
    value = policy.model_dump(mode="json", exclude_none=False)
    value["risk_marker"] = _encode_opaque(policy.risk_marker)
    return value


def _consent_from_metadata(value: object) -> RepositoryConsentPolicy:
    raw = _nested_mapping(value)
    if raw.get("risk_marker") is not None:
        raw["risk_marker"] = _decode_opaque(raw["risk_marker"], "risk marker")
    return RepositoryConsentPolicy.model_validate(raw)


def _canonical_json_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.astimezone(UTC).isoformat()
    return normalized[:-6] + "Z" if normalized.endswith("+00:00") else normalized


class RepositoryWorkItemRequest(BaseModel):
    """AU-side typed projection of a repository-development request.

    The shape is intentionally flat at the WorkItem boundary.  Repository
    Manager may carry richer nested contract records; only the immutable
    correlations, policy digests, and admission fields needed for durable
    orchestration cross this package boundary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
        revalidate_instances="always",
    )

    contract_version: Literal["1"] = CONTRACT_VERSION
    request_id: str
    idempotency_key: str
    operation: RepositoryOperation
    repository_id: str
    base_ref: str
    # RMDD-27 native reservation identity.  ``branch`` is distinct from
    # ``base_ref`` when a contract supplies it; a missing branch is represented
    # as ``base_ref`` only for non-exclusive jobs and is rejected by the native
    # authority for branch-exclusive admission.
    branch: str | None = None
    base_sha: str
    owner_id: str
    session_id: str
    tenant_id: str
    fairness_group: str = "default"
    dependencies: tuple[str, ...] = ()
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    resource_class: str = "light-check"
    concurrency_key: str = "light-check"
    profile_version: str | None = None
    resolved_profile_authority: (
        Literal["repository_manager:resource_profile_registry:v1"] | None
    ) = None
    concurrency_limit: StrictInt | None = Field(default=None, ge=1)
    repository_exclusive: StrictBool = False
    branch_exclusive: StrictBool = False
    disk_policy_key: str | None = None
    fairness_cost: StrictInt | None = Field(default=None, ge=1)
    cpu_weight: StrictInt = Field(default=1, ge=1, le=1000)
    memory_mib: StrictInt = Field(default=256, ge=1, le=1_048_576)
    disk_mib: StrictInt = Field(default=256, ge=1, le=10_485_760)
    process_slots: StrictInt = Field(default=1, ge=1, le=256)
    host_labels: tuple[str, ...] = ()
    preferred_target: RepositoryTargetPolicy = Field(
        default_factory=RepositoryTargetPolicy
    )
    required_target: RepositoryTargetPolicy | None = None
    anti_affinity: tuple[str, ...] = ()
    queue_deadline: datetime | None = None
    disk_low_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    disk_high_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    consent: RepositoryConsentPolicy = Field(default_factory=RepositoryConsentPolicy)
    lane_id: str | None = None
    candidate_id: str | None = None
    generation_id: str | None = None
    target_kind: Literal["local", "inventory_alias"] = "local"
    target_alias: str | None = None
    retry_class: str | None = None
    validation_stages: tuple[str, ...] = ()
    config_digest: str | None = None
    input_digest: str | None = None
    correlation_id: str | None = None
    # RMDD-28: the sole typed carrier for a development-lane allocation/cleanup
    # intent. Genuinely immutable -- this whole model is
    # ``ConfigDict(frozen=True)``, so any attempted post-construction mutation
    # (``request.lane_intent = ...``) raises, it is not merely a convention.
    # Never populated from ``consent``/``preferred_target``/``correlation_id``;
    # those remain ``extra="forbid"`` typed models of their own and cannot
    # carry an opaque ``lane_event`` payload.
    lane_intent: DevelopmentLaneIntent | None = None
    lane_cleanup_intent: DevelopmentLaneCleanupIntent | None = None
    # RMDD-29: operation-specific input is a second, independent additive
    # typed extension living beside RMDD-28's lane intent above -- neither
    # is projected into correlation_id, consent, target policy, or a generic
    # mapping field, and neither substitutes for the other. A lane.lifecycle/
    # lane.cleanup WorkItem carries lane_intent/lane_cleanup_intent; a build
    # WorkItem may separately carry operation_payload; both remain None on
    # every other operation kind.
    operation_payload: RepositoryOperationPayload | None = None

    @field_validator(
        "request_id",
        "idempotency_key",
        "repository_id",
        "base_ref",
        "owner_id",
        "session_id",
        "tenant_id",
        "fairness_group",
        "resource_class",
        "concurrency_key",
    )
    @classmethod
    def validate_strings(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("branch", "profile_version", "disk_policy_key")
    @classmethod
    def validate_optional_admission_strings(
        cls, value: str | None, info: Any
    ) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator(
        "dependencies",
        "validation_stages",
        "host_labels",
        "anti_affinity",
        mode="before",
    )
    @classmethod
    def normalize_sequences(cls, value: object, info: Any) -> tuple[str, ...]:
        if info.field_name in {"host_labels", "anti_affinity"}:
            return _sorted_opaque_sequence(value, info.field_name)
        return _opaque_sequence(value, info.field_name)

    @field_validator("base_sha")
    @classmethod
    def validate_sha(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(value):
            raise ValueError(
                "base_sha must be exactly 40 lowercase hexadecimal characters"
            )
        return value

    @field_validator("config_digest", "input_digest")
    @classmethod
    def validate_digests(cls, value: str | None, info: Any) -> str | None:
        return _validate_digest(value, info.field_name)

    @field_validator("target_alias")
    @classmethod
    def validate_alias(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _nonblank(value, "target_alias")
        if any(token in value for token in ("/", "\\", "@", ":")):
            raise ValueError(
                "target_alias must be an inventory alias, not connection data"
            )
        return value

    @field_validator(
        "lane_id", "candidate_id", "generation_id", "correlation_id", "retry_class"
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("queue_deadline")
    @classmethod
    def validate_queue_deadline(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("queue_deadline must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> RepositoryWorkItemRequest:
        if (
            self.operation != RepositoryOperation.BUILD
            and self.operation_payload is not None
        ):
            raise ValueError(
                "operation_payload discriminator does not match the operation"
            )
        if self.resolved_profile_authority is not None and (
            self.profile_version is None
            or self.disk_policy_key is None
            or self.fairness_cost is None
        ):
            raise ValueError(
                "resolved profile authority requires the complete resolved profile projection"
            )
        if self.target_kind == "local" and self.target_alias is not None:
            raise ValueError("local target must not carry target_alias")
        if self.target_kind == "inventory_alias" and self.target_alias is None:
            raise ValueError("inventory_alias target requires target_alias")
        if self.branch_exclusive and self.branch is None:
            raise ValueError("branch_exclusive requires an explicit branch")
        if (
            self.operation
            in {
                RepositoryOperation.RELEASE,
                RepositoryOperation.WORKSPACE_PUSH,
            }
            and not self.consent.allow_push
        ):
            raise ValueError("release or workspace push requires explicit push consent")
        if (
            self.required_target is not None
            and self.preferred_target.kind == "local"
            and self.required_target.kind == "inventory_alias"
            and self.preferred_target.alias is not None
        ):
            raise ValueError(
                "preferred and required targets cannot express conflicting forms"
            )
        low = self.disk_low_watermark_mib
        high = self.disk_high_watermark_mib
        if low is not None and high is not None and low > high:
            raise ValueError("disk low watermark must not exceed high watermark")
        if self.lane_intent is not None and self.lane_cleanup_intent is not None:
            raise ValueError(
                "lane_intent and lane_cleanup_intent are mutually exclusive"
            )
        if self.operation == RepositoryOperation.LANE_LIFECYCLE:
            if self.lane_intent is None:
                raise ValueError("lane.lifecycle requires a typed lane_intent")
        elif self.lane_intent is not None:
            raise ValueError(
                "lane_intent is only valid on a lane.lifecycle WorkItem request"
            )
        if self.operation == RepositoryOperation.LANE_CLEANUP:
            if self.lane_cleanup_intent is None:
                raise ValueError("lane.cleanup requires a typed lane_cleanup_intent")
        elif self.lane_cleanup_intent is not None:
            raise ValueError(
                "lane_cleanup_intent is only valid on a lane.cleanup WorkItem request"
            )
        return self

    def immutable_digest(self) -> str:
        """Digest the complete v1 request projection used for deduplication."""

        return _digest(self.model_dump(mode="json", exclude_none=False))

    @classmethod
    def from_contract(cls, contract: object) -> RepositoryWorkItemRequest:
        """Adapt RMDD's richer request model without importing Repository Manager."""

        raw = _as_mapping(contract)
        if raw.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
            raise RepositoryWorkItemError(
                "unsupported repository-development contract version"
            )
        repository = _nested_mapping(raw.get("repository"))
        resources = _nested_mapping(raw.get("resources"))
        target = _nested_mapping(raw.get("target"))
        validation = _nested_mapping(raw.get("validation_policy"))
        consent = RepositoryConsentPolicy.from_contract(raw.get("consent"))
        preferred_target = RepositoryTargetPolicy.from_contract(
            resources.get("preferred_target")
        )
        required_target_raw = resources.get("required_target")
        required_target = (
            RepositoryTargetPolicy.from_contract(required_target_raw)
            if required_target_raw is not None
            else None
        )
        operation = raw.get("operation")
        if isinstance(operation, StrEnum):
            operation = operation.value
        target_kind = str(target.get("kind") or "local")
        if target_kind == "remote":
            target_kind = "inventory_alias"
        input_digest = raw.get("input_digest")
        if input_digest is None:
            input_digest = _digest(raw)
        return cls(
            request_id=raw.get("request_id") or raw.get("id"),
            idempotency_key=raw.get("idempotency_key"),
            operation=operation,
            repository_id=repository.get("repository_id") or raw.get("repository_id"),
            base_ref=raw.get("base_ref"),
            branch=raw.get("branch"),
            base_sha=raw.get("base_sha"),
            owner_id=raw.get("owner_id"),
            session_id=raw.get("session_id"),
            tenant_id=raw.get("tenant_id") or raw.get("tenant"),
            fairness_group=raw.get("fairness_group")
            or resources.get("fairness_group")
            or "default",
            dependencies=raw.get("dependencies") or raw.get("depends_on") or (),
            priority=raw.get("priority", resources.get("priority", 0)),
            resource_class=resources.get("resource_class")
            or raw.get("resource_class")
            or "light-check",
            concurrency_key=resources.get("concurrency_key")
            or raw.get("concurrency_key")
            or "light-check",
            profile_version=resources.get("profile_version")
            or raw.get("profile_version"),
            resolved_profile_authority=resources.get("resolved_profile_authority")
            or raw.get("resolved_profile_authority"),
            concurrency_limit=resources.get("concurrency_limit")
            if resources.get("concurrency_limit") is not None
            else raw.get("concurrency_limit"),
            repository_exclusive=resources.get(
                "repository_exclusive", raw.get("repository_exclusive", False)
            ),
            branch_exclusive=resources.get(
                "branch_exclusive", raw.get("branch_exclusive", False)
            ),
            disk_policy_key=resources.get("disk_policy_key")
            or raw.get("disk_policy_key"),
            fairness_cost=resources.get("fairness_cost")
            if resources.get("fairness_cost") is not None
            else raw.get("fairness_cost"),
            cpu_weight=resources.get("cpu_weight", 1),
            memory_mib=resources.get("memory_mib", 256),
            disk_mib=resources.get("disk_mib", 256),
            process_slots=resources.get("process_slots", 1),
            host_labels=resources.get("host_labels") or (),
            preferred_target=preferred_target,
            required_target=required_target,
            anti_affinity=resources.get("anti_affinity") or (),
            queue_deadline=resources.get("queue_deadline"),
            disk_low_watermark_mib=resources.get("disk_low_watermark_mib"),
            disk_high_watermark_mib=resources.get("disk_high_watermark_mib"),
            consent=consent,
            lane_id=raw.get("lane_id"),
            candidate_id=raw.get("candidate_id"),
            generation_id=raw.get("generation_id"),
            target_kind=target_kind,
            target_alias=target.get("alias") or target.get("target_alias"),
            retry_class=raw.get("retry_class"),
            validation_stages=validation.get("stages") or (),
            config_digest=raw.get("config_digest"),
            input_digest=input_digest,
            correlation_id=raw.get("correlation_id"),
            operation_payload=raw.get("operation_payload"),
            lane_intent=raw.get("lane_intent"),
            lane_cleanup_intent=raw.get("lane_cleanup_intent"),
        )


class RepositoryLease(BaseModel):
    """Typed lease/fence projection returned to Repository Manager workers."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    owner: str
    epoch: int = Field(ge=0)
    fencing_token: int = Field(ge=0)
    attempt: int = Field(ge=1)
    heartbeat_at: float | None = None
    expires_at: float | None = None


class RepositoryWorkItemView(BaseModel):
    """Tenant-scoped read-only view of a durable repository WorkItem."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    contract_version: Literal["1"] = CONTRACT_VERSION
    job_id: str
    work_item_id: str
    request_id: str
    operation: RepositoryOperation
    kind: RepositoryWorkItemKind
    state: RepositoryJobState
    repository_id: str
    tenant_id: str
    owner_id: str
    session_id: str
    base_ref: str
    base_sha: str
    target_kind: str
    target_alias: str | None = None
    lane_id: str | None = None
    candidate_id: str | None = None
    generation_id: str | None = None
    dependencies: tuple[str, ...] = ()
    input_digest: str
    config_digest: str | None = None
    correlation_id: str | None = None
    operation_payload_kind: str | None = None
    operation_payload_version: str | None = None
    operation_payload_digest: str | None = None
    resource_class: str = "light-check"
    concurrency_key: str = "light-check"
    fairness_group: str = "default"
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    cpu_weight: StrictInt = Field(default=1, ge=1)
    memory_mib: StrictInt = Field(default=256, ge=1)
    disk_mib: StrictInt = Field(default=256, ge=1)
    process_slots: StrictInt = Field(default=1, ge=1)
    host_labels: tuple[str, ...] = ()
    preferred_target: RepositoryTargetPolicy = Field(
        default_factory=RepositoryTargetPolicy
    )
    required_target: RepositoryTargetPolicy | None = None
    anti_affinity: tuple[str, ...] = ()
    queue_deadline: datetime | None = None
    disk_low_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    disk_high_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    consent: RepositoryConsentPolicy = Field(default_factory=RepositoryConsentPolicy)
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    checkpoint: str | None = None
    retry_class: str | None = None
    result_ref: str | None = None
    error_ref: str | None = None
    lease: RepositoryLease | None = None


class RepositoryWorkItemResult(BaseModel):
    """Typed terminal/result projection consumable by RMDD adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    contract_version: Literal["1"] = CONTRACT_VERSION
    job_id: str
    work_item_id: str
    request_id: str
    operation: RepositoryOperation
    state: RepositoryJobState
    repository_id: str
    tenant_id: str
    operation_payload_kind: str | None = None
    operation_payload_version: str | None = None
    operation_payload_digest: str | None = None
    target_kind: str = "local"
    target_alias: str | None = None
    lane_id: str | None = None
    candidate_id: str | None = None
    generation_id: str | None = None
    input_digest: str
    config_digest: str | None = None
    resource_class: str = "light-check"
    concurrency_key: str = "light-check"
    fairness_group: str = "default"
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    cpu_weight: StrictInt = Field(default=1, ge=1)
    memory_mib: StrictInt = Field(default=256, ge=1)
    disk_mib: StrictInt = Field(default=256, ge=1)
    process_slots: StrictInt = Field(default=1, ge=1)
    host_labels: tuple[str, ...] = ()
    preferred_target: RepositoryTargetPolicy = Field(
        default_factory=RepositoryTargetPolicy
    )
    required_target: RepositoryTargetPolicy | None = None
    anti_affinity: tuple[str, ...] = ()
    queue_deadline: datetime | None = None
    disk_low_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    disk_high_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    consent: RepositoryConsentPolicy = Field(default_factory=RepositoryConsentPolicy)
    attempt: int = Field(ge=0)
    checkpoint: str | None = None
    retry_class: str | None = None
    result_ref: str | None = None
    error_ref: str | None = None
    failure_class: str | None = None
    refusal_code: str | None = None

    @model_validator(mode="after")
    def validate_failure_or_refusal(self) -> RepositoryWorkItemResult:
        if self.state in {
            RepositoryJobState.SUBMITTED,
            RepositoryJobState.READY,
            RepositoryJobState.LEASED,
            RepositoryJobState.RUNNING,
        }:
            raise ValueError("repository result state must be terminal")
        if self.failure_class is not None and self.refusal_code is not None:
            raise ValueError(
                "a repository result cannot carry both failure_class and refusal_code"
            )
        return self


class RepositoryWorkItemHandle(BaseModel):
    """Submission response; the handle is stable across process restarts."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    contract_version: Literal["1"] = CONTRACT_VERSION
    job_id: str
    work_item_id: str
    request_id: str
    state: RepositoryJobState
    input_digest: str
    deduplicated: bool = False


def repository_job_id(scope: str, idempotency_key: str) -> str:
    """Derive a stable opaque ``rmjob:<uuid>`` from the authenticated scope."""

    scope = _nonblank(scope, "scope")
    idempotency_key = _nonblank(idempotency_key, "idempotency_key")
    value = uuid.uuid5(_IDEMPOTENCY_NAMESPACE, f"{scope}\0{idempotency_key}")
    return f"rmjob:{value}"


def repository_work_item_id(job_id: str) -> str:
    """Convert a full public job handle to its full durable WorkItem ID."""

    match = _JOB_ID_RE.fullmatch(job_id)
    if not match:
        raise RepositoryWorkItemError("job_id must use the full rmjob:<uuid> form")
    return f"workitem:repository_manager:{match.group('uuid').lower()}"


def _job_id_from_identifier(identifier: str) -> str:
    if _JOB_ID_RE.fullmatch(identifier):
        return f"rmjob:{_JOB_ID_RE.fullmatch(identifier).group('uuid').lower()}"  # type: ignore[union-attr]
    match = _WORK_ITEM_ID_RE.fullmatch(identifier)
    if match:
        return f"rmjob:{match.group('uuid').lower()}"
    raise RepositoryWorkItemError(
        "repository identifier must be a full rmjob or WorkItem ID"
    )


def repository_work_item_kind(
    operation: RepositoryOperation | str,
) -> RepositoryWorkItemKind:
    """Return the one registered WorkItem kind for an operation family."""

    try:
        operation = RepositoryOperation(operation)
    except ValueError as exc:
        raise RepositoryWorkItemError(
            f"unknown repository operation: {operation!r}"
        ) from exc
    return _OPERATION_TO_KIND[operation]


def _operation_value(operation: RepositoryOperation | str) -> str:
    return RepositoryOperation(operation).value


def _native_priority_bucket(priority: int) -> int:
    """Project RMDD's 0..10,000 priority onto WorkItem's 0..3 buckets.

    The complete strict request priority remains in the immutable repository
    extension record and view.  The generic WorkItem queue has four discrete
    buckets, so admission clamps the request to the nearest representable
    urgency without silently changing the repository contract projection.
    """

    return min(3, max(0, priority))


def _work_item_dependency_id(identifier: str) -> str:
    if _WORK_ITEM_ID_RE.fullmatch(identifier):
        return identifier.lower()
    if _JOB_ID_RE.fullmatch(identifier):
        return repository_work_item_id(identifier)
    raise RepositoryWorkItemError(
        "repository dependencies must use full rmjob:<uuid> or "
        "workitem:repository_manager:<uuid> IDs"
    )


def _scoped_idempotency_key(request: RepositoryWorkItemRequest) -> str:
    return "repository:" + _digest(
        {"tenant_id": request.tenant_id, "idempotency_key": request.idempotency_key}
    )


def _request_metadata(
    request: RepositoryWorkItemRequest,
    *,
    job_id: str,
    input_digest: str,
    dependencies: Sequence[str],
    resolved_profile_projection: bool = False,
) -> dict[str, Any]:
    """Build the bounded, privacy-safe WorkItem extension record."""

    operation_payload = request.operation_payload
    serialized_payload = (
        operation_payload.model_dump(mode="json", exclude_none=False)
        if operation_payload is not None
        else None
    )
    if serialized_payload is not None:
        assert operation_payload is not None
        computed_payload_digest = payload_digest(serialized_payload)
        if operation_payload.payload_digest != computed_payload_digest:
            raise RepositoryWorkItemConflict(
                "input_conflict: operation payload digest does not match its body"
            )
        encoded_size = len(
            json.dumps(
                serialized_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if encoded_size > MAX_OPERATION_PAYLOAD_BYTES:
            raise RepositoryWorkItemError("operation payload exceeds its durable bound")
        clean_payload, privacy_report = PersistencePrivacyGuard().sanitize(
            serialized_payload
        )
        if privacy_report.changed or clean_payload != serialized_payload:
            raise RepositoryWorkItemError(
                "operation payload fails persistence privacy validation"
            )

    resource_reservation = {
        "schema_version": "1",
        "profile_name": _encode_opaque(request.resource_class),
        "profile_version": _encode_opaque(request.profile_version),
        "cpu_weight": request.cpu_weight,
        "memory_mib": request.memory_mib,
        "disk_mib": request.disk_mib,
        "process_slots": request.process_slots,
        "host_labels": _encode_opaque_sequence(request.host_labels),
        "anti_affinity": _encode_opaque_sequence(request.anti_affinity),
        "preferred_target": _target_policy_metadata(request.preferred_target),
        "required_target": (
            _target_policy_metadata(request.required_target)
            if request.required_target is not None
            else None
        ),
        "repository_id": _encode_opaque(request.repository_id),
        "concurrency_key": _encode_opaque(request.concurrency_key),
        "fairness_group": _encode_opaque(request.fairness_group),
        "disk_low_watermark_mib": request.disk_low_watermark_mib,
        "disk_high_watermark_mib": request.disk_high_watermark_mib,
        # WorkItem admission identity and the later reservation input
        # fingerprint are deliberately separate.  The former is known at
        # submission; native reservation recomputes the latter from the
        # fenced attempt/host/TTL request.
        "work_item_input_fingerprint": "v1:" + request.immutable_digest(),
        **(
            {"resolved_profile_authority": request.resolved_profile_authority}
            if resolved_profile_projection
            else {}
        ),
        "concurrency_limit": request.concurrency_limit,
        "repository_exclusive": request.repository_exclusive,
        "branch_exclusive": request.branch_exclusive,
        "disk_policy_key": _encode_opaque(request.disk_policy_key),
        "fairness_cost": request.fairness_cost,
        "branch": _encode_opaque(request.branch or request.base_ref),
        "branch_explicit": request.branch is not None,
        "base_ref": _encode_opaque(request.base_ref),
        "target_kind": request.target_kind,
        "target_alias": _encode_opaque(request.target_alias),
    }
    extension_metadata: dict[str, Any] = {
        "resource_reservation": resource_reservation,
    }
    if serialized_payload is not None:
        assert operation_payload is not None
        extension_metadata.update(
            {
                "operation_payload": serialized_payload,
                "operation_payload_digest": operation_payload.payload_digest,
            }
        )

    return {
        _METADATA_KEY: {
            "contract_version": CONTRACT_VERSION,
            "job_id": _encode_opaque(job_id),
            "request_id": _encode_opaque(request.request_id),
            "tenant_id": _encode_opaque(request.tenant_id),
            "idempotency_scope_digest": _digest(request.tenant_id),
            "immutable_input_digest": input_digest,
            "source_input_digest": request.input_digest,
            "operation": _operation_value(request.operation),
            **extension_metadata,
            "repository_id": _encode_opaque(request.repository_id),
            "base_ref": _encode_opaque(request.base_ref),
            "branch": _encode_opaque(request.branch),
            "base_sha": request.base_sha,
            "owner_id": _encode_opaque(request.owner_id),
            "session_id": _encode_opaque(request.session_id),
            "resource_class": _encode_opaque(request.resource_class),
            "concurrency_key": _encode_opaque(request.concurrency_key),
            "fairness_group": _encode_opaque(request.fairness_group),
            "priority": request.priority,
            "cpu_weight": request.cpu_weight,
            "memory_mib": request.memory_mib,
            "disk_mib": request.disk_mib,
            "process_slots": request.process_slots,
            "host_labels": _encode_opaque_sequence(request.host_labels),
            "preferred_target": _target_policy_metadata(request.preferred_target),
            "required_target": (
                _target_policy_metadata(request.required_target)
                if request.required_target is not None
                else None
            ),
            "anti_affinity": _encode_opaque_sequence(request.anti_affinity),
            "queue_deadline": (_canonical_json_datetime(request.queue_deadline)),
            "disk_low_watermark_mib": request.disk_low_watermark_mib,
            "disk_high_watermark_mib": request.disk_high_watermark_mib,
            "target_kind": request.target_kind,
            "target_alias": _encode_opaque(request.target_alias),
            "lane_id": _encode_opaque(request.lane_id),
            "candidate_id": _encode_opaque(request.candidate_id),
            "generation_id": _encode_opaque(request.generation_id),
            "retry_class": _encode_opaque(request.retry_class),
            "config_digest": request.config_digest,
            "validation_stages": _encode_opaque_sequence(request.validation_stages),
            "dependencies": _encode_opaque_sequence(dependencies),
            "consent": _consent_metadata(request.consent),
        }
    }


def _native_admission_projection(
    request: RepositoryWorkItemRequest, *, now: float | None
) -> dict[str, Any]:
    """Map explicit C-01 consent to the native WorkItem gate."""

    external_push = request.operation in {
        RepositoryOperation.RELEASE,
        RepositoryOperation.WORKSPACE_PUSH,
    }
    destructive_repair = (
        request.operation == RepositoryOperation.REPAIR
        and request.consent.allow_destructive_cleanup
    )
    if not (external_push or destructive_repair):
        return {
            "consent_required": False,
            "consent_scope": "",
            "consent_subject": "",
            "consent_basis": "",
            "consent_granted_at": None,
            "consent_expires_at": None,
        }
    admission_time = time.time() if now is None else float(now)
    return {
        "consent_required": True,
        "consent_scope": f"repository:{_operation_value(request.operation)}",
        "consent_subject": request.repository_id,
        "consent_basis": request.consent.risk_marker or "risk-acknowledged",
        "consent_granted_at": admission_time,
        "consent_expires_at": None,
    }


def _metadata_record(
    row: Mapping[str, Any], *, include_payload: bool = True
) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RepositoryWorkItemConflict("repository WorkItem metadata is missing")
    record = metadata.get(_METADATA_KEY)
    if (
        not isinstance(record, Mapping)
        or record.get("contract_version") != CONTRACT_VERSION
    ):
        raise RepositoryWorkItemConflict(
            "WorkItem is not a repository-development v1 record"
        )
    result = dict(record)
    # The generic persistence privacy guard can conservatively recognize a
    # UUID's all-numeric groups as a credit-card-shaped substring and redact
    # part of ``job_id``.  The durable WorkItem ID is the canonical identity,
    # so repair only that known sanitizer marker rather than allowing an
    # otherwise valid submission to become unreadable after persistence.
    item_id = str(row.get("id") or "")
    match = _WORK_ITEM_ID_RE.fullmatch(item_id)
    stored_job_id = result.get("job_id")
    if (
        match is not None
        and isinstance(stored_job_id, str)
        and "[REDACTED_" in stored_job_id
    ):
        result["job_id"] = f"rmjob:{match.group('uuid').lower()}"
    for field in (
        "job_id",
        "request_id",
        "repository_id",
        "base_ref",
        "owner_id",
        "session_id",
        "resource_class",
        "concurrency_key",
        "fairness_group",
        "target_alias",
        "lane_id",
        "candidate_id",
        "generation_id",
        "retry_class",
    ):
        if field in result:
            result[field] = _decode_opaque(result[field], field)
    for field in ("host_labels", "anti_affinity", "validation_stages", "dependencies"):
        if field in result:
            result[field] = _decode_opaque_sequence(result[field], field)
    raw_payload = result.get("operation_payload")
    stored_payload_digest = result.get("operation_payload_digest")
    if raw_payload is not None:
        if not include_payload:
            # Ordinary projections may carry only the closed discriminator
            # summary.  Never parse or retain the executable body on this
            # path; exact input requires the native capability seam below.
            if isinstance(raw_payload, Mapping):
                result["operation_payload"] = {
                    "kind": raw_payload.get("kind"),
                    "schema_version": raw_payload.get("schema_version"),
                }
            else:
                result["operation_payload"] = None
            return result
        try:
            typed_payload = operation_payload_from_mapping(raw_payload)
        except (TypeError, ValueError) as exc:
            raise RepositoryWorkItemConflict(
                "input_conflict: repository WorkItem operation payload is invalid"
            ) from exc
        if result.get("operation") != RepositoryOperation.BUILD.value:
            raise RepositoryWorkItemConflict(
                "input_conflict: operation payload discriminator does not match operation"
            )
        computed_payload_digest = payload_digest(typed_payload)
        raw_payload_digest = (
            raw_payload.get("payload_digest")
            if isinstance(raw_payload, Mapping)
            else None
        )
        if (
            stored_payload_digest != computed_payload_digest
            or raw_payload_digest != computed_payload_digest
            or typed_payload.payload_digest != computed_payload_digest
        ):
            raise RepositoryWorkItemConflict(
                "input_conflict: repository WorkItem operation payload digest mismatch"
            )
        result["operation_payload"] = typed_payload.model_dump(
            mode="json", exclude_none=False
        )
        result["operation_payload_digest"] = computed_payload_digest
    elif stored_payload_digest is not None:
        raise RepositoryWorkItemConflict(
            "input_conflict: operation payload digest has no body"
        )
    return result


def _assert_idempotent(
    row: Mapping[str, Any],
    request: RepositoryWorkItemRequest,
    *,
    job_id: str,
    input_digest: str,
) -> None:
    expected_kind = repository_work_item_kind(request.operation).value
    record = _metadata_record(row)
    if row.get("id") != repository_work_item_id(job_id):
        raise RepositoryWorkItemConflict(
            "repository WorkItem identity does not match job handle"
        )
    if row.get("tenant") != request.tenant_id:
        raise RepositoryWorkItemConflict("idempotency key is scoped to another tenant")
    if row.get("kind") != expected_kind or record.get("job_id") != job_id:
        raise RepositoryWorkItemConflict(
            "idempotency key is bound to another operation"
        )
    if record.get("immutable_input_digest") != input_digest:
        raise RepositoryWorkItemConflict(
            "input_conflict: idempotency key was reused with changed immutable repository input"
        )


def _assert_repository_dependencies_same_tenant(
    engine: Any, dependency_ids: Sequence[str], *, tenant: str
) -> None:
    """Reject missing or cross-tenant repository prerequisites before admission."""

    for dependency_id in dependency_ids:
        dependency = get_work_item(engine, dependency_id)
        if dependency is None or dependency.get("tenant") != tenant:
            raise RepositoryWorkItemError(
                "repository dependency is missing or outside the authenticated tenant"
            )


def submit_repository_work_item(
    engine: Any,
    request: RepositoryWorkItemRequest | Mapping[str, Any] | BaseModel,
    *,
    job_id: str | None = None,
    now: float | None = None,
    max_attempts: int = 3,
    resolved_profile_projection: bool = False,
) -> RepositoryWorkItemHandle:
    """Atomically submit or deduplicate one repository-development WorkItem.

    The durable identity is derived from authenticated tenant scope plus the
    client idempotency key.  The engine's ``create_node_if_absent`` operation
    arbitrates concurrent first writers; a winner with a different immutable
    request digest is rejected rather than overwritten.
    """

    raw_request = (
        dict(vars(request))
        if isinstance(request, RepositoryWorkItemRequest)
        else _as_mapping(request)
    )
    raw_resources = _nested_mapping(raw_request.get("resources"))
    supplied_authority = raw_resources.get(
        "resolved_profile_authority"
    ) or raw_request.get("resolved_profile_authority")
    if resolved_profile_projection:
        if supplied_authority != "repository_manager:resource_profile_registry:v1":
            raise RepositoryWorkItemError(
                "resolved_profile_projection requires the trusted profile authority marker"
            )
        authority_source = {**raw_request, **raw_resources}
        required_authority_fields = (
            "resource_class",
            "concurrency_key",
            "profile_version",
            "concurrency_limit",
            "repository_exclusive",
            "branch_exclusive",
            "disk_policy_key",
            "fairness_cost",
            "cpu_weight",
            "memory_mib",
            "disk_mib",
            "process_slots",
            "host_labels",
            "anti_affinity",
            "preferred_target",
            "required_target",
            "disk_low_watermark_mib",
            "disk_high_watermark_mib",
            "fairness_group",
        )
        # Nullable policy values (concurrency_limit and disk watermarks) are
        # deliberately checked for presence only: a trusted profile may
        # explicitly resolve them to None.  Identity, profile, dimensions,
        # and cost fields must carry a concrete value before the Pydantic model
        # validator runs, otherwise the boundary would leak ValidationError
        # instead of the stable repository-authority error vocabulary.
        required_non_null_fields = {
            "resource_class",
            "concurrency_key",
            "profile_version",
            "disk_policy_key",
            "fairness_cost",
            "cpu_weight",
            "memory_mib",
            "disk_mib",
            "process_slots",
            "preferred_target",
            "fairness_group",
        }
        missing_authority_fields = [
            field
            for field in required_authority_fields
            if field not in authority_source
            or (field in required_non_null_fields and authority_source[field] is None)
        ]
        if missing_authority_fields:
            raise RepositoryWorkItemError(
                "trusted resolved profile projection is incomplete: "
                + ", ".join(missing_authority_fields)
            )
    elif supplied_authority is not None:
        raise RepositoryWorkItemError(
            "resolved profile authority is reserved for the trusted RM projection path"
        )
    try:
        typed_request = (
            RepositoryWorkItemRequest.model_validate(request)
            if isinstance(request, RepositoryWorkItemRequest)
            else RepositoryWorkItemRequest.from_contract(request)
        )
    except ValidationError as exc:
        raise RepositoryWorkItemError(
            "repository request is invalid at the WorkItem authority boundary"
        ) from exc
    if typed_request.operation == RepositoryOperation.BUILD and (
        typed_request.operation_payload is not None
    ):
        if typed_request.operation_payload.repository_id != typed_request.repository_id:
            raise RepositoryWorkItemError(
                "operation payload repository identity disagrees with WorkItem"
            )
        if typed_request.operation_payload.base_sha != typed_request.base_sha:
            raise RepositoryWorkItemError(
                "operation payload base SHA disagrees with WorkItem"
            )
    if resolved_profile_projection and typed_request.resolved_profile_authority is None:
        raise RepositoryWorkItemError(
            "trusted resolved profile projection was not preserved by contract adaptation"
        )
    _require_tenant_for_engine(engine, typed_request.tenant_id)
    derived_job_id = repository_job_id(
        typed_request.tenant_id, typed_request.idempotency_key
    )
    if job_id is not None and job_id != derived_job_id:
        raise RepositoryWorkItemConflict(
            "job_id must be the deterministic handle for tenant scope and idempotency key"
        )
    job_id = derived_job_id
    item_id = repository_work_item_id(job_id)
    input_digest = typed_request.immutable_digest()
    existing = get_work_item(engine, item_id)
    if existing is not None:
        _assert_idempotent(
            existing, typed_request, job_id=job_id, input_digest=input_digest
        )
        view = _view_from_row(existing)
        return RepositoryWorkItemHandle(
            job_id=job_id,
            work_item_id=item_id,
            request_id=typed_request.request_id,
            state=view.state,
            input_digest=input_digest,
            deduplicated=True,
        )

    dependencies = tuple(
        sorted(
            {_work_item_dependency_id(value) for value in typed_request.dependencies}
        )
    )
    _assert_repository_dependencies_same_tenant(
        engine, dependencies, tenant=typed_request.tenant_id
    )
    kind = repository_work_item_kind(typed_request.operation)
    _, created = submit_work_item_atomic(
        engine,
        kind=kind.value,
        queue=kind.value,
        payload_ref=job_id,
        tenant=typed_request.tenant_id,
        depends_on=dependencies,
        priority=_native_priority_bucket(typed_request.priority),
        deadline_unix=(
            typed_request.queue_deadline.timestamp()
            if typed_request.queue_deadline is not None
            else None
        ),
        resource_class=typed_request.resource_class,
        fairness_group=typed_request.fairness_group,
        max_attempts=max_attempts,
        idempotency_key=_scoped_idempotency_key(typed_request),
        correlation_id=typed_request.correlation_id or typed_request.request_id,
        description=f"repository operation: {_operation_value(typed_request.operation)}",
        created_by=typed_request.owner_id,
        metadata=_request_metadata(
            typed_request,
            job_id=job_id,
            input_digest=input_digest,
            dependencies=dependencies,
            resolved_profile_projection=resolved_profile_projection,
        ),
        work_item_id=item_id,
        now=now,
        **_native_admission_projection(typed_request, now=now),
    )
    stored = get_work_item(engine, item_id)
    if stored is None:
        raise WorkItemBackendUnavailable(
            "repository WorkItem disappeared after submission"
        )
    _assert_idempotent(stored, typed_request, job_id=job_id, input_digest=input_digest)
    view = _view_from_row(stored)
    return RepositoryWorkItemHandle(
        job_id=job_id,
        work_item_id=item_id,
        request_id=typed_request.request_id,
        state=view.state,
        input_digest=input_digest,
        deduplicated=not created,
    )


def _state_value(value: object) -> str:
    state = str(value or "")
    return "dead-letter" if state == "dead_letter" else state


def _view_from_row(row: Mapping[str, Any]) -> RepositoryWorkItemView:
    record = _metadata_record(row, include_payload=False)
    operation_payload = record.get("operation_payload")
    payload_kind = (
        str(operation_payload.get("kind"))
        if isinstance(operation_payload, Mapping)
        else None
    )
    payload_version = (
        str(operation_payload.get("schema_version"))
        if isinstance(operation_payload, Mapping)
        else None
    )
    lease = None
    if row.get("lease_owner"):
        lease = RepositoryLease(
            owner=str(row["lease_owner"]),
            epoch=int(row.get("lease_epoch") or 0),
            fencing_token=int(row.get("fencing_token") or 0),
            attempt=max(1, int(row.get("attempt") or 0)),
            heartbeat_at=(
                float(row["heartbeat_at"])
                if row.get("heartbeat_at") is not None
                else None
            ),
            expires_at=(
                float(row["lease_expires_at"])
                if row.get("lease_expires_at") is not None
                else None
            ),
        )
    return RepositoryWorkItemView(
        job_id=record["job_id"],
        work_item_id=str(row["id"]),
        request_id=record["request_id"],
        operation=record["operation"],
        kind=row["kind"],
        state=_state_value(row.get("status")),
        repository_id=record["repository_id"],
        tenant_id=str(row.get("tenant") or ""),
        owner_id=record["owner_id"],
        session_id=record["session_id"],
        base_ref=_decode_opaque(record["base_ref"], "base_ref") or "",
        base_sha=record["base_sha"],
        target_kind=record["target_kind"],
        target_alias=_decode_opaque(record.get("target_alias"), "target alias"),
        lane_id=record.get("lane_id"),
        candidate_id=record.get("candidate_id"),
        generation_id=record.get("generation_id"),
        resource_class=str(
            record.get("resource_class") or row.get("resource_class") or "light-check"
        ),
        concurrency_key=str(
            record.get("concurrency_key")
            or record.get("resource_class")
            or row.get("resource_class")
            or "light-check"
        ),
        fairness_group=str(
            record.get("fairness_group") or row.get("fairness_group") or "default"
        ),
        priority=int(record.get("priority") or row.get("prio_bucket") or 0),
        cpu_weight=int(record.get("cpu_weight") or 1),
        memory_mib=int(record.get("memory_mib") or 256),
        disk_mib=int(record.get("disk_mib") or 256),
        process_slots=int(record.get("process_slots") or 1),
        host_labels=tuple(record.get("host_labels") or ()),
        preferred_target=_target_policy_from_metadata(
            record.get("preferred_target") or {}
        ),
        required_target=(
            _target_policy_from_metadata(record["required_target"])
            if record.get("required_target") is not None
            else None
        ),
        anti_affinity=tuple(record.get("anti_affinity") or ()),
        queue_deadline=(
            datetime.fromisoformat(str(record["queue_deadline"]))
            if record.get("queue_deadline")
            else None
        ),
        disk_low_watermark_mib=record.get("disk_low_watermark_mib"),
        disk_high_watermark_mib=record.get("disk_high_watermark_mib"),
        consent=_consent_from_metadata(record.get("consent") or {}),
        retry_class=record.get("retry_class"),
        dependencies=tuple(row.get("depends_on") or ()),
        input_digest=record["immutable_input_digest"],
        config_digest=record.get("config_digest"),
        correlation_id=row.get("correlation_id"),
        operation_payload_kind=payload_kind,
        operation_payload_version=payload_version,
        operation_payload_digest=record.get("operation_payload_digest"),
        attempt=int(row.get("attempt") or 0),
        max_attempts=max(1, int(row.get("max_attempts") or 1)),
        checkpoint=row.get("checkpoint_id"),
        result_ref=row.get("result_ref"),
        error_ref=row.get("error_ref"),
        lease=lease,
    )


def _require_tenant(tenant: str) -> str:
    return _nonblank(tenant, "tenant")


def _bound_authority_tenant(engine: Any) -> str | None:
    """Return a tenant already bound by the native authority, when exposed.

    The repository adapter receives a tenant from its authenticated boundary,
    but a host may also expose a tenant-bound WorkItem authority.  In that
    case the caller's value is only a claim and must agree with the bound
    authority; it must never widen a query or mutation to another tenant.
    Lightweight test authorities commonly expose ``tenant`` directly, while
    graph clients expose the verified value through ``_verified_tenant``.
    """

    authority = _work_item._authority(engine)
    for name in ("bound_tenant", "tenant", "tenant_id", "tenant_ref"):
        value = getattr(authority, name, None)
        if isinstance(value, str) and value.strip():
            return _require_tenant(value)
    verified = getattr(authority, "_verified_tenant", None)
    if callable(verified):
        value = verified()
        if isinstance(value, str) and value.strip():
            return _require_tenant(value)
    return None


def _require_tenant_for_engine(engine: Any, tenant: str) -> str:
    """Require the supplied tenant to match any already-bound authority."""

    requested = _require_tenant(tenant)
    bound = _bound_authority_tenant(engine)
    if bound is not None and bound != requested:
        raise RepositoryWorkItemError(
            "requested tenant does not match the authenticated WorkItem authority"
        )
    return requested


def get_repository_work_item(
    engine: Any,
    identifier: str,
    *,
    tenant: str,
    owner_id: str | None = None,
) -> RepositoryWorkItemView | None:
    """Read one repository job only inside the authenticated tenant scope."""

    tenant = _require_tenant_for_engine(engine, tenant)
    job_id = _job_id_from_identifier(identifier)
    row = get_work_item(engine, repository_work_item_id(job_id))
    if row is None or row.get("tenant") != tenant:
        return None
    record = _metadata_record(row, include_payload=False)
    if owner_id is not None and record.get("owner_id") != _nonblank(
        owner_id, "owner_id"
    ):
        return None
    return _view_from_row(row)


def get_repository_operation_payload(
    engine: Any,
    identifier: str,
    *,
    tenant: str,
    owner_id: str,
) -> RepositoryBuildExecutionPayloadV1 | None:
    """Fail closed until EG supplies one atomic native exact-input operation.

    The legacy owner-shaped signature is retained only as a compatibility
    boundary.  It never inspects the engine, owner string, or public metadata;
    native authority integration will replace this path after the EG schema
    freeze.
    """

    del engine, identifier, tenant, owner_id
    raise RepositoryWorkItemError(TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE)


def get_repository_operation_payload_for_claim(
    engine: Any,
    identifier: str,
    *,
    tenant: str,
    claim: Mapping[str, Any],
) -> RepositoryBuildExecutionPayloadV1 | None:
    """Reject the retired public tuple claim path before any row read.

    Fencing tuples are mutation CAS evidence, not authentication.  Native
    worker callers must use the future EG-native atomic exact-input operation.
    """

    del engine, identifier, tenant, claim
    raise RepositoryWorkItemError(TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE)


_T = TypeVar("_T", bound=str)


def _repository_rows(
    engine: Any,
    *,
    tenant: str,
    limit: int,
    kinds: Sequence[str] = _REPOSITORY_KINDS,
    statuses: Sequence[str] = (),
    cursor: tuple[float, str] | None = None,
) -> list[dict[str, Any]]:
    """Read one bounded, keyset-paginated tenant-scoped repository page.

    ``cursor`` is the last ``(created_at, id)`` pair returned.  The explicit
    tie-breaker keeps rows with equal timestamps from being skipped or repeated
    when a host receives many submissions in one clock tick.
    """

    fields = ["w.id AS id"] + [f"w.{field} AS {field}" for field in _work_item_fields()]
    where = ["w.kind IN $kinds", "w.tenant = $tenant"]
    params: dict[str, Any] = {
        "kinds": list(kinds),
        "tenant": tenant,
        "limit": int(limit),
    }
    if statuses:
        where.append("w.status IN $statuses")
        params["statuses"] = list(statuses)
    if cursor is not None:
        where.append(
            "(w.created_at > $cursor_created_at OR "
            "(w.created_at = $cursor_created_at AND w.id > $cursor_id))"
        )
        params["cursor_created_at"], params["cursor_id"] = cursor
    rows = _work_item._authority(engine).query_cypher(
        "MATCH (w:WorkItem) WHERE "
        + " AND ".join(where)
        + " RETURN "
        + ", ".join(fields)
        + " ORDER BY w.created_at ASC, w.id ASC LIMIT $limit",
        params,
    )
    return [dict(row) for row in rows or []]


def _row_cursor(row: Mapping[str, Any]) -> tuple[float, str]:
    """Return the stable keyset cursor for one native WorkItem row."""

    created_at = row.get("created_at")
    return (float(created_at) if created_at is not None else 0.0, str(row["id"]))


def _repair_repository_row(
    engine: Any, row: Mapping[str, Any], *, tenant: str, now: float | None
) -> bool:
    """Repair one durable repository child and report whether state changed."""

    child_id = str(row.get("id") or "")
    if not child_id:
        return False
    dependencies = tuple(str(value) for value in row.get("depends_on") or () if value)
    if not dependencies:
        return False
    _assert_repository_dependencies_same_tenant(engine, dependencies, tenant=tenant)
    mutated = False
    edge_type = _work_item._task_depends_on_edge_type()
    for parent_id in dependencies:
        # ``depends_on`` is the durable admission projection, but the native
        # DAG edge is also part of the graph view.  Native link implementations
        # use MERGE semantics, so replaying this write is idempotent after a
        # crash at any point in create -> edge -> reverse-index ordering.
        _work_item._link(engine, child_id, parent_id, edge_type)
        mutated = (
            _work_item._append_downstream(engine, parent_id, child_id, now=now)
            or mutated
        )
    return (
        _work_item._reconcile_dependency_readiness(engine, child_id, now=now) or mutated
    )


def reconcile_repository_work_items(
    engine: Any,
    *,
    tenant: str,
    limit: int = 100,
    now: float | None = None,
) -> int:
    """Backfill dependency indexes and repair readiness after restart.

    A process can terminate after the durable child create but before its
    reverse dependency edges are indexed. This idempotent, tenant-scoped pass
    re-adds every parent->child index and reconciles the child count/status;
    native parent commits can then release the child normally. Only submitted
    dependency-bearing candidates are scanned. ``limit`` bounds each keyset
    page; the pass advances through every page to exhaustion, and its return
    value is the number of candidates whose index, count, or status actually
    changed. Missing or cross-tenant dependencies fail closed and are never
    indexed.
    """

    tenant = _require_tenant_for_engine(engine, tenant)
    if not 1 <= int(limit) <= _MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIST_LIMIT}")
    timestamp = now
    repaired = 0
    cursor: tuple[float, str] | None = None
    page_size = int(limit)
    while True:
        rows = _repository_rows(
            engine,
            tenant=tenant,
            limit=page_size,
            statuses=("submitted",),
            cursor=cursor,
        )
        if not rows:
            break
        for row in rows:
            if _repair_repository_row(engine, row, tenant=tenant, now=timestamp):
                repaired += 1
        cursor = _row_cursor(rows[-1])
        if len(rows) < page_size:
            break
    return repaired


def _reconcile_repository_target(
    engine: Any, identifier: str, *, tenant: str, now: float | None
) -> bool:
    """Repair one requested child without relying on a global query prefix."""

    tenant = _require_tenant_for_engine(engine, tenant)
    job_id = _job_id_from_identifier(identifier)
    row = get_work_item(engine, repository_work_item_id(job_id))
    if row is None or row.get("tenant") != tenant:
        return False
    return _repair_repository_row(engine, row, tenant=tenant, now=now)


def list_repository_work_items(
    engine: Any,
    *,
    tenant: str,
    operation: RepositoryOperation | str | None = None,
    states: Sequence[RepositoryJobState | str] = (),
    repository_id: str | None = None,
    lane_id: str | None = None,
    candidate_id: str | None = None,
    generation_id: str | None = None,
    correlation_id: str | None = None,
    owner_id: str | None = None,
    limit: int = 100,
) -> list[RepositoryWorkItemView]:
    """List only repository WorkItems for one authenticated tenant.

    The native query filters by the registered kind set before returning rows,
    so unrelated orchestration work is never selected for a domain listing.
    Correlation filters that live in the bounded extension metadata are applied
    after each bounded native page. Keyset pagination continues until the
    requested number of matching rows is found or the tenant-scoped result set
    is exhausted, so an older nonmatching prefix cannot hide a match.
    """

    tenant = _require_tenant_for_engine(engine, tenant)
    if not 1 <= int(limit) <= _MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIST_LIMIT}")
    reconcile_repository_work_items(engine, tenant=tenant, limit=limit)
    kinds = (
        [repository_work_item_kind(operation).value]
        if operation is not None
        else list(_REPOSITORY_KINDS)
    )
    normalized_states = [_state_value(state) for state in states]
    result: list[RepositoryWorkItemView] = []
    cursor: tuple[float, str] | None = None
    page_size = min(_MAX_LIST_LIMIT, max(limit, 100))
    native_statuses = tuple(
        "dead_letter" if state == "dead-letter" else state
        for state in normalized_states
    )
    while len(result) < limit:
        rows = _repository_rows(
            engine,
            tenant=tenant,
            kinds=kinds,
            statuses=native_statuses,
            limit=page_size,
            cursor=cursor,
        )
        if not rows:
            break
        for row in rows:
            view = _view_from_row(row)
            if repository_id is not None and view.repository_id != repository_id:
                continue
            if lane_id is not None and view.lane_id != lane_id:
                continue
            if candidate_id is not None and view.candidate_id != candidate_id:
                continue
            if generation_id is not None and view.generation_id != generation_id:
                continue
            if correlation_id is not None and view.correlation_id != correlation_id:
                continue
            if owner_id is not None and view.owner_id != owner_id:
                continue
            result.append(view)
            if len(result) >= limit:
                break
        cursor = _row_cursor(rows[-1])
        if len(rows) < page_size:
            break
    return result


def _work_item_fields() -> tuple[str, ...]:
    from agent_utilities.orchestration import work_item

    return work_item._FIELDS


def claim_repository_work_item(
    engine: Any,
    identifier: str,
    *,
    tenant: str,
    token: str,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Claim and start one repository job through the native fenced lease."""

    _reconcile_repository_target(engine, identifier, tenant=tenant, now=now)
    view = get_repository_work_item(engine, identifier, tenant=tenant)
    if view is None:
        return None
    claim = claim_specific(
        engine,
        view.work_item_id,
        token=_nonblank(token, "token"),
        now=now,
        lease_ttl_s=lease_ttl_s,
    )
    if claim is None or not mark_running(engine, view.work_item_id, claim, now=now):
        return None
    return {**claim, "job_id": view.job_id, "tenant": view.tenant_id}


def claim_next_repository_work_item(
    engine: Any,
    *,
    tenant: str,
    kind: RepositoryWorkItemKind | str,
    token: str,
    resource_class: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Claim the next job in one repository kind queue.

    Requiring a kind prevents a repository worker from accidentally claiming
    unrelated graph orchestration work.
    """

    tenant = _require_tenant_for_engine(engine, tenant)
    try:
        kind_value = RepositoryWorkItemKind(kind).value
    except ValueError as exc:
        raise RepositoryWorkItemError(
            f"unknown repository WorkItem kind: {kind!r}"
        ) from exc
    reconcile_repository_work_items(engine, tenant=tenant, limit=_MAX_LIST_LIMIT)
    claim = claim_next(
        engine,
        resource_class=resource_class,
        queue=kind_value,
        tenant=tenant,
        token=_nonblank(token, "token"),
        now=now,
        lease_ttl_s=lease_ttl_s,
    )
    if claim is None:
        return None
    view = get_repository_work_item(engine, claim["work_item_id"], tenant=tenant)
    if view is None:
        raise WorkItemBackendUnavailable(
            "native claim returned a non-repository WorkItem"
        )
    return {**claim, "job_id": view.job_id, "tenant": view.tenant_id}


def _claim_for_view(
    engine: Any, identifier: str, claim: Mapping[str, Any], *, tenant: str
) -> RepositoryWorkItemView:
    view = get_repository_work_item(engine, identifier, tenant=tenant)
    if view is None:
        raise RepositoryWorkItemError(
            "repository WorkItem is missing or outside tenant scope"
        )
    if claim.get("work_item_id") != view.work_item_id:
        raise RepositoryWorkItemError(
            "claim does not belong to the requested repository job"
        )
    claim_job_id = claim.get("job_id")
    if claim_job_id is not None and claim_job_id != view.job_id:
        raise RepositoryWorkItemError(
            "claim does not belong to the requested repository job"
        )
    if claim.get("tenant") not in (None, view.tenant_id):
        raise RepositoryWorkItemError(
            "claim tenant does not match authenticated tenant"
        )
    return view


def heartbeat_repository_work_item(
    engine: Any,
    identifier: str,
    claim: Mapping[str, Any],
    *,
    tenant: str,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> bool:
    """Renew a repository job lease; a stale fence returns ``False``."""

    view = _claim_for_view(engine, identifier, claim, tenant=tenant)
    return heartbeat(
        engine,
        view.work_item_id,
        dict(claim),
        now=now,
        lease_ttl_s=lease_ttl_s,
    )


def checkpoint_repository_work_item(
    engine: Any,
    identifier: str,
    claim: Mapping[str, Any],
    checkpoint_id: str,
    *,
    tenant: str,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> bool:
    """Persist a checkpoint only under the current WorkItem fence."""

    view = _claim_for_view(engine, identifier, claim, tenant=tenant)
    return checkpoint_work_item(
        engine,
        view.work_item_id,
        dict(claim),
        checkpoint_id,
        now=now,
        lease_ttl_s=lease_ttl_s,
    )


def _failure_ref(
    *,
    failure_class: str | None,
    refusal_code: str | None,
    error_ref: str | None,
) -> str | None:
    if not failure_class and not refusal_code:
        return error_ref
    parts = [failure_class or "", refusal_code or "", error_ref or ""]
    if any(any(ord(char) < 0x20 for char in part) for part in parts):
        raise RepositoryWorkItemError(
            "failure/refusal references must not contain control characters"
        )
    return "repository-error:v1:" + ":".join(parts)


def commit_repository_work_item(
    engine: Any,
    identifier: str,
    claim: Mapping[str, Any],
    *,
    tenant: str,
    outcome: Literal["succeeded", "failed", "cancelled"],
    result_ref: str | None = None,
    error_ref: str | None = None,
    failure_class: str | None = None,
    refusal_code: str | None = None,
    retryable: bool = True,
    now: float | None = None,
) -> str:
    """Commit a repository result through the native fenced terminal verb."""

    view = _claim_for_view(engine, identifier, claim, tenant=tenant)
    if failure_class is not None:
        failure_class = _nonblank(failure_class, "failure_class")
        if ":" in failure_class:
            raise RepositoryWorkItemError("failure_class must not contain ':'")
    if refusal_code is not None:
        refusal_code = _nonblank(refusal_code, "refusal_code")
        if ":" in refusal_code:
            raise RepositoryWorkItemError("refusal_code must not contain ':'")
    if failure_class is not None and refusal_code is not None:
        raise RepositoryWorkItemError(
            "a repository result cannot carry both failure_class and refusal_code"
        )
    return commit_result(
        engine,
        view.work_item_id,
        dict(claim),
        outcome=outcome,
        result_ref=result_ref,
        error_ref=_failure_ref(
            failure_class=failure_class,
            refusal_code=refusal_code,
            error_ref=error_ref,
        ),
        retryable=retryable,
        now=now,
    )


def cancel_repository_work_item(
    engine: Any,
    identifier: str,
    *,
    tenant: str,
    reason: str = "",
    now: float | None = None,
) -> bool:
    """Cancel a tenant-owned repository job using the native cancel verb."""

    view = get_repository_work_item(engine, identifier, tenant=tenant)
    if view is None:
        return False
    return cancel_work_item(engine, view.work_item_id, reason=reason, now=now)


def repository_result_from_view(
    view: RepositoryWorkItemView,
) -> RepositoryWorkItemResult:
    """Convert a durable view into the typed RMDD result projection."""

    failure_class = None
    refusal_code = None
    error_ref = view.error_ref
    if error_ref and error_ref.startswith("repository-error:v1:"):
        encoded = error_ref.split(":", 4)
        if len(encoded) == 5:
            _, _, encoded_failure, encoded_refusal, encoded_error = encoded
            failure_class = encoded_failure or None
            refusal_code = encoded_refusal or None
            error_ref = encoded_error or None
    return RepositoryWorkItemResult(
        job_id=view.job_id,
        work_item_id=view.work_item_id,
        request_id=view.request_id,
        operation=view.operation,
        state=view.state,
        repository_id=view.repository_id,
        tenant_id=view.tenant_id,
        operation_payload_kind=view.operation_payload_kind,
        operation_payload_version=view.operation_payload_version,
        operation_payload_digest=view.operation_payload_digest,
        target_kind=view.target_kind,
        target_alias=view.target_alias,
        lane_id=view.lane_id,
        candidate_id=view.candidate_id,
        generation_id=view.generation_id,
        input_digest=view.input_digest,
        config_digest=view.config_digest,
        resource_class=view.resource_class,
        concurrency_key=view.concurrency_key,
        fairness_group=view.fairness_group,
        priority=view.priority,
        cpu_weight=view.cpu_weight,
        memory_mib=view.memory_mib,
        disk_mib=view.disk_mib,
        process_slots=view.process_slots,
        host_labels=view.host_labels,
        preferred_target=view.preferred_target,
        required_target=view.required_target,
        anti_affinity=view.anti_affinity,
        queue_deadline=view.queue_deadline,
        disk_low_watermark_mib=view.disk_low_watermark_mib,
        disk_high_watermark_mib=view.disk_high_watermark_mib,
        consent=view.consent,
        attempt=view.attempt,
        checkpoint=view.checkpoint,
        retry_class=view.retry_class,
        result_ref=view.result_ref,
        error_ref=error_ref,
        failure_class=failure_class,
        refusal_code=refusal_code,
    )


__all__ = [
    "CONTRACT_VERSION",
    "RepositoryJobState",
    "RepositoryLease",
    "RepositoryOperation",
    "RepositoryBuildExecutionPayloadV1",
    "RepositoryOperationPayload",
    "RepositoryTargetPolicy",
    "RepositoryWorkItemConflict",
    "RepositoryWorkItemError",
    "RepositoryWorkItemHandle",
    "RepositoryWorkItemKind",
    "RepositoryWorkItemRequest",
    "RepositoryWorkItemResult",
    "RepositoryWorkItemView",
    "TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE",
    "cancel_repository_work_item",
    "checkpoint_repository_work_item",
    "claim_next_repository_work_item",
    "claim_repository_work_item",
    "commit_repository_work_item",
    "get_repository_operation_payload",
    "get_repository_operation_payload_for_claim",
    "get_repository_work_item",
    "heartbeat_repository_work_item",
    "list_repository_work_items",
    "repository_job_id",
    "reconcile_repository_work_items",
    "repository_result_from_view",
    "repository_work_item_id",
    "repository_work_item_kind",
    "submit_repository_work_item",
]
