#!/usr/bin/python
from __future__ import annotations

"""Durable, idempotent ScaleIntent reconciliation.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling

This module owns the AU-side transaction seam between a durable scale intent
and a typed actuator.  It deliberately does not know Kubernetes, Docker,
Swarm, or a graph client's wire shape.  A deployment supplies a durable
``ScaleIntentLedger`` and a typed, idempotent ``ScaleActuator``.

The ordering is intentional and safety-critical:

1. persist the complete intent identity before an actuator is called;
2. acquire the controller lease and persist ``started`` under its fence;
3. call the actuator with the same execution key on every retry;
4. persist the typed result, then persist exactly one observation;
5. return ``verified`` only after the observation is durable.

If a process crashes between steps 3 and 4, the next delivery sees the
``started`` execution and retries the same execution key.  The actuator
contract must make that key idempotent.  If a process crashes after step 4,
the next delivery records the missing observation without calling the
actuator again.  A dry-run uses the explicit ``simulated`` state and can
never claim that replicas changed.
"""

import hashlib
import json
import math
import re
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

CONTRACT_VERSION: Final[Literal["1"]] = "1"
MAX_REASON_LENGTH = 512
MAX_DETAIL_LENGTH = 512
MAX_RETRY_ATTEMPTS = 8
MAX_RETRY_BACKOFF_S = 3_600.0
MAX_LEASE_TTL_S = 900.0

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class _ScaleModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        use_enum_values=True,
    )


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value.strip()) is None:
        raise ValueError(f"{field_name} must be a bounded identifier")
    return value.strip()


def _digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase digest")
    return value


def _derived_id(prefix: str, value: str) -> str:
    """Build a bounded deterministic identifier from an external key."""

    candidate = f"{prefix}:{value}"
    if _IDENTIFIER_RE.fullmatch(candidate) is not None:
        return candidate
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _finite(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _result_digest(result: ScaleActuationResult) -> str:
    """Canonical digest for the exact typed result being observed."""

    material = json.dumps(
        result.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ScaleControllerMode(StrEnum):
    NATIVE = "native"
    DELEGATED = "delegated"


class ScaleIntentState(StrEnum):
    PERSISTED = "persisted"
    STARTED = "started"
    SIMULATED = "simulated"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OBSERVED = "observed"
    VERIFIED = "verified"
    DENIED = "denied"


class ScaleObservationStatus(StrEnum):
    CONVERGED = "converged"
    FAILED = "failed"
    SIMULATED = "simulated"


class ScaleFence(_ScaleModel):
    """Authority fence carried by the intent, lease, actuator, and outcome."""

    lease_id: str
    lease_epoch: StrictInt = Field(ge=1)
    fence_token: StrictInt = Field(ge=1)
    expires_at: datetime

    _validate_lease = field_validator("lease_id")(
        lambda value: _identifier(value, "lease_id")
    )
    _validate_expiry = field_validator("expires_at")(
        lambda value: _aware(value, "expires_at")
    )


class ScaleTargetBinding(_ScaleModel):
    """The complete target identity; names alone are never sufficient."""

    cluster_id: str
    runtime_id: str
    workload_id: str
    target_uid: str
    target_resource_version: str

    _validate_cluster = field_validator("cluster_id")(
        lambda value: _identifier(value, "cluster_id")
    )
    _validate_runtime = field_validator("runtime_id")(
        lambda value: _identifier(value, "runtime_id")
    )
    _validate_workload = field_validator("workload_id")(
        lambda value: _identifier(value, "workload_id")
    )
    _validate_uid = field_validator("target_uid")(
        lambda value: _identifier(value, "target_uid")
    )
    _validate_resource_version = field_validator("target_resource_version")(
        lambda value: _identifier(value, "target_resource_version")
    )


class ScaleRetryPolicy(_ScaleModel):
    """Durable bounded retry policy; retries reuse one execution key."""

    max_attempts: StrictInt = Field(default=3, ge=1, le=MAX_RETRY_ATTEMPTS)
    backoff_s: float = Field(default=5.0, ge=0, le=MAX_RETRY_BACKOFF_S)

    @field_validator("backoff_s")
    @classmethod
    def finite_backoff(cls, value: float) -> float:
        return _finite(value, "retry backoff")


class ScaleIntentRecord(_ScaleModel):
    """Immutable desired change with all identity required for replay safety."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    intent_id: str
    intent_revision: StrictInt = Field(ge=1)
    expected_unit_revision: StrictInt = Field(ge=1)
    desired_replicas: StrictInt = Field(ge=0)
    min_replicas: StrictInt = Field(ge=0)
    max_replicas: StrictInt = Field(ge=0)
    controller_mode: ScaleControllerMode
    replica_writer_id: str
    execution_key: str
    fence: ScaleFence
    target: ScaleTargetBinding
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH)
    retry_policy: ScaleRetryPolicy = Field(default_factory=ScaleRetryPolicy)
    created_at: datetime

    _validate_intent = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_writer = field_validator("replica_writer_id")(
        lambda value: _identifier(value, "replica_writer_id")
    )
    _validate_execution = field_validator("execution_key")(
        lambda value: _identifier(value, "execution_key")
    )
    _validate_created = field_validator("created_at")(
        lambda value: _aware(value, "created_at")
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> ScaleIntentRecord:
        if self.max_replicas < self.min_replicas:
            raise ValueError("scale intent max_replicas must cover min_replicas")
        if not self.min_replicas <= self.desired_replicas <= self.max_replicas:
            raise ValueError("scale intent desired_replicas violates bounds")
        return self

    def identity_digest(self) -> str:
        """Digest all request identity, including fence and target version."""

        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ControllerLease(_ScaleModel):
    """Controller ownership lease; calls with another lease are fenced."""

    lease_instance_id: str
    controller_id: str
    fence: ScaleFence
    attempt: StrictInt = Field(ge=1)
    acquired_at: datetime
    expires_at: datetime

    _validate_instance = field_validator("lease_instance_id")(
        lambda value: _identifier(value, "lease_instance_id")
    )
    _validate_controller = field_validator("controller_id")(
        lambda value: _identifier(value, "controller_id")
    )
    _validate_acquired = field_validator("acquired_at")(
        lambda value: _aware(value, "acquired_at")
    )
    _validate_expiry = field_validator("expires_at")(
        lambda value: _aware(value, "expires_at")
    )

    @model_validator(mode="after")
    def validate_times(self) -> ControllerLease:
        if self.expires_at <= self.acquired_at:
            raise ValueError("controller lease expires_at must be after acquired_at")
        return self


class ScaleActuationRequest(_ScaleModel):
    """Typed, idempotent actuator input; no runtime-specific dicts."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    intent_id: str
    intent_revision: StrictInt = Field(ge=1)
    expected_unit_revision: StrictInt = Field(ge=1)
    execution_key: str
    desired_replicas: StrictInt = Field(ge=0)
    controller_mode: ScaleControllerMode
    replica_writer_id: str
    target: ScaleTargetBinding
    lease: ControllerLease
    dry_run: StrictBool = False

    _validate_intent = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_execution = field_validator("execution_key")(
        lambda value: _identifier(value, "execution_key")
    )
    _validate_writer = field_validator("replica_writer_id")(
        lambda value: _identifier(value, "replica_writer_id")
    )


class ScaleActuationResult(_ScaleModel):
    """Typed actuator result; ``simulated`` is never represented as scaled."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    execution_key: str
    target: ScaleTargetBinding
    state: ScaleIntentState
    scaled: StrictBool
    observed_replicas: StrictInt | None = Field(default=None, ge=0)
    retryable: StrictBool = False
    error_code: str | None = None
    detail: str = Field(default="", max_length=MAX_DETAIL_LENGTH)
    actuator_operation_id: str | None = None

    _validate_execution = field_validator("execution_key")(
        lambda value: _identifier(value, "execution_key")
    )
    _validate_error = field_validator("error_code")(
        lambda value: None if value is None else _identifier(value, "error_code")
    )
    _validate_operation = field_validator("actuator_operation_id")(
        lambda value: (
            None if value is None else _identifier(value, "actuator_operation_id")
        )
    )

    @model_validator(mode="after")
    def validate_result(self) -> ScaleActuationResult:
        if self.state == ScaleIntentState.SIMULATED.value:
            if self.scaled or self.error_code is not None:
                raise ValueError("simulated result must not claim scaled or failed")
        elif self.state == ScaleIntentState.SUCCEEDED.value:
            if not self.scaled or self.observed_replicas is None:
                raise ValueError("succeeded result requires scaled observation")
        elif self.state == ScaleIntentState.FAILED.value:
            if self.scaled or self.error_code is None:
                raise ValueError("failed result requires an error and no scaling")
        else:
            raise ValueError("actuator result must be succeeded, failed, or simulated")
        return self


class ScaleExecutionRecord(_ScaleModel):
    """Durable execution row, including the last typed result for crash repair."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    execution_id: str
    intent_id: str
    intent_revision: StrictInt = Field(ge=1)
    execution_key: str
    target: ScaleTargetBinding
    state: ScaleIntentState
    attempt: StrictInt = Field(ge=0)
    dry_run: StrictBool = False
    actuator_called: StrictBool = False
    scaled: StrictBool = False
    # A retry keeps the prior typed result attached for auditability while
    # ``attempt`` advances under a newer fence.  This marker lets the ledger
    # accept only that newer retry result, never an unbound duplicate.
    result_attempt: StrictInt | None = Field(default=None, ge=1)
    lease: ControllerLease | None = None
    result: ScaleActuationResult | None = None
    observation_id: str | None = None
    identity_digest: str
    created_at: datetime
    updated_at: datetime

    _validate_execution_id = field_validator("execution_id")(
        lambda value: _identifier(value, "execution_id")
    )
    _validate_intent = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_execution = field_validator("execution_key")(
        lambda value: _identifier(value, "execution_key")
    )
    _validate_identity = field_validator("identity_digest")(
        lambda value: _digest(value, "identity_digest")
    )
    _validate_created = field_validator("created_at")(
        lambda value: _aware(value, "created_at")
    )
    _validate_updated = field_validator("updated_at")(
        lambda value: _aware(value, "updated_at")
    )


class ScaleObservation(_ScaleModel):
    """Exactly-once durable convergence or failure observation."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    observation_id: str
    execution_key: str
    intent_id: str
    intent_revision: StrictInt = Field(ge=1)
    target: ScaleTargetBinding
    status: ScaleObservationStatus
    scaled: StrictBool
    observed_replicas: StrictInt | None = Field(default=None, ge=0)
    outcome_digest: str
    observed_at: datetime
    verified: StrictBool = True
    detail: str = Field(default="", max_length=MAX_DETAIL_LENGTH)

    _validate_observation = field_validator("observation_id")(
        lambda value: _identifier(value, "observation_id")
    )
    _validate_execution = field_validator("execution_key")(
        lambda value: _identifier(value, "execution_key")
    )
    _validate_intent = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_outcome = field_validator("outcome_digest")(
        lambda value: _digest(value, "outcome_digest")
    )
    _validate_observed = field_validator("observed_at")(
        lambda value: _aware(value, "observed_at")
    )

    @model_validator(mode="after")
    def validate_status(self) -> ScaleObservation:
        if not self.verified:
            raise ValueError("durable observations must be verified")
        if self.status == ScaleObservationStatus.CONVERGED.value:
            if not self.scaled or self.observed_replicas is None:
                raise ValueError("converged observation requires scaled replicas")
        elif self.status in {
            ScaleObservationStatus.FAILED.value,
            ScaleObservationStatus.SIMULATED.value,
        }:
            if self.scaled:
                raise ValueError("failed or simulated observation cannot be scaled")
        return self


class ScalePolicyDecision(_ScaleModel):
    """Policy result; break-glass is a separate authorization, never implicit."""

    decision_id: str
    allowed: StrictBool
    reason: str = Field(default="", max_length=MAX_REASON_LENGTH)

    _validate_decision = field_validator("decision_id")(
        lambda value: _identifier(value, "decision_id")
    )


class BreakGlassAuthorization(_ScaleModel):
    """Explicit, scoped authorization for an otherwise denied intent."""

    authorization_id: str
    actor_id: str
    approval_ref: str
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH)
    target: ScaleTargetBinding
    authorized_until: datetime
    authorized: StrictBool = True

    _validate_authorization = field_validator("authorization_id")(
        lambda value: _identifier(value, "authorization_id")
    )
    _validate_actor = field_validator("actor_id")(
        lambda value: _identifier(value, "actor_id")
    )
    _validate_approval = field_validator("approval_ref")(
        lambda value: _identifier(value, "approval_ref")
    )
    _validate_expiry = field_validator("authorized_until")(
        lambda value: _aware(value, "authorized_until")
    )


class BreakGlassAudit(_ScaleModel):
    """Durable audit receipt emitted before a break-glass actuator call."""

    audit_id: str
    authorization_id: str
    intent_id: str
    actor_id: str
    approval_ref: str
    target: ScaleTargetBinding
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH)
    authorized_until: datetime
    recorded_at: datetime
    audit_digest: str

    _validate_audit = field_validator("audit_id")(
        lambda value: _identifier(value, "audit_id")
    )
    _validate_authorization = field_validator("authorization_id")(
        lambda value: _identifier(value, "authorization_id")
    )
    _validate_intent = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_actor = field_validator("actor_id")(
        lambda value: _identifier(value, "actor_id")
    )
    _validate_approval = field_validator("approval_ref")(
        lambda value: _identifier(value, "approval_ref")
    )
    _validate_recorded = field_validator("recorded_at")(
        lambda value: _aware(value, "recorded_at")
    )
    _validate_authorized_until = field_validator("authorized_until")(
        lambda value: _aware(value, "authorized_until")
    )
    _validate_digest = field_validator("audit_digest")(
        lambda value: _digest(value, "audit_digest")
    )


class ScaleReconcileResult(_ScaleModel):
    """Process-local presentation of a durable reconciliation result."""

    state: ScaleIntentState
    replayed: StrictBool = False
    actuator_called: StrictBool = False
    scaled: StrictBool = False
    execution: ScaleExecutionRecord
    observation: ScaleObservation | None = None
    break_glass_audit: BreakGlassAudit | None = None
    retry_after_s: float | None = Field(default=None, ge=0)
    detail: str = Field(default="", max_length=MAX_DETAIL_LENGTH)


class ScaleIntentConflict(RuntimeError):
    """An execution key or intent revision was replayed with another payload."""


class ScaleLeaseUnavailable(RuntimeError):
    """Another controller owns the live lease or the fence is stale."""


class ScaleLeaseLost(RuntimeError):
    """A result or observation arrived under a stale controller lease."""


class ScaleObservationConflict(RuntimeError):
    """The same execution key attempted a different terminal observation."""


@runtime_checkable
class ScaleActuator(Protocol):
    """Typed actuator; implementations must deduplicate ``execution_key``."""

    name: str

    def apply(self, request: ScaleActuationRequest) -> ScaleActuationResult:
        """Apply or idempotently replay one typed request; never return a dict."""
        ...  # ABSTRACT-OK


@runtime_checkable
class ScalePolicy(Protocol):
    def decide(self, intent: ScaleIntentRecord) -> ScalePolicyDecision:
        """Return a durable-policy verdict before actuation."""
        ...  # ABSTRACT-OK


class AllowScalePolicy:
    """Explicit allow fixture; deployments should inject governance."""

    def decide(self, intent: ScaleIntentRecord) -> ScalePolicyDecision:
        return ScalePolicyDecision(
            decision_id=_derived_id("decision", intent.execution_key),
            allowed=True,
            reason="allowed by injected test policy",
        )


class DenyScalePolicy:
    """Fail-closed policy used when a deployment has not injected governance."""

    def decide(self, intent: ScaleIntentRecord) -> ScalePolicyDecision:
        return ScalePolicyDecision(
            decision_id=_derived_id("decision", intent.execution_key),
            allowed=False,
            reason="no scale policy was injected",
        )


@runtime_checkable
class ScaleIntentLedger(Protocol):
    """Durable store required by the reconcile ordering contract."""

    def persist_intent(self, intent: ScaleIntentRecord) -> PersistedIntent:
        """Atomically insert/deduplicate the intent before any actuator call."""
        ...  # ABSTRACT-OK

    def acquire_controller_lease(
        self,
        intent: ScaleIntentRecord,
        controller_id: str,
        *,
        now: datetime,
    ) -> ControllerLease | None:
        """Return a fenced lease or ``None`` when another controller owns it."""
        ...  # ABSTRACT-OK

    def mark_started(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        lease: ControllerLease,
        *,
        now: datetime,
    ) -> ScaleExecutionRecord:
        """Persist started state under the live lease before applying a side effect."""
        ...  # ABSTRACT-OK

    def record_result(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        lease: ControllerLease,
        result: ScaleActuationResult,
        *,
        now: datetime,
    ) -> ScaleExecutionRecord:
        """Persist one typed result idempotently under the execution key."""
        ...  # ABSTRACT-OK

    def record_observation(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        observation: ScaleObservation,
        *,
        now: datetime,
    ) -> ScaleObservation:
        """Persist exactly one convergence/failure observation."""
        ...  # ABSTRACT-OK

    def get_observation(self, execution_key: str) -> ScaleObservation | None:
        """Read the durable terminal observation for crash/replay repair."""
        ...  # ABSTRACT-OK

    def record_break_glass(
        self,
        intent: ScaleIntentRecord,
        authorization: BreakGlassAuthorization,
        *,
        now: datetime,
    ) -> BreakGlassAudit:
        """Persist the authorization audit before bypassing a policy denial."""
        ...  # ABSTRACT-OK


class PersistedIntent(_ScaleModel):
    execution: ScaleExecutionRecord
    replayed: StrictBool = False


class MemoryScaleIntentLedger:
    """Deterministic fixture ledger implementing the durable protocol.

    This is intentionally an in-memory test seam, not a production durability
    claim.  A graph/database adapter must preserve the same atomic ordering,
    identity-conflict behavior, fence checks, and terminal idempotency.
    """

    def __init__(self) -> None:
        self._intents: dict[tuple[str, int], tuple[str, str]] = {}
        self._executions: dict[str, ScaleExecutionRecord] = {}
        self._leases: dict[str, ControllerLease] = {}
        self._observations: dict[str, ScaleObservation] = {}
        self._break_glass: dict[tuple[str, str], BreakGlassAudit] = {}
        self.events: list[str] = []
        self._lock = threading.RLock()

    def persist_intent(self, intent: ScaleIntentRecord) -> PersistedIntent:
        with self._lock:
            identity = intent.identity_digest()
            intent_key = (intent.intent_id, intent.intent_revision)
            prior = self._intents.get(intent_key)
            if prior is not None and prior != (intent.execution_key, identity):
                raise ScaleIntentConflict(
                    "intent revision replayed with different identity"
                )
            existing = self._executions.get(intent.execution_key)
            if existing is not None:
                if existing.identity_digest != identity:
                    raise ScaleIntentConflict(
                        "execution key replayed with different identity"
                    )
                self.events.append("persist_intent:replayed")
                return PersistedIntent(execution=existing, replayed=True)
            if prior is not None and prior[0] != intent.execution_key:
                raise ScaleIntentConflict(
                    "intent revision is bound to another execution key"
                )
            now = intent.created_at
            execution = ScaleExecutionRecord(
                execution_id=_derived_id("execution", intent.execution_key),
                intent_id=intent.intent_id,
                intent_revision=intent.intent_revision,
                execution_key=intent.execution_key,
                target=intent.target,
                state=ScaleIntentState.PERSISTED,
                attempt=0,
                dry_run=False,
                actuator_called=False,
                scaled=False,
                result_attempt=None,
                identity_digest=identity,
                created_at=now,
                updated_at=now,
            )
            self._intents[intent_key] = (intent.execution_key, identity)
            self._executions[intent.execution_key] = execution
            self.events.append("persist_intent")
            return PersistedIntent(execution=execution, replayed=False)

    def acquire_controller_lease(
        self,
        intent: ScaleIntentRecord,
        controller_id: str,
        *,
        now: datetime,
    ) -> ControllerLease | None:
        with self._lock:
            now = _aware(now, "now")
            if now >= intent.fence.expires_at:
                raise ScaleLeaseUnavailable("scale intent fence is expired")
            current_execution = self._executions.get(intent.execution_key)
            if (
                current_execution is None
                or current_execution.identity_digest != intent.identity_digest()
            ):
                raise ScaleIntentConflict(
                    "controller lease requires the persisted intent identity"
                )
            current = self._leases.get(intent.execution_key)
            if current is not None and current.expires_at > now:
                if current.controller_id == controller_id and not (
                    current_execution.state == ScaleIntentState.FAILED.value
                    and current_execution.result is not None
                    and current_execution.result.retryable
                ):
                    self.events.append("lease:replayed")
                    return current
                if current.controller_id != controller_id:
                    return None
            prior_attempt = self._executions[intent.execution_key].attempt
            lease = ControllerLease(
                lease_instance_id=f"lease:{uuid.uuid4().hex}",
                controller_id=_identifier(controller_id, "controller_id"),
                fence=intent.fence,
                attempt=prior_attempt + 1,
                acquired_at=now,
                expires_at=min(
                    intent.fence.expires_at,
                    now + timedelta(seconds=MAX_LEASE_TTL_S),
                ),
            )
            self._leases[intent.execution_key] = lease
            self.events.append("lease:acquired")
            return lease

    def mark_started(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        lease: ControllerLease,
        *,
        now: datetime,
    ) -> ScaleExecutionRecord:
        with self._lock:
            self._check_lease(intent, execution, lease, now)
            current = self._executions[intent.execution_key]
            if current.state in {
                ScaleIntentState.SIMULATED.value,
                ScaleIntentState.SUCCEEDED.value,
                ScaleIntentState.OBSERVED.value,
                ScaleIntentState.VERIFIED.value,
            }:
                return current
            if current.state == ScaleIntentState.FAILED.value:
                if current.result is None or not current.result.retryable:
                    raise ScaleIntentConflict(
                        "non-retryable execution cannot be started again"
                    )
                if lease.attempt <= current.attempt:
                    raise ScaleLeaseLost("retry must use a newer controller lease")
            updated = current.model_copy(
                update={
                    "state": ScaleIntentState.STARTED.value,
                    "attempt": lease.attempt,
                    "lease": lease,
                    "updated_at": _aware(now, "now"),
                }
            )
            self._executions[intent.execution_key] = updated
            self.events.append("execution:started")
            return updated

    def record_result(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        lease: ControllerLease,
        result: ScaleActuationResult,
        *,
        now: datetime,
    ) -> ScaleExecutionRecord:
        with self._lock:
            self._check_lease(intent, execution, lease, now)
            if (
                result.execution_key != intent.execution_key
                or result.target != intent.target
            ):
                raise ScaleIntentConflict(
                    "actuator result identity does not match intent"
                )
            current = self._executions[intent.execution_key]
            if current.result is not None:
                same_result = current.result == result
                same_attempt = current.result_attempt == lease.attempt
                retry_attempt = (
                    current.result.retryable
                    and current.result_attempt is not None
                    and lease.attempt > current.result_attempt
                    and current.state == ScaleIntentState.STARTED.value
                )
                if not same_result and not retry_attempt:
                    raise ScaleIntentConflict(
                        "execution key returned a different result"
                    )
                if same_result and same_attempt:
                    return current
            state = result.state
            updated = current.model_copy(
                update={
                    "state": state,
                    "dry_run": state == ScaleIntentState.SIMULATED.value,
                    "actuator_called": state != ScaleIntentState.SIMULATED.value,
                    "scaled": result.scaled,
                    "result_attempt": lease.attempt,
                    "lease": lease,
                    "result": result,
                    "updated_at": _aware(now, "now"),
                }
            )
            self._executions[intent.execution_key] = updated
            self.events.append(f"execution:result:{state}")
            return updated

    def record_observation(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        observation: ScaleObservation,
        *,
        now: datetime,
    ) -> ScaleObservation:
        with self._lock:
            if (
                observation.execution_key != intent.execution_key
                or observation.target != intent.target
            ):
                raise ScaleIntentConflict("observation identity does not match intent")
            if (
                observation.intent_id != intent.intent_id
                or observation.intent_revision != intent.intent_revision
            ):
                raise ScaleIntentConflict("observation intent identity does not match")
            current_execution = self._executions.get(intent.execution_key)
            if current_execution is None:
                raise ScaleIntentConflict("observation requires a persisted intent")
            if (
                execution.execution_key != current_execution.execution_key
                or execution.identity_digest != current_execution.identity_digest
            ):
                raise ScaleIntentConflict("observation execution identity is stale")
            current = self._observations.get(intent.execution_key)
            if current is not None:
                if current.outcome_digest != observation.outcome_digest:
                    raise ScaleObservationConflict("observation replay changed outcome")
                return current
            if current_execution.result is None:
                raise ScaleIntentConflict(
                    "observation requires a durable actuator result"
                )
            result = current_execution.result
            expected_status = (
                ScaleObservationStatus.CONVERGED.value
                if result.state == ScaleIntentState.SUCCEEDED.value
                else ScaleObservationStatus.SIMULATED.value
                if result.state == ScaleIntentState.SIMULATED.value
                else ScaleObservationStatus.FAILED.value
            )
            if (
                observation.status != expected_status
                or observation.scaled != result.scaled
                or observation.observed_replicas != result.observed_replicas
                or observation.outcome_digest != _result_digest(result)
            ):
                raise ScaleObservationConflict(
                    "observation does not match the durable actuator result"
                )
            self._observations[intent.execution_key] = observation
            self._executions[intent.execution_key] = current_execution.model_copy(
                update={
                    "state": ScaleIntentState.VERIFIED.value,
                    "observation_id": observation.observation_id,
                    "updated_at": _aware(now, "now"),
                }
            )
            self.events.append("observation:verified")
            return observation

    def record_break_glass(
        self,
        intent: ScaleIntentRecord,
        authorization: BreakGlassAuthorization,
        *,
        now: datetime,
    ) -> BreakGlassAudit:
        with self._lock:
            now = _aware(now, "now")
            if not authorization.authorized or now >= authorization.authorized_until:
                raise ScaleIntentConflict("break-glass authorization is not active")
            if authorization.target != intent.target:
                raise ScaleIntentConflict("break-glass target does not match intent")
            persisted = self._executions.get(intent.execution_key)
            if (
                persisted is None
                or persisted.identity_digest != intent.identity_digest()
            ):
                raise ScaleIntentConflict(
                    "break-glass requires the persisted intent identity"
                )
            key = (intent.execution_key, authorization.authorization_id)
            material = json.dumps(
                {
                    "authorization_id": authorization.authorization_id,
                    "intent_id": intent.intent_id,
                    "actor_id": authorization.actor_id,
                    "approval_ref": authorization.approval_ref,
                    "reason": authorization.reason,
                    "target": authorization.target.model_dump(mode="json"),
                    "authorized_until": authorization.authorized_until.isoformat(),
                    "authorized": authorization.authorized,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            audit_digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
            existing = self._break_glass.get(key)
            if existing is not None:
                if existing.audit_digest != audit_digest:
                    raise ScaleIntentConflict(
                        "break-glass authorization replay changed its identity"
                    )
                return existing
            audit = BreakGlassAudit(
                audit_id=_derived_id("audit", authorization.authorization_id),
                authorization_id=authorization.authorization_id,
                intent_id=intent.intent_id,
                actor_id=authorization.actor_id,
                approval_ref=authorization.approval_ref,
                target=authorization.target,
                reason=authorization.reason,
                authorized_until=authorization.authorized_until,
                recorded_at=now,
                audit_digest=audit_digest,
            )
            self._break_glass[key] = audit
            self.events.append("break_glass:audit")
            return audit

    def get_observation(self, execution_key: str) -> ScaleObservation | None:
        with self._lock:
            return self._observations.get(execution_key)

    def _check_lease(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        lease: ControllerLease,
        now: datetime,
    ) -> None:
        now = _aware(now, "now")
        current = self._leases.get(intent.execution_key)
        if current is None or current.lease_instance_id != lease.lease_instance_id:
            raise ScaleLeaseLost("controller lease is no longer current")
        if now >= lease.expires_at:
            raise ScaleLeaseLost("controller lease expired")
        if execution.execution_key != intent.execution_key:
            raise ScaleIntentConflict("execution key is not bound to intent")
        persisted = self._executions.get(intent.execution_key)
        if (
            persisted is None
            or persisted.identity_digest != intent.identity_digest()
            or execution.identity_digest != persisted.identity_digest
            or execution.target != intent.target
            or execution.intent_id != intent.intent_id
            or execution.intent_revision != intent.intent_revision
        ):
            raise ScaleIntentConflict("execution identity is not bound to intent")
        if lease.fence != intent.fence:
            raise ScaleLeaseLost("controller fence changed")


class ScaleIntentReconciler:
    """One-at-a-time durable intent reconciler with replay-safe retries."""

    def __init__(
        self,
        ledger: ScaleIntentLedger,
        actuator: ScaleActuator,
        *,
        policy: ScalePolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ledger, ScaleIntentLedger):
            raise TypeError("scale ledger must implement the durable contract")
        if not isinstance(actuator, ScaleActuator):
            raise TypeError("scale actuator must implement the typed contract")
        self.ledger = ledger
        self.actuator = actuator
        self.policy = policy if policy is not None else DenyScalePolicy()
        if not isinstance(self.policy, ScalePolicy):
            raise TypeError("scale policy must implement the policy contract")
        self.clock = clock or (lambda: datetime.now(UTC))

    def reconcile(
        self,
        intent: ScaleIntentRecord,
        *,
        controller_id: str,
        dry_run: bool = False,
        break_glass: BreakGlassAuthorization | None = None,
    ) -> ScaleReconcileResult:
        now = _aware(self.clock(), "clock")
        if controller_id != intent.replica_writer_id:
            raise ScaleLeaseUnavailable("controller is not the declared replica writer")
        if now >= intent.fence.expires_at:
            raise ScaleLeaseUnavailable("scale intent fence is expired")
        persisted = self.ledger.persist_intent(intent)
        execution = persisted.execution
        if dry_run and execution.state == ScaleIntentState.VERIFIED.value:
            existing = self._existing_observation(intent.execution_key)
            if (
                existing is None
                or existing.status != ScaleObservationStatus.SIMULATED.value
            ):
                raise ScaleIntentConflict(
                    "dry-run cannot reuse an execution that may have scaled"
                )
        elif dry_run and execution.state not in {
            ScaleIntentState.PERSISTED.value,
            ScaleIntentState.SIMULATED.value,
        }:
            raise ScaleIntentConflict(
                "dry-run cannot reuse a live or retrying execution key"
            )
        if execution.state in {
            ScaleIntentState.OBSERVED.value,
            ScaleIntentState.VERIFIED.value,
        }:
            observation = self._existing_observation(intent.execution_key)
            if observation is None:
                raise ScaleObservationConflict(
                    "verified execution is missing its durable observation"
                )
            return ScaleReconcileResult(
                state=ScaleIntentState.VERIFIED,
                replayed=True,
                actuator_called=False,
                scaled=bool(observation and observation.scaled),
                execution=execution,
                observation=observation,
                detail="terminal execution replayed",
            )
        if (
            execution.state
            in {
                ScaleIntentState.SIMULATED.value,
                ScaleIntentState.SUCCEEDED.value,
            }
            and execution.result is not None
        ):
            observation = self._observe_result(intent, execution, execution.result, now)
            return ScaleReconcileResult(
                state=ScaleIntentState.VERIFIED,
                replayed=True,
                actuator_called=False,
                scaled=observation.scaled,
                execution=self._verified_execution(execution, observation),
                observation=observation,
                detail="durable result replayed into missing observation",
            )

        # A durable non-retryable result (or an exhausted retry budget) is
        # terminal even if the process died before writing its observation.
        # Repair that observation without consulting policy or invoking an
        # actuator again; the execution key is already consumed.
        if (
            execution.state == ScaleIntentState.FAILED.value
            and execution.result is not None
            and (
                not execution.result.retryable
                or execution.attempt >= intent.retry_policy.max_attempts
            )
        ):
            observation = self._observe_result(intent, execution, execution.result, now)
            return ScaleReconcileResult(
                state=ScaleIntentState.VERIFIED,
                replayed=True,
                actuator_called=False,
                scaled=False,
                execution=self._verified_execution(execution, observation),
                observation=observation,
                detail="terminal actuator result replayed into missing observation",
            )

        if (
            execution.state == ScaleIntentState.STARTED.value
            and execution.result is None
            and execution.attempt >= intent.retry_policy.max_attempts
        ):
            # The side effect may have happened, but its durable result did
            # not.  Do not exceed the declared retry budget or invent a
            # failure observation that could conceal an external change.
            raise ScaleIntentConflict(
                "retry budget exhausted before an actuator result was durable"
            )

        decision = self.policy.decide(intent)
        audit: BreakGlassAudit | None = None
        if not decision.allowed:
            if break_glass is None:
                return ScaleReconcileResult(
                    state=ScaleIntentState.DENIED,
                    replayed=persisted.replayed,
                    execution=execution,
                    detail=f"policy denied: {decision.reason}",
                )
            self._validate_break_glass(intent, break_glass, now)
            audit = self.ledger.record_break_glass(intent, break_glass, now=now)

        lease = self.ledger.acquire_controller_lease(intent, controller_id, now=now)
        if lease is None:
            raise ScaleLeaseUnavailable("another controller owns the scale lease")

        if dry_run:
            result = ScaleActuationResult(
                execution_key=intent.execution_key,
                target=intent.target,
                state=ScaleIntentState.SIMULATED,
                scaled=False,
                detail="dry-run: actuator not called",
            )
            execution = self.ledger.record_result(
                intent,
                execution,
                lease,
                result,
                now=now,
            )
            observation = self._observe_result(intent, execution, result, now)
            return ScaleReconcileResult(
                state=ScaleIntentState.VERIFIED,
                replayed=persisted.replayed,
                actuator_called=False,
                scaled=False,
                execution=self._verified_execution(execution, observation),
                observation=observation,
                break_glass_audit=audit,
                detail="dry-run simulated; no replicas changed",
            )

        execution = self.ledger.mark_started(intent, execution, lease, now=now)
        request = ScaleActuationRequest(
            intent_id=intent.intent_id,
            intent_revision=intent.intent_revision,
            expected_unit_revision=intent.expected_unit_revision,
            execution_key=intent.execution_key,
            desired_replicas=intent.desired_replicas,
            controller_mode=intent.controller_mode,
            replica_writer_id=intent.replica_writer_id,
            target=intent.target,
            lease=lease,
            dry_run=False,
        )
        try:
            result = self.actuator.apply(request)
        except Exception as exc:  # noqa: BLE001 — typed failure is persisted, never raised as success
            result = ScaleActuationResult(
                execution_key=intent.execution_key,
                target=intent.target,
                state=ScaleIntentState.FAILED,
                scaled=False,
                retryable=True,
                error_code="actuator_exception",
                detail=str(exc)[:MAX_DETAIL_LENGTH],
            )
        if not isinstance(result, ScaleActuationResult):
            raise TypeError("scale actuator returned a non-typed result")
        if result.state == ScaleIntentState.SIMULATED.value:
            raise ScaleIntentConflict(
                "a live actuator must not return a simulated result"
            )
        if (
            result.state == ScaleIntentState.SUCCEEDED.value
            and result.observed_replicas != intent.desired_replicas
        ):
            result = ScaleActuationResult(
                execution_key=intent.execution_key,
                target=intent.target,
                state=ScaleIntentState.FAILED,
                scaled=False,
                observed_replicas=result.observed_replicas,
                retryable=False,
                error_code="observed_replicas_mismatch",
                detail="actuator did not converge to the requested replica count",
            )
        execution = self.ledger.record_result(
            intent,
            execution,
            lease,
            result,
            now=now,
        )
        if result.state == ScaleIntentState.FAILED.value and result.retryable:
            if lease.attempt < intent.retry_policy.max_attempts:
                return ScaleReconcileResult(
                    state=ScaleIntentState.FAILED,
                    replayed=persisted.replayed,
                    actuator_called=True,
                    scaled=False,
                    execution=execution,
                    break_glass_audit=audit,
                    retry_after_s=intent.retry_policy.backoff_s,
                    detail="retryable actuator failure persisted",
                )
        observation = self._observe_result(intent, execution, result, now)
        return ScaleReconcileResult(
            state=ScaleIntentState.VERIFIED,
            replayed=persisted.replayed,
            actuator_called=True,
            scaled=result.scaled,
            execution=self._verified_execution(execution, observation),
            observation=observation,
            break_glass_audit=audit,
            detail=result.detail,
        )

    @staticmethod
    def _verified_execution(
        execution: ScaleExecutionRecord,
        observation: ScaleObservation,
    ) -> ScaleExecutionRecord:
        """Reflect the durable observation transition in the returned view."""

        return execution.model_copy(
            update={
                "state": ScaleIntentState.VERIFIED.value,
                "observation_id": observation.observation_id,
            }
        )

    def _observe_result(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        result: ScaleActuationResult,
        now: datetime,
    ) -> ScaleObservation:
        status = (
            ScaleObservationStatus.CONVERGED
            if result.state == ScaleIntentState.SUCCEEDED.value
            else ScaleObservationStatus.SIMULATED
            if result.state == ScaleIntentState.SIMULATED.value
            else ScaleObservationStatus.FAILED
        )
        outcome_digest = _result_digest(result)
        observation = ScaleObservation(
            observation_id=_derived_id("observation", intent.execution_key),
            execution_key=intent.execution_key,
            intent_id=intent.intent_id,
            intent_revision=intent.intent_revision,
            target=intent.target,
            status=status,
            scaled=result.scaled,
            observed_replicas=result.observed_replicas,
            outcome_digest=outcome_digest,
            observed_at=now,
            detail=result.detail,
        )
        return self.ledger.record_observation(
            intent,
            execution,
            observation,
            now=now,
        )

    def _observe_failure(
        self,
        intent: ScaleIntentRecord,
        execution: ScaleExecutionRecord,
        now: datetime,
    ) -> ScaleObservation:
        if execution.result is None:
            raise ScaleIntentConflict("terminal failure lacks durable actuator result")
        return self._observe_result(intent, execution, execution.result, now)

    def _existing_observation(self, execution_key: str) -> ScaleObservation | None:
        return self.ledger.get_observation(execution_key)

    @staticmethod
    def _validate_break_glass(
        intent: ScaleIntentRecord,
        authorization: BreakGlassAuthorization,
        now: datetime,
    ) -> None:
        if not authorization.authorized:
            raise ScaleIntentConflict("break-glass authorization is not active")
        if authorization.target != intent.target:
            raise ScaleIntentConflict("break-glass authorization target mismatch")
        if now >= authorization.authorized_until:
            raise ScaleIntentConflict("break-glass authorization is expired")


__all__ = [
    "AllowScalePolicy",
    "BreakGlassAudit",
    "BreakGlassAuthorization",
    "ControllerLease",
    "DenyScalePolicy",
    "MemoryScaleIntentLedger",
    "PersistedIntent",
    "ScaleActuationRequest",
    "ScaleActuationResult",
    "ScaleActuator",
    "ScaleControllerMode",
    "ScaleExecutionRecord",
    "ScaleFence",
    "ScaleIntentConflict",
    "ScaleIntentLedger",
    "ScaleIntentRecord",
    "ScaleIntentReconciler",
    "ScaleIntentState",
    "ScaleLeaseLost",
    "ScaleLeaseUnavailable",
    "ScaleObservation",
    "ScaleObservationConflict",
    "ScaleObservationStatus",
    "ScalePolicy",
    "ScalePolicyDecision",
    "ScaleReconcileResult",
    "ScaleRetryPolicy",
    "ScaleTargetBinding",
]
