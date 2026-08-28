"""Typed, append-only economics and SLO control-plane contracts.

This module deliberately models *references and facts*, not telemetry payloads.  A
collector keeps raw spans, logs, and provider responses in its own retention tier;
the control plane receives only an opaque source reference, a content digest, and
bounded numeric facts.  The models are frozen so a correction is a new record and
never an in-place mutation of accounting or SLO history.

CONCEPT:AU-OS.control-plane.economics — auditable usage, pricing, and SLO read
models with deterministic identities and tenant-scoped projections.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

SCHEMA_VERSION: Literal["1"] = "1"
MAX_REF_LENGTH = 256
MAX_DIMENSIONS = 16
MAX_FACT_QUANTITY = 10**15
MAX_PAGE_SIZE = 500
MAX_FACT_IDS_PER_AGGREGATE = 100_000
MAX_RETENTION_SECONDS = 366 * 24 * 60 * 60
MAX_LATE_EVENT_SECONDS = 7 * 24 * 60 * 60
MAX_RECONCILIATION_ITEMS = 10_000
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SENSITIVE_RE = re.compile(
    r"(?:authorization|bearer|cookie|password|passwd|private[_-]?key|secret|"
    r"token(?:=|:)|api[_-]?key|raw[_-]?(?:body|span|log))",
    re.IGNORECASE,
)

MetricKind = Literal[
    "tokens",
    "requests",
    "latency_ms",
    "compute_ms",
    "storage_bytes",
    "errors",
    "queue_ms",
    "ingest_rows",
]
MAX_METRIC_KINDS = 8
WindowGranularity = Literal["hour", "day"]
SliKind = Literal[
    "availability",
    "error_rate",
    "latency_ms",
    "queue_age_ms",
    "freshness_ms",
    "throughput",
]


class EconomicsContractError(ValueError):
    """Raised when a domain invariant cannot be represented safely."""


class EconomicsConflict(EconomicsContractError):
    """Raised when an append-only identity is reused with different content."""


class TenantScopeError(EconomicsContractError):
    """Raised when a read or allocation crosses a tenant boundary."""


class LateEventRejected(EconomicsContractError):
    """Raised when policy rejects an event older than the closed watermark."""


def _canonical(value: object) -> bytes:
    """Encode a small model projection deterministically for identity hashing."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: object) -> str:
    """Return the content-addressed digest used by all durable identities."""

    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _as_json(model: ProtocolModel) -> dict[str, object]:
    return model.model_dump(mode="json", exclude_none=True)


def _ensure_ref(value: str, *, field_name: str) -> str:
    if len(value) > MAX_REF_LENGTH or not _REF_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a bounded opaque reference")
    if _SENSITIVE_RE.search(value):
        raise ValueError(f"{field_name} must not contain credentials or raw telemetry")
    return value


def _ensure_digest(value: str, *, field_name: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a sha256:<64 lowercase hex> digest")
    return value


def _ensure_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    normalized = value.astimezone(UTC)
    if normalized.year < 2000 or normalized.year > 2300:
        raise ValueError(f"{field_name} is outside the supported time range")
    return normalized


def _ensure_nonnegative_finite(value: int, *, field_name: str) -> int:
    # StrictInt prevents bool and float coercion.  Keep this explicit because the
    # boundary must document why NaN/Inf/negative telemetry is not admitted.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a finite integer")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


class DimensionRef(ProtocolModel):
    """Bounded, non-secret grouping dimension carried with a usage fact."""

    key: Annotated[str, Field(min_length=1, max_length=64)]
    value: Annotated[str, Field(min_length=1, max_length=256)]

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not _KEY_RE.fullmatch(value) or _SENSITIVE_RE.search(value):
            raise ValueError("dimension key is invalid or sensitive")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        if not value.strip() or _SENSITIVE_RE.search(value):
            raise ValueError("dimension value is invalid or sensitive")
        return value


class AllocationRef(ProtocolModel):
    """Tenant-owned allocation coordinates; no cross-tenant joins are implicit."""

    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    project_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    cost_center_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    workload_ref: (
        Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)] | None
    ) = None

    @field_validator("tenant_id", "project_ref", "cost_center_ref", "workload_ref")
    @classmethod
    def validate_refs(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))


class UsageFact(ProtocolModel):
    """One immutable, deduplicable usage observation.

    ``fact_id`` and ``fact_digest`` are derived from the event identity.  A caller
    may provide them when replaying a persisted record, but a mismatched value is
    rejected rather than trusted.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    fact_id: Annotated[str, Field(min_length=1, max_length=80)] = ""
    fact_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    allocation: AllocationRef
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    occurred_at: datetime
    metric: MetricKind
    quantity: Annotated[StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY)]
    input_quantity: Annotated[StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY)] | None = (
        None
    )
    output_quantity: Annotated[StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY)] | None = (
        None
    )
    service_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    meter_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    sample_sequence: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)] | None = None
    dimensions: Annotated[
        tuple[DimensionRef, ...], Field(max_length=MAX_DIMENSIONS)
    ] = ()

    @field_validator("tenant_id", "source_ref", "service_ref", "meter_ref")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _ensure_digest(value, field_name="source_digest")

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        return _ensure_utc(value, field_name="occurred_at")

    @field_validator("quantity", "input_quantity", "output_quantity")
    @classmethod
    def validate_quantities(cls, value: int | None, info: object) -> int | None:
        if value is None:
            return None
        return _ensure_nonnegative_finite(
            value, field_name=str(getattr(info, "field_name", "quantity"))
        )

    def _check_coherence(self) -> None:
        """Cross-field invariants extracted from ``validate_identity_and_coherence``.

        Split out so the identity-assignment half of that validator stays a
        flat, low-complexity sequence; this half carries the branching.
        """
        if self.allocation.tenant_id != self.tenant_id:
            raise TenantScopeError("allocation tenant_id must match usage tenant_id")
        keys = [dimension.key for dimension in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError("usage dimensions must have unique keys")
        if self.metric == "tokens":
            if (self.input_quantity is None) != (self.output_quantity is None):
                raise ValueError(
                    "token input/output quantities must be supplied together"
                )
            if (
                self.input_quantity is not None
                and self.quantity != self.input_quantity + self.output_quantity  # type: ignore[operator]
            ):
                raise ValueError("token quantity must equal input plus output")
        elif self.input_quantity is not None or self.output_quantity is not None:
            raise ValueError("input/output quantities are valid only for token facts")

    @model_validator(mode="after")
    def validate_identity_and_coherence(self) -> UsageFact:
        self._check_coherence()
        identity = self.identity_payload()
        expected_id = "usage:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        expected_digest = content_digest({"kind": "usage_fact", "identity": identity})
        if self.fact_id and self.fact_id != expected_id:
            raise EconomicsConflict(
                "fact_id does not match the immutable fact identity"
            )
        if self.fact_digest and self.fact_digest != expected_digest:
            raise EconomicsConflict(
                "fact_digest does not match the immutable fact identity"
            )
        object.__setattr__(self, "fact_id", expected_id)
        object.__setattr__(self, "fact_digest", expected_digest)
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "allocation": _as_json(self.allocation),
            "source_ref": self.source_ref,
            "source_digest": self.source_digest,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "metric": self.metric,
            "quantity": self.quantity,
            "input_quantity": self.input_quantity,
            "output_quantity": self.output_quantity,
            "service_ref": self.service_ref,
            "meter_ref": self.meter_ref,
            "sample_sequence": self.sample_sequence,
            "dimensions": [_as_json(item) for item in self.dimensions],
        }


class PriceRate(ProtocolModel):
    """Price in integer micros of the card currency per base usage unit."""

    metric: MetricKind
    micros_per_unit: Annotated[StrictInt, Field(ge=0, le=10**12)]

    @field_validator("micros_per_unit")
    @classmethod
    def validate_rate(cls, value: int) -> int:
        return _ensure_nonnegative_finite(value, field_name="micros_per_unit")


class PriceCard(ProtocolModel):
    """Immutable, versioned pricing authority used by deterministic aggregation."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    card_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    card_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    provider_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    service_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    version: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    effective_from: datetime
    effective_to: datetime | None = None
    rates: Annotated[
        tuple[PriceRate, ...],
        Field(min_length=1, max_length=MAX_METRIC_KINDS),
    ]
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @field_validator("provider_ref", "service_ref", "source_ref")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _ensure_digest(value, field_name="source_digest")

    @field_validator("effective_from", "effective_to")
    @classmethod
    def validate_effective_times(
        cls, value: datetime | None, info: object
    ) -> datetime | None:
        if value is None:
            return None
        return _ensure_utc(
            value, field_name=str(getattr(info, "field_name", "effective_time"))
        )

    @model_validator(mode="after")
    def validate_immutability(self) -> PriceCard:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("price card effective_to must follow effective_from")
        metrics = [rate.metric for rate in self.rates]
        if len(metrics) != len(set(metrics)):
            raise ValueError("price card rates must contain each metric at most once")
        identity = self.identity_payload()
        expected_id = "price:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        expected_digest = content_digest({"kind": "price_card", "identity": identity})
        if self.card_id and self.card_id != expected_id:
            raise EconomicsConflict("card_id does not match immutable pricing identity")
        if self.card_digest and self.card_digest != expected_digest:
            raise EconomicsConflict(
                "card_digest does not match immutable pricing identity"
            )
        object.__setattr__(self, "card_id", expected_id)
        object.__setattr__(self, "card_digest", expected_digest)
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_ref": self.provider_ref,
            "service_ref": self.service_ref,
            "currency": self.currency,
            "version": self.version,
            "effective_from": self.effective_from.astimezone(UTC).isoformat(),
            "effective_to": self.effective_to.astimezone(UTC).isoformat()
            if self.effective_to
            else None,
            "rates": [
                _as_json(rate)
                for rate in sorted(self.rates, key=lambda item: item.metric)
            ],
            "source_ref": self.source_ref,
            "source_digest": self.source_digest,
        }


class UsageWindow(ProtocolModel):
    """UTC-aligned hourly or daily bucket."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    granularity: WindowGranularity
    window_start: datetime
    window_end: datetime | None = None
    window_id: Annotated[str, Field(min_length=1, max_length=80)] = ""

    @field_validator("window_start", "window_end")
    @classmethod
    def validate_window_times(
        cls, value: datetime | None, info: object
    ) -> datetime | None:
        if value is None:
            return None
        return _ensure_utc(
            value, field_name=str(getattr(info, "field_name", "window_time"))
        )

    @staticmethod
    def _aligned_window_end(
        start: datetime, granularity: WindowGranularity, window_end: datetime | None
    ) -> datetime:
        """Validate hour/day alignment and return the expected window end."""
        if start.minute or start.second or start.microsecond:
            raise ValueError("window_start must be aligned to an hour")
        if granularity == "day" and start.hour:
            raise ValueError("daily window_start must be aligned to UTC midnight")
        duration = timedelta(hours=1 if granularity == "hour" else 24)
        expected_end = start + duration
        if window_end is not None and window_end != expected_end:
            raise ValueError("window_end does not match granularity")
        return expected_end

    @model_validator(mode="after")
    def validate_alignment(self) -> UsageWindow:
        start = self.window_start.astimezone(UTC)
        expected_end = self._aligned_window_end(
            start, self.granularity, self.window_end
        )
        expected_id = f"window:{self.granularity}:{start.isoformat()}"
        if self.window_id and self.window_id != expected_id:
            raise EconomicsConflict("window_id does not match aligned window")
        object.__setattr__(self, "window_start", start)
        object.__setattr__(self, "window_end", expected_end)
        object.__setattr__(self, "window_id", expected_id)
        return self


class MetricTotal(ProtocolModel):
    metric: MetricKind
    quantity: Annotated[
        StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY * MAX_FACT_IDS_PER_AGGREGATE)
    ]
    cost_micros: Annotated[StrictInt, Field(ge=0, le=10**18)]

    @field_validator("quantity", "cost_micros")
    @classmethod
    def validate_totals(cls, value: int, info: object) -> int:
        return _ensure_nonnegative_finite(
            value, field_name=str(getattr(info, "field_name", "total"))
        )


class WindowAggregate(ProtocolModel):
    """Deterministic summary of a bounded set of usage facts."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    aggregate_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    aggregate_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    allocation: AllocationRef
    window: UsageWindow
    fact_ids: Annotated[
        tuple[str, ...], Field(min_length=1, max_length=MAX_FACT_IDS_PER_AGGREGATE)
    ]
    price_card_id: Annotated[str, Field(min_length=1, max_length=100)]
    price_card_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    totals: Annotated[
        tuple[MetricTotal, ...],
        Field(min_length=1, max_length=MAX_METRIC_KINDS),
    ]

    @field_validator("tenant_id", "price_card_id")
    @classmethod
    def validate_aggregate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))

    @field_validator("price_card_digest")
    @classmethod
    def validate_price_digest(cls, value: str) -> str:
        return _ensure_digest(value, field_name="price_card_digest")

    @model_validator(mode="after")
    def validate_aggregate_identity(self) -> WindowAggregate:
        if self.allocation.tenant_id != self.tenant_id:
            raise TenantScopeError("aggregate allocation crosses tenant scope")
        if len(set(self.fact_ids)) != len(self.fact_ids):
            raise ValueError("aggregate fact_ids must be unique")
        metrics = [total.metric for total in self.totals]
        if len(metrics) != len(set(metrics)):
            raise ValueError("aggregate metrics must be unique")
        identity = {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "allocation": _as_json(self.allocation),
            "window": _as_json(self.window),
            "fact_ids": sorted(self.fact_ids),
            "price_card_id": self.price_card_id,
            "price_card_digest": self.price_card_digest,
            "totals": [
                _as_json(total)
                for total in sorted(self.totals, key=lambda item: item.metric)
            ],
        }
        expected_id = (
            "aggregate:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        expected_digest = content_digest(
            {"kind": "window_aggregate", "identity": identity}
        )
        if self.aggregate_id and self.aggregate_id != expected_id:
            raise EconomicsConflict("aggregate_id does not match aggregate identity")
        if self.aggregate_digest and self.aggregate_digest != expected_digest:
            raise EconomicsConflict(
                "aggregate_digest does not match aggregate identity"
            )
        object.__setattr__(self, "aggregate_id", expected_id)
        object.__setattr__(self, "aggregate_digest", expected_digest)
        object.__setattr__(self, "fact_ids", tuple(sorted(self.fact_ids)))
        object.__setattr__(
            self, "totals", tuple(sorted(self.totals, key=lambda item: item.metric))
        )
        return self


class WatermarkPolicy(ProtocolModel):
    """Explicit late-event and correction policy; no silent data loss."""

    policy_version: Annotated[str, Field(min_length=1, max_length=64)]
    allowed_lateness_seconds: Annotated[
        StrictInt, Field(ge=0, le=MAX_LATE_EVENT_SECONDS)
    ] = 3600
    max_correction_age_seconds: Annotated[
        StrictInt, Field(ge=0, le=MAX_LATE_EVENT_SECONDS)
    ] = 86400
    late_event_mode: Literal["reject", "correction"] = "correction"
    close_delay_seconds: Annotated[
        StrictInt, Field(ge=0, le=MAX_LATE_EVENT_SECONDS)
    ] = 300

    @field_validator("policy_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _ensure_ref(value, field_name="policy_version")

    @model_validator(mode="after")
    def validate_lateness_bounds(self) -> WatermarkPolicy:
        if self.max_correction_age_seconds < self.allowed_lateness_seconds:
            raise ValueError("correction age cannot be shorter than allowed lateness")
        return self


class Watermark(ProtocolModel):
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    watermark_at: datetime
    observed_at: datetime
    policy_version: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("tenant_id", "source_ref", "policy_version")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))

    @field_validator("watermark_at", "observed_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _ensure_utc(value, field_name=str(getattr(info, "field_name", "time")))

    @model_validator(mode="after")
    def validate_order(self) -> Watermark:
        if self.observed_at < self.watermark_at:
            raise ValueError("observed_at cannot precede watermark_at")
        return self


class UsageCorrection(ProtocolModel):
    """Append-only replacement/void record for a late or corrected fact."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    correction_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    original_fact_id: Annotated[str, Field(min_length=1, max_length=80)]
    original_fact_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    replacement_fact_id: Annotated[str, Field(min_length=1, max_length=80)] | None = (
        None
    )
    replacement_fact_digest: (
        Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None
    reason: Literal[
        "late_event", "source_correction", "privacy_retraction", "reconciliation"
    ]
    created_at: datetime

    @field_validator("tenant_id", "original_fact_id", "replacement_fact_id")
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "id")))

    @field_validator("original_fact_digest", "replacement_fact_digest")
    @classmethod
    def validate_digests(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _ensure_digest(
            value, field_name=str(getattr(info, "field_name", "digest"))
        )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _ensure_utc(value, field_name="created_at")

    @model_validator(mode="after")
    def validate_replacement(self) -> UsageCorrection:
        if (self.replacement_fact_id is None) != (self.replacement_fact_digest is None):
            raise ValueError("replacement fact id and digest must be supplied together")
        if self.replacement_fact_id == self.original_fact_id:
            raise ValueError("a correction cannot replace a fact with itself")
        identity = {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "original_fact_id": self.original_fact_id,
            "original_fact_digest": self.original_fact_digest,
            "replacement_fact_id": self.replacement_fact_id,
            "replacement_fact_digest": self.replacement_fact_digest,
            "reason": self.reason,
        }
        expected = "correction:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        if self.correction_id and self.correction_id != expected:
            raise EconomicsConflict(
                "correction_id does not match immutable correction identity"
            )
        object.__setattr__(self, "correction_id", expected)
        return self


class RetentionPolicy(ProtocolModel):
    """Retention is explicit and produces tombstones instead of hidden deletes."""

    policy_version: Annotated[str, Field(min_length=1, max_length=64)]
    usage_retention_seconds: Annotated[StrictInt, Field(gt=0, le=MAX_RETENTION_SECONDS)]
    aggregate_retention_seconds: Annotated[
        StrictInt, Field(gt=0, le=MAX_RETENTION_SECONDS)
    ]
    tombstone_after_seconds: Annotated[StrictInt, Field(gt=0, le=MAX_RETENTION_SECONDS)]
    legal_hold: StrictBool = False

    @field_validator("policy_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _ensure_ref(value, field_name="policy_version")


class UsageTombstone(ProtocolModel):
    """Auditable deletion marker for a fact or aggregate."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    tombstone_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    target_kind: Literal["usage_fact", "window_aggregate", "slo_rollup"]
    target_id: Annotated[str, Field(min_length=1, max_length=100)]
    target_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    reason: Literal["retention", "privacy_retraction", "legal_request"]
    effective_at: datetime
    policy_version: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("tenant_id", "target_id", "policy_version")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))

    @field_validator("target_digest")
    @classmethod
    def validate_target_digest(cls, value: str) -> str:
        return _ensure_digest(value, field_name="target_digest")

    @field_validator("effective_at")
    @classmethod
    def validate_effective_at(cls, value: datetime) -> datetime:
        return _ensure_utc(value, field_name="effective_at")

    @model_validator(mode="after")
    def validate_identity(self) -> UsageTombstone:
        identity = {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "target_digest": self.target_digest,
            "reason": self.reason,
            "effective_at": self.effective_at.astimezone(UTC).isoformat(),
            "policy_version": self.policy_version,
        }
        expected = "tombstone:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        if self.tombstone_id and self.tombstone_id != expected:
            raise EconomicsConflict("tombstone_id does not match immutable identity")
        object.__setattr__(self, "tombstone_id", expected)
        return self


class SloObjective(ProtocolModel):
    """Versioned SLO objective; a new target is a new objective version."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    objective_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    service_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    sli: SliKind
    window_granularity: WindowGranularity
    version: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    target_micros: Annotated[StrictInt, Field(ge=0, le=1_000_000)]
    comparison: Literal["gte", "lte"]
    latency_threshold_ms: Annotated[StrictInt, Field(gt=0, le=86_400_000)] | None = None
    percentile_micros: Annotated[StrictInt, Field(ge=0, le=1_000_000)] | None = None

    @field_validator("tenant_id", "service_ref")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))

    def _check_target_shape(self) -> None:
        """SLI-dependent shape/comparison invariants for ``validate_target``."""
        if self.sli == "latency_ms" and self.latency_threshold_ms is None:
            raise ValueError("latency SLO requires latency_threshold_ms")
        if self.sli != "latency_ms" and self.latency_threshold_ms is not None:
            raise ValueError("latency_threshold_ms is valid only for latency SLOs")
        if self.percentile_micros is not None and self.sli != "latency_ms":
            raise ValueError("percentile is valid only for latency SLOs")
        expected_comparison = (
            "gte" if self.sli in {"availability", "throughput"} else "lte"
        )
        if self.comparison != expected_comparison:
            raise ValueError(
                f"{self.sli} SLOs must use {expected_comparison} comparison"
            )

    @model_validator(mode="after")
    def validate_target(self) -> SloObjective:
        self._check_target_shape()
        identity = {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "service_ref": self.service_ref,
            "sli": self.sli,
            "window_granularity": self.window_granularity,
            "version": self.version,
            "target_micros": self.target_micros,
            "comparison": self.comparison,
            "latency_threshold_ms": self.latency_threshold_ms,
            "percentile_micros": self.percentile_micros,
        }
        expected = "slo:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        if self.objective_id and self.objective_id != expected:
            raise EconomicsConflict(
                "objective_id does not match immutable SLO identity"
            )
        object.__setattr__(self, "objective_id", expected)
        return self


class SloWindowRollup(ProtocolModel):
    """Bounded SLO window summary with no raw spans or logs."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    rollup_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    rollup_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    objective_id: Annotated[str, Field(min_length=1, max_length=100)]
    window: UsageWindow
    good_events: Annotated[
        StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY * MAX_FACT_IDS_PER_AGGREGATE)
    ]
    total_events: Annotated[
        StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY * MAX_FACT_IDS_PER_AGGREGATE)
    ]
    bad_events: Annotated[
        StrictInt, Field(ge=0, le=MAX_FACT_QUANTITY * MAX_FACT_IDS_PER_AGGREGATE)
    ]
    measured_micros: Annotated[StrictInt, Field(ge=0, le=1_000_000)] | None = None
    source_fact_digests: Annotated[
        tuple[str, ...], Field(max_length=MAX_FACT_IDS_PER_AGGREGATE)
    ] = ()
    status: Literal["met", "breached", "insufficient_data"]

    @field_validator("tenant_id", "objective_id")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "id")))

    @field_validator("source_fact_digests")
    @classmethod
    def validate_source_digests(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _ensure_digest(value, field_name="source_fact_digest")
        if len(values) != len(set(values)):
            raise ValueError("source fact digests must be unique")
        return tuple(sorted(values))

    def _check_rollup_status(self) -> None:
        """Event-count/status consistency invariants for ``validate_rollup``."""
        if self.good_events > self.total_events or self.bad_events > self.total_events:
            raise ValueError("SLO good/bad events cannot exceed total events")
        if self.total_events == 0:
            if self.status != "insufficient_data" or self.measured_micros is not None:
                raise ValueError(
                    "empty SLO rollup must be explicitly insufficient_data"
                )
        elif self.status == "insufficient_data" or self.measured_micros is None:
            raise ValueError("populated SLO rollup requires a bounded measurement")

    @model_validator(mode="after")
    def validate_rollup(self) -> SloWindowRollup:
        self._check_rollup_status()
        identity = {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "objective_id": self.objective_id,
            "window": _as_json(self.window),
            "good_events": self.good_events,
            "total_events": self.total_events,
            "bad_events": self.bad_events,
            "measured_micros": self.measured_micros,
            "source_fact_digests": sorted(self.source_fact_digests),
            "status": self.status,
        }
        expected_id = (
            "slo-rollup:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        expected_digest = content_digest({"kind": "slo_rollup", "identity": identity})
        if self.rollup_id and self.rollup_id != expected_id:
            raise EconomicsConflict(
                "rollup_id does not match immutable rollup identity"
            )
        if self.rollup_digest and self.rollup_digest != expected_digest:
            raise EconomicsConflict(
                "rollup_digest does not match immutable rollup identity"
            )
        object.__setattr__(self, "rollup_id", expected_id)
        object.__setattr__(self, "rollup_digest", expected_digest)
        return self


class TenantReadScope(ProtocolModel):
    """The tenant boundary is part of every read-model request and cursor."""

    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    principal_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]

    @field_validator("tenant_id", "principal_ref")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "ref")))


class KeysetCursor(ProtocolModel):
    """Opaque cursor bound to tenant and query shape; offset pagination is forbidden."""

    tenant_id: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    query_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    last_key: Annotated[str, Field(min_length=1, max_length=256)]

    @field_validator("tenant_id", "last_key")
    @classmethod
    def validate_cursor_refs(cls, value: str, info: object) -> str:
        return _ensure_ref(value, field_name=str(getattr(info, "field_name", "cursor")))

    @field_validator("query_digest")
    @classmethod
    def validate_query_digest(cls, value: str) -> str:
        return _ensure_digest(value, field_name="query_digest")


class UsageReadRequest(ProtocolModel):
    scope: TenantReadScope
    granularity: WindowGranularity | None = None
    allocation_project_ref: (
        Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)] | None
    ) = None
    allocation_cost_center_ref: (
        Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)] | None
    ) = None
    limit: Annotated[StrictInt, Field(gt=0, le=MAX_PAGE_SIZE)] = 100
    after: KeysetCursor | None = None

    @field_validator("allocation_project_ref", "allocation_cost_center_ref")
    @classmethod
    def validate_allocation_filters(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _ensure_ref(
            value, field_name=str(getattr(info, "field_name", "allocation"))
        )

    @model_validator(mode="after")
    def validate_cursor_scope(self) -> UsageReadRequest:
        if self.after is not None and self.after.tenant_id != self.scope.tenant_id:
            raise TenantScopeError("cursor tenant does not match read scope")
        return self

    def query_digest(self) -> str:
        return content_digest(
            {
                "tenant_id": self.scope.tenant_id,
                "principal_ref": self.scope.principal_ref,
                "granularity": self.granularity,
                "project_ref": self.allocation_project_ref,
                "cost_center_ref": self.allocation_cost_center_ref,
            }
        )


class UsageReadPage(ProtocolModel):
    scope: TenantReadScope
    rows: Annotated[tuple[WindowAggregate, ...], Field(max_length=MAX_PAGE_SIZE)]
    next_cursor: KeysetCursor | None = None
    exhausted: StrictBool

    @model_validator(mode="after")
    def validate_page(self) -> UsageReadPage:
        if any(row.tenant_id != self.scope.tenant_id for row in self.rows):
            raise TenantScopeError("usage read page contains a foreign tenant row")
        if (
            self.next_cursor is not None
            and self.next_cursor.tenant_id != self.scope.tenant_id
        ):
            raise TenantScopeError("next cursor contains a foreign tenant")
        return self


class SloReadRequest(ProtocolModel):
    scope: TenantReadScope
    objective_id: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    limit: Annotated[StrictInt, Field(gt=0, le=MAX_PAGE_SIZE)] = 100
    after: KeysetCursor | None = None

    @field_validator("objective_id")
    @classmethod
    def validate_objective_id(cls, value: str | None) -> str | None:
        return (
            _ensure_ref(value, field_name="objective_id") if value is not None else None
        )

    @model_validator(mode="after")
    def validate_cursor_scope(self) -> SloReadRequest:
        if self.after is not None and self.after.tenant_id != self.scope.tenant_id:
            raise TenantScopeError("cursor tenant does not match SLO read scope")
        return self

    def query_digest(self) -> str:
        return content_digest(
            {
                "tenant_id": self.scope.tenant_id,
                "principal_ref": self.scope.principal_ref,
                "objective_id": self.objective_id,
            }
        )


class SloReadPage(ProtocolModel):
    scope: TenantReadScope
    rows: Annotated[tuple[SloWindowRollup, ...], Field(max_length=MAX_PAGE_SIZE)]
    next_cursor: KeysetCursor | None = None
    exhausted: StrictBool

    @model_validator(mode="after")
    def validate_page(self) -> SloReadPage:
        if any(row.tenant_id != self.scope.tenant_id for row in self.rows):
            raise TenantScopeError("SLO read page contains a foreign tenant row")
        if (
            self.next_cursor is not None
            and self.next_cursor.tenant_id != self.scope.tenant_id
        ):
            raise TenantScopeError("next cursor contains a foreign tenant")
        return self


class SampleRef(ProtocolModel):
    """Reference to an expected collector sample; raw sample content is external."""

    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    sample_sequence: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)] | None = None

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        return _ensure_ref(value, field_name="source_ref")

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _ensure_digest(value, field_name="source_digest")


class ReconciliationReport(ProtocolModel):
    """Deterministic gap/drift result for a bounded sample manifest."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    report_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    expected_count: Annotated[StrictInt, Field(ge=0, le=MAX_RECONCILIATION_ITEMS)]
    observed_count: Annotated[StrictInt, Field(ge=0, le=MAX_RECONCILIATION_ITEMS)]
    missing: Annotated[
        tuple[SampleRef, ...], Field(max_length=MAX_RECONCILIATION_ITEMS)
    ] = ()
    unexpected: Annotated[
        tuple[SampleRef, ...], Field(max_length=MAX_RECONCILIATION_ITEMS)
    ] = ()
    duplicate_samples: Annotated[
        tuple[SampleRef, ...], Field(max_length=MAX_RECONCILIATION_ITEMS)
    ] = ()
    duplicate_fact_ids: Annotated[
        tuple[str, ...], Field(max_length=MAX_RECONCILIATION_ITEMS)
    ] = ()
    status: Literal["complete", "gap", "drift"]

    @field_validator("duplicate_fact_ids")
    @classmethod
    def validate_duplicate_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _ensure_ref(value, field_name="duplicate_fact_id")
        return tuple(sorted(set(values)))

    @field_validator("duplicate_samples")
    @classmethod
    def validate_duplicate_samples(
        cls, values: tuple[SampleRef, ...]
    ) -> tuple[SampleRef, ...]:
        keys = [
            (item.source_ref, item.source_digest, item.sample_sequence)
            for item in values
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate sample references must be unique")
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    item.source_ref,
                    item.source_digest,
                    item.sample_sequence or -1,
                ),
            )
        )

    def _check_complete_shape(self) -> None:
        if (
            self.missing
            or self.unexpected
            or self.duplicate_samples
            or self.duplicate_fact_ids
        ):
            raise ValueError("complete reconciliation cannot contain gaps or drift")

    def _check_gap_shape(self) -> None:
        if not self.missing:
            raise ValueError("gap reconciliation must expose missing samples")

    def _check_drift_shape(self) -> None:
        if not (self.unexpected or self.duplicate_samples or self.duplicate_fact_ids):
            raise ValueError(
                "drift reconciliation must expose unexpected or duplicate samples"
            )

    def _check_status_shape(self) -> None:
        """Status/content consistency invariants for ``validate_report``."""
        if self.status == "complete":
            self._check_complete_shape()
        elif self.status == "gap":
            self._check_gap_shape()
        elif self.status == "drift":
            self._check_drift_shape()

    @model_validator(mode="after")
    def validate_report(self) -> ReconciliationReport:
        self._check_status_shape()
        identity = {
            "schema_version": self.schema_version,
            "expected_count": self.expected_count,
            "observed_count": self.observed_count,
            "missing": [_as_json(item) for item in self.missing],
            "unexpected": [_as_json(item) for item in self.unexpected],
            "duplicate_samples": [_as_json(item) for item in self.duplicate_samples],
            "duplicate_fact_ids": sorted(self.duplicate_fact_ids),
            "status": self.status,
        }
        expected_digest = content_digest(
            {"kind": "sample_reconciliation", "identity": identity}
        )
        if self.report_digest and self.report_digest != expected_digest:
            raise EconomicsConflict(
                "report_digest does not match reconciliation identity"
            )
        object.__setattr__(self, "report_digest", expected_digest)
        return self


class LateEventDecision(ProtocolModel):
    """Explicit admission result for an event relative to a watermark."""

    fact_id: Annotated[str, Field(min_length=1, max_length=80)]
    status: Literal["accepted", "correction_required", "rejected"]
    reason: Literal[
        "on_time", "within_lateness", "outside_lateness", "correction_window_expired"
    ]
    watermark_at: datetime

    @field_validator("fact_id")
    @classmethod
    def validate_fact_id(cls, value: str) -> str:
        return _ensure_ref(value, field_name="fact_id")

    @field_validator("watermark_at")
    @classmethod
    def validate_watermark_at(cls, value: datetime) -> datetime:
        return _ensure_utc(value, field_name="watermark_at")


__all__ = [
    "AllocationRef",
    "DimensionRef",
    "EconomicsConflict",
    "EconomicsContractError",
    "KeysetCursor",
    "LateEventDecision",
    "LateEventRejected",
    "MetricKind",
    "MetricTotal",
    "PriceCard",
    "PriceRate",
    "ReconciliationReport",
    "RetentionPolicy",
    "SampleRef",
    "SliKind",
    "SloObjective",
    "SloReadPage",
    "SloReadRequest",
    "SloWindowRollup",
    "TenantReadScope",
    "UsageCorrection",
    "UsageFact",
    "UsageReadPage",
    "UsageReadRequest",
    "UsageTombstone",
    "UsageWindow",
    "Watermark",
    "WatermarkPolicy",
    "WindowAggregate",
    "WindowGranularity",
    "content_digest",
]
