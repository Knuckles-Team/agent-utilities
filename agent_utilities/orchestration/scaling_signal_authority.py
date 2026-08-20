#!/usr/bin/python
from __future__ import annotations

"""Authenticated, bounded signal reads for the scaling authority.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling

This module is the signal-side boundary for NE-165.  A metrics backend may
collect high-rate samples, but the scaling authority accepts only an
authenticated, scope-bound, fresh, finite sample page.  Samples stay in the
metrics path; a caller explicitly derives a bounded :class:`SignalSummary`
and decision evidence before anything can cross the graph boundary.

The provider API is deliberately symbolic.  It accepts a bounded query and
opaque cursor supplied by a trusted backend adapter, not an untrusted raw
PromQL client.  Backend adapters must stream deterministic pages for a fixed
snapshot and must not materialize a million-resource catalog in the provider.
"""

import hashlib
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Literal, Protocol, runtime_checkable

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

# Global hard bounds.  Deployments may choose stricter policy values, never
# looser ones.  These limits keep the provider safe for million-resource
# catalogs without allocating a fleet-sized result in one read.
MAX_QUERY_LENGTH = 4_096
MAX_QUERY_TIMEOUT_S = 10.0
MAX_QUERY_WINDOW_S = 86_400.0
MAX_PAGE_SIZE = 256
MAX_BATCH_ITEMS = 1_024
MAX_SIGNAL_KINDS_PER_QUERY = 16
MAX_SAMPLE_FRESHNESS_S = 300.0
MAX_FUTURE_SKEW_S = 30.0
MAX_CURSOR_LENGTH = 256
MAX_REFERENCE_LENGTH = 256

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CURSOR_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,256}$")
_SENSITIVE_QUERY_RE = re.compile(
    r"(?:authorization|bearer\s|password\s*=|secret\s*=|token\s*=|"
    r"api[_-]?key\s*=|private[_-]?key\s*=|credential\s*=)",
    re.IGNORECASE,
)


class SignalKind(StrEnum):
    """Finite normalized vocabulary shared by signal adapters."""

    CPU = "cpu"
    RAM = "ram"
    GPU = "gpu"
    KV_CACHE = "kv_cache"
    DISK = "disk"
    NETWORK = "network"
    TOKEN = "token"
    REQUEST = "request"
    LATENCY = "latency"
    QUEUE = "queue"
    SHARD = "shard"


class SignalAggregation(StrEnum):
    """Allowed aggregation semantics; adapters may not invent operators."""

    LAST = "last"
    MEAN = "mean"
    MAX = "max"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"
    SUM = "sum"
    COUNT = "count"


class SignalSourceKind(StrEnum):
    """Known source families; source identity is separately scope-bound."""

    PROMETHEUS = "prometheus"
    TELEMETRY = "telemetry"
    ENGINE = "engine"
    LOCAL = "local"


class _SignalModel(BaseModel):
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


def _reference(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) == 0
        or len(value) > MAX_REFERENCE_LENGTH
    ):
        raise ValueError(f"{field_name} must be a bounded opaque reference")
    if any(char.isspace() or ord(char) < 0x20 for char in value):
        raise ValueError(f"{field_name} must not contain whitespace or controls")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    lowered = value.lower()
    if any(
        secret in lowered for secret in ("bearer", "token=", "secret=", "password=")
    ):
        raise ValueError(f"{field_name} must not contain credentials")
    return value


def _digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase digest")
    return value


def _aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _finite_non_negative(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


def query_digest(query: str) -> str:
    """Return the deterministic digest that binds a read to its exact query."""

    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("signal query is required")
    if len(query.encode("utf-8")) > MAX_QUERY_LENGTH:
        raise ValueError("signal query exceeds the bounded query length")
    if any(ord(char) < 0x20 and char not in "\t" for char in query):
        raise ValueError("signal query contains a control character")
    if _SENSITIVE_QUERY_RE.search(query):
        raise ValueError("signal query contains credential material")
    return query


class SignalSourceBinding(_SignalModel):
    """Authenticated source identity and the exact scope it may report."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    source_id: str
    source_kind: SignalSourceKind
    service_scope: str
    tenant_scope: str
    unit_id: str
    source_epoch: StrictInt = Field(ge=1)
    auth_evidence_ref: str

    _validate_source = field_validator("source_id")(
        lambda value: _identifier(value, "source_id")
    )
    _validate_service = field_validator("service_scope")(
        lambda value: _identifier(value, "service_scope")
    )
    _validate_tenant = field_validator("tenant_scope")(
        lambda value: _identifier(value, "tenant_scope")
    )
    _validate_unit = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_evidence = field_validator("auth_evidence_ref")(
        lambda value: _reference(value, "auth_evidence_ref")
    )


class SignalReadPolicy(_SignalModel):
    """Deployment caps; every value is bounded by the global hard maximum."""

    max_query_length: StrictInt = Field(
        default=MAX_QUERY_LENGTH, ge=1, le=MAX_QUERY_LENGTH
    )
    max_timeout_s: float = Field(
        default=MAX_QUERY_TIMEOUT_S, gt=0, le=MAX_QUERY_TIMEOUT_S
    )
    max_window_s: float = Field(default=MAX_QUERY_WINDOW_S, gt=0, le=MAX_QUERY_WINDOW_S)
    max_page_size: StrictInt = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    max_batch_items: StrictInt = Field(
        default=MAX_BATCH_ITEMS, ge=1, le=MAX_BATCH_ITEMS
    )
    max_signal_kinds: StrictInt = Field(
        default=MAX_SIGNAL_KINDS_PER_QUERY,
        ge=1,
        le=MAX_SIGNAL_KINDS_PER_QUERY,
    )
    max_freshness_s: float = Field(
        default=MAX_SAMPLE_FRESHNESS_S,
        gt=0,
        le=MAX_SAMPLE_FRESHNESS_S,
    )
    max_future_skew_s: float = Field(
        default=MAX_FUTURE_SKEW_S,
        ge=0,
        le=MAX_FUTURE_SKEW_S,
    )

    @field_validator(
        "max_timeout_s",
        "max_window_s",
        "max_freshness_s",
        "max_future_skew_s",
    )
    @classmethod
    def finite_caps(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("signal policy caps must be finite")
        return value


class SignalReadRequest(_SignalModel):
    """Bounded, scope-bound batch read; cursor pagination is snapshot based."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    request_id: str
    source_id: str
    service_scope: str
    tenant_scope: str
    unit_id: str
    signals: tuple[SignalKind, ...]
    aggregation: SignalAggregation
    query: str
    query_digest: str
    window_start: datetime
    window_end: datetime
    timeout_s: float = Field(default=MAX_QUERY_TIMEOUT_S, gt=0, le=MAX_QUERY_TIMEOUT_S)
    page_size: StrictInt = Field(default=MAX_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    max_items: StrictInt = Field(default=MAX_BATCH_ITEMS, ge=1, le=MAX_BATCH_ITEMS)
    cursor: str | None = None
    snapshot_digest: str | None = None

    _validate_request = field_validator("request_id")(
        lambda value: _identifier(value, "request_id")
    )
    _validate_source = field_validator("source_id")(
        lambda value: _identifier(value, "source_id")
    )
    _validate_service = field_validator("service_scope")(
        lambda value: _identifier(value, "service_scope")
    )
    _validate_tenant = field_validator("tenant_scope")(
        lambda value: _identifier(value, "tenant_scope")
    )
    _validate_unit = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_query = field_validator("query")(_validate_query)
    _validate_query_digest = field_validator("query_digest")(
        lambda value: _digest(value, "query_digest")
    )
    _validate_window_start = field_validator("window_start")(
        lambda value: _aware_datetime(value, "window_start")
    )
    _validate_window_end = field_validator("window_end")(
        lambda value: _aware_datetime(value, "window_end")
    )
    _validate_cursor = field_validator("cursor")(
        lambda value: None if value is None else _validate_cursor_value(value)
    )
    _validate_snapshot = field_validator("snapshot_digest")(
        lambda value: None if value is None else _digest(value, "snapshot_digest")
    )

    @model_validator(mode="after")
    def validate_request(self) -> SignalReadRequest:
        if self.query_digest != query_digest(self.query):
            raise ValueError("query_digest does not match the exact query")
        if len(self.signals) == 0 or len(self.signals) > MAX_SIGNAL_KINDS_PER_QUERY:
            raise ValueError("signal batch vocabulary is out of bounds")
        if len(set(self.signals)) != len(self.signals):
            raise ValueError("signal batch contains duplicate signal kinds")
        if self.window_end <= self.window_start:
            raise ValueError("signal window_end must be after window_start")
        if self.window_end - self.window_start > timedelta(seconds=MAX_QUERY_WINDOW_S):
            raise ValueError("signal query window exceeds the bounded duration")
        if self.page_size > self.max_items:
            raise ValueError("signal page_size cannot exceed max_items")
        return self


def _validate_cursor_value(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_CURSOR_LENGTH:
        raise ValueError("signal cursor exceeds the bounded length")
    if _CURSOR_RE.fullmatch(value) is None:
        raise ValueError("signal cursor contains unsupported characters")
    return value


class SignalSample(_SignalModel):
    """One authenticated observation; invalid values never become no-data zero."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    sample_id: str
    sample_digest: str
    source_id: str
    source_kind: SignalSourceKind
    source_epoch: StrictInt = Field(ge=1)
    auth_evidence_ref: str
    service_scope: str
    tenant_scope: str
    unit_id: str
    signal: SignalKind
    aggregation: SignalAggregation
    value: float
    sample_time: datetime
    event_time: datetime
    freshness_s: float = Field(ge=0, le=MAX_SAMPLE_FRESHNESS_S)
    confidence: float = Field(ge=0, le=1)
    evidence_ref: str
    sequence: StrictInt = Field(ge=1)
    query_digest: str

    _validate_sample = field_validator("sample_id")(
        lambda value: _identifier(value, "sample_id")
    )
    _validate_sample_digest = field_validator("sample_digest")(
        lambda value: _digest(value, "sample_digest")
    )
    _validate_source = field_validator("source_id")(
        lambda value: _identifier(value, "source_id")
    )
    _validate_auth_ref = field_validator("auth_evidence_ref")(
        lambda value: _reference(value, "auth_evidence_ref")
    )
    _validate_service = field_validator("service_scope")(
        lambda value: _identifier(value, "service_scope")
    )
    _validate_tenant = field_validator("tenant_scope")(
        lambda value: _identifier(value, "tenant_scope")
    )
    _validate_unit = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_evidence = field_validator("evidence_ref")(
        lambda value: _reference(value, "evidence_ref")
    )
    _validate_query_digest = field_validator("query_digest")(
        lambda value: _digest(value, "query_digest")
    )
    _validate_sample_time = field_validator("sample_time")(
        lambda value: _aware_datetime(value, "sample_time")
    )
    _validate_event_time = field_validator("event_time")(
        lambda value: _aware_datetime(value, "event_time")
    )

    @field_validator("value", mode="before")
    @classmethod
    def finite_value(cls, value: float) -> float:
        return _finite_non_negative(value, "signal value")

    @field_validator("freshness_s", "confidence", mode="before")
    @classmethod
    def finite_metadata(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("signal freshness/confidence must be finite")
        return value

    @model_validator(mode="after")
    def validate_event_order(self) -> SignalSample:
        if self.event_time > self.sample_time:
            raise ValueError("event_time cannot be after sample_time")
        return self


class SignalBackendPage:
    """Streaming backend response; ``samples`` must not be pre-materialized."""

    __slots__ = ("samples", "next_cursor", "has_more", "snapshot_digest")

    def __init__(
        self,
        samples: Iterable[SignalSample],
        *,
        next_cursor: str | None,
        has_more: bool,
        snapshot_digest: str,
    ) -> None:
        self.samples = samples
        self.next_cursor = next_cursor
        self.has_more = has_more
        self.snapshot_digest = snapshot_digest


class SignalPage(_SignalModel):
    """Bounded authenticated page with a stable snapshot cursor."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    request_id: str
    source_id: str
    snapshot_digest: str
    samples: tuple[SignalSample, ...]
    next_cursor: str | None = None
    has_more: StrictBool = False

    _validate_request = field_validator("request_id")(
        lambda value: _identifier(value, "request_id")
    )
    _validate_source = field_validator("source_id")(
        lambda value: _identifier(value, "source_id")
    )
    _validate_snapshot = field_validator("snapshot_digest")(
        lambda value: _digest(value, "snapshot_digest")
    )
    _validate_cursor = field_validator("next_cursor")(
        lambda value: None if value is None else _validate_cursor_value(value)
    )

    @model_validator(mode="after")
    def validate_page(self) -> SignalPage:
        if self.has_more != (self.next_cursor is not None):
            raise ValueError("has_more and next_cursor must agree")
        if len(self.samples) > MAX_PAGE_SIZE:
            raise ValueError("signal page exceeds the bounded cardinality")
        return self

    def graph_projection(self) -> dict[str, Any]:
        """Expose page metadata only; samples remain outside graph mutations."""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "source_id": self.source_id,
            "snapshot_digest": self.snapshot_digest,
            "sample_count": len(self.samples),
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }


class SignalSummary(_SignalModel):
    """Bounded aggregate suitable for conversion to NE-164 evidence."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    summary_id: str
    summary_digest: str
    source_id: str
    source_kind: SignalSourceKind
    service_scope: str
    tenant_scope: str
    unit_id: str
    signal: SignalKind
    aggregation: SignalAggregation
    value: float
    sample_count: StrictInt = Field(ge=1, le=MAX_BATCH_ITEMS)
    freshness_s: float = Field(ge=0, le=MAX_SAMPLE_FRESHNESS_S)
    confidence: float = Field(ge=0, le=1)
    evidence_ref: str
    query_digest: str
    window_start: datetime
    window_end: datetime

    _validate_summary = field_validator("summary_id")(
        lambda value: _identifier(value, "summary_id")
    )
    _validate_digest = field_validator("summary_digest")(
        lambda value: _digest(value, "summary_digest")
    )
    _validate_source = field_validator("source_id")(
        lambda value: _identifier(value, "source_id")
    )
    _validate_service = field_validator("service_scope")(
        lambda value: _identifier(value, "service_scope")
    )
    _validate_tenant = field_validator("tenant_scope")(
        lambda value: _identifier(value, "tenant_scope")
    )
    _validate_unit = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_evidence = field_validator("evidence_ref")(
        lambda value: _reference(value, "evidence_ref")
    )
    _validate_query_digest = field_validator("query_digest")(
        lambda value: _digest(value, "query_digest")
    )
    _validate_window_start = field_validator("window_start")(
        lambda value: _aware_datetime(value, "window_start")
    )
    _validate_window_end = field_validator("window_end")(
        lambda value: _aware_datetime(value, "window_end")
    )

    @field_validator("value", mode="before")
    @classmethod
    def finite_summary_value(cls, value: float) -> float:
        return _finite_non_negative(value, "summary value")

    @field_validator("freshness_s", "confidence", mode="before")
    @classmethod
    def finite_summary_metadata(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("summary freshness/confidence must be finite")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> SignalSummary:
        if self.window_end <= self.window_start:
            raise ValueError("summary window_end must be after window_start")
        return self

    def graph_projection(self) -> dict[str, Any]:
        """Bounded graph evidence; exact query text and raw samples stay out."""

        return {
            "schema_version": self.schema_version,
            "summary_id": self.summary_id,
            "summary_digest": self.summary_digest,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "service_scope": self.service_scope,
            "tenant_scope": self.tenant_scope,
            "unit_id": self.unit_id,
            "signal": self.signal,
            "aggregation": self.aggregation,
            "sample_count": self.sample_count,
            "freshness_s": self.freshness_s,
            "confidence": self.confidence,
            "evidence_ref": self.evidence_ref,
            "query_digest": self.query_digest,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
        }


class SignalDecisionEvidence(_SignalModel):
    """Decision-only binding that may retain the exact query text."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    request_id: str
    query: str
    query_digest: str
    summary_digest: str
    snapshot_digest: str
    window_start: datetime
    window_end: datetime
    sample_count: StrictInt = Field(ge=1, le=MAX_BATCH_ITEMS)

    _validate_request = field_validator("request_id")(
        lambda value: _identifier(value, "request_id")
    )
    _validate_query = field_validator("query")(_validate_query)
    _validate_query_digest = field_validator("query_digest")(
        lambda value: _digest(value, "query_digest")
    )
    _validate_summary_digest = field_validator("summary_digest")(
        lambda value: _digest(value, "summary_digest")
    )
    _validate_snapshot_digest = field_validator("snapshot_digest")(
        lambda value: _digest(value, "snapshot_digest")
    )
    _validate_window_start = field_validator("window_start")(
        lambda value: _aware_datetime(value, "window_start")
    )
    _validate_window_end = field_validator("window_end")(
        lambda value: _aware_datetime(value, "window_end")
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> SignalDecisionEvidence:
        if self.query_digest != query_digest(self.query):
            raise ValueError("decision query_digest does not match exact query")
        if self.window_end <= self.window_start:
            raise ValueError("decision window_end must be after window_start")
        return self


@runtime_checkable
class SignalBackend(Protocol):
    """Streaming read seam implemented by a trusted metrics adapter."""

    def read(self, request: SignalReadRequest) -> SignalBackendPage:
        """Return one bounded deterministic page for the request snapshot."""
        ...  # ABSTRACT-OK


@runtime_checkable
class SignalAuthenticator(Protocol):
    """Deployment-injected source attestation verifier."""

    def verify(self, sample: SignalSample, binding: SignalSourceBinding) -> bool:
        """Return true only when the source attests to this exact sample."""
        ...  # ABSTRACT-OK


class OpaqueReferenceAuthenticator:
    """Minimal fixture authenticator; deployments should inject cryptographic verification."""

    def verify(self, sample: SignalSample, binding: SignalSourceBinding) -> bool:
        return sample.auth_evidence_ref == binding.auth_evidence_ref


class ReplayGuard:
    """Bounded source sequence/digest guard; it never stores a fleet of samples."""

    def __init__(self, *, max_entries: int = MAX_BATCH_ITEMS * 4) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise ValueError("replay guard max_entries must be an integer")
        if max_entries < 1 or max_entries > MAX_BATCH_ITEMS * 16:
            raise ValueError("replay guard max_entries is out of bounds")
        self._max_entries = max_entries
        self._highest_sequence: dict[tuple[str, int], int] = {}
        self._sample_ids: OrderedDict[tuple[str, int, str], None] = OrderedDict()
        self._digests: OrderedDict[tuple[str, int, str], None] = OrderedDict()
        self._lock = threading.Lock()

    def accept(self, samples: Sequence[SignalSample]) -> bool:
        """Atomically accept a fresh batch or reject it in its entirety."""

        if len(samples) > MAX_BATCH_ITEMS:
            return False
        with self._lock:
            pending_highest: dict[tuple[str, int], int] = {}
            pending_ids: set[tuple[str, int, str]] = set()
            pending_digests: set[tuple[str, int, str]] = set()
            for sample in samples:
                source_key = (sample.source_id, sample.source_epoch)
                sample_key = (*source_key, sample.sample_id)
                digest_key = (*source_key, sample.sample_digest)
                current = pending_highest.get(
                    source_key, self._highest_sequence.get(source_key, 0)
                )
                if sample.sequence <= current:
                    return False
                if (
                    sample_key in self._sample_ids
                    or sample_key in pending_ids
                    or digest_key in self._digests
                    or digest_key in pending_digests
                ):
                    return False
                pending_highest[source_key] = sample.sequence
                pending_ids.add(sample_key)
                pending_digests.add(digest_key)
            for source_key, sequence in pending_highest.items():
                self._highest_sequence[source_key] = sequence
            for sample_key in pending_ids:
                self._sample_ids[sample_key] = None
                self._sample_ids.move_to_end(sample_key)
            for digest_key in pending_digests:
                self._digests[digest_key] = None
                self._digests.move_to_end(digest_key)
            while len(self._sample_ids) > self._max_entries:
                self._sample_ids.popitem(last=False)
            while len(self._digests) > self._max_entries:
                self._digests.popitem(last=False)
            return True


class AuthenticatedSignalProvider:
    """Fail-closed, scope-bound provider over a streaming signal backend."""

    name = "authenticated-signal"

    def __init__(
        self,
        backend: SignalBackend,
        binding: SignalSourceBinding,
        *,
        policy: SignalReadPolicy | None = None,
        authenticator: SignalAuthenticator | None = None,
        replay_guard: ReplayGuard | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(backend, SignalBackend):
            raise TypeError("signal backend must implement the streaming contract")
        self._backend = backend
        self._binding = binding
        self._policy = policy or SignalReadPolicy()
        self._authenticator = authenticator or OpaqueReferenceAuthenticator()
        if not isinstance(self._authenticator, SignalAuthenticator):
            raise TypeError("signal authenticator must implement the verifier contract")
        self._replay_guard = replay_guard or ReplayGuard()
        self._clock = clock or (lambda: datetime.now(UTC))

    def read_batch(self, request: SignalReadRequest) -> SignalPage:
        """Read one bounded page; malformed or unauthorized pages fail closed."""

        self._validate_request_scope(request)
        try:
            backend_page = self._backend.read(request)
        except Exception:
            # A backend outage is no data, never zero and never a scale-down.
            raise
        if not isinstance(backend_page, SignalBackendPage):
            raise ValueError("signal backend returned an invalid page")
        snapshot_digest = _digest(backend_page.snapshot_digest, "snapshot_digest")
        if (
            request.snapshot_digest is not None
            and snapshot_digest != request.snapshot_digest
        ):
            raise ValueError("signal backend changed snapshots during pagination")
        if backend_page.has_more != (backend_page.next_cursor is not None):
            raise ValueError("backend page cursor/has_more contract is invalid")
        if backend_page.next_cursor is not None:
            _validate_cursor_value(backend_page.next_cursor)

        samples: list[SignalSample] = []
        for sample in backend_page.samples:
            if len(samples) >= request.page_size:
                raise ValueError("signal backend exceeded page cardinality")
            self._validate_sample(sample, request)
            samples.append(sample)
        if len(samples) > request.max_items:
            raise ValueError("signal backend exceeded batch cardinality")
        if not self._replay_guard.accept(samples):
            raise ValueError("replayed or out-of-order signal sample")
        return SignalPage(
            request_id=request.request_id,
            source_id=request.source_id,
            snapshot_digest=snapshot_digest,
            samples=tuple(samples),
            next_cursor=backend_page.next_cursor,
            has_more=backend_page.has_more,
        )

    def _validate_request_scope(self, request: SignalReadRequest) -> None:
        if request.source_id != self._binding.source_id:
            raise ValueError("signal request source is not authenticated")
        if request.service_scope != self._binding.service_scope:
            raise ValueError("signal request crosses service scope")
        if request.tenant_scope != self._binding.tenant_scope:
            raise ValueError("signal request crosses tenant scope")
        if request.unit_id != self._binding.unit_id:
            raise ValueError("signal request crosses scale-unit scope")
        if len(request.query.encode("utf-8")) > self._policy.max_query_length:
            raise ValueError("signal query exceeds policy length")
        if request.timeout_s > self._policy.max_timeout_s:
            raise ValueError("signal query timeout exceeds policy")
        if request.page_size > self._policy.max_page_size:
            raise ValueError("signal page exceeds policy cardinality")
        if request.max_items > self._policy.max_batch_items:
            raise ValueError("signal batch exceeds policy cardinality")
        if len(request.signals) > self._policy.max_signal_kinds:
            raise ValueError("signal vocabulary exceeds policy cardinality")
        if request.window_end - request.window_start > timedelta(
            seconds=self._policy.max_window_s
        ):
            raise ValueError("signal query window exceeds policy duration")

    def _validate_sample(
        self, sample: SignalSample, request: SignalReadRequest
    ) -> None:
        if not isinstance(sample, SignalSample):
            raise ValueError("signal backend returned an invalid sample")
        if sample.source_id != self._binding.source_id:
            raise ValueError("signal sample source is not authenticated")
        if sample.source_kind != self._binding.source_kind:
            raise ValueError("signal sample source kind is not authenticated")
        if sample.source_epoch != self._binding.source_epoch:
            raise ValueError("signal sample source epoch is stale")
        if sample.auth_evidence_ref != self._binding.auth_evidence_ref:
            raise ValueError("signal sample attestation does not match source")
        if (
            sample.service_scope != request.service_scope
            or sample.tenant_scope != request.tenant_scope
            or sample.unit_id != request.unit_id
        ):
            raise ValueError("signal sample crosses an authenticated scope")
        if sample.signal not in request.signals:
            raise ValueError("signal sample is outside the requested vocabulary")
        if sample.aggregation != request.aggregation:
            raise ValueError("signal sample aggregation differs from request")
        if sample.query_digest != request.query_digest:
            raise ValueError("signal sample is bound to a different query")
        if not self._authenticator.verify(sample, self._binding):
            raise ValueError("signal sample authentication failed")
        now = _aware_datetime(self._clock(), "clock")
        age = (now - sample.sample_time).total_seconds()
        if age < -self._policy.max_future_skew_s:
            raise ValueError("signal sample timestamp is from the future")
        if (
            age > self._policy.max_freshness_s
            or sample.freshness_s > self._policy.max_freshness_s
        ):
            raise ValueError("stale signal sample")
        if (
            sample.sample_time < request.window_start
            or sample.sample_time > request.window_end
        ):
            raise ValueError("signal sample is outside the requested window")


def summarize_signal_page(
    request: SignalReadRequest,
    page: SignalPage,
) -> tuple[SignalSummary, SignalDecisionEvidence] | None:
    """Aggregate one bounded page; empty pages remain missing data, never zero."""

    if page.request_id != request.request_id or page.source_id != request.source_id:
        raise ValueError("signal page is not bound to the request")
    if (
        page.snapshot_digest != request.snapshot_digest
        and request.snapshot_digest is not None
    ):
        raise ValueError("signal page snapshot differs from request")
    if page.has_more:
        raise ValueError("cannot summarize an incomplete signal page")
    if not page.samples:
        return None
    signals = {sample.signal for sample in page.samples}
    if len(signals) != 1 or next(iter(signals)) not in request.signals:
        raise ValueError("signal page contains mixed or unrequested signals")
    for sample in page.samples:
        if sample.aggregation != request.aggregation:
            raise ValueError("signal page aggregation differs from request")
    values = [sample.value for sample in page.samples]
    ordered = sorted(page.samples, key=lambda sample: sample.sample_time)
    if request.aggregation == SignalAggregation.LAST.value:
        value = ordered[-1].value
    elif request.aggregation == SignalAggregation.MEAN.value:
        value = sum(values) / len(values)
    elif request.aggregation == SignalAggregation.MAX.value:
        value = max(values)
    elif request.aggregation in {
        SignalAggregation.P50.value,
        SignalAggregation.P95.value,
        SignalAggregation.P99.value,
    }:
        ordered_values = sorted(values)
        percentile = float(request.aggregation[1:]) / 100
        index = min(
            len(ordered_values) - 1, math.ceil(percentile * len(ordered_values)) - 1
        )
        value = ordered_values[index]
    elif request.aggregation == SignalAggregation.SUM.value:
        value = sum(values)
    elif request.aggregation == SignalAggregation.COUNT.value:
        value = float(len(values))
    else:  # RATE: samples are already normalized rates from the source.
        value = sum(values) / len(values)

    digest_material = "|".join(
        [
            request.query_digest,
            page.snapshot_digest,
            *sorted(sample.sample_digest for sample in page.samples),
        ]
    )
    summary_digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()
    first = page.samples[0]
    summary = SignalSummary(
        summary_id=f"summary:{first.unit_id}:{first.signal}:{summary_digest[:16]}",
        summary_digest=summary_digest,
        source_id=first.source_id,
        source_kind=first.source_kind,
        service_scope=first.service_scope,
        tenant_scope=first.tenant_scope,
        unit_id=first.unit_id,
        signal=first.signal,
        aggregation=first.aggregation,
        value=value,
        sample_count=len(page.samples),
        freshness_s=max(sample.freshness_s for sample in page.samples),
        confidence=min(sample.confidence for sample in page.samples),
        evidence_ref=f"evidence:summary:{summary_digest[:24]}",
        query_digest=request.query_digest,
        window_start=request.window_start,
        window_end=request.window_end,
    )
    evidence = SignalDecisionEvidence(
        request_id=request.request_id,
        query=request.query,
        query_digest=request.query_digest,
        summary_digest=summary.summary_digest,
        snapshot_digest=page.snapshot_digest,
        window_start=request.window_start,
        window_end=request.window_end,
        sample_count=summary.sample_count,
    )
    return summary, evidence


__all__ = [
    "CONTRACT_VERSION",
    "MAX_BATCH_ITEMS",
    "MAX_FUTURE_SKEW_S",
    "MAX_PAGE_SIZE",
    "MAX_QUERY_LENGTH",
    "MAX_QUERY_TIMEOUT_S",
    "MAX_QUERY_WINDOW_S",
    "MAX_SAMPLE_FRESHNESS_S",
    "AuthenticatedSignalProvider",
    "OpaqueReferenceAuthenticator",
    "ReplayGuard",
    "SignalAggregation",
    "SignalBackend",
    "SignalBackendPage",
    "SignalDecisionEvidence",
    "SignalKind",
    "SignalPage",
    "SignalReadPolicy",
    "SignalReadRequest",
    "SignalSample",
    "SignalSourceBinding",
    "SignalSourceKind",
    "SignalSummary",
    "SignalAuthenticator",
    "query_digest",
    "summarize_signal_page",
]
