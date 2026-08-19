"""NE-165 authenticated and bounded scaling-signal contract fixtures.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling

The fixtures intentionally use a streaming fake backend.  They exercise the
authority boundary without starting Prometheus, a graph engine, or a live
service.
"""

from datetime import datetime, timedelta, timezone
import math

import pytest
from pydantic import ValidationError

from agent_utilities.orchestration.scaling_signal_authority import (
    AuthenticatedSignalProvider,
    OpaqueReferenceAuthenticator,
    SignalAggregation,
    SignalBackendPage,
    SignalKind,
    SignalReadPolicy,
    SignalReadRequest,
    SignalSourceBinding,
    SignalSourceKind,
    SignalSample,
    SignalSummary,
    query_digest,
    summarize_signal_page,
)

pytestmark = pytest.mark.concept("AU-OS.scaling.reactive-replica-autoscaling")

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
WINDOW_START = NOW - timedelta(minutes=2)
WINDOW_END = NOW
SNAPSHOT_DIGEST = "b" * 64
QUERY = "sum(cpu_usage_seconds_total)"


class _Backend:
    def __init__(self, page: SignalBackendPage):
        self.samples = tuple(page.samples)
        self.next_cursor = page.next_cursor
        self.has_more = page.has_more
        self.snapshot_digest = page.snapshot_digest
        self.requests: list[SignalReadRequest] = []

    def read(self, request: SignalReadRequest) -> SignalBackendPage:
        self.requests.append(request)
        return SignalBackendPage(
            iter(self.samples),
            next_cursor=self.next_cursor,
            has_more=self.has_more,
            snapshot_digest=self.snapshot_digest,
        )


def _binding() -> SignalSourceBinding:
    return SignalSourceBinding(
        source_id="source:metrics",
        source_kind=SignalSourceKind.TELEMETRY,
        service_scope="service:gateway",
        tenant_scope="tenant:acme",
        unit_id="unit:gateway",
        source_epoch=3,
        auth_evidence_ref="attestation:metrics:3",
    )


def _request(**overrides: object) -> SignalReadRequest:
    values: dict[str, object] = {
        "request_id": "read:gateway:1",
        "source_id": "source:metrics",
        "service_scope": "service:gateway",
        "tenant_scope": "tenant:acme",
        "unit_id": "unit:gateway",
        "signals": (SignalKind.CPU,),
        "aggregation": SignalAggregation.MEAN,
        "query": QUERY,
        "query_digest": query_digest(QUERY),
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "timeout_s": 3.0,
        "page_size": 16,
        "max_items": 64,
    }
    values.update(overrides)
    return SignalReadRequest(**values)


def _sample(
    *,
    sample_id: str = "sample:1",
    sequence: int = 1,
    value: float = 42.0,
    sample_time: datetime = NOW - timedelta(seconds=10),
    event_time: datetime | None = None,
    service_scope: str = "service:gateway",
    tenant_scope: str = "tenant:acme",
    unit_id: str = "unit:gateway",
    source_id: str = "source:metrics",
    source_kind: SignalSourceKind = SignalSourceKind.TELEMETRY,
    source_epoch: int = 3,
    auth_evidence_ref: str = "attestation:metrics:3",
    freshness_s: float = 10.0,
    confidence: float = 0.95,
    evidence_ref: str | None = None,
    query: str = QUERY,
    sample_digest: str | None = None,
) -> SignalSample:
    return SignalSample(
        sample_id=sample_id,
        sample_digest=sample_digest or (hex(sequence)[2:] * 64)[:64],
        source_id=source_id,
        source_kind=source_kind,
        source_epoch=source_epoch,
        auth_evidence_ref=auth_evidence_ref,
        service_scope=service_scope,
        tenant_scope=tenant_scope,
        unit_id=unit_id,
        signal=SignalKind.CPU,
        aggregation=SignalAggregation.MEAN,
        value=value,
        sample_time=sample_time,
        event_time=event_time or sample_time,
        freshness_s=freshness_s,
        confidence=confidence,
        evidence_ref=evidence_ref or f"evidence:sample:{sample_id.replace(':', '-')}",
        sequence=sequence,
        query_digest=query_digest(query),
    )


def _provider(
    page: SignalBackendPage,
    *,
    authenticator: object | None = None,
) -> tuple[AuthenticatedSignalProvider, SignalReadRequest]:
    backend = _Backend(page)
    return (
        AuthenticatedSignalProvider(
            backend,
            _binding(),
            authenticator=authenticator,  # type: ignore[arg-type]
            clock=lambda: NOW,
        ),
        _request(),
    )


def test_signal_vocabulary_is_finite_and_request_is_query_digest_bound() -> None:
    assert {kind.value for kind in SignalKind} == {
        "cpu",
        "ram",
        "gpu",
        "kv_cache",
        "disk",
        "network",
        "token",
        "request",
        "latency",
        "queue",
        "shard",
    }
    assert {aggregation.value for aggregation in SignalAggregation} == {
        "last",
        "mean",
        "max",
        "p50",
        "p95",
        "p99",
        "rate",
        "sum",
        "count",
    }
    with pytest.raises(ValidationError, match="query_digest"):
        _request(query_digest="c" * 64)
    with pytest.raises(ValidationError, match="credential"):
        _request(query="sum(rate(requests_total{token=\"secret\"}[5m]))")


def test_invalid_values_timestamps_and_metadata_fail_closed() -> None:
    for value in (-1.0, math.nan, math.inf):
        with pytest.raises(ValidationError):
            _sample(value=value)
    with pytest.raises(ValidationError, match="event_time"):
        _sample(event_time=NOW)
    with pytest.raises(ValidationError, match="confidence"):
        _sample(confidence=1.1)
    with pytest.raises(ValidationError, match="opaque reference"):
        _sample(evidence_ref="bearer=not-a-reference")  # type: ignore[call-arg]


def test_authenticated_batch_read_and_summary_keep_samples_out_of_graph() -> None:
    sample = _sample()
    provider, request = _provider(
        SignalBackendPage(iter((sample,)), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST)
    )
    page = provider.read_batch(request)
    assert page.samples == (sample,)
    projection = page.graph_projection()
    assert projection["sample_count"] == 1
    assert "samples" not in projection

    summary, decision_evidence = summarize_signal_page(request, page)  # type: ignore[misc]
    assert isinstance(summary, SignalSummary)
    assert summary.value == 42.0
    assert "query" not in summary.graph_projection()
    assert "samples" not in summary.graph_projection()
    assert decision_evidence.query == QUERY
    assert decision_evidence.query_digest == query_digest(QUERY)
    assert decision_evidence.window_start == WINDOW_START
    assert decision_evidence.window_end == WINDOW_END


def test_missing_batch_is_not_interpreted_as_zero() -> None:
    provider, request = _provider(
        SignalBackendPage(iter(()), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST)
    )
    page = provider.read_batch(request)
    assert page.samples == ()
    assert summarize_signal_page(request, page) is None


def test_scope_source_freshness_and_authentication_are_enforced() -> None:
    stale = _sample(sample_time=NOW - timedelta(minutes=10), freshness_s=10)
    provider, request = _provider(
        SignalBackendPage(iter((stale,)), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST)
    )
    with pytest.raises(ValueError, match="stale"):
        provider.read_batch(request)

    cross_service = _sample(service_scope="service:other")
    provider, request = _provider(
        SignalBackendPage(iter((cross_service,)), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST)
    )
    with pytest.raises(ValueError, match="scope"):
        provider.read_batch(request)

    spoofed = _sample(source_id="source:spoof")
    provider, request = _provider(
        SignalBackendPage(iter((spoofed,)), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST)
    )
    with pytest.raises(ValueError, match="source"):
        provider.read_batch(request)

    class _RejectingAuthenticator(OpaqueReferenceAuthenticator):
        def verify(self, sample: SignalSample, binding: SignalSourceBinding) -> bool:
            return False

    provider, request = _provider(
        SignalBackendPage(
            iter((_sample(),)),
            next_cursor=None,
            has_more=False,
            snapshot_digest=SNAPSHOT_DIGEST,
        ),
        authenticator=_RejectingAuthenticator(),
    )
    with pytest.raises(ValueError, match="authentication"):
        provider.read_batch(request)


def test_replayed_and_out_of_order_samples_are_rejected_without_zero_fallback() -> None:
    sample = _sample()
    page = SignalBackendPage(
        iter((sample,)), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST
    )
    provider, request = _provider(page)
    provider.read_batch(request)
    with pytest.raises(ValueError, match="replayed"):
        provider.read_batch(request)

    first = _sample(sample_id="sample:1", sequence=1)
    second = _sample(sample_id="sample:2", sequence=2, sample_digest="d" * 64)
    provider, request = _provider(
        SignalBackendPage(
            iter((second, first)),
            next_cursor=None,
            has_more=False,
            snapshot_digest=SNAPSHOT_DIGEST,
        )
    )
    with pytest.raises(ValueError, match="replayed"):
        provider.read_batch(request)


def test_snapshot_pagination_is_bounded_and_deterministic() -> None:
    first = _sample(sample_id="sample:1", sequence=1)
    second = _sample(
        sample_id="sample:2",
        sequence=2,
        sample_time=NOW - timedelta(seconds=5),
        sample_digest="d" * 64,
    )

    class _PagedBackend:
        def read(self, request: SignalReadRequest) -> SignalBackendPage:
            if request.cursor is None:
                return SignalBackendPage(
                    iter((first,)),
                    next_cursor="cursor:1",
                    has_more=True,
                    snapshot_digest=SNAPSHOT_DIGEST,
                )
            return SignalBackendPage(
                iter((second,)),
                next_cursor=None,
                has_more=False,
                snapshot_digest=SNAPSHOT_DIGEST,
            )

    provider = AuthenticatedSignalProvider(
        _PagedBackend(), _binding(), clock=lambda: NOW
    )
    first_request = _request(page_size=1, max_items=1)
    first_page = provider.read_batch(first_request)
    assert first_page.next_cursor == "cursor:1"
    second_request = first_request.model_copy(
        update={"cursor": "cursor:1", "snapshot_digest": SNAPSHOT_DIGEST}
    )
    second_page = provider.read_batch(second_request)
    assert second_page.samples == (second,)

    class _ChangingBackend:
        def read(self, request: SignalReadRequest) -> SignalBackendPage:
            return SignalBackendPage(
                iter((second,)),
                next_cursor=None,
                has_more=False,
                snapshot_digest="e" * 64,
            )

    provider = AuthenticatedSignalProvider(
        _ChangingBackend(), _binding(), clock=lambda: NOW
    )
    with pytest.raises(ValueError, match="snapshots"):
        provider.read_batch(second_request)


def test_query_and_catalog_bounds_are_rejected_before_backend_read() -> None:
    with pytest.raises(ValidationError, match="query length"):
        _request(query="x" * 4_097, query_digest=query_digest("x" * 4_097))
    with pytest.raises(ValidationError, match="window"):
        _request(
            window_start=NOW - timedelta(days=2),
            window_end=NOW,
        )
    with pytest.raises(ValidationError):
        _request(page_size=257)
    with pytest.raises(ValidationError, match="duplicate"):
        _request(signals=(SignalKind.CPU, SignalKind.CPU))

    backend = _Backend(
        SignalBackendPage(
            iter((_sample(),)), next_cursor=None, has_more=False, snapshot_digest=SNAPSHOT_DIGEST
        )
    )
    request = _request(service_scope="service:other")
    provider = AuthenticatedSignalProvider(backend, _binding(), clock=lambda: NOW)
    with pytest.raises(ValueError, match="service scope"):
        provider.read_batch(request)
    assert backend.requests == []

    policy = SignalReadPolicy(max_signal_kinds=1)
    provider = AuthenticatedSignalProvider(
        backend, _binding(), policy=policy, clock=lambda: NOW
    )
    with pytest.raises(ValueError, match="vocabulary"):
        provider.read_batch(_request(signals=(SignalKind.CPU, SignalKind.RAM)))
