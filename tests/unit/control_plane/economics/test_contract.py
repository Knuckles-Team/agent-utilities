"""Focused contract fixtures for NE-090.

The suite is intentionally small: persistence adapters and live telemetry are
validated by the owning integration tracks.  These fixtures pin the domain's
fail-closed identity, accounting, SLO, tenant, and reconciliation rules.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.economics import (
    AllocationRef,
    EconomicsContractError,
    InMemoryEconomicsRepository,
    PriceCard,
    PriceRate,
    SampleRef,
    SloObjective,
    TenantReadScope,
    UsageFact,
    UsageReadRequest,
    Watermark,
    WatermarkPolicy,
    aggregate_usage,
    build_slo_rollup,
    decide_late_event,
    reconcile_samples,
    window_for,
)


NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def usage_fact(*, sequence: int = 1, digest: str = DIGEST_A) -> UsageFact:
    return UsageFact(
        tenant_id="tenant:alpha",
        allocation=AllocationRef(
            tenant_id="tenant:alpha",
            project_ref="project:one",
            cost_center_ref="cost:center-a",
        ),
        source_ref="metrics:collector-a",
        source_digest=digest,
        occurred_at=NOW,
        metric="tokens",
        quantity=12,
        input_quantity=8,
        output_quantity=4,
        service_ref="service:inference",
        meter_ref="meter:tokens",
        sample_sequence=sequence,
    )


def price_card() -> PriceCard:
    return PriceCard(
        provider_ref="provider:local",
        service_ref="service:inference",
        currency="USD",
        version=1,
        effective_from=datetime(2026, 8, 19, 0, tzinfo=UTC),
        rates=(PriceRate(metric="tokens", micros_per_unit=2),),
        source_ref="pricing:catalog-v1",
        source_digest=DIGEST_B,
    )


def test_fact_identity_is_stable_and_exact_replay_is_not_double_counted() -> None:
    fact = usage_fact()
    replay = usage_fact()
    assert fact.fact_id == replay.fact_id
    assert fact.fact_digest == replay.fact_digest

    repository = InMemoryEconomicsRepository()
    assert repository.append_usage_fact(fact).disposition == "inserted"
    assert repository.append_usage_fact(replay).replayed


def test_cross_tenant_allocation_and_raw_secret_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="allocation tenant_id"):
        UsageFact(
            tenant_id="tenant:alpha",
            allocation=AllocationRef(
                tenant_id="tenant:other",
                project_ref="project:one",
                cost_center_ref="cost:center-a",
            ),
            source_ref="metrics:collector-a",
            source_digest=DIGEST_A,
            occurred_at=NOW,
            metric="requests",
            quantity=1,
            service_ref="service:inference",
            meter_ref="meter:requests",
        )
    with pytest.raises(ValidationError):
        UsageFact(
            tenant_id="tenant:alpha",
            allocation=AllocationRef(
                tenant_id="tenant:alpha",
                project_ref="project:one",
                cost_center_ref="cost:center-a",
            ),
            source_ref="metrics:collector-a?token=secret",
            source_digest=DIGEST_A,
            occurred_at=NOW,
            metric="requests",
            quantity=1,
            service_ref="service:inference",
            meter_ref="meter:requests",
            raw_span="must-not-cross-boundary",  # type: ignore[call-arg]
        )


def test_hourly_aggregate_uses_integer_price_and_rejects_duplicate_fact() -> None:
    fact = usage_fact()
    aggregate = aggregate_usage(
        (fact,), window=window_for(NOW, "hour"), price_card=price_card()
    )
    assert aggregate.totals[0].quantity == 12
    assert aggregate.totals[0].cost_micros == 24
    with pytest.raises(EconomicsContractError, match="double count"):
        aggregate_usage(
            (fact, fact), window=window_for(NOW, "hour"), price_card=price_card()
        )


def test_watermark_requires_explicit_late_event_decision() -> None:
    watermark = Watermark(
        tenant_id="tenant:alpha",
        source_ref="metrics:collector-a",
        watermark_at=datetime(2026, 8, 19, 13, tzinfo=UTC),
        observed_at=datetime(2026, 8, 19, 13, 1, tzinfo=UTC),
        policy_version="watermark:v1",
    )
    decision = decide_late_event(
        usage_fact(), watermark, WatermarkPolicy(policy_version="watermark:v1")
    )
    assert decision.status == "correction_required"
    assert decision.reason == "within_lateness"


def test_slo_rollup_and_tenant_keyset_are_bounded() -> None:
    objective = SloObjective(
        tenant_id="tenant:alpha",
        service_ref="service:inference",
        sli="availability",
        window_granularity="hour",
        version=1,
        target_micros=999_000,
        comparison="gte",
    )
    rollup = build_slo_rollup(
        objective,
        window=window_for(NOW, "hour"),
        good_events=99,
        total_events=100,
        bad_events=1,
        source_fact_digests=(DIGEST_A,),
    )
    assert rollup.status == "breached"
    with pytest.raises(ValidationError):
        UsageReadRequest(
            scope=TenantReadScope(
                tenant_id="tenant:alpha", principal_ref="principal:one"
            ),
            limit=501,
        )


def test_reconciliation_exposes_gaps_and_drift() -> None:
    missing = SampleRef(
        source_ref="metrics:collector-a", source_digest=DIGEST_B, sample_sequence=2
    )
    report = reconcile_samples(
        (
            SampleRef(
                source_ref="metrics:collector-a",
                source_digest=DIGEST_A,
                sample_sequence=1,
            ),
            missing,
        ),
        (usage_fact(),),
    )
    assert report.status == "gap"
    assert report.missing == (missing,)


def test_cursor_scope_mismatch_is_denied() -> None:
    with pytest.raises(ValidationError, match="cursor tenant"):
        UsageReadRequest(
            scope=TenantReadScope(
                tenant_id="tenant:alpha", principal_ref="principal:one"
            ),
            after={
                "tenant_id": "tenant:other",
                "query_digest": DIGEST_A,
                "last_key": "aggregate:row",
            },
        )
