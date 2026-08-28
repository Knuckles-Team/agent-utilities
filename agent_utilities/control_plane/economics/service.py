"""Pure deterministic economics/SLO operations.

The functions here are side-effect free until a caller explicitly invokes a
typed :class:`EconomicsRepository` method.  They reject ambiguous input rather
than silently dropping records, double-counting facts, or treating missing data
as zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Literal

from .models import (
    AllocationRef,
    EconomicsContractError,
    LateEventDecision,
    MetricKind,
    MetricTotal,
    PriceCard,
    ReconciliationReport,
    SampleRef,
    SloObjective,
    SloWindowRollup,
    UsageFact,
    UsageWindow,
    Watermark,
    WatermarkPolicy,
    WindowAggregate,
)


def window_for(
    occurred_at: datetime, granularity: Literal["hour", "day"]
) -> UsageWindow:
    """Return an aligned UTC window; only hourly and daily aggregation is valid."""

    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    normalized = occurred_at.astimezone(UTC)
    if granularity == "hour":
        start = normalized.replace(minute=0, second=0, microsecond=0)
    elif granularity == "day":
        start = normalized.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("granularity must be hour or day")
    return UsageWindow(granularity=granularity, window_start=start)


def decide_late_event(
    fact: UsageFact,
    watermark: Watermark | None,
    policy: WatermarkPolicy,
) -> LateEventDecision:
    """Classify lateness explicitly before admission or correction.

    A fact older than the closed watermark is never silently discarded.  Depending
    on policy it is admitted on time, marked for an append-only correction, or
    rejected with a durable reason.
    """

    if watermark is None:
        return LateEventDecision(
            fact_id=fact.fact_id,
            status="accepted",
            reason="on_time",
            watermark_at=fact.occurred_at,
        )
    if watermark.tenant_id != fact.tenant_id or watermark.source_ref != fact.source_ref:
        raise EconomicsContractError("watermark scope does not match fact source")
    age_seconds = (watermark.watermark_at - fact.occurred_at).total_seconds()
    if age_seconds <= 0:
        return LateEventDecision(
            fact_id=fact.fact_id,
            status="accepted",
            reason="on_time",
            watermark_at=watermark.watermark_at,
        )
    if age_seconds <= policy.allowed_lateness_seconds:
        if policy.late_event_mode == "correction":
            return LateEventDecision(
                fact_id=fact.fact_id,
                status="correction_required",
                reason="within_lateness",
                watermark_at=watermark.watermark_at,
            )
        return LateEventDecision(
            fact_id=fact.fact_id,
            status="rejected",
            reason="within_lateness",
            watermark_at=watermark.watermark_at,
        )
    if (
        policy.late_event_mode == "correction"
        and age_seconds <= policy.max_correction_age_seconds
    ):
        return LateEventDecision(
            fact_id=fact.fact_id,
            status="correction_required",
            reason="outside_lateness",
            watermark_at=watermark.watermark_at,
        )
    return LateEventDecision(
        fact_id=fact.fact_id,
        status="rejected",
        reason="correction_window_expired",
        watermark_at=watermark.watermark_at,
    )


def _validate_aggregate_scope(
    materialized: tuple[UsageFact, ...], window: UsageWindow, price_card: PriceCard
) -> tuple[str, AllocationRef, datetime]:
    """``(tenant_id, allocation, window_end)`` after whole-aggregate scope checks."""
    tenant_id = materialized[0].tenant_id
    allocation = materialized[0].allocation
    if allocation.tenant_id != tenant_id:
        raise ValueError("usage allocation crosses tenant scope")
    if window.window_end is None:
        raise EconomicsContractError("aggregate window is missing its end boundary")
    window_end = window.window_end
    if price_card.effective_from > window.window_start or (
        price_card.effective_to is not None and price_card.effective_to < window_end
    ):
        raise ValueError("price card does not cover the full aggregate window")
    return tenant_id, allocation, window_end


def _check_fact_scope(
    fact: UsageFact, tenant_id: str, allocation: AllocationRef
) -> None:
    if fact.tenant_id != tenant_id or fact.allocation != allocation:
        raise EconomicsContractError("aggregate cannot mix tenants or allocations")


def _check_fact_window_and_price(
    fact: UsageFact,
    price_card: PriceCard,
    rates: dict[MetricKind, int],
    window: UsageWindow,
    window_end: datetime,
) -> None:
    if fact.service_ref != price_card.service_ref:
        raise EconomicsContractError("price card service does not match usage fact")
    if not (window.window_start <= fact.occurred_at < window_end):
        raise EconomicsContractError("aggregate input contains an out-of-window fact")
    if fact.metric not in rates:
        raise EconomicsContractError(f"price card has no rate for {fact.metric}")


def aggregate_usage(
    facts: Iterable[UsageFact],
    *,
    window: UsageWindow,
    price_card: PriceCard,
) -> WindowAggregate:
    """Aggregate one exact window with integer arithmetic and stable ordering.

    Every supplied fact must belong to the requested window and allocation scope.
    This fail-closed rule prevents a caller from accidentally dropping out-of-
    window events and then treating the result as a complete accounting record.
    """

    materialized = tuple(facts)
    if not materialized:
        raise ValueError("an accounting aggregate requires at least one usage fact")
    tenant_id, allocation, window_end = _validate_aggregate_scope(
        materialized, window, price_card
    )
    seen: set[str] = set()
    totals: dict[MetricKind, int] = defaultdict(int)
    costs: dict[MetricKind, int] = defaultdict(int)
    rates = {rate.metric: rate.micros_per_unit for rate in price_card.rates}

    for fact in materialized:
        if fact.fact_id in seen:
            raise EconomicsContractError(
                "duplicate fact would double count an aggregate"
            )
        seen.add(fact.fact_id)
        _check_fact_scope(fact, tenant_id, allocation)
        _check_fact_window_and_price(fact, price_card, rates, window, window_end)
        totals[fact.metric] += fact.quantity
        costs[fact.metric] += fact.quantity * rates[fact.metric]

    metrics = tuple(
        MetricTotal(metric=metric, quantity=totals[metric], cost_micros=costs[metric])
        for metric in sorted(totals)
    )
    return WindowAggregate(
        tenant_id=tenant_id,
        allocation=allocation,
        window=window,
        fact_ids=tuple(sorted(seen)),
        price_card_id=price_card.card_id,
        price_card_digest=price_card.card_digest,
        totals=metrics,
    )


def _validate_daily_completeness(
    day: UsageWindow, hours: tuple[WindowAggregate, ...]
) -> None:
    """The 24-hour completeness rule: a missing hour is a gap, not a zero."""
    expected = {day.window_start + timedelta(hours=index) for index in range(24)}
    actual = {item.window.window_start for item in hours}
    if actual != expected or len(hours) != 24:
        raise EconomicsContractError("daily aggregate requires all 24 hourly windows")


def _validate_daily_scope_consistency(
    hours: tuple[WindowAggregate, ...],
) -> WindowAggregate:
    first = hours[0]
    if any(
        item.tenant_id != first.tenant_id
        or item.allocation != first.allocation
        or item.price_card_id != first.price_card_id
        or item.price_card_digest != first.price_card_digest
        for item in hours
    ):
        raise EconomicsContractError(
            "daily aggregate cannot mix allocation or price-card scope"
        )
    return first


def _validate_daily_fact_ids(hours: tuple[WindowAggregate, ...]) -> list[str]:
    fact_ids = [fact_id for item in hours for fact_id in item.fact_ids]
    if len(fact_ids) != len(set(fact_ids)):
        raise EconomicsContractError("hourly inputs overlap and would double count")
    return fact_ids


def _sum_daily_totals(hours: tuple[WindowAggregate, ...]) -> tuple[MetricTotal, ...]:
    quantity_by_metric: dict[MetricKind, int] = defaultdict(int)
    cost_by_metric: dict[MetricKind, int] = defaultdict(int)
    for item in hours:
        for total in item.totals:
            quantity_by_metric[total.metric] += total.quantity
            cost_by_metric[total.metric] += total.cost_micros
    return tuple(
        MetricTotal(
            metric=metric,
            quantity=quantity_by_metric[metric],
            cost_micros=cost_by_metric[metric],
        )
        for metric in sorted(quantity_by_metric)
    )


def aggregate_daily_from_hours(
    hourly: Iterable[WindowAggregate],
    *,
    day: UsageWindow,
) -> WindowAggregate:
    """Compose an aligned daily aggregate from hourly aggregates.

    The 24-hour completeness rule is deliberate: a missing hour is a gap, not a
    zero.  A caller can choose to publish a separate reconciliation result instead
    of manufacturing an apparently complete daily total.
    """

    if day.granularity != "day":
        raise ValueError("daily composition requires a day window")
    hours = tuple(hourly)
    _validate_daily_completeness(day, hours)
    first = _validate_daily_scope_consistency(hours)
    fact_ids = _validate_daily_fact_ids(hours)
    totals = _sum_daily_totals(hours)
    return WindowAggregate(
        tenant_id=first.tenant_id,
        allocation=first.allocation,
        window=day,
        fact_ids=tuple(sorted(fact_ids)),
        price_card_id=first.price_card_id,
        price_card_digest=first.price_card_digest,
        totals=totals,
    )


def _validate_slo_counts(
    objective: SloObjective, good_events: int, total_events: int, bad_events: int
) -> None:
    if good_events < 0 or total_events < 0 or bad_events < 0:
        raise ValueError("SLO counts must be non-negative")
    if good_events > total_events or bad_events > total_events:
        raise ValueError("SLO good/bad counts cannot exceed total")
    if (
        objective.sli in {"availability", "error_rate"}
        and good_events + bad_events != total_events
    ):
        raise ValueError("availability/error-rate rollup must account for every event")


def _resolve_slo_measurement(
    objective: SloObjective,
    good_events: int,
    total_events: int,
    bad_events: int,
    measured_micros: int | None,
) -> tuple[int | None, Literal["met", "breached", "insufficient_data"]]:
    """``(measured_micros, status)`` per the objective's SLI accounting rule."""
    if total_events == 0:
        if measured_micros is not None:
            raise ValueError("missing SLO measurements cannot be represented as zero")
        return None, "insufficient_data"
    if objective.sli == "availability":
        measured_micros = (good_events * 1_000_000) // total_events
    elif objective.sli == "error_rate":
        measured_micros = (bad_events * 1_000_000) // total_events
    elif measured_micros is None:
        raise ValueError(
            "latency/queue/freshness/throughput rollup requires a measurement"
        )
    if measured_micros < 0 or measured_micros > 1_000_000:
        raise ValueError("SLO measurement must be in bounded micros")
    met = (
        measured_micros >= objective.target_micros
        if objective.comparison == "gte"
        else measured_micros <= objective.target_micros
    )
    return measured_micros, ("met" if met else "breached")


def _validated_slo_digests(source_fact_digests: Iterable[str]) -> tuple[str, ...]:
    supplied_digests = tuple(source_fact_digests)
    digests = tuple(sorted(set(supplied_digests)))
    if len(digests) != len(supplied_digests):
        raise EconomicsContractError("SLO source digests must be unique")
    if len(digests) > 100_000:
        raise ValueError("SLO source digest set exceeds the bounded read model")
    for digest in digests:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("SLO source facts must be opaque sha256 digests")
    return digests


def build_slo_rollup(
    objective: SloObjective,
    *,
    window: UsageWindow,
    good_events: int,
    total_events: int,
    bad_events: int,
    measured_micros: int | None = None,
    source_fact_digests: Iterable[str] = (),
) -> SloWindowRollup:
    """Build a bounded SLO rollup without admitting raw spans or logs."""

    if objective.window_granularity != window.granularity:
        raise ValueError(
            "SLO objective and rollup windows have different granularities"
        )
    _validate_slo_counts(objective, good_events, total_events, bad_events)
    measured_micros, status = _resolve_slo_measurement(
        objective, good_events, total_events, bad_events, measured_micros
    )
    digests = _validated_slo_digests(source_fact_digests)
    return SloWindowRollup(
        tenant_id=objective.tenant_id,
        objective_id=objective.objective_id,
        window=window,
        good_events=good_events,
        total_events=total_events,
        bad_events=bad_events,
        measured_micros=measured_micros,
        source_fact_digests=digests,
        status=status,
    )


_SampleKey = tuple[str, str, int | None]


def _reconcile_expected_index(
    expected_items: tuple[SampleRef, ...],
) -> tuple[dict[_SampleKey, SampleRef], Counter[_SampleKey]]:
    expected_keys = {
        (item.source_ref, item.source_digest, item.sample_sequence): item
        for item in expected_items
    }
    expected_key_counts = Counter(
        (item.source_ref, item.source_digest, item.sample_sequence)
        for item in expected_items
    )
    return expected_keys, expected_key_counts


def _reconcile_observed_index(
    observed_items: tuple[UsageFact, ...],
) -> tuple[tuple[SampleRef, ...], tuple[_SampleKey, ...], Counter[_SampleKey]]:
    observed_refs = tuple(
        SampleRef(
            source_ref=fact.source_ref,
            source_digest=fact.source_digest,
            sample_sequence=fact.sample_sequence,
        )
        for fact in observed_items
    )
    observed_keys_in_order = tuple(
        (item.source_ref, item.source_digest, item.sample_sequence)
        for item in observed_refs
    )
    observed_counts = Counter(observed_keys_in_order)
    return observed_refs, observed_keys_in_order, observed_counts


def _reconcile_missing_and_unexpected(
    expected_keys: dict[_SampleKey, SampleRef],
    observed_keys_in_order: tuple[_SampleKey, ...],
    observed_refs: tuple[SampleRef, ...],
) -> tuple[tuple[SampleRef, ...], tuple[SampleRef, ...]]:
    observed_keys = set(observed_keys_in_order)
    missing = tuple(
        expected_keys[key]
        for key in sorted(expected_keys.keys())
        if key not in observed_keys
    )
    unexpected = tuple(
        observed_refs[index]
        for index, key in enumerate(observed_keys_in_order)
        if key not in expected_keys and key not in observed_keys_in_order[:index]
    )
    return missing, unexpected


def _reconcile_duplicates(
    expected_items: tuple[SampleRef, ...],
    observed_items: tuple[UsageFact, ...],
    expected_key_counts: Counter[_SampleKey],
    observed_counts: Counter[_SampleKey],
    observed_refs: tuple[SampleRef, ...],
) -> tuple[tuple[SampleRef, ...], tuple[str, ...]]:
    duplicate_sample_keys = sorted(
        {key for key, count in expected_key_counts.items() if count > 1}
        | {key for key, count in observed_counts.items() if count > 1}
    )
    sample_by_key = {
        (item.source_ref, item.source_digest, item.sample_sequence): item
        for item in (*expected_items, *observed_refs)
    }
    duplicate_samples = tuple(sample_by_key[key] for key in duplicate_sample_keys)
    duplicate_fact_ids = tuple(
        fact_id
        for fact_id, count in Counter(fact.fact_id for fact in observed_items).items()
        if count > 1
    )
    return duplicate_samples, duplicate_fact_ids


def reconcile_samples(
    expected: Iterable[SampleRef],
    observed: Iterable[UsageFact],
) -> ReconciliationReport:
    """Expose missing and drifted samples deterministically.

    ``observed`` is bounded before materialization.  Duplicate identities are
    reported as drift instead of being silently deduplicated for accounting.
    """

    expected_items = tuple(expected)
    observed_items = tuple(observed)
    if len(expected_items) > 10_000 or len(observed_items) > 10_000:
        raise ValueError("sample reconciliation exceeds its bounded input size")

    expected_keys, expected_key_counts = _reconcile_expected_index(expected_items)
    observed_refs, observed_keys_in_order, observed_counts = _reconcile_observed_index(
        observed_items
    )
    missing, unexpected = _reconcile_missing_and_unexpected(
        expected_keys, observed_keys_in_order, observed_refs
    )
    duplicate_samples, duplicate_fact_ids = _reconcile_duplicates(
        expected_items,
        observed_items,
        expected_key_counts,
        observed_counts,
        observed_refs,
    )

    status: Literal["complete", "gap", "drift"] = (
        "gap"
        if missing
        else "drift"
        if unexpected or duplicate_samples or duplicate_fact_ids
        else "complete"
    )
    return ReconciliationReport(
        expected_count=len(expected_items),
        observed_count=len(observed_items),
        missing=missing,
        unexpected=unexpected,
        duplicate_samples=duplicate_samples,
        duplicate_fact_ids=duplicate_fact_ids,
        status=status,
    )


def usage_query_digest(request: object) -> str:
    """Stable helper for adapters that need to bind a cursor to a query shape."""

    if not hasattr(request, "query_digest"):
        raise TypeError("request must expose query_digest()")
    return request.query_digest()  # type: ignore[no-any-return]


__all__ = [
    "aggregate_daily_from_hours",
    "aggregate_usage",
    "build_slo_rollup",
    "decide_late_event",
    "reconcile_samples",
    "usage_query_digest",
    "window_for",
]
