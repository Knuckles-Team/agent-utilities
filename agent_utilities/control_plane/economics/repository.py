"""Typed persistence seam for the economics and SLO control plane.

The production adapter owns transactions and durable storage.  This module only
specifies the authority boundary and provides a small deterministic in-memory
fixture for contract tests.  It intentionally has no telemetry, SQL, gateway, or
Kubernetes dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import Field

from agent_utilities.protocols.epistemic_operations import ProtocolModel

from .models import (
    EconomicsConflict,
    KeysetCursor,
    PriceCard,
    SloReadPage,
    SloReadRequest,
    SloWindowRollup,
    TenantScopeError,
    UsageCorrection,
    UsageFact,
    UsageReadPage,
    UsageReadRequest,
    UsageTombstone,
    Watermark,
    WindowAggregate,
)

AppendDisposition = Literal["inserted", "replayed"]


class AppendReceipt(ProtocolModel):
    """Outcome of an idempotent append; no caller-provided status is trusted."""

    identity: str = Field(min_length=1, max_length=100)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    disposition: AppendDisposition

    @property
    def replayed(self) -> bool:
        return self.disposition == "replayed"


class RepositoryUnavailable(RuntimeError):
    """The authority adapter could not complete a transaction."""


@runtime_checkable
class EconomicsRepository(Protocol):
    """Minimal typed repository contract implemented by a durable adapter.

    Every append is idempotent by the model's derived identity.  An adapter must
    return ``replayed`` only when the stored digest is byte-for-byte identical;
    an identity collision with a different digest is a conflict.
    """

    def append_usage_fact(self, fact: UsageFact) -> AppendReceipt: ...

    def append_correction(self, correction: UsageCorrection) -> AppendReceipt: ...

    def append_window_aggregate(self, aggregate: WindowAggregate) -> AppendReceipt: ...

    def append_slo_rollup(self, rollup: SloWindowRollup) -> AppendReceipt: ...

    def put_price_card(self, card: PriceCard) -> AppendReceipt: ...

    def append_tombstone(self, tombstone: UsageTombstone) -> AppendReceipt: ...

    def advance_watermark(self, watermark: Watermark) -> AppendReceipt: ...

    def read_usage_windows(self, request: UsageReadRequest) -> UsageReadPage: ...

    def read_slo_rollups(self, request: SloReadRequest) -> SloReadPage: ...

    def get_watermark(self, *, tenant_id: str, source_ref: str) -> Watermark | None: ...


_T = TypeVar("_T")


def _append(
    store: dict[str, tuple[str, _T]],
    *,
    identity: str,
    digest: str,
    value: _T,
) -> AppendReceipt:
    previous = store.get(identity)
    if previous is not None:
        previous_digest, _ = previous
        if previous_digest != digest:
            raise EconomicsConflict(
                "append-only identity "
                f"{identity!r} already exists with a different digest"
            )
        return AppendReceipt(identity=identity, digest=digest, disposition="replayed")
    store[identity] = (digest, value)
    return AppendReceipt(identity=identity, digest=digest, disposition="inserted")


class InMemoryEconomicsRepository:
    """Deterministic fixture that demonstrates the repository contract.

    It is intentionally not a production backend.  Its value is in making replay,
    conflict, tenant isolation, keyset pagination, and monotonic watermarks
    testable without emulating an external telemetry system.
    """

    def __init__(self) -> None:
        self._facts: dict[str, tuple[str, UsageFact]] = {}
        self._corrections: dict[str, tuple[str, UsageCorrection]] = {}
        self._aggregates: dict[str, tuple[str, WindowAggregate]] = {}
        self._rollups: dict[str, tuple[str, SloWindowRollup]] = {}
        self._cards: dict[str, tuple[str, PriceCard]] = {}
        self._tombstones: dict[str, tuple[str, UsageTombstone]] = {}
        self._watermarks: dict[tuple[str, str], Watermark] = {}
        self._watermark_history: dict[tuple[str, str], dict[str, Watermark]] = {}

    def append_usage_fact(self, fact: UsageFact) -> AppendReceipt:
        return _append(
            self._facts,
            identity=fact.fact_id,
            digest=fact.fact_digest,
            value=fact,
        )

    def append_correction(self, correction: UsageCorrection) -> AppendReceipt:
        return _append(
            self._corrections,
            identity=correction.correction_id,
            digest=correction.original_fact_digest,
            value=correction,
        )

    def append_window_aggregate(self, aggregate: WindowAggregate) -> AppendReceipt:
        return _append(
            self._aggregates,
            identity=aggregate.aggregate_id,
            digest=aggregate.aggregate_digest,
            value=aggregate,
        )

    def append_slo_rollup(self, rollup: SloWindowRollup) -> AppendReceipt:
        return _append(
            self._rollups,
            identity=rollup.rollup_id,
            digest=rollup.rollup_digest,
            value=rollup,
        )

    def put_price_card(self, card: PriceCard) -> AppendReceipt:
        # A card's derived identity contains its version and effective interval;
        # replacing a version can therefore never silently mutate pricing.
        for _, (_, previous) in self._cards.items():
            if previous.card_id == card.card_id:
                continue
            if (
                previous.provider_ref == card.provider_ref
                and previous.service_ref == card.service_ref
                and previous.version == card.version
                and _intervals_overlap(previous, card)
            ):
                raise EconomicsConflict(
                    "price-card versions may not overlap with a different digest"
                )
        return _append(
            self._cards,
            identity=card.card_id,
            digest=card.card_digest,
            value=card,
        )

    def append_tombstone(self, tombstone: UsageTombstone) -> AppendReceipt:
        return _append(
            self._tombstones,
            identity=tombstone.tombstone_id,
            digest=tombstone.target_digest,
            value=tombstone,
        )

    def advance_watermark(self, watermark: Watermark) -> AppendReceipt:
        key = (watermark.tenant_id, watermark.source_ref)
        digest = _watermark_digest(watermark)
        history = self._watermark_history.setdefault(key, {})
        if digest in history:
            return AppendReceipt(
                identity=f"watermark:{digest}",
                digest=digest,
                disposition="replayed",
            )
        previous = self._watermarks.get(key)
        if previous is not None:
            if previous.policy_version != watermark.policy_version:
                raise EconomicsConflict("watermark policy cannot change in place")
            if watermark.watermark_at < previous.watermark_at:
                raise EconomicsConflict("watermark cannot move backwards")
            if watermark.watermark_at == previous.watermark_at:
                raise EconomicsConflict("same watermark requires exact replay")
        history[digest] = watermark
        self._watermarks[key] = watermark
        return AppendReceipt(
            identity=f"watermark:{digest}",
            digest=digest,
            disposition="inserted",
        )

    def get_watermark(self, *, tenant_id: str, source_ref: str) -> Watermark | None:
        return self._watermarks.get((tenant_id, source_ref))

    def read_usage_windows(self, request: UsageReadRequest) -> UsageReadPage:
        query_digest = request.query_digest()
        if request.after is not None and request.after.query_digest != query_digest:
            raise TenantScopeError("cursor is bound to a different usage query")
        rows = [
            aggregate
            for _, aggregate in self._aggregates.values()
            if aggregate.tenant_id == request.scope.tenant_id
            and (
                request.granularity is None
                or aggregate.window.granularity == request.granularity
            )
            and (
                request.allocation_project_ref is None
                or aggregate.allocation.project_ref == request.allocation_project_ref
            )
            and (
                request.allocation_cost_center_ref is None
                or aggregate.allocation.cost_center_ref
                == request.allocation_cost_center_ref
            )
        ]
        rows.sort(key=lambda row: (row.window.window_start, row.aggregate_id))
        if request.after is not None:
            rows = [row for row in rows if _aggregate_key(row) > request.after.last_key]
        page_rows = tuple(rows[: request.limit])
        exhausted = len(rows) <= request.limit
        next_cursor = None
        if not exhausted and page_rows:
            next_cursor = KeysetCursor(
                tenant_id=request.scope.tenant_id,
                query_digest=query_digest,
                last_key=_aggregate_key(page_rows[-1]),
            )
        return UsageReadPage(
            scope=request.scope,
            rows=page_rows,
            next_cursor=next_cursor,
            exhausted=exhausted,
        )

    def read_slo_rollups(self, request: SloReadRequest) -> SloReadPage:
        query_digest = request.query_digest()
        if request.after is not None and request.after.query_digest != query_digest:
            raise TenantScopeError("cursor is bound to a different SLO query")
        rows = [
            rollup
            for _, rollup in self._rollups.values()
            if rollup.tenant_id == request.scope.tenant_id
            and (
                request.objective_id is None
                or rollup.objective_id == request.objective_id
            )
        ]
        rows.sort(key=lambda row: (row.window.window_start, row.rollup_id))
        if request.after is not None:
            rows = [row for row in rows if _rollup_key(row) > request.after.last_key]
        page_rows = tuple(rows[: request.limit])
        exhausted = len(rows) <= request.limit
        next_cursor = None
        if not exhausted and page_rows:
            next_cursor = KeysetCursor(
                tenant_id=request.scope.tenant_id,
                query_digest=query_digest,
                last_key=_rollup_key(page_rows[-1]),
            )
        return SloReadPage(
            scope=request.scope,
            rows=page_rows,
            next_cursor=next_cursor,
            exhausted=exhausted,
        )

    def iter_facts(self, *, tenant_id: str) -> Iterable[UsageFact]:
        """Fixture-only bounded tenant-scoped fact iteration."""

        for _, fact in sorted(self._facts.values(), key=lambda item: item[1].fact_id):
            if fact.tenant_id == tenant_id:
                yield fact


def _watermark_digest(watermark: Watermark) -> str:
    from .models import content_digest

    return content_digest(watermark.model_dump(mode="json"))


def _aggregate_key(aggregate: WindowAggregate) -> str:
    return f"{aggregate.window.window_start.isoformat()}/{aggregate.aggregate_id}"


def _rollup_key(rollup: SloWindowRollup) -> str:
    return f"{rollup.window.window_start.isoformat()}/{rollup.rollup_id}"


def _intervals_overlap(first: PriceCard, second: PriceCard) -> bool:
    first_end = first.effective_to
    second_end = second.effective_to
    # An open interval is treated as extending past every bounded interval.  Two
    # open cards for the same version are therefore always a conflict.
    return (first_end is None or second.effective_from < first_end) and (
        second_end is None or first.effective_from < second_end
    )


__all__ = [
    "AppendDisposition",
    "AppendReceipt",
    "EconomicsRepository",
    "InMemoryEconomicsRepository",
    "RepositoryUnavailable",
]
