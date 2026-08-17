"""GOC-21 known-bad proofs: fencing rejection + bounded-wait fairness.

Two properties the lane brief requires as evidence, each proved against a
KNOWN-BAD input (never merely a happy-path pass):

1. An expired OR wrong-epoch/wrong-fence lease holder is REJECTED, not
   silently allowed to keep working (`TestFencingRejectsStaleHolder`).
2. A low-priority flood of unbounded size cannot delay a high-priority
   request beyond the stated bound
   `ceil(active_tenants_in_priority / weight)` rounds
   (`TestBoundedFairnessAgainstFlood`), independent of the flood's size.
"""

from __future__ import annotations

import math

import pytest

from agent_utilities.core.resource_priority import PriorityClass
from agent_utilities.orchestration.capacity_leases import (
    CapacityCell,
    CapacityExhausted,
    CapacityLedger,
    FairRequest,
    HierarchicalFairScheduler,
    IdempotencyConflict,
    LeaseExpiredError,
    LeaseNotFound,
    StaleEpoch,
    StaleFence,
    TenantBackpressure,
)


def _cell(*, epoch: int = 1, capacity: int = 10, reserved_floor: int = 2) -> CapacityCell:
    return CapacityCell(
        cell_id="cell-llm-1",
        resource_class="llm_generator",
        capacity=capacity,
        reserved_floor=reserved_floor,
        epoch=epoch,
        policy_digest="d" * 64,
    )


# ── 1. fencing / expiry known-bad proofs ─────────────────────────────────────


class TestFencingRejectsStaleHolder:
    def test_expired_holder_rejected_on_renew(self) -> None:
        ledger = CapacityLedger()
        cell = _cell()
        lease = ledger.try_acquire(
            cell,
            lease_id="lease-1",
            work_item_id="wi-1",
            tenant_ref="tenant-a",
            actor_digest="actor-a",
            amount=1,
            priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-1",
            now_ms=0,
            ttl_ms=1_000,
        )
        # Never renewed; far past its TTL.
        with pytest.raises(LeaseExpiredError):
            ledger.try_renew(
                lease.lease_id,
                fence_token=lease.fence_token,
                lease_epoch=lease.lease_epoch,
                cell_epoch=cell.epoch,
                now_ms=5_000,
                ttl_ms=1_000,
            )

    def test_expired_holder_rejected_on_release(self) -> None:
        ledger = CapacityLedger()
        cell = _cell()
        lease = ledger.try_acquire(
            cell,
            lease_id="lease-1b",
            work_item_id="wi-1",
            tenant_ref="tenant-a",
            actor_digest="actor-a",
            amount=1,
            priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-1b",
            now_ms=0,
            ttl_ms=1_000,
        )
        with pytest.raises(LeaseExpiredError):
            ledger.try_release(
                lease.lease_id,
                fence_token=lease.fence_token,
                lease_epoch=lease.lease_epoch,
                cell_epoch=cell.epoch,
                now_ms=5_000,
            )

    def test_stale_epoch_holder_rejected_even_when_unexpired(self) -> None:
        """The classic fencing defect: an old cell-authority epoch is rejected
        even though the lease's own TTL has not lapsed — the cell failed over
        / was repartitioned and this holder's authority no longer applies."""
        ledger = CapacityLedger()
        old_cell = _cell(epoch=1)
        lease = ledger.try_acquire(
            old_cell,
            lease_id="lease-2",
            work_item_id="wi-2",
            tenant_ref="tenant-a",
            actor_digest="actor-a",
            amount=1,
            priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-2",
            now_ms=0,
            ttl_ms=1_000_000,  # huge TTL: NOT a timestamp-expiry story
        )
        with pytest.raises(StaleEpoch) as excinfo:
            ledger.try_renew(
                lease.lease_id,
                fence_token=lease.fence_token,
                lease_epoch=lease.lease_epoch,  # still 1
                cell_epoch=2,  # cell failed over to epoch 2
                now_ms=500,
                ttl_ms=1_000,
            )
        assert excinfo.value.cell_epoch == 2
        assert excinfo.value.caller_epoch == 1

    def test_forged_fence_token_rejected(self) -> None:
        ledger = CapacityLedger()
        cell = _cell()
        lease = ledger.try_acquire(
            cell,
            lease_id="lease-3",
            work_item_id="wi-3",
            tenant_ref="tenant-a",
            actor_digest="actor-a",
            amount=1,
            priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-3",
            now_ms=0,
            ttl_ms=10_000,
        )
        with pytest.raises(StaleFence):
            ledger.try_renew(
                lease.lease_id,
                fence_token=lease.fence_token + 1,  # forged/guessed
                lease_epoch=lease.lease_epoch,
                cell_epoch=cell.epoch,
                now_ms=100,
                ttl_ms=1_000,
            )

    def test_reclaim_frees_capacity_without_cooperating_holder(self) -> None:
        """A holder that never calls release (the crashed-worker case) does
        not permanently strand capacity: `reclaim_expired` frees it on TTL
        lapse, and the freed capacity is immediately usable by a NEW holder —
        proof that the expired original cannot keep it fenced forever
        merely by never showing back up."""
        ledger = CapacityLedger()
        cell = _cell(capacity=10, reserved_floor=0)
        ledger.try_acquire(
            cell,
            lease_id="hog",
            work_item_id="wi-hog",
            tenant_ref="tenant-hog",
            actor_digest="actor-hog",
            amount=10,
            priority=PriorityClass.BACKGROUND_INGESTION,
            idempotency_key="idem-hog",
            now_ms=0,
            ttl_ms=1_000,
        )
        with pytest.raises(CapacityExhausted):
            ledger.try_acquire(
                cell,
                lease_id="second",
                work_item_id="wi-2",
                tenant_ref="tenant-b",
                actor_digest="actor-b",
                amount=1,
                priority=PriorityClass.INTERACTIVE,
                idempotency_key="idem-2",
                now_ms=500,
                ttl_ms=1_000,
            )
        reclaimed = ledger.reclaim_expired(2_000)
        assert reclaimed == ["hog"]
        lease = ledger.try_acquire(
            cell,
            lease_id="third",
            work_item_id="wi-3",
            tenant_ref="tenant-b",
            actor_digest="actor-b",
            amount=1,
            priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-3",
            now_ms=2_100,
            ttl_ms=1_000,
        )
        assert lease.amount == 1

    def test_unknown_lease_id_rejected(self) -> None:
        ledger = CapacityLedger()
        with pytest.raises(LeaseNotFound):
            ledger.try_renew(
                "no-such-lease",
                fence_token=1,
                lease_epoch=1,
                cell_epoch=1,
                now_ms=0,
                ttl_ms=1_000,
            )

    def test_idempotent_replay_returns_same_lease(self) -> None:
        ledger = CapacityLedger()
        cell = _cell()
        first = ledger.try_acquire(
            cell, lease_id="lease-idem", work_item_id="wi-1", tenant_ref="tenant-a",
            actor_digest="actor-a", amount=1, priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-shared", now_ms=0, ttl_ms=1_000,
        )
        second = ledger.try_acquire(
            cell, lease_id="lease-idem-DIFFERENT-id", work_item_id="wi-1", tenant_ref="tenant-a",
            actor_digest="actor-a", amount=1, priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-shared", now_ms=1, ttl_ms=1_000,
        )
        assert first.fence_token == second.fence_token
        assert first.lease_id == second.lease_id

    def test_idempotency_conflict_on_shape_mismatch(self) -> None:
        ledger = CapacityLedger()
        cell = _cell()
        ledger.try_acquire(
            cell, lease_id="lease-a", work_item_id="wi-1", tenant_ref="tenant-a",
            actor_digest="actor-a", amount=1, priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-shared", now_ms=0, ttl_ms=1_000,
        )
        with pytest.raises(IdempotencyConflict):
            ledger.try_acquire(
                cell, lease_id="lease-b", work_item_id="wi-1", tenant_ref="tenant-a",
                actor_digest="actor-a", amount=999, priority=PriorityClass.INTERACTIVE,
                idempotency_key="idem-shared", now_ms=1, ttl_ms=1_000,
            )

    def test_background_flood_cannot_touch_reserved_floor(self) -> None:
        ledger = CapacityLedger()
        cell = _cell(capacity=10, reserved_floor=2)
        for i in range(8):  # fills the entire spare (10 - 2) pool
            ledger.try_acquire(
                cell, lease_id=f"flood-{i}", work_item_id="wi-flood", tenant_ref="tenant-flood",
                actor_digest="actor-flood", amount=1, priority=PriorityClass.BACKGROUND_INGESTION,
                idempotency_key=f"idem-flood-{i}", now_ms=0, ttl_ms=1_000_000,
            )
        with pytest.raises(CapacityExhausted):
            ledger.try_acquire(
                cell, lease_id="flood-8", work_item_id="wi-flood", tenant_ref="tenant-flood",
                actor_digest="actor-flood", amount=1, priority=PriorityClass.BACKGROUND_INGESTION,
                idempotency_key="idem-flood-8", now_ms=0, ttl_ms=1_000_000,
            )
        interactive = ledger.try_acquire(
            cell, lease_id="interactive-1", work_item_id="wi-int", tenant_ref="tenant-int",
            actor_digest="actor-int", amount=2, priority=PriorityClass.INTERACTIVE,
            idempotency_key="idem-int-1", now_ms=0, ttl_ms=1_000,
        )
        assert interactive.amount == 2


# ── 2. bounded-wait fairness against an unbounded flood ─────────────────────


class TestBoundedFairnessAgainstFlood:
    def test_flood_cannot_starve_interactive_beyond_stated_bound(self) -> None:
        """KNOWN-BAD PROOF: submit a `MAX_TENANT_BACKLOG`-sized (this module's
        own bounded-queue-memory ceiling — the largest flood a single tenant
        can even hold pending at once) BACKGROUND_INGESTION flood from ONE
        tenant, then submit ONE INTERACTIVE request from a different tenant.
        The stated bound is
        `ceil(active_interactive_tenants / weight) = ceil(1/1) = 1` round: the
        interactive request must be admitted within the very first round,
        regardless of the flood's size.
        """
        from agent_utilities.orchestration.capacity_leases import MAX_TENANT_BACKLOG

        cell = _cell(capacity=10, reserved_floor=2)
        ledger = CapacityLedger()
        scheduler = HierarchicalFairScheduler(cell, ledger)

        for i in range(MAX_TENANT_BACKLOG):
            scheduler.submit(
                FairRequest(f"flood-{i}", "tenant-flood", PriorityClass.BACKGROUND_INGESTION)
            )
        scheduler.submit(FairRequest("vip-1", "tenant-vip", PriorityClass.INTERACTIVE))

        bound = scheduler.worst_case_wait_rounds(PriorityClass.INTERACTIVE, "tenant-vip")
        assert bound == 1

        admitted_round_1 = scheduler.run_round(now_ms=0)
        vip_leases = [lease for lease in admitted_round_1 if lease.tenant_ref == "tenant-vip"]
        assert len(vip_leases) == 1, (
            "interactive request must be admitted within its proved 1-round "
            f"bound regardless of a {MAX_TENANT_BACKLOG}-request background flood"
        )
        # And the flood, of that entire size, could only ever take the spare
        # pool (8 units) in round 1 — never the reserved floor the interactive
        # request drew from.
        flood_leases = [lease for lease in admitted_round_1 if lease.tenant_ref == "tenant-flood"]
        assert len(flood_leases) <= (cell.capacity - cell.reserved_floor)

    def test_bound_scales_with_distinct_tenants_not_queue_depth(self) -> None:
        """Doubling ONE tenant's backlog does not change ANY tenant's bound;
        adding a SECOND distinct interactive tenant does (by exactly the
        formula), proving the bound is driven by tenant count, not flood size.
        """
        cell = _cell(capacity=100, reserved_floor=50)
        ledger = CapacityLedger()
        scheduler = HierarchicalFairScheduler(cell, ledger)
        scheduler.submit(FairRequest("a-1", "tenant-a", PriorityClass.INTERACTIVE))
        scheduler.submit(FairRequest("b-1", "tenant-b", PriorityClass.INTERACTIVE))
        bound_before = scheduler.worst_case_wait_rounds(PriorityClass.INTERACTIVE, "tenant-a")
        assert bound_before == math.ceil(2 / 1)

        # Flood tenant-b's OWN backlog deeper (up to this module's bounded-queue
        # ceiling) — tenant-a's bound must not move.
        from agent_utilities.orchestration.capacity_leases import MAX_TENANT_BACKLOG

        for i in range(MAX_TENANT_BACKLOG - 1):  # -1: "b-1" already queued above
            scheduler.submit(FairRequest(f"b-{i}", "tenant-b", PriorityClass.INTERACTIVE))
        bound_after_same_tenant_flood = scheduler.worst_case_wait_rounds(
            PriorityClass.INTERACTIVE, "tenant-a"
        )
        assert bound_after_same_tenant_flood == bound_before

        # A THIRD distinct tenant does move the bound, by the stated formula.
        scheduler.submit(FairRequest("c-1", "tenant-c", PriorityClass.INTERACTIVE))
        bound_with_third_tenant = scheduler.worst_case_wait_rounds(
            PriorityClass.INTERACTIVE, "tenant-a"
        )
        assert bound_with_third_tenant == math.ceil(3 / 1)

    def test_weighted_tenant_gets_proportionally_more_turns(self) -> None:
        cell = _cell(capacity=100, reserved_floor=100)
        ledger = CapacityLedger()
        scheduler = HierarchicalFairScheduler(
            cell, ledger, weights={"tenant-heavy": 3}
        )
        for i in range(10):
            scheduler.submit(FairRequest(f"heavy-{i}", "tenant-heavy", PriorityClass.INTERACTIVE))
            scheduler.submit(FairRequest(f"light-{i}", "tenant-light", PriorityClass.INTERACTIVE))

        admitted = scheduler.run_round(now_ms=0)
        heavy_count = sum(1 for lease in admitted if lease.tenant_ref == "tenant-heavy")
        light_count = sum(1 for lease in admitted if lease.tenant_ref == "tenant-light")
        assert heavy_count == 3
        assert light_count == 1

    def test_backpressure_sheds_instead_of_buffering_unbounded(self) -> None:
        cell = _cell()
        ledger = CapacityLedger()
        scheduler = HierarchicalFairScheduler(cell, ledger)
        from agent_utilities.orchestration.capacity_leases import MAX_TENANT_BACKLOG

        for i in range(MAX_TENANT_BACKLOG):
            scheduler.submit(
                FairRequest(f"r-{i}", "tenant-x", PriorityClass.BACKGROUND_INGESTION)
            )
        with pytest.raises(TenantBackpressure):
            scheduler.submit(
                FairRequest("overflow", "tenant-x", PriorityClass.BACKGROUND_INGESTION)
            )

    def test_multi_round_flood_never_exceeds_bound_across_many_rounds(self) -> None:
        """A running flood across MANY rounds still never delays a freshly
        arriving interactive request past its 1-round bound."""
        cell = _cell(capacity=10, reserved_floor=2)
        ledger = CapacityLedger()
        scheduler = HierarchicalFairScheduler(cell, ledger)
        now_ms = 0
        for round_i in range(20):
            for i in range(100):
                scheduler.submit(
                    FairRequest(
                        f"flood-{round_i}-{i}", "tenant-flood", PriorityClass.BACKGROUND_INGESTION
                    )
                )
            scheduler.run_round(now_ms=now_ms)
            now_ms += 100

        # Now a fresh interactive tenant arrives mid-flood.
        scheduler.submit(FairRequest("late-vip", "tenant-late-vip", PriorityClass.INTERACTIVE))
        admitted = scheduler.run_round(now_ms=now_ms)
        assert any(lease.tenant_ref == "tenant-late-vip" for lease in admitted)
