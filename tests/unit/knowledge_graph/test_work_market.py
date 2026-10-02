"""The harness-evolution spec's graph-driven work market, end to end through the typed EG surfaces.

signal → one Gap + its WorkItem → derived offer → fenced claim (one winner) →
terminal outcome → engine-read evidence on the Gap → reopen on NEW evidence only,
with a failure cooldown. Deterministic: the fake EG market's clock and every
``now_ms`` are fixed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.research import gaps, work_market
from tests.unit.fleet_autonomy_fakes import verified_fleet_session
from tests.unit.work_market_fakes import FakeWorkMarket, attach_market


@pytest.fixture
def market() -> FakeWorkMarket:
    return FakeWorkMarket(now_ms=1_000)


@pytest.fixture
def engine(market: FakeWorkMarket) -> SimpleNamespace:
    engine = SimpleNamespace()
    attach_market(engine, market)
    return engine


def _signal(engine: SimpleNamespace, *refs: str) -> dict:
    gap = gaps.submit_gap(
        engine,
        source=gaps.SOURCE_FAILURE,
        signature="timeout",
        statement="tool calls time out under load",
        severity=0.9,
        evidence_refs=list(refs),
    )
    assert gap is not None
    return gap


def _view(market: FakeWorkMarket, tenant: str = "fleet-autonomy") -> dict:
    return market.gaps.get(tenant=tenant, gap_id="gap:failure:timeout")


def test_a_signal_becomes_one_gap_with_its_claimable_work_item(engine, market):
    with verified_fleet_session():
        gap = _signal(engine, "trace:1")
        again = _signal(engine, "trace:1")
        merged = _signal(engine, "trace:1", "trace:2")
    assert gap["id"] == "gap:failure:timeout"
    assert gap["priority_bucket"] == 0
    assert market.items[gap["work_item_id"]]["status"] == "ready"
    assert again["revision"] == gap["revision"], (
        "an unseen-evidence-free signal changes nothing"
    )
    assert merged["work_item_id"] == gap["work_item_id"]
    assert len(market.items) == 1, "one Gap, one WorkItem"
    assert len(merged["evidence"]) == 2


def test_gaps_are_tenant_scoped(engine, market):
    with verified_fleet_session("tenant-a"):
        _signal(engine, "trace:1")
    with verified_fleet_session("tenant-b"):
        assert gaps.get_gap(engine, "gap:failure:timeout") is None
        assert gaps.open_gaps(engine) == []
        own = _signal(engine, "trace:1")
    assert own["work_item_id"] != _view(market, "tenant-a")["work_item_id"]


def test_the_sweep_prices_from_the_gaps_own_evidence(engine, market):
    with verified_fleet_session():
        gap = _signal(engine, "trace:1")
        counts = work_market.reconcile_market(engine, now_ms=1_500)
        again = work_market.reconcile_market(engine, now_ms=1_600)
    assert counts["priced"] == 1 and again["priced"] == 0
    offer = _view(market)["offer"]
    assert offer["work_item_id"] == gap["work_item_id"]
    assert offer["offer"]["evidence_digests"] == [e["digest"] for e in gap["evidence"]]
    # 9_000_000 utility micros x 500_000 ppm / 40_000 cost microunits
    assert offer["utility_rate"] == 112_500_000


def test_two_claimants_of_one_gap_item_cannot_both_win(engine, market):
    with verified_fleet_session():
        gap = _signal(engine, "trace:1")
        first = work_market.claim_gap_work(engine, gap, worker="w-1", now_ms=2_000)
        second = work_market.claim_gap_work(engine, gap, worker="w-2", now_ms=2_000)
    assert first is not None and first["lease_holder_ref"] == "w-1"
    assert second is None


def test_a_completed_item_writes_its_evidence_back_to_the_gap(engine, market):
    with verified_fleet_session():
        gap = _signal(engine, "trace:1")
        assert work_market.settle_gap_work(engine, gap["id"]) == "pending"
        work_market.claim_gap_work(engine, gap, worker="w-1", now_ms=2_000)
        market.finish(gap["work_item_id"], "w-1", "succeeded")
        counts = work_market.reconcile_market(engine, now_ms=3_000)
        closed = gaps.get_gap(engine, gap["id"])
    assert counts["settled"] == 1
    assert closed["status"] == gaps.STATUS_RESOLVED
    outcome = closed["evidence"][-1]
    assert (outcome["kind"], outcome["reference"]) == (
        "work_item_outcome",
        gap["work_item_id"],
    )


def test_a_failed_attempt_reopens_only_on_new_evidence_and_cools_down(engine, market):
    with verified_fleet_session():
        gap = _signal(engine, "trace:1")
        work_market.claim_gap_work(engine, gap, worker="w-1", now_ms=2_000)
        market.finish(gap["work_item_id"], "w-1", "failed")
        market.now_ms = 3_000
        assert work_market.settle_gap_work(engine, gap["id"]) == "deferred"
        assert _signal(engine, "trace:1")["status"] == gaps.STATUS_DEFERRED
        reopened = _signal(engine, "trace:9")
        work_market.reconcile_market(engine, now_ms=4_000)
    assert reopened["generation"] == 2
    assert reopened["work_item_id"] != gap["work_item_id"]
    offer = _view(market)["offer"]["offer"]
    assert offer["cooldown_until_ms"] == 3_000 + work_market.FAILURE_COOLDOWN_MS


def test_a_closed_gaps_unrun_work_item_is_cancelled_by_the_sweep(engine, market):
    with verified_fleet_session():
        gap = _signal(engine, "trace:1")
        assert gaps.link_gap_to_spec(engine, gap["id"], "spec:1")
        assert gaps.mark_gap_resolved(
            engine, gap["id"], reference="loop:develop:spec:1"
        )
        counts = work_market.reconcile_market(engine, now_ms=5_000)
        record = work_market.reconcile_market(engine, now_ms=5_100)
    assert counts["cancelled"] == 1
    assert record["settled"] == 1, "the cancellation is then recorded as evidence"
    assert market.items[gap["work_item_id"]]["status"] == "cancelled"
    assert _view(market)["status"] == gaps.STATUS_RESOLVED


def test_the_loop_stage_skips_without_the_typed_surface():
    assert work_market.run_market_stage(SimpleNamespace(), now_ms=1) == {
        "skipped": "connected engine has no typed Gap surface"
    }
    with pytest.raises(gaps.GapAuthorityUnavailable):
        gaps.gap_client(SimpleNamespace())


def test_open_gaps_is_a_listing_not_a_ranking(engine, market):
    with verified_fleet_session():
        for signature, severity in (("b-high", 0.95), ("a-low", 0.1)):
            gaps.submit_gap(
                engine,
                source=gaps.SOURCE_AUDIT,
                signature=signature,
                statement=f"finding {signature}",
                severity=severity,
            )
        listed = [g["id"] for g in gaps.open_gaps(engine)]
    assert listed == ["gap:audit:a-low", "gap:audit:b-high"], "EG key order, no AU sort"
