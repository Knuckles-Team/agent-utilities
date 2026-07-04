"""Background throttle + foreground pause (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

from __future__ import annotations

import threading
import time

from agent_utilities.core.background_throttle import BackgroundThrottle


def test_foreground_flag_reentrant():
    t = BackgroundThrottle()
    assert not t.foreground_active
    t.set_foreground(True)
    t.set_foreground(True)
    assert t.foreground_active
    t.set_foreground(False)
    assert t.foreground_active  # still one holder
    t.set_foreground(False)
    assert not t.foreground_active


def test_foreground_context_manager():
    t = BackgroundThrottle()
    with t.foreground():
        assert t.foreground_active
    assert not t.foreground_active


def test_background_slot_skips_when_foreground_and_no_wait():
    t = BackgroundThrottle()
    t.set_foreground(True)
    with t.background_slot(wait_foreground=False) as ok:
        assert ok is False  # caller should skip this cycle


def test_background_slot_acquires_when_idle():
    t = BackgroundThrottle(max_concurrent=1)
    with t.background_slot() as ok:
        assert ok is True


def test_background_slot_yields_until_foreground_clears():
    t = BackgroundThrottle(max_concurrent=1)
    t.set_foreground(True)
    proceeded = threading.Event()

    def worker():
        with t.background_slot(fg_poll=0.02) as ok:
            if ok:
                proceeded.set()

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.1)
    assert not proceeded.is_set()  # blocked by foreground
    t.set_foreground(False)
    th.join(timeout=2)
    assert proceeded.is_set()  # proceeded once foreground cleared


def test_bulk_ingest_flag_reentrant():
    t = BackgroundThrottle()
    assert not t.bulk_ingest_active
    t.set_bulk_ingest(True)
    t.set_bulk_ingest(True)
    assert t.bulk_ingest_active
    t.set_bulk_ingest(False)
    assert t.bulk_ingest_active  # still one holder
    t.set_bulk_ingest(False)
    assert not t.bulk_ingest_active


def test_bulk_ingest_context_manager():
    t = BackgroundThrottle()
    with t.bulk_ingest():
        assert t.bulk_ingest_active
    assert not t.bulk_ingest_active


def test_should_yield_background_covers_foreground_and_ingest():
    t = BackgroundThrottle()
    assert not t.should_yield_background
    with t.foreground():
        assert t.should_yield_background  # interactive
    assert not t.should_yield_background
    with t.bulk_ingest():
        assert t.should_yield_background  # bulk ingest
    assert not t.should_yield_background
    # Both at once, independently reentrant.
    with t.foreground(), t.bulk_ingest():
        assert t.should_yield_background


def test_wait_while_busy_blocks_then_returns_when_cleared():
    t = BackgroundThrottle()
    t.set_bulk_ingest(True)
    returned = threading.Event()
    result = {}

    def worker():
        result["cleared"] = t.wait_while_busy(poll=0.02, max_wait=5.0)
        returned.set()

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.1)
    assert not returned.is_set()  # still paused — ingest active
    t.set_bulk_ingest(False)
    th.join(timeout=2)
    assert returned.is_set()
    assert result["cleared"] is True  # returned because gate cleared, not timeout


def test_wait_while_busy_gives_up_after_max_wait():
    # A permanently-busy gate must never starve background work forever.
    t = BackgroundThrottle()
    t.set_foreground(True)
    t0 = time.time()
    cleared = t.wait_while_busy(poll=0.02, max_wait=0.1)
    assert cleared is False
    assert time.time() - t0 >= 0.1
    t.set_foreground(False)


# ── enrichment-lane fairness (CONCEPT:AU-KG.ontology.capability-card-backfill-lane, MUST-FIX) ────────────────────
#
# Regression guards for the dedicated OWL card-enrichment lane. The bug: the
# enrichment_backfill task was wrapped in an OUTER ``background_slot()`` at the
# task-dispatch boundary (engine_tasks ``_execute_claimed_task``) AND the
# per-batch ``_tick_enrichment`` loop acquired the slot AGAIN per batch — a
# double-acquire. Holding one outer permit for a whole 16*64=1024-symbol tick
# (a) capped the lane at ``_BACKGROUND_MAX_CONCURRENT`` no matter how many spare
# workers it could expand into, and (b) starved the other ~17 background/maint
# ticks for the full tick. The fix: enrichment is NOT wrapped at dispatch; the
# per-batch acquire/release in ``_tick_enrichment`` is the only gate, so the
# permit is yielded BETWEEN batches and other background work interleaves.
#
# These tests drive the REAL ``BackgroundThrottle`` the way the fixed code does:
# an enrichment "tick" is modelled as a loop that acquires ``background_slot()``
# per batch and releases it between batches (mirroring engine_tasks line ~1595).


def _enrichment_tick(throttle, *, batches, batch_work, progress, hold_between=0.0):
    """Model ``_tick_enrichment``: acquire a background_slot PER BATCH, do the
    batch's work, then RELEASE before looping. Mirrors the fixed dispatch where
    enrichment is NOT wrapped in an outer per-tick slot."""
    for _ in range(batches):
        with throttle.background_slot(acquire_timeout=5.0) as ok:
            assert ok, "enrichment batch should eventually get a slot"
            progress.append(1)
            if batch_work:
                time.sleep(batch_work)
        if hold_between:
            time.sleep(hold_between)


def test_two_concurrent_enrichment_dispatches_both_progress():
    """Two concurrent enrichment_backfill dispatches must BOTH make progress
    (not serialize to one) even when the shared semaphore is the binding cap.

    Because each tick releases its permit between batches, two enrichment
    workers interleave their batches rather than one running to completion
    while the other is fully starved.
    """
    # cap=1 is the worst case: the ONLY way both can progress is if each
    # releases between batches so the other can grab the freed permit.
    t = BackgroundThrottle(max_concurrent=1)
    progress_a: list[int] = []
    progress_b: list[int] = []

    def worker(progress):
        _enrichment_tick(
            t, batches=6, batch_work=0.01, progress=progress, hold_between=0.005
        )

    th_a = threading.Thread(target=worker, args=(progress_a,))
    th_b = threading.Thread(target=worker, args=(progress_b,))
    th_a.start()
    th_b.start()
    th_a.join(timeout=10)
    th_b.join(timeout=10)
    assert not th_a.is_alive() and not th_b.is_alive()
    # Both ticks completed all their batches — neither was starved to zero.
    assert sum(progress_a) == 6
    assert sum(progress_b) == 6


def test_long_enrichment_tick_does_not_block_other_background_task():
    """A long enrichment tick (many batches) must NOT hold the background
    permit for its whole duration — another background task acquires a slot
    while the enrichment tick is mid-flight (between batches).

    This is the fairness regression the MUST-FIX targets: under the old
    per-tick outer hold (cap=1), the other background task could only run AFTER
    the entire 1024-symbol tick finished.
    """
    t = BackgroundThrottle(max_concurrent=1)
    other_acquired = threading.Event()
    enrichment_done = threading.Event()
    enrichment_progress: list[int] = []

    def long_enrichment():
        # A "long" tick: many batches with per-batch work + a yield gap between.
        _enrichment_tick(
            t,
            batches=20,
            batch_work=0.01,
            progress=enrichment_progress,
            hold_between=0.01,
        )
        enrichment_done.set()

    def other_background():
        # A different background task wanting a single slot.
        with t.background_slot(acquire_timeout=5.0) as ok:
            if ok:
                other_acquired.set()

    th_enrich = threading.Thread(target=long_enrichment)
    th_enrich.start()
    # Let the enrichment tick get going (a few batches in).
    time.sleep(0.05)
    th_other = threading.Thread(target=other_background)
    th_other.start()
    # The other background task must acquire its slot WELL BEFORE the long
    # enrichment tick finishes — proving the permit is yielded between batches.
    assert other_acquired.wait(timeout=2.0), (
        "other background task was starved by the enrichment tick "
        "(permit not released between batches)"
    )
    assert not enrichment_done.is_set(), (
        "enrichment tick already finished — test did not exercise mid-tick interleaving"
    )
    th_enrich.join(timeout=10)
    th_other.join(timeout=10)
    assert not th_enrich.is_alive() and not th_other.is_alive()
    assert sum(enrichment_progress) == 20
