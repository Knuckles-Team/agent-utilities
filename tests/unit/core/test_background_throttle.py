"""Background throttle + foreground pause (CONCEPT:KG-2.7)."""

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
