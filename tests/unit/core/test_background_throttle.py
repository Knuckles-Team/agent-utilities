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
