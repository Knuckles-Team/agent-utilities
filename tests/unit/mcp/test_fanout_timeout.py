"""Per-target timeout for graph fan-out (CONCEPT:KG-2.63 follow-up).

A slow/hung backend must not stall a ``target='all'`` fan-out: ``fanout_execute``
runs each target concurrently under a per-target wall-clock budget, returning the
fast results and marking the slow one as a timeout error (partial success) — instead
of the old sequential loop that blocked on the slowest backend.
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.mcp.kg_server import fanout_execute

pytestmark = pytest.mark.concept("KG-2.63")


def _fast(name, _engine):
    return f"ok:{name}"


def test_fast_targets_all_return():
    results, errors = fanout_execute([("a", None), ("b", None)], _fast, timeout=5)
    assert results == {"a": "ok:a", "b": "ok:b"}
    assert errors == {}


def test_slow_target_times_out_while_others_return():
    def fn(name, _engine):
        if name == "slow":
            time.sleep(10)
            return "never"
        return f"ok:{name}"

    started = time.time()
    results, errors = fanout_execute(
        [("fast1", None), ("slow", None), ("fast2", None)], fn, timeout=1
    )
    elapsed = time.time() - started

    assert elapsed < 4  # did NOT wait 10s for the slow target
    assert results == {"fast1": "ok:fast1", "fast2": "ok:fast2"}
    assert "slow" in errors and "timed out" in errors["slow"]


def test_raising_target_is_captured_not_propagated():
    def fn(name, _engine):
        if name == "bad":
            raise RuntimeError("boom")
        return "ok"

    results, errors = fanout_execute([("ok1", None), ("bad", None)], fn, timeout=5)
    assert results == {"ok1": "ok"}
    assert "bad" in errors and "boom" in errors["bad"]


def test_empty_entries():
    assert fanout_execute([], _fast, timeout=5) == ({}, {})
