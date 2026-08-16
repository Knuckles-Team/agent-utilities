"""Tests for burst-mode message coalescing (CONCEPT:AU-ECO.messaging.burst-mode-coalescing)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.messaging.coalescer import BurstCoalescer


@pytest.mark.asyncio
async def test_burst_collapses_to_one_flush() -> None:
    flushes: list[tuple[str, list[Any]]] = []

    async def on_flush(key: str, items: list[Any]) -> None:
        flushes.append((key, items))

    # GOC-70: `BurstCoalescer` debounces on real wall-clock time (no injectable
    # clock), so this test's correctness depends on each 0.02s submit gap
    # staying well under `window_s` even under scheduler contention (e.g.
    # pytest-xdist workers sharing 2 CPUs delay this process's event loop).
    # window_s=0.3 keeps a 15x margin over the 0.02s gap — the same shape of
    # timing-dependent-property-as-invariant risk as the eg `ops==400` bug,
    # mitigated here with a margin generous enough that only a genuine hang
    # (not ordinary scheduling jitter) could still trip it.
    c = BurstCoalescer(on_flush, window_s=0.3, max_wait_s=5)
    for i in range(5):
        await c.submit("telegram:42", i)
        await asyncio.sleep(0.02)  # faster than the window → one batch
    await asyncio.sleep(0.6)  # comfortably past window_s once the burst ends
    assert len(flushes) == 1
    assert flushes[0] == ("telegram:42", [0, 1, 2, 3, 4])


@pytest.mark.asyncio
async def test_separate_bursts_flush_separately() -> None:
    flushes: list[list[Any]] = []

    async def on_flush(key: str, items: list[Any]) -> None:
        flushes.append(items)

    c = BurstCoalescer(on_flush, window_s=0.08, max_wait_s=5)
    await c.submit("k", "a")
    await asyncio.sleep(0.2)  # window elapses → flush #1
    await c.submit("k", "b")
    await asyncio.sleep(0.2)  # flush #2
    assert flushes == [["a"], ["b"]]


@pytest.mark.asyncio
async def test_hard_cap_flushes_a_nonstop_typer() -> None:
    flushes: list[list[Any]] = []

    async def on_flush(key: str, items: list[Any]) -> None:
        flushes.append(items)

    # Submit faster than the window so it keeps resetting, but max_wait still forces a flush.
    # GOC-70: the original window_s=0.05 vs a 0.03s submit gap left only a
    # ~1.7x margin — thin enough that ordinary scheduler jitter on a
    # contended/low-core host could let a gap exceed the window and flush
    # early, or (less likely) delay a submit past max_wait before the hard
    # cap's own check fires. Widened to a ~4x window/gap margin (0.16s vs
    # 0.04s) and max_wait_s=0.2s, which the 7-submit loop (7 * 0.04s = 0.28s
    # total) still comfortably crosses with slack to spare before the loop
    # ends, so the hard-cap path is exercised reliably rather than by luck.
    c = BurstCoalescer(on_flush, window_s=0.16, max_wait_s=0.2)
    await c.submit("k", 1)
    for i in range(2, 9):
        await asyncio.sleep(0.04)  # < window, so the quiet window never elapses
        await c.submit("k", i)  # eventually crosses max_wait → flush
    await asyncio.sleep(0.4)
    assert flushes and 1 in flushes[0]


@pytest.mark.asyncio
async def test_keys_are_independent() -> None:
    flushes: dict[str, list[Any]] = {}

    async def on_flush(key: str, items: list[Any]) -> None:
        flushes[key] = items

    c = BurstCoalescer(on_flush, window_s=0.08, max_wait_s=5)
    await c.submit("a", 1)
    await c.submit("b", 2)
    await asyncio.sleep(0.2)
    assert flushes == {"a": [1], "b": [2]}


@pytest.mark.asyncio
async def test_flush_does_not_cancel_its_own_handler() -> None:
    """Regression (ECO-4.74): the debounce-timer flush must not cancel itself.

    _wait_and_flush (the timer task) calls _flush, which popped + cancelled the timer — i.e.
    cancelled THIS running task — killing on_flush at its first await. This killed every
    coalesced reply. on_flush must run to completion even though it awaits.
    """
    from agent_utilities.messaging.coalescer import BurstCoalescer

    completed = []

    async def on_flush(key, items):
        await asyncio.sleep(0.3)  # an await is where the bogus cancellation struck
        completed.append((key, len(items)))

    c = BurstCoalescer(on_flush, window_s=0.2, max_wait_s=5)
    await c.submit("k", {"x": 1})
    await asyncio.sleep(1.0)
    assert completed == [("k", 1)], "on_flush was cancelled mid-flight"
