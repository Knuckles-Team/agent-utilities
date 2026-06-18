"""Tests for burst-mode message coalescing (CONCEPT:ECO-4.63)."""

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

    c = BurstCoalescer(on_flush, window_s=0.1, max_wait_s=5)
    for i in range(5):
        await c.submit("telegram:42", i)
        await asyncio.sleep(0.02)  # faster than the window → one batch
    await asyncio.sleep(0.2)
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
    c = BurstCoalescer(on_flush, window_s=0.05, max_wait_s=0.12)
    await c.submit("k", 1)
    for i in range(2, 9):
        await asyncio.sleep(0.03)  # < window, so the quiet window never elapses
        await c.submit("k", i)  # eventually crosses max_wait → flush
    await asyncio.sleep(0.1)
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
