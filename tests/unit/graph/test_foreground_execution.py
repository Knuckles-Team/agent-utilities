"""The canonical execution engine must ramp background work down."""

from __future__ import annotations

import pytest

from agent_utilities.core import background_throttle
from agent_utilities.orchestration.engine import _foreground_execution


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch):
    throttle = background_throttle.BackgroundThrottle()
    monkeypatch.setattr(background_throttle, "_throttle", throttle)
    return throttle


@pytest.mark.asyncio
async def test_foreground_execution_marks_the_shared_lease_during_a_run() -> None:
    seen: list[bool] = []

    @_foreground_execution
    async def run() -> str:
        seen.append(background_throttle.get_throttle().foreground_active)
        return "done"

    assert await run() == "done"
    assert seen == [True]
    assert not background_throttle.get_throttle().foreground_active


@pytest.mark.asyncio
async def test_foreground_execution_keeps_the_lease_for_the_full_stream() -> None:
    seen: list[bool] = []

    @_foreground_execution
    async def stream():
        seen.append(background_throttle.get_throttle().foreground_active)
        yield 1
        seen.append(background_throttle.get_throttle().foreground_active)
        yield 2

    assert [item async for item in stream()] == [1, 2]
    assert seen == [True, True]
    assert not background_throttle.get_throttle().foreground_active
