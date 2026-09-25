"""A WorkItem harness run defers provenance until its fenced L5 commit."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_utilities.layers.trace_ownership import (
    l5_terminal_owns_trace,
    l5_terminal_trace_owner,
)
from agent_utilities.orchestration import agent_runner


def test_trace_owner_is_scoped_and_restored() -> None:
    assert not l5_terminal_owns_trace()
    with l5_terminal_trace_owner():
        assert l5_terminal_owns_trace()
    assert not l5_terminal_owns_trace()


def test_work_item_trace_owner_never_calls_legacy_writer(monkeypatch: Any) -> None:
    calls: list[str] = []

    async def legacy_write(*_args: Any, **_kwargs: Any) -> bool:
        calls.append("legacy")
        return True

    monkeypatch.setattr(agent_runner, "run_blocking_ordered", legacy_write)

    async def record() -> bool:
        return await agent_runner._record_execution_trace_ordered(
            None, "run-one", "agent-one", "task"
        )

    with l5_terminal_trace_owner():
        assert asyncio.run(record()) is True
    assert calls == []
    assert asyncio.run(record()) is True
    assert calls == ["legacy"]


def test_work_item_trace_owner_emits_no_premature_receipt(monkeypatch: Any) -> None:
    seen: list[str] = []

    async def sink(event: Any) -> None:
        seen.append(event.stage)

    async def emit() -> None:
        await agent_runner._emit_checkpoint_event(sink, "run-one", False, True)
        await agent_runner._emit_terminal_events(sink, "run-one", False, True, None)

    with l5_terminal_trace_owner():
        asyncio.run(emit())
    assert seen == []
