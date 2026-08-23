"""CONCEPT:AU-KG.query.object-graph-mapper

``PipelineRunner`` regression coverage for two production bugs:

1. A retryable ``PARTIAL_MATERIALIZATION`` wire error (as the epistemic-graph
   engine actually emits it — see the live production log excerpt in the
   bug report) used to abort the whole pipeline instead of resuming from the
   engine's own ``completeness_cursor``. These tests drive the REAL
   ``PipelineRunner.run`` entrypoint with fake phases that raise the exact
   wire payload (never a mock standing in for
   ``engine_tasks._retryable_partial_materialization`` itself — that parser
   is imported and reused unmodified, not re-implemented here).
2. ``PipelineRunner.get_status()`` used to return a hardcoded "everything is
   complete" payload regardless of what actually ran.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.pipeline import runner as runner_mod
from agent_utilities.knowledge_graph.pipeline.runner import PipelineRunner
from agent_utilities.knowledge_graph.pipeline.types import (
    PipelineContext,
    PipelinePhase,
)
from agent_utilities.models.knowledge_graph import PipelineConfig

_SNAPSHOT = 68887


def _materialization_error(node_offset: int, *, snapshot: int = _SNAPSHOT) -> Exception:
    """The exact retryable wire payload from the production log excerpt."""
    return Exception(
        json.dumps(
            {
                "code": "PARTIAL_MATERIALIZATION",
                "completeness_cursor": {
                    "edge_offset": 0,
                    "node_offset": node_offset,
                },
                "phase": "partial",
                "retryable": True,
                "source_snapshot_version": snapshot,
            }
        )
    )


def _make_ctx() -> PipelineContext:
    return PipelineContext(
        config=PipelineConfig(workspace_path="/tmp/does-not-matter"),
        graph=MagicMock(spec=GraphComputeEngine),
        backend=None,
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Retries must not actually block the test suite."""
    fake_sleep = AsyncMock()
    monkeypatch.setattr(runner_mod.asyncio, "sleep", fake_sleep)
    return fake_sleep


@pytest.mark.asyncio
async def test_resumes_from_retryable_partial_materialization_cursor(
    _no_real_sleep: AsyncMock,
) -> None:
    """An advancing cursor lets the phase resume and the pipeline complete."""
    calls: list[int] = []
    offsets = iter([0, 2048, 4096])

    async def flaky_scan(ctx: PipelineContext, deps: dict[str, Any]) -> list[str]:
        del ctx, deps
        offset = next(offsets)
        calls.append(offset)
        if offset < 4096:
            raise _materialization_error(offset)
        return ["file:a.py", "file:b.py"]

    phase = PipelinePhase(name="scan", deps=[], execute_fn=flaky_scan)
    runner = PipelineRunner([phase])
    ctx = _make_ctx()

    results = await runner.run(ctx)

    assert calls == [0, 2048, 4096]
    assert results["scan"].success is True
    assert results["scan"].output == ["file:a.py", "file:b.py"]
    assert results["scan"].error is None
    # asyncio.sleep was used to back off between resumes, not a busy loop.
    assert _no_real_sleep.await_count == 2


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: Exception("not json at all"),
        lambda: Exception(
            json.dumps(
                {
                    "code": "PARTIAL_MATERIALIZATION",
                    "completeness_cursor": {"edge_offset": 0, "node_offset": 4096},
                    "phase": "partial",
                    "retryable": False,
                    "source_snapshot_version": _SNAPSHOT,
                }
            )
        ),
        lambda: Exception(json.dumps({"code": "SOME_OTHER_ERROR", "retryable": True})),
    ],
    ids=["malformed-non-json", "non-retryable", "wrong-code"],
)
@pytest.mark.asyncio
async def test_non_retryable_or_malformed_materialization_error_still_fails(
    make_error: Any,
) -> None:
    """Only the exact retryable wire payload resumes — everything else still
    aborts the pipeline through the ordinary failure path, on the FIRST
    attempt (no broadened catch, no retry)."""
    calls = 0

    async def always_fails(ctx: PipelineContext, deps: dict[str, Any]) -> None:
        del ctx, deps
        nonlocal calls
        calls += 1
        raise make_error()

    phase = PipelinePhase(name="scan", deps=[], execute_fn=always_fails)
    runner = PipelineRunner([phase])
    ctx = _make_ctx()

    with pytest.raises(Exception):  # noqa: B017 - re-raised original exception, any subtype
        await runner.run(ctx)

    assert calls == 1
    assert ctx.results["scan"].success is False
    assert ctx.results["scan"].error is not None


@pytest.mark.asyncio
async def test_non_advancing_cursor_terminates(_no_real_sleep: AsyncMock) -> None:
    """A cursor that stops advancing must terminate loudly, not loop forever."""
    calls = 0

    async def stuck_scan(ctx: PipelineContext, deps: dict[str, Any]) -> None:
        del ctx, deps
        nonlocal calls
        calls += 1
        raise _materialization_error(4096)

    phase = PipelinePhase(name="scan", deps=[], execute_fn=stuck_scan)
    runner = PipelineRunner([phase])
    ctx = _make_ctx()

    with pytest.raises(RuntimeError, match="stopped advancing"):
        await runner.run(ctx)

    # One attempt establishes the cursor, the second observes no progress and
    # aborts — nowhere near looping forever or exhausting the attempt bound.
    assert calls == 2
    assert ctx.results["scan"].success is False
    assert "stopped advancing" in ctx.results["scan"].error


@pytest.mark.asyncio
async def test_snapshot_version_change_mid_resume_aborts(
    _no_real_sleep: AsyncMock,
) -> None:
    """A cursor is only valid against the snapshot it was issued for."""
    calls = 0

    async def snapshot_drifts(ctx: PipelineContext, deps: dict[str, Any]) -> None:
        del ctx, deps
        nonlocal calls
        calls += 1
        # Cursor advances each time, but the snapshot version changes on the
        # second attempt -- the earlier cursor no longer corresponds to the
        # current graph state and must not be treated as a valid resume.
        raise _materialization_error(4096 * calls, snapshot=_SNAPSHOT + calls - 1)

    phase = PipelinePhase(name="scan", deps=[], execute_fn=snapshot_drifts)
    runner = PipelineRunner([phase])
    ctx = _make_ctx()

    with pytest.raises(RuntimeError, match="source_snapshot_version changed"):
        await runner.run(ctx)

    assert calls == 2
    assert ctx.results["scan"].success is False


@pytest.mark.asyncio
async def test_exceeding_max_attempts_terminates(
    monkeypatch: pytest.MonkeyPatch, _no_real_sleep: AsyncMock
) -> None:
    """An always-advancing cursor that never finishes still terminates (bounded
    attempts, not an unbounded resume loop)."""
    monkeypatch.setattr(runner_mod, "_MATERIALIZATION_MAX_ATTEMPTS", 3)
    calls = 0

    async def perpetually_partial(ctx: PipelineContext, deps: dict[str, Any]) -> None:
        del ctx, deps
        nonlocal calls
        calls += 1
        raise _materialization_error(calls * 1024)

    phase = PipelinePhase(name="scan", deps=[], execute_fn=perpetually_partial)
    runner = PipelineRunner([phase])
    ctx = _make_ctx()

    with pytest.raises(RuntimeError, match="did not finish materializing"):
        await runner.run(ctx)

    assert calls == 3
    assert ctx.results["scan"].success is False


@pytest.mark.asyncio
async def test_get_status_is_idle_before_any_run() -> None:
    async def noop(ctx: PipelineContext, deps: dict[str, Any]) -> None:
        del ctx, deps

    runner = PipelineRunner([PipelinePhase(name="memory", deps=[], execute_fn=noop)])

    assert runner.get_status() == {"status": "idle", "phases": {}}


@pytest.mark.asyncio
async def test_get_status_reports_a_failed_phase_as_failed() -> None:
    """The core Bug 2 assertion: a real failure must read back as failed, not
    a hardcoded 'complete' (AGENTS.md 'Fail closed')."""

    async def ok_memory(ctx: PipelineContext, deps: dict[str, Any]) -> str:
        del ctx, deps
        return "ok"

    async def broken_scan(ctx: PipelineContext, deps: dict[str, Any]) -> None:
        del ctx, deps
        raise Exception("not json at all")

    phases = [
        PipelinePhase(name="memory", deps=[], execute_fn=ok_memory),
        PipelinePhase(name="scan", deps=["memory"], execute_fn=broken_scan),
        PipelinePhase(name="parse", deps=["scan"], execute_fn=ok_memory),
    ]
    runner = PipelineRunner(phases)
    ctx = _make_ctx()

    with pytest.raises(Exception, match="not json at all"):  # noqa: B017
        await runner.run(ctx)

    status = runner.get_status()
    assert status["status"] == "failed"
    assert status["phases"]["memory"] == "complete"
    assert status["phases"]["scan"] == "failed"
    # "parse" never ran (scan failed first) -- absent, never fabricated.
    assert "parse" not in status["phases"]


@pytest.mark.asyncio
async def test_get_status_reports_complete_on_a_fully_successful_run() -> None:
    async def ok(ctx: PipelineContext, deps: dict[str, Any]) -> str:
        del ctx, deps
        return "ok"

    runner = PipelineRunner([PipelinePhase(name="memory", deps=[], execute_fn=ok)])
    ctx = _make_ctx()

    await runner.run(ctx)

    assert runner.get_status() == {"status": "complete", "phases": {"memory": "complete"}}


def test_asyncio_import_present() -> None:
    """Sanity: the runner module actually imports asyncio (used for the
    resume backoff) rather than relying on it being imported transitively."""
    assert runner_mod.asyncio is asyncio
