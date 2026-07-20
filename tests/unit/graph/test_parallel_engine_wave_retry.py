"""In-wave agent retry (SWARM-5) on the declarative ResiliencePolicy.

CONCEPT:AU-ORCH.execution.parallel-engine-visualizer / CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating — the hand-rolled while-loop in
``ParallelEngine._execute_wave`` migrated onto ``run_with_resilience``.
These tests pin the preserved semantics: failed results retry up to
``max_retries`` extra attempts, the LAST attempt's result is kept on
exhaustion, ``metadata['retries']`` counts the extra attempts, and an
exception inside ``_execute_agent`` becomes a failed (retryable) result.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.graph.parallel_engine import ParallelEngine
from agent_utilities.models.execution_manifest import (
    AgentExecutionResult,
    AgentSpec,
    ExecutionManifest,
)


def _scheduler() -> MagicMock:
    scheduler = MagicMock()
    proc = MagicMock()
    proc.id = "proc-1"
    scheduler.submit = AsyncMock(return_value=proc)
    scheduler.wait_for_running = AsyncMock()
    scheduler.complete = AsyncMock()
    scheduler.fail = AsyncMock()
    return scheduler


def _manifest(agent_id: str, max_retries: int) -> ExecutionManifest:
    return ExecutionManifest(
        query="q",
        agents=[
            AgentSpec(
                agent_id=agent_id, role="r", task_template="t", max_retries=max_retries
            )
        ],
    )


async def test_retries_then_succeeds_and_counts_retries():
    engine = ParallelEngine(engine=None)
    calls = {"n": 0}

    async def flaky(agent, manifest, graph_deps, wave_results, proc):
        calls["n"] += 1
        ok = calls["n"] >= 2
        return AgentExecutionResult(
            agent_id=agent.agent_id,
            role=agent.role,
            success=ok,
            error="" if ok else "transient",
        )

    engine._execute_agent = flaky  # type: ignore[method-assign]
    manifest = _manifest("a1", max_retries=1)
    wave = await engine._execute_wave(
        manifest.agents, 0, _scheduler(), manifest, None, []
    )
    res = wave.results[0]
    assert res.success is True
    assert calls["n"] == 2
    assert res.metadata["retries"] == 1


async def test_exhaustion_keeps_last_failed_result():
    engine = ParallelEngine(engine=None)

    async def always_fail(agent, manifest, graph_deps, wave_results, proc):
        return AgentExecutionResult(
            agent_id=agent.agent_id, role=agent.role, success=False, error="nope"
        )

    engine._execute_agent = always_fail  # type: ignore[method-assign]
    manifest = _manifest("a2", max_retries=1)
    wave = await engine._execute_wave(
        manifest.agents, 0, _scheduler(), manifest, None, []
    )
    res = wave.results[0]
    assert res.success is False
    assert res.error == "nope"
    assert res.metadata["retries"] == 1


async def test_execute_agent_exception_becomes_failed_result():
    engine = ParallelEngine(engine=None)

    async def raises(agent, manifest, graph_deps, wave_results, proc):
        raise RuntimeError("kaput")

    engine._execute_agent = raises  # type: ignore[method-assign]
    scheduler = _scheduler()
    manifest = _manifest("a3", max_retries=0)
    wave = await engine._execute_wave(manifest.agents, 0, scheduler, manifest, None, [])
    res = wave.results[0]
    assert res.success is False
    assert res.error == "kaput"
    assert "retries" not in res.metadata  # single attempt — no retry metadata
    scheduler.fail.assert_awaited()


async def test_first_attempt_success_has_no_retry_metadata():
    engine = ParallelEngine(engine=None)

    async def immediate(agent, manifest, graph_deps, wave_results, proc):
        return AgentExecutionResult(
            agent_id=agent.agent_id, role=agent.role, success=True, error=""
        )

    engine._execute_agent = immediate  # type: ignore[method-assign]
    manifest = _manifest("a4", max_retries=3)
    wave = await engine._execute_wave(
        manifest.agents, 0, _scheduler(), manifest, None, []
    )
    res = wave.results[0]
    assert res.success is True
    assert "retries" not in res.metadata


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
