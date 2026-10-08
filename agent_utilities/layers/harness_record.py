"""L5 recording of a harness :class:`RunOutcome`.

The run/trace recording already exists: ``run_agent`` writes the ``RunTrace``
through its ordered trace writer, and :class:`~agent_utilities.usage.recorder.UsageRecorder`
records token and cost usage. This module feeds a harness outcome into those
same writers. A native run already wrote its trace, so it is not written twice.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.layers.harness_port import HarnessPort, HarnessRequest, RunOutcome

#: The L4 statuses mapped onto the RunTrace status vocabulary.
_TRACE_STATUS = {
    "completed": "completed",
    "degraded": "degraded",
    "failed": "failed",
    "timeout": "failed",
    "refused": "failed",
}


async def _write_trace(
    engine: Any, request: HarnessRequest, outcome: RunOutcome
) -> bool:
    from agent_utilities.orchestration.agent_runner import (
        _record_execution_trace_ordered,
    )

    return await _record_execution_trace_ordered(
        engine,
        outcome.run_id,
        outcome.agent_name,
        request.task,
        status=_TRACE_STATUS[outcome.status],
        error=outcome.error,
        duration_ms=outcome.duration_ms,
        result_preview=outcome.final_text[:500],
        model_name=outcome.model,
        execution_mode=f"harness:{outcome.harness}",
    )


def _write_usage(request: HarnessRequest, outcome: RunOutcome) -> bool:
    from agent_utilities.usage.recorder import get_usage_recorder

    usage = outcome.usage
    return get_usage_recorder().record_run(
        run_id=outcome.run_id,
        query=request.task,
        status=_TRACE_STATUS[outcome.status],
        duration_ms=outcome.duration_ms,
        token_usage=None if usage is None else usage.token_usage(),
        model=outcome.model,
    )


async def record_outcome(
    engine: Any, request: HarnessRequest, outcome: RunOutcome
) -> bool:
    """Record one harness outcome through the existing L5 writers.

    Returns True when the ``RunTrace`` exists (written now or by the runtime).
    """
    if outcome.recorded:
        return True
    if outcome.usage is not None:
        _write_usage(request, outcome)
    return await _write_trace(engine, request, outcome)


async def run_and_record(
    port: HarnessPort, request: HarnessRequest, *, engine: Any = None
) -> RunOutcome:
    """Run one request on ``port`` and record its outcome at L5."""
    outcome = await port.run(request)
    recorded = await record_outcome(engine, request, outcome)
    return outcome.model_copy(update={"recorded": recorded})


__all__ = ["record_outcome", "run_and_record"]
