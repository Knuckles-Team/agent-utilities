"""The ``native`` harness: the in-process pydantic-ai path behind :class:`HarnessPort`.

The adapter calls :func:`agent_utilities.orchestration.agent_runner.run_agent`
unchanged. That path keeps typed tool calls, structured output, token
accounting, the Langfuse/OTel trace export and the RunTrace write. The adapter
only asks for the rich envelope (``include_run_summary=True``) and reads it
into a :class:`RunOutcome`. It never writes a second trace.

It grants no native sub-agent allowance (AU-CONTROL-R018): a request naming
``max_subagents`` above zero is refused closed rather than silently run with
fewer sub-agents than the committed plan granted.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent_utilities.layers.harness_port import (
    NATIVE_HARNESS,
    HarnessRequest,
    OutcomeStatus,
    RunOutcome,
    refused,
)

Runner = Callable[..., Awaitable[str]]

#: ``run_summary.outcome`` values mapped onto the L4 status vocabulary.
_STATUS: dict[str, OutcomeStatus] = {
    "ok": "completed",
    "degraded": "degraded",
    "failed": "failed",
    "timeout": "timeout",
}


def _default_runner() -> Runner:
    from agent_utilities.orchestration.agent_runner import run_agent

    return run_agent


def _envelope(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"output": raw}
    return parsed if isinstance(parsed, dict) else {"output": raw}


def _structured(text: str, response_format: str) -> Any:
    if response_format != "json":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _failure_text(summary: dict[str, Any]) -> str | None:
    failure = summary.get("failure")
    if not failure:
        return None
    if isinstance(failure, dict):
        return str(failure.get("translated") or failure.get("raw") or failure)
    return str(failure)


def native_outcome(request: HarnessRequest, raw: str, duration_ms: float) -> RunOutcome:
    """Read one ``run_agent`` envelope into a :class:`RunOutcome`."""
    envelope = _envelope(raw)
    summary = envelope.get("run_summary") or {}
    text = str(envelope.get("output", ""))
    return RunOutcome(
        run_id=str(envelope.get("run_id") or request.run_id),
        harness=NATIVE_HARNESS,
        agent_name=request.agent_name,
        status=_STATUS.get(str(summary.get("outcome")), "completed"),
        final_text=text,
        structured_output=_structured(text, request.response_format),
        trace_ref=summary.get("trace_ref"),
        duration_ms=duration_ms,
        error=_failure_text(summary),
        recorded=bool(envelope.get("provenance_recorded")),
    )


class NativeHarness:
    """The default L4 adapter over the pydantic-ai ``run_agent`` path."""

    def __init__(self, *, engine: Any = None, runner: Runner | None = None) -> None:
        self._engine = engine
        self._runner = runner

    @property
    def name(self) -> str:
        return NATIVE_HARNESS

    def _call(self, request: HarnessRequest) -> Awaitable[str]:
        runner = self._runner or _default_runner()
        return runner(
            agent_name=request.agent_name,
            task=request.task,
            max_steps=request.max_steps,
            engine=self._engine,
            allowed_tools=list(request.allowed_tools) or None,
            response_format=request.response_format,
            run_id=request.run_id,
            include_run_summary=True,
        )

    async def run(self, request: HarnessRequest) -> RunOutcome:
        if request.max_subagents > 0:
            # AU-CONTROL-R018: this adapter's run_agent path has no native
            # sub-agent tool to grant the allowance through yet. Fail closed
            # rather than silently run with fewer sub-agents than committed.
            return refused(
                NATIVE_HARNESS,
                request,
                "native harness has no sub-agent tool to grant max_subagents "
                f"{request.max_subagents} through; request max_subagents=0",
            )
        start = time.monotonic()
        try:
            raw = await asyncio.wait_for(self._call(request), request.timeout_s)
        except TimeoutError:
            return RunOutcome(
                run_id=request.run_id,
                harness=NATIVE_HARNESS,
                agent_name=request.agent_name,
                status="timeout",
                duration_ms=(time.monotonic() - start) * 1000,
                error=f"native run exceeded {request.timeout_s:.0f}s",
            )
        return native_outcome(request, raw, (time.monotonic() - start) * 1000)


__all__ = ["NativeHarness", "native_outcome"]
