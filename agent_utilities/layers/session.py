"""Shared run lifecycle for every :class:`HarnessPort` adapter.

An adapter supplies three things: :meth:`HarnessRuntime.describe`, a
:meth:`HarnessRuntime.preflight` that proves the harness is installed and
configured, and a :meth:`HarnessRuntime.drive` coroutine that runs the harness
and reports normalized events through :class:`RunContext`. This base owns the
rest once, for all adapters: negotiation, spec-digest stamping, event
sequencing, the wall-clock deadline, cancellation, typed failure mapping,
trace completeness and the uncertain-outcome rule (RF-ADR-010 §7-§8).
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pydantic import JsonValue

from agent_utilities.layers.contracts import (
    UNAVAILABLE_USAGE,
    EvidenceClass,
    HarnessDescriptor,
    HarnessError,
    HarnessOutcomeUncertain,
    HarnessRunFailed,
    NegotiatedRunSpec,
    RunEvent,
    RunEventKind,
    RunResult,
    RunSpec,
    RunStatus,
    UsageRecord,
)
from agent_utilities.layers.negotiation import HarnessPolicy, negotiate
from agent_utilities.layers.ports import RunHandle, SandboxLease

#: Bound on retained normalized events per run (older runs are not kept).
MAX_TRACE_EVENTS = 10_000


@dataclass(frozen=True, slots=True)
class DriveOutcome:
    """What an adapter's :meth:`HarnessRuntime.drive` reports on normal return."""

    status: RunStatus
    output: str = ""
    usage: UsageRecord = UNAVAILABLE_USAGE
    error_kind: str | None = None
    error: str = ""
    provider_session: str | None = None
    #: ``False`` when the harness stream ended without its terminal record.
    trace_complete: bool = True
    gap_reason: str = ""


@dataclass(slots=True)
class RunContext:
    """The adapter's view of one run: its spec, workspace and event sink."""

    negotiated: NegotiatedRunSpec
    lease: SandboxLease | None
    queue: asyncio.Queue[RunEvent | None]
    events: list[RunEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    provider_session: str | None = None
    #: The typed error that ended the run, with the adapter's cause chained.
    failure: HarnessError | None = None

    @property
    def spec(self) -> RunSpec:
        return self.negotiated.spec

    @property
    def workspace(self) -> str | None:
        return None if self.lease is None else self.lease.workspace

    def emit(
        self,
        kind: RunEventKind,
        evidence: EvidenceClass,
        *,
        name: str = "",
        detail: str = "",
        data: dict[str, JsonValue] | None = None,
    ) -> RunEvent:
        """Append one normalized event, stamped with the spec digest."""
        event = RunEvent(
            run_id=self.spec.run_id,
            seq=len(self.events),
            kind=kind,
            evidence=evidence,
            fidelity=self.negotiated.fidelity,
            spec_digest=self.negotiated.spec_digest,
            name=name,
            detail=detail[:20_000],
            data=dict(data or {}),
        )
        if len(self.events) < MAX_TRACE_EVENTS:
            self.events.append(event)
            self.queue.put_nowait(event)
        return event

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


@dataclass(slots=True)
class _RunState:
    context: RunContext
    task: asyncio.Task[RunResult]
    cancel_requested: bool = False


class HarnessRuntime(abc.ABC):
    """Base :class:`HarnessPort` implementation; adapters implement three hooks."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}

    # -- adapter hooks ------------------------------------------------------

    @abc.abstractmethod
    def describe(self) -> HarnessDescriptor:
        """The provider claim for this harness."""

    @abc.abstractmethod
    def preflight(self, spec: RunSpec) -> None:
        """Raise ``HarnessNotInstalled``/``HarnessNotConfigured`` when unusable."""

    @abc.abstractmethod
    async def drive(self, run: RunContext) -> DriveOutcome:
        """Run the harness to completion, emitting normalized events."""

    async def interrupt(self, run: RunContext) -> None:
        """Ask the harness to stop before the drive task is cancelled.

        The default relies on task cancellation alone; process-backed
        adapters override it to terminate their child first.
        """
        run.emit("step", "observation", name="interrupt")

    # -- HarnessPort --------------------------------------------------------

    def negotiate(self, spec: RunSpec, policy: HarnessPolicy) -> NegotiatedRunSpec:
        self.preflight(spec)
        return negotiate(spec, self.describe(), policy)

    async def start(
        self, negotiated: NegotiatedRunSpec, lease: SandboxLease | None
    ) -> RunHandle:
        self._require_own(negotiated)
        run_id = negotiated.spec.run_id
        if run_id in self._runs:
            raise HarnessError(f"run {run_id!r} was already started")
        context = RunContext(negotiated=negotiated, lease=lease, queue=asyncio.Queue())
        task = asyncio.create_task(self._supervise(context))
        self._runs[run_id] = _RunState(context=context, task=task)
        return RunHandle(
            run_id=run_id,
            harness=negotiated.harness,
            spec_digest=negotiated.spec_digest,
            workspace=context.workspace,
        )

    async def events(self, handle: RunHandle) -> AsyncIterator[RunEvent]:
        queue = self._state(handle).context.queue
        while True:
            event = await queue.get()
            if event is None:
                return
            yield event

    async def cancel(self, handle: RunHandle) -> None:
        state = self._state(handle)
        if state.task.done():
            return
        state.cancel_requested = True
        await self.interrupt(state.context)
        state.task.cancel()
        await asyncio.wait({state.task})

    async def result(self, handle: RunHandle) -> RunResult:
        return await asyncio.shield(self._state(handle).task)

    def trace(self, handle: RunHandle) -> tuple[RunEvent, ...]:
        return tuple(self._state(handle).context.events)

    def failure(self, handle: RunHandle) -> HarnessError | None:
        """The typed error that ended ``handle`` (its cause chained), if any."""
        return self._state(handle).context.failure

    # -- internals ----------------------------------------------------------

    def _require_own(self, negotiated: NegotiatedRunSpec) -> None:
        desc = self.describe()
        if negotiated.harness != desc.name:
            raise HarnessError("the negotiated spec belongs to another harness")
        if negotiated.spec_digest != negotiated.spec.digest():
            raise HarnessError("the negotiated spec digest does not match its spec")

    def _state(self, handle: RunHandle) -> _RunState:
        state = self._runs.get(handle.run_id)
        if state is None or handle.harness != self.describe().name:
            raise HarnessError(f"unknown run {handle.run_id!r}")
        return state

    async def _supervise(self, run: RunContext) -> RunResult:
        run.emit("started", "observation", name=run.negotiated.harness)
        deadline = run.spec.budget.max_wall_s
        try:
            outcome = await asyncio.wait_for(self.drive(run), timeout=deadline)
        except asyncio.CancelledError:
            outcome = self._cancelled_outcome(run)
        except TimeoutError:
            await self.interrupt(run)
            outcome = _after_dispatch(run, "deadline", "wall-clock budget exceeded")
        except HarnessError as exc:
            run.failure = exc
            outcome = _typed_outcome(run, exc)
        except BaseException:
            run.queue.put_nowait(None)
            raise
        return self._finish(run, self._honour_cancel(run, outcome))

    def _honour_cancel(self, run: RunContext, outcome: DriveOutcome) -> DriveOutcome:
        """A run the caller cancelled is reported as cancelled even when the
        interrupted harness ended its stream before the task cancellation
        landed (a terminated child reports a failure exit first)."""
        state = self._runs.get(run.spec.run_id)
        if outcome.status == "succeeded" or state is None:
            return outcome
        if not state.cancel_requested:
            return outcome
        return _after_dispatch(
            run, "cancelled", "cancelled by the caller", clean="cancelled"
        )

    def _cancelled_outcome(self, run: RunContext) -> DriveOutcome:
        state = self._runs.get(run.spec.run_id)
        if state is None or not state.cancel_requested:
            raise asyncio.CancelledError
        return _after_dispatch(
            run, "cancelled", "cancelled by the caller", clean="cancelled"
        )

    def _finish(self, run: RunContext, outcome: DriveOutcome) -> RunResult:
        complete = outcome.trace_complete and outcome.status == "succeeded"
        run.emit(
            "completed",
            "observation",
            name=outcome.status,
            detail=outcome.error,
            data={"trace": "complete" if complete else "incomplete"},
        )
        run.queue.put_nowait(None)
        return RunResult(
            run_id=run.spec.run_id,
            spec_digest=run.negotiated.spec_digest,
            harness=run.negotiated.harness,
            status=outcome.status,
            output=outcome.output,
            usage=outcome.usage,
            error_kind=outcome.error_kind,
            error=outcome.error,
            provider_session=outcome.provider_session or run.provider_session,
            environment=run.negotiated.environment,
            fidelity=run.negotiated.fidelity,
            trace="complete" if complete else "incomplete",
            high_watermark=len(run.events) - 1,
            gap_reason="" if complete else (outcome.gap_reason or outcome.error),
        )


def _typed_outcome(run: RunContext, exc: HarnessError) -> DriveOutcome:
    """Map a typed adapter error to its outcome (uncertainty rule of §7)."""
    kind = type(exc).__name__
    if isinstance(exc, HarnessOutcomeUncertain):
        return _uncertain(run, kind, str(exc))
    if isinstance(exc, HarnessRunFailed):
        return DriveOutcome(
            status="failed",
            error_kind=kind,
            error=str(exc),
            provider_session=run.provider_session,
            trace_complete=False,
            gap_reason=str(exc),
        )
    return _after_dispatch(run, kind, str(exc))


def _uncertain(run: RunContext, kind: str, message: str) -> DriveOutcome:
    return DriveOutcome(
        status="outcome_uncertain",
        error_kind=kind,
        error=message,
        provider_session=run.provider_session,
        trace_complete=False,
        gap_reason=message,
    )


def _after_dispatch(
    run: RunContext, kind: str, message: str, *, clean: RunStatus = "failed"
) -> DriveOutcome:
    """An interruption after dispatch is uncertain unless the run is effect-free."""
    if run.spec.side_effects != "none":
        return _uncertain(run, kind, message)
    return DriveOutcome(
        status=clean,
        error_kind=kind,
        error=message,
        provider_session=run.provider_session,
        trace_complete=False,
        gap_reason=message,
    )


__all__ = ["MAX_TRACE_EVENTS", "DriveOutcome", "HarnessRuntime", "RunContext"]
