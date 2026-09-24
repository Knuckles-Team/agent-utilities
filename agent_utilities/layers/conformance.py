"""The harness conformance kit every :class:`HarnessPort` adapter must pass.

RF-ADR-010 §6.2/§12: no adapter runs until it passes one shared kit. The kit
checks, per adapter: descriptor honesty, session lifecycle, streaming events,
tool-call surfacing, cancellation, sandbox boundary and error typing.

A target supplies *rigs* -- a fresh harness plus a spec and a lease for one
scenario -- and states where its evidence comes from
(:data:`~agent_utilities.layers.contracts.EvidenceSource`): the real harness,
a transcript recorded from a live run, or a synthetic transcript authored from
vendor documentation. Every result carries that label so a double can never be
reported as a real-harness pass.

:class:`TranscriptLauncher` is the recorded-transcript double for CLI
harnesses: it implements :class:`~agent_utilities.layers.cli_process.
ProcessLauncher` and replays JSONL through the adapter's real parsing path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.layers.contracts import (
    EvidenceSource,
    HarnessError,
    HarnessNotConfigured,
    HarnessNotInstalled,
    HarnessRefused,
    RunSpec,
)
from agent_utilities.layers.execution import RunOutcome, run_to_completion
from agent_utilities.layers.negotiation import HarnessPolicy
from agent_utilities.layers.ports import HarnessPort, RunHandle, SandboxLease

# ---------------------------------------------------------------------------
# Recorded-transcript double for CLI harnesses
# ---------------------------------------------------------------------------

#: Event-loop turns a replayed child takes to exit after SIGTERM; enough for
#: the adapter's reader to observe end-of-stream and the exit status first.
TERMINATION_YIELDS = 5


@dataclass(slots=True)
class _ReplayedProcess:
    lines_: tuple[str, ...]
    exit_code: int
    stderr: str
    hold: asyncio.Event | None
    terminated: bool = False

    async def lines(self) -> AsyncIterator[str]:
        for line in self.lines_:
            await asyncio.sleep(0)
            yield line
        if self.hold is not None:
            await self.hold.wait()

    async def terminate(self) -> None:
        """Like a real child: the stream closes and the exit status is
        observable before ``terminate`` returns to the adapter."""
        self.terminated = True
        if self.hold is not None:
            self.hold.set()
        for _ in range(TERMINATION_YIELDS):
            await asyncio.sleep(0)

    async def wait(self) -> int:
        return -15 if self.terminated else self.exit_code

    def stderr_tail(self) -> str:
        return self.stderr


@dataclass(slots=True)
class TranscriptLauncher:
    """Replays a recorded JSONL transcript as if the harness had printed it.

    ``installed=False`` makes binary resolution fail exactly like a host
    without the harness. ``hold_open`` keeps the stream open after the last
    line until the adapter terminates the child (cancellation scenarios).
    """

    lines: Sequence[str]
    exit_code: int = 0
    stderr: str = ""
    installed: bool = True
    hold_open: bool = False
    launches: list[dict[str, Any]] = field(default_factory=list)
    processes: list[_ReplayedProcess] = field(default_factory=list)

    def resolve(self, binary: str) -> str:
        if not self.installed:
            raise HarnessNotInstalled(f"{binary!r} is not installed on this host")
        return f"/recorded/bin/{binary}"

    async def launch(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdin_text: str,
    ) -> _ReplayedProcess:
        self.launches.append(
            {"argv": tuple(argv), "cwd": cwd, "env": dict(env), "stdin": stdin_text}
        )
        process = _ReplayedProcess(
            lines_=tuple(self.lines),
            exit_code=self.exit_code,
            stderr=self.stderr,
            hold=asyncio.Event() if self.hold_open else None,
        )
        self.processes.append(process)
        return process


# ---------------------------------------------------------------------------
# Targets and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rig:
    """One scenario: a fresh harness, the spec to run and its lease."""

    harness: HarnessPort
    spec: RunSpec
    lease: SandboxLease | None
    #: Observations the scenario's double recorded (launches, terminations).
    observe: Callable[[], dict[str, Any]] = dict


RigFactory = Callable[[str], Rig]


@dataclass(frozen=True, slots=True)
class ConformanceTarget:
    """What an adapter provides to the kit.

    Scenarios: ``success`` (a run with at least one tool call named
    ``tool_name``), ``blocking`` (a run that never finishes on its own),
    ``failing`` (a run the harness reports as failed before any tool call) and
    ``unavailable`` (the harness is not installed or not configured).
    """

    name: str
    evidence: EvidenceSource
    policy: HarnessPolicy
    tool_name: str
    scenarios: Mapping[str, RigFactory]


@dataclass(frozen=True, slots=True)
class CheckResult:
    check: str
    harness: str
    evidence: EvidenceSource
    passed: bool
    detail: str = ""


class ConformanceFailure(AssertionError):
    """A harness adapter failed a conformance check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceFailure(message)


async def _run(target: ConformanceTarget, scenario: str) -> tuple[Rig, RunOutcome]:
    rig = target.scenarios[scenario](f"conformance-{target.name}-{scenario}")
    outcome = await run_to_completion(
        rig.harness, rig.spec, policy=target.policy, lease=rig.lease
    )
    return rig, outcome


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


async def check_descriptor_honesty(target: ConformanceTarget) -> str:
    rig = target.scenarios["success"]("conformance-describe")
    desc = rig.harness.describe()
    _require(desc.name == target.name, "descriptor name differs from the target")
    _require(bool(desc.environment_modes), "no execution environment declared")
    _require(
        desc.account_modes == frozenset({"api_key", "subscription"}),
        "every ruled adapter supports api_key and subscription accounts",
    )
    remote = "provider-managed-remote" in desc.environment_modes
    _require(
        not (remote and "local-sandbox" in desc.environment_modes),
        "a provider-managed remote harness cannot claim local sandbox containment",
    )
    metered = desc.enforceable_budgets - {"wall_time"}
    _require(
        not metered or desc.usage_quality == "measured",
        "a metered budget is enforceable only with measured usage",
    )
    _require(
        desc.skill_proof != "none" or desc.max_skills == 0,
        "skills are delivered without a proof",
    )
    return f"fidelity={desc.fidelity} usage={desc.usage_quality}"


async def check_session_lifecycle(target: ConformanceTarget) -> str:
    rig, outcome = await _run(target, "success")
    result, trace = outcome.result, outcome.trace
    _require(result.status == "succeeded", f"status {result.status}: {result.error}")
    _require(result.trace == "complete", "a successful run left an incomplete trace")
    _require(trace[0].kind == "started", "the first event is not 'started'")
    _require(trace[-1].kind == "completed", "the last event is not 'completed'")
    _require(
        [event.seq for event in trace] == list(range(len(trace))),
        "event sequence numbers are not contiguous",
    )
    _require(result.high_watermark == trace[-1].seq, "high watermark is wrong")
    digest = rig.spec.digest()
    _require(
        all(event.spec_digest == digest for event in trace),
        "an event is not stamped with the negotiated spec digest",
    )
    fidelity = rig.harness.describe().fidelity
    _require(
        all(event.fidelity == fidelity for event in trace),
        "an event misstates the adapter fidelity",
    )
    return f"{len(trace)} events, output {len(result.output)} chars"


async def check_streaming_events(target: ConformanceTarget) -> str:
    rig = target.scenarios["success"]("conformance-stream")
    negotiated = rig.harness.negotiate(rig.spec, target.policy)
    handle = await rig.harness.start(negotiated, rig.lease)
    streamed = [event async for event in rig.harness.events(handle)]
    await rig.harness.result(handle)
    _require(
        streamed == list(rig.harness.trace(handle)),
        "the streamed events differ from the recorded trace",
    )
    return f"{len(streamed)} events streamed in order"


async def check_tool_call_surfacing(target: ConformanceTarget) -> str:
    rig, outcome = await _run(target, "success")
    calls = [event for event in outcome.trace if event.kind == "tool_call"]
    if rig.harness.describe().fidelity == "final-output":
        _require(not calls, "a final-output harness claimed tool-call visibility")
        return "final-output fidelity: tool calls are not observable"
    _require(
        any(target.tool_name in event.name for event in calls),
        f"the planted tool call {target.tool_name!r} was not surfaced",
    )
    _require(
        all(event.evidence == "observation" for event in calls),
        "a tool call is not recorded as an observation",
    )
    results = [event for event in outcome.trace if event.kind == "tool_result"]
    _require(bool(results), "no tool result was surfaced")
    return f"{len(calls)} tool calls, {len(results)} results"


#: Bound on waiting for a blocking run to become live before cancelling it.
LIVE_EVENT_TIMEOUT_S = 120.0


async def _first_live_event(harness: HarnessPort, handle: RunHandle) -> None:
    """Return once the run emitted something after ``started`` (it is live)."""
    async for event in harness.events(handle):
        if event.kind != "started":
            return


async def check_cancellation(target: ConformanceTarget) -> str:
    rig = target.scenarios["blocking"]("conformance-cancel")
    negotiated = rig.harness.negotiate(rig.spec, target.policy)
    handle = await rig.harness.start(negotiated, rig.lease)
    await asyncio.wait_for(_first_live_event(rig.harness, handle), LIVE_EVENT_TIMEOUT_S)
    await rig.harness.cancel(handle)
    result = await rig.harness.result(handle)
    expected = "cancelled" if rig.spec.side_effects == "none" else "outcome_uncertain"
    _require(result.status == expected, f"cancelled run ended {result.status}")
    _require(result.trace == "incomplete", "a cancelled run claims a complete trace")
    _require(bool(rig.observe().get("stopped")), "the harness was not told to stop")
    return f"status={result.status}"


async def check_sandbox_boundary(target: ConformanceTarget) -> str:
    rig, outcome = await _run(target, "success")
    env = outcome.result.environment
    _require(
        env in rig.harness.describe().environment_modes,
        f"ran in undeclared environment {env}",
    )
    observed = rig.observe()
    workspace = rig.lease.workspace if rig.lease is not None else None
    for path in observed.get("paths", ()):
        _require(
            workspace is not None and str(path).startswith(workspace),
            f"{path} is outside the leased workspace",
        )
    return f"environment={env}, {len(observed.get('paths', ()))} confined paths"


async def check_error_typing(target: ConformanceTarget) -> str:
    rig = target.scenarios["unavailable"]("conformance-unavailable")
    try:
        rig.harness.negotiate(rig.spec, target.policy)
    except (HarnessNotInstalled, HarnessNotConfigured) as exc:
        unavailable = type(exc).__name__
    else:
        raise ConformanceFailure("an unavailable harness negotiated a run")
    working = target.scenarios["success"]("conformance-refused")
    refused_spec = working.spec.model_copy(
        update={"required_capabilities": frozenset({"resume"})}
    )
    try:
        working.harness.negotiate(refused_spec, target.policy)
    except HarnessRefused as exc:
        _require(bool(exc.reasons), "a refusal named no reason")
    else:
        _require(
            "resume" in working.harness.describe().capabilities,
            "an unsupported required capability was not refused",
        )
    _rig, outcome = await _run(target, "failing")
    _require(outcome.result.status == "failed", "a failing run was not 'failed'")
    _require(
        isinstance(outcome.failure, HarnessError) and bool(outcome.result.error_kind),
        "a failed run carries no typed error",
    )
    return f"unavailable={unavailable}, failed={outcome.result.error_kind}"


CHECKS: dict[str, Callable[[ConformanceTarget], Awaitable[str]]] = {
    "descriptor_honesty": check_descriptor_honesty,
    "session_lifecycle": check_session_lifecycle,
    "streaming_events": check_streaming_events,
    "tool_call_surfacing": check_tool_call_surfacing,
    "cancellation": check_cancellation,
    "sandbox_boundary": check_sandbox_boundary,
    "error_typing": check_error_typing,
}


async def run_check(target: ConformanceTarget, check: str) -> CheckResult:
    """Run one named check; a failure is raised, never downgraded."""
    detail = await CHECKS[check](target)
    return CheckResult(
        check=check,
        harness=target.name,
        evidence=target.evidence,
        passed=True,
        detail=detail,
    )


__all__ = [
    "CHECKS",
    "CheckResult",
    "ConformanceFailure",
    "ConformanceTarget",
    "Rig",
    "RigFactory",
    "TranscriptLauncher",
    "run_check",
]
