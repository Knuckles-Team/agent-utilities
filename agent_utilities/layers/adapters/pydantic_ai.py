"""The in-process adapter: AU's own pydantic-ai(+harness) agent runtime.

This is the first :class:`HarnessPort` adapter and it wraps the existing
runtime rather than a second one: every run goes through
``Orchestrator.execute_agent`` (the one delegation choke point, with its
authority keep-alive, grounding contract and tool contract), and the runtime's
``progress_sink`` stream is normalized into :class:`RunEvent` records.

Fidelity is declared ``tool-calls``: the progress stream surfaces routing,
tool calls, tool results and checkpoints, not every model request, and the
runtime does not return token usage, so ``usage_quality="unavailable"``.
Both are recorded truthfully rather than claimed (RF-ADR-010 §3, §8).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    HarnessError,
    HarnessNotConfigured,
    HarnessRefused,
    HarnessRunFailed,
    RunEventKind,
    RunSpec,
    VendorTerms,
)
from agent_utilities.layers.session import DriveOutcome, HarnessRuntime, RunContext

HARNESS_NAME = "pydantic-ai"

#: ``AgentExecutionRequest`` options the runtime honours, carried in
#: :attr:`RunSpec.runtime_options` and forwarded verbatim.
RUNTIME_OPTIONS = frozenset(
    {
        "max_steps",
        "return_mermaid",
        "context",
        "context_ref",
        "session_id",
        "open_channel",
        "memento_source",
        "execution_profile",
        "reasoning_effort",
        "model_class",
        "response_format",
        "include_run_summary",
        "skill_name",
        "tool_server",
        "execution_mode",
        "grounding",
        "budget_tokens",
    }
)

DESCRIPTOR = HarnessDescriptor(
    name=HARNESS_NAME,
    version="au-orchestrator/progress-sink",
    fidelity="tool-calls",
    capabilities=frozenset(
        {"mcp_client", "skills", "tool_allowlist", "cancellation", "sub_agents"}
    ),
    usage_quality="unavailable",
    enforceable_budgets=frozenset({"wall_time"}),
    account_modes=frozenset({"api_key", "subscription"}),
    environment_modes=frozenset({"caller-managed-host"}),
    tool_proof="runtime_contract",
    skill_proof="runtime_contract",
    max_skills=1,
    reconciliation="none",
    vendor_terms=VendorTerms(
        subscription_automation_allowed=True,
        note="model access follows AU's configured providers",
    ),
    runtime_options=RUNTIME_OPTIONS,
)

#: Runtime progress stage -> normalized event kind.
_STAGE_KIND: dict[str, RunEventKind] = {
    "start": "step",
    "route": "step",
    "evidence_gate": "step",
    "checkpoint": "step",
    "synthesis": "step",
    "done": "step",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "failure": "error",
}


class AgentRunner(Protocol):
    """The AU runtime surface this adapter drives."""

    async def execute_agent(
        self, agent_name: str, task: str, **options: Any
    ) -> str: ...


def _options(spec: RunSpec) -> dict[str, Any]:
    toolset = spec.toolset
    options: dict[str, Any] = dict(spec.runtime_options)
    options.update(
        run_id=spec.run_id,
        allowed_tools=None
        if toolset.allowed_tools is None
        else list(toolset.allowed_tools),
        required_tools=list(toolset.required_tools) or None,
        cred_ref=spec.account_ref,
    )
    if toolset.context_endpoint is not None:
        options["context_endpoint"] = toolset.context_endpoint
    elif toolset.mcp_servers:
        options["tool_server"] = toolset.mcp_servers[0].name
    if toolset.skills:
        options["skill_name"] = toolset.skills[0].name
    return options


def _progress_sink(run: RunContext) -> Callable[[Any], Awaitable[None]]:
    async def sink(event: Any) -> None:
        kind = _STAGE_KIND.get(str(getattr(event, "stage", "")))
        if kind is None:
            return
        evidence = getattr(event, "evidence", None)
        run.emit(
            kind,
            "observation",
            name=str(getattr(event, "stage", "")),
            detail=str(getattr(event, "detail", "")),
            data={
                "status": str(getattr(event, "status", "")),
                "evidence_keys": sorted(evidence) if isinstance(evidence, dict) else [],
            },
        )

    return sink


def _typed_failure(run: RunContext, exc: Exception) -> HarnessError:
    """A runtime failure is clean until a tool call may have had effects."""
    message = f"{type(exc).__name__}: {exc}"
    if any(event.kind == "tool_call" for event in run.events):
        return HarnessError(f"the runtime failed after tool use: {message}")
    return HarnessRunFailed(f"the runtime failed: {message}")


class PydanticAiHarness(HarnessRuntime):
    """:class:`HarnessPort` over AU's in-process agent runtime."""

    def __init__(self, runner: AgentRunner) -> None:
        super().__init__()
        if not callable(getattr(runner, "execute_agent", None)):
            raise TypeError("an AU agent runner with execute_agent is required")
        self._runner = runner

    def describe(self) -> HarnessDescriptor:
        return DESCRIPTOR

    def preflight(self, spec: RunSpec) -> None:
        if spec.account_mode == "api_key" and not spec.account_ref:
            raise HarnessNotConfigured("pydantic-ai api_key mode needs an account_ref")
        if len(spec.toolset.endpoints()) > 1:
            raise HarnessRefused(
                HARNESS_NAME,
                ("the in-process runtime binds at most one MCP server per run",),
            )

    async def drive(self, run: RunContext) -> DriveOutcome:
        spec = run.spec
        try:
            output = await self._runner.execute_agent(
                spec.agent_ref,
                spec.task,
                progress_sink=_progress_sink(run),
                **_options(spec),
            )
        except Exception as exc:
            raise _typed_failure(run, exc) from exc
        text = str(output)
        run.emit("message", "claim", detail=text)
        return DriveOutcome(status="succeeded", output=text)


__all__ = ["DESCRIPTOR", "HARNESS_NAME", "RUNTIME_OPTIONS", "PydanticAiHarness"]
