"""The L4 harness port: one typed contract between L3 agent graphs and runtimes.

An L3 node names a harness (``AgentSpec.harness``; default ``native``). The
harness registry resolves that name to a :class:`HarnessPort`. Every port takes
one :class:`HarnessRequest` and returns one :class:`RunOutcome`. The L5 run
recording (:mod:`agent_utilities.layers.harness_record`) consumes that outcome.

Adapters on main:

* ``native`` -- :class:`~agent_utilities.layers.harness_native.NativeHarness`
  over the pydantic-ai ``run_agent`` path.
* ``claude-code`` -- :class:`~agent_utilities.layers.harness_cli.ClaudeCodeHarness`
  over headless ``claude -p --output-format json``.

Extension point: a new runtime (for example a LangGraph graph runner) is one
class with a ``name`` and an async ``run(request) -> RunOutcome``, registered
through :meth:`HarnessRegistry.register`. No caller changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent_utilities.orchestration.run_identity import new_run_id

#: The harness every L3 node uses unless its spec names another one.
NATIVE_HARNESS = "native"

OutcomeStatus = Literal["completed", "degraded", "failed", "timeout", "refused"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HarnessRequest(_Frozen):
    """One L3 node invocation, independent of the runtime that serves it."""

    agent_name: str = Field(min_length=1)
    task: str = Field(min_length=1)
    run_id: str = Field(default_factory=new_run_id)
    workspace: Path | None = None
    timeout_s: float = Field(default=600.0, gt=0)
    model: str | None = None
    response_format: Literal["text", "json"] = "text"
    max_steps: int = Field(default=30, ge=1)
    allowed_tools: tuple[str, ...] = ()


class UsageReport(_Frozen):
    """Token and cost accounting as the runtime reported it."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float | None = None

    def token_usage(self) -> dict[str, int]:
        """The shape :class:`~agent_utilities.usage.recorder.UsageRecorder` reads."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


class DiffStat(_Frozen):
    """The working-tree change a run left in its workspace."""

    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


class RunOutcome(_Frozen):
    """The typed terminal outcome of one harness run (the L5 input)."""

    run_id: str
    harness: str
    agent_name: str
    status: OutcomeStatus
    final_text: str = ""
    structured_output: JsonValue | None = None
    exit_code: int | None = None
    transcript_ref: str | None = None
    trace_ref: str | None = None
    diff_stat: DiffStat | None = None
    usage: UsageReport | None = None
    model: str = ""
    duration_ms: float = 0.0
    error: str | None = None
    #: True when the runtime already wrote the RunTrace itself.
    recorded: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


@runtime_checkable
class HarnessPort(Protocol):
    """The L4 contract every runtime adapter implements."""

    @property
    def name(self) -> str: ...

    async def run(self, request: HarnessRequest) -> RunOutcome: ...


def refused(port_name: str, request: HarnessRequest, reason: str) -> RunOutcome:
    """A typed refusal: the harness did not start."""
    return RunOutcome(
        run_id=request.run_id,
        harness=port_name,
        agent_name=request.agent_name,
        status="refused",
        error=reason,
    )


__all__ = [
    "NATIVE_HARNESS",
    "DiffStat",
    "HarnessPort",
    "HarnessRequest",
    "OutcomeStatus",
    "RunOutcome",
    "UsageReport",
    "refused",
]
