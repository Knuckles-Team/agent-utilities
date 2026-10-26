"""Typed state of one agent run: the request, the plan it receives, the result.

The harness never chooses any field of a :class:`RunPlan`. The host (arbora,
answering from epistemic-graph) supplies the plan through
:func:`agent_utilities.host_ports.register_run_decision_provider`. Without a
host, :func:`default_run_plan` builds the documented fallback plan from the
request and the graph configuration only.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import BinaryContent
from pydantic_ai.usage import RunUsage

from agent_utilities.harness.plan import PlanState

from .metrics import RunMetrics

if TYPE_CHECKING:
    from agent_utilities.harness.client_capabilities import ClientToolLimits
    from agent_utilities.host_ports import ClientCapabilityTransport

RunStatus = Literal["pending", "running", "completed", "stopped", "failed"]

#: Fallback budgets when the host gives none.
DEFAULT_REQUEST_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_STEPS = 30
DEFAULT_RETRIES = 1


class RunPlanUnavailable(RuntimeError):
    """The run has no model: neither the host plan nor the caller named one."""


class RunPlanFailure(Exception):
    """Trusted host failure after measured work, before a plan can return.

    Raise this from the run decision provider only with actual reported usage.
    The run loop validates and snapshots the counters, then handles ``error``
    as the original failure or cancellation. This grants no execution and
    supplies no replacement budget. Model output and request metadata cannot
    set it. A provider without a usage ledger must raise its original error.
    """

    def __init__(
        self,
        error: BaseException,
        *,
        initial_usage: RunUsage,
        initial_elapsed_seconds: float = 0.0,
    ) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        super().__init__(str(error))
        self.error = error
        self.initial_usage = initial_usage
        self.initial_elapsed_seconds = initial_elapsed_seconds


@dataclass(frozen=True)
class RunBudget:
    """Limits the run loop enforces; it never widens them.

    ``candidate_slots`` is the candidate slot count of the stage
    allocation the run decision carries; the harness evolution cycle
    reads its candidate budget k from it and abstains below two.
    ``max_concurrent_requests`` caps concurrent model calls on dependency and
    recursive paths; waiting parents do not hold these permits.
    """

    request_limit: int | None = DEFAULT_REQUEST_LIMIT
    tool_calls_limit: int | None = None
    total_tokens_limit: int | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    candidate_slots: int | None = None
    max_attachment_bytes: int = 1_048_576
    max_attachments: int = 8
    max_concurrent_requests: int = 4


@dataclass(frozen=True)
class StopRule:
    """When the loop stops before the plan runs out.

    ``max_steps`` caps the plan steps executed. ``stop_markers`` stop the loop
    after a step whose output contains any marker.
    """

    max_steps: int = DEFAULT_MAX_STEPS
    stop_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessConfig:
    """In-run settings the host passes through to the agent framework."""

    retries: int = DEFAULT_RETRIES
    response_format: Literal["text", "json"] = "text"
    model_settings: Mapping[str, Any] = field(default_factory=dict)
    plan_scaffold: Literal["on", "off"] = "off"
    client_tools: tuple[str, ...] = ()
    client_tool_limits: ClientToolLimits | None = None


ToolOutcome = Literal["ok", "refused", "failed"]


@dataclass(frozen=True, slots=True)
class ToolStep:
    """One host-bound call in a received dependency plan.

    The harness only carries the declared server, tool and JSON-shaped
    arguments. The host owns the caller, transport, credentials and policy.
    """

    server: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolStepResult:
    """Structured result returned by a host-bound :class:`ToolCaller`."""

    result: Any = None
    outcome: ToolOutcome = "ok"
    error: str | None = None

    def __post_init__(self) -> None:
        # Accept the natural ``ToolStepResult("ok", value)`` positional form
        # as well as the named ``result=..., outcome=...`` form. The latter is
        # the stable wire shape; this small normalization keeps host adapters
        # from having to coordinate constructor ordering.
        if self.outcome not in ("ok", "refused", "failed"):
            if isinstance(self.result, str) and self.result in (
                "ok",
                "refused",
                "failed",
            ):
                positional_outcome = self.result
                positional_result = self.outcome
                object.__setattr__(self, "result", positional_result)
                object.__setattr__(self, "outcome", positional_outcome)
            else:
                raise ValueError("tool result outcome must be ok, refused or failed")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("tool result error must be a string or None")

    @property
    def status(self) -> ToolOutcome:
        """Compatibility spelling for callers that use status terminology."""

        return self.outcome

    @property
    def output(self) -> Any:
        """Compatibility spelling for the structured result value."""

        return self.result


ToolCaller = Callable[[str, str, Mapping[str, Any], str], Awaitable[ToolStepResult]]


class ToolCallerUnavailable(RuntimeError):
    """A tool-step plan was received without a host-bound caller."""


class BindingUnresolved(RuntimeError):
    """A tool-step argument could not resolve from a completed prerequisite."""

    def __init__(
        self,
        task_id: str,
        argument: str,
        reference: str,
        reason: str,
    ) -> None:
        self.task_id = task_id
        self.argument = argument
        self.reference = reference
        self.reason = reason
        super().__init__(
            f"task {task_id!r} argument {argument!r}: {reason} ({reference})"
        )


class ToolCallRefused(RuntimeError):
    """The host-bound caller refused a declared tool step."""


class ToolCallFailed(RuntimeError):
    """The host-bound caller failed a declared tool step."""


@dataclass(frozen=True)
class RunRequest:
    """What the caller asks for. The host turns it into a :class:`RunPlan`."""

    task: str
    agent_name: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mode: str = "ask"
    context: str | None = None
    model_id: str | None = None
    allowed_tools: tuple[str, ...] | None = None
    max_steps: int | None = None
    budget_tokens: int | None = None
    response_format: Literal["text", "json"] = "text"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attachments: tuple[BinaryContent, ...] = ()


@dataclass(frozen=True)
class RunTask:
    """One received task and its direct prerequisites, not a planning decision."""

    task_id: str
    prompt: str = ""
    depends_on: tuple[str, ...] = ()
    call: ToolStep | None = None


@dataclass(frozen=True)
class DependencyPlan:
    """An admitted, finite DAG to execute with a bounded rolling ready queue.

    The host supplies declaration order, dependencies, the final output task
    and the concurrency limit. The harness neither adds tasks nor picks them.
    All tasks must contribute to ``output_task_id`` through dependency edges.
    """

    tasks: tuple[RunTask, ...]
    output_task_id: str
    max_concurrency: int = 1
    plan_id: str = ""


@dataclass(frozen=True)
class RunPlan:
    """Every decision of one run, as received from the host.

    ``model`` is a model id (resolved through the host model provider) or a
    ready pydantic-ai model. ``steps`` are the prompts run in order; each step
    sees the earlier steps' messages. ``toolsets`` are ready pydantic-ai
    toolsets; ``tool_names`` are host operations bound through the native
    toolset provider. ``dependency_plan`` opts into bounded rolling execution
    instead of serial ``steps``; the two are mutually exclusive.

    ``initial_usage`` and ``initial_elapsed_seconds`` are measured consumption
    supplied only by the trusted host, before execution (for example planning).
    Keep ``budget`` at its original bounds: the harness charges this consumption
    once. Model proposals and request metadata must never supply these fields.
    """

    model: Any
    execution_constraint: Literal["text-only"] | None = field(
        default=None, kw_only=True
    )
    instructions: str = ""
    steps: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    toolsets: tuple[Any, ...] = ()
    budget: RunBudget = field(default_factory=RunBudget)
    stop: StopRule = field(default_factory=StopRule)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    client_transport: ClientCapabilityTransport | None = field(
        default=None, repr=False, compare=False
    )
    input_media_types: tuple[str, ...] = ()
    dependency_plan: DependencyPlan | None = None
    initial_usage: RunUsage = field(default_factory=RunUsage)
    initial_elapsed_seconds: float = 0.0


@dataclass
class GraphState:
    """Mutable state of one run while the loop executes it."""

    run_id: str
    query: str
    mode: str = "ask"
    status: RunStatus = "pending"
    step: int = 0
    outputs: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    scaffold: PlanState | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)
    event_sequence: int = 0

    def add_usage(self, usage: Any) -> None:
        for key in ("requests", "tool_calls", "input_tokens", "output_tokens"):
            self.usage[key] = self.usage.get(key, 0) + int(getattr(usage, key, 0) or 0)
        self.usage["total_tokens"] = self.usage.get("input_tokens", 0) + self.usage.get(
            "output_tokens", 0
        )


@dataclass
class GraphDeps:
    """Dependencies the agent's tools see through ``RunContext.deps``."""

    state: GraphState
    plan: RunPlan
    config: Mapping[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    dependency_outputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one run."""

    run_id: str
    output: str
    status: RunStatus
    steps: int
    usage: Mapping[str, int]
    error: str | None = None
    execution_mode: Literal["direct", "graph"] = "graph"
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output": self.output,
            "status": self.status,
            "steps": self.steps,
            "usage": dict(self.usage),
            "error": self.error,
            "execution_mode": self.execution_mode,
            "metrics": dict(self.metrics),
        }


def _validate_failure_outcome(result: RunResult) -> None:
    """Only a failed terminal state can travel through the exception port."""
    if type(result.status) is not str or result.status != "failed":
        raise ValueError("result status must be failed")
    if type(result.steps) is not int or result.steps < 0:
        raise ValueError("result steps must be a nonnegative integer")
    if result.error is not None and type(result.error) is not str:
        raise ValueError("result error must be a string or None")
    if type(result.execution_mode) is not str or result.execution_mode not in (
        "direct",
        "graph",
    ):
        raise ValueError("result execution_mode must be direct or graph")


def _validate_public_usage(counters: Mapping[str, int]) -> None:
    """Accept a complete public ledger or preserve an unreported empty one."""
    if not counters:
        return
    if set(counters) != {
        "requests",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }:
        raise ValueError("result usage must contain the five public counters")
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("result usage counters must be nonnegative integers")
    if counters["total_tokens"] != counters["input_tokens"] + counters["output_tokens"]:
        raise ValueError("result usage total_tokens must match input and output")


def _failed_result_snapshot(result: RunResult) -> RunResult:
    """Copy before validation so the retained receipt cannot change afterward."""
    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult")
    if not isinstance(result.usage, Mapping):
        raise ValueError("result usage must be a mapping")
    snapshot = RunResult(
        run_id=result.run_id,
        output=result.output,
        status=result.status,
        steps=result.steps,
        usage=MappingProxyType(dict(result.usage)),
        error=result.error,
        execution_mode=result.execution_mode,
        metrics=MappingProxyType(dict(result.metrics)),
    )
    if type(snapshot.run_id) is not str or not snapshot.run_id:
        raise ValueError("result run_id must be a nonempty string")
    if type(snapshot.output) is not str:
        raise ValueError("result output must be a string")
    _validate_failure_outcome(snapshot)
    _validate_public_usage(snapshot.usage)
    return snapshot


class RunFailed(RuntimeError):
    """A failed executor result with its authoritative aggregate receipt.

    The message and RuntimeError catch contract stay unchanged. The snapshot
    grants no execution or retry. Hosts must match its run identity before
    using its counters, and must not add them to the terminal ledger again.
    """

    def __init__(self, result: RunResult) -> None:
        snapshot = _failed_result_snapshot(result)
        super().__init__(snapshot.error or "agent run failed")
        self._result: RunResult = snapshot

    @property
    def result(self) -> RunResult:
        return self._result

    @property
    def run_id(self) -> str:
        return self._result.run_id

    @property
    def usage(self) -> Mapping[str, int]:
        return self._result.usage

    def __reduce__(self) -> tuple[type[RunFailed], tuple[RunResult], dict[str, Any]]:
        # RuntimeError reconstructs from its message by default. Rebuild this
        # typed carrier from plain scalar data instead of a mapping proxy.
        state = {key: value for key, value in self.__dict__.items() if key != "_result"}
        return type(self), (RunResult(**self.result.as_dict()),), state


def default_run_plan(request: RunRequest, config: Mapping[str, Any]) -> RunPlan:
    """The fallback plan when no host decision provider is registered.

    * model: ``request.model_id``, else ``config["agent_model"]``; none fails closed;
    * instructions: the named specialist's prompt, else ``config["system_prompt"]``;
    * steps: the task alone, prefixed by ``request.context`` when given;
    * tools: only ``request.allowed_tools`` (bound through the host) plus the
      toolsets the graph was built with; nothing is discovered;
    * budget, stop rule and harness: the module defaults above, narrowed by
      ``request.max_steps`` and ``request.budget_tokens``.
    """
    model = request.model_id or config.get("agent_model")
    if not model:
        raise RunPlanUnavailable(
            "no model: register a run decision provider or name agent_model"
        )
    specialists = config.get("specialists") or {}
    instructions = str(
        specialists.get(request.agent_name) or config.get("system_prompt") or ""
    )
    prompt = request.task
    if request.context:
        prompt = f"{request.context}\n\n{request.task}"
    return RunPlan(
        model=model,
        instructions=instructions,
        steps=(prompt,),
        tool_names=tuple(request.allowed_tools or ()),
        toolsets=tuple(config.get("toolsets") or ()),
        budget=RunBudget(total_tokens_limit=request.budget_tokens),
        stop=StopRule(max_steps=request.max_steps or DEFAULT_MAX_STEPS),
        harness=HarnessConfig(response_format=request.response_format),
    )


__all__ = [
    "BindingUnresolved",
    "DependencyPlan",
    "GraphDeps",
    "GraphState",
    "HarnessConfig",
    "RunBudget",
    "RunFailed",
    "RunPlan",
    "RunPlanUnavailable",
    "RunRequest",
    "RunResult",
    "RunStatus",
    "RunTask",
    "StopRule",
    "ToolCaller",
    "ToolCallerUnavailable",
    "ToolCallFailed",
    "ToolCallRefused",
    "ToolOutcome",
    "ToolStep",
    "ToolStepResult",
    "default_run_plan",
]
