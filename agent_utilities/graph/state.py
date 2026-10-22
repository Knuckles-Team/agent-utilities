"""Typed state of one agent run: the request, the plan it receives, the result.

The harness never chooses any field of a :class:`RunPlan`. The host (arbora,
answering from epistemic-graph) supplies the plan through
:func:`agent_utilities.host_ports.register_run_decision_provider`. Without a
host, :func:`default_run_plan` builds the documented fallback plan from the
request and the graph configuration only.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai import BinaryContent

from agent_utilities.harness.plan import PlanState

RunStatus = Literal["pending", "running", "completed", "stopped", "failed"]

#: Fallback budgets when the host gives none.
DEFAULT_REQUEST_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_STEPS = 30
DEFAULT_RETRIES = 1


class RunPlanUnavailable(RuntimeError):
    """The run has no model: neither the host plan nor the caller named one."""


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
    prompt: str
    depends_on: tuple[str, ...] = ()


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
    """

    model: Any
    instructions: str = ""
    steps: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    toolsets: tuple[Any, ...] = ()
    budget: RunBudget = field(default_factory=RunBudget)
    stop: StopRule = field(default_factory=StopRule)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    input_media_types: tuple[str, ...] = ()
    dependency_plan: DependencyPlan | None = None


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
    dependency_outputs: Mapping[str, str] = field(default_factory=dict)


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output": self.output,
            "status": self.status,
            "steps": self.steps,
            "usage": dict(self.usage),
            "error": self.error,
            "execution_mode": self.execution_mode,
        }


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
    "DependencyPlan",
    "GraphDeps",
    "GraphState",
    "HarnessConfig",
    "RunBudget",
    "RunPlan",
    "RunPlanUnavailable",
    "RunRequest",
    "RunResult",
    "RunStatus",
    "RunTask",
    "StopRule",
    "default_run_plan",
]
