"""The agent run loop: apply a received run plan with pydantic-ai.

The loop takes a :class:`RunPlan` from the host decision provider (or the
documented fallback in :func:`default_run_plan`), binds the model and tools
through ``host_ports``, runs each plan step in order on one message history,
enforces the plan's budget, timeout and stop rule, and emits one event per
transition to the host run event sink. It chooses nothing and stores nothing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from agent_utilities import host_ports
from agent_utilities.harness.plan import PlanState
from agent_utilities.harness.plan_runtime import PlanScaffold

from .checkpoint import CheckpointUnavailable, RunCheckpoint
from .conversation_history import read_conversation_history
from .state import (
    GraphDeps,
    GraphState,
    RunPlan,
    RunRequest,
    RunResult,
    default_run_plan,
)


async def resolve_plan(request: RunRequest, config: Mapping[str, Any]) -> RunPlan:
    """The host's plan for ``request``, else the fallback plan."""
    provider = host_ports.run_decision_provider()
    if provider is None:
        return default_run_plan(request, config)
    plan = provider(request)
    if inspect.isawaitable(plan):
        plan = await plan
    if not isinstance(plan, RunPlan):
        raise TypeError("the run decision provider must return a RunPlan")
    return plan


def build_agent(
    plan: RunPlan,
    run_id: str,
    *,
    scaffold: PlanState | None = None,
    publisher: host_ports.ScaffoldPublisher | None = None,
) -> Agent[GraphDeps, Any]:
    """One pydantic-ai agent bound to the plan's model, instructions and tools."""
    if plan.harness.plan_scaffold not in ("on", "off"):
        raise ValueError("plan_scaffold must be on or off")
    capabilities = (
        [PlanScaffold(scaffold if scaffold is not None else PlanState(), publisher)]
        if plan.harness.plan_scaffold == "on"
        else []
    )
    model = (
        host_ports.resolve_model(plan.model)
        if isinstance(plan.model, str)
        else plan.model
    )
    toolsets = list(plan.toolsets)
    if plan.tool_names:
        toolsets.append(
            host_ports.build_native_toolset(
                list(plan.tool_names), toolset_id=f"run-{run_id}"
            )
        )
    output_type: Any = dict[str, Any] if plan.harness.response_format == "json" else str
    return Agent(
        model,
        capabilities=capabilities,
        output_type=output_type,
        instructions=plan.instructions or None,
        deps_type=GraphDeps,
        toolsets=toolsets or None,
        retries=plan.harness.retries,
        model_settings=dict(plan.harness.model_settings) or None,  # type: ignore[arg-type]
    )


def _limits(plan: RunPlan) -> UsageLimits:
    return UsageLimits(
        request_limit=plan.budget.request_limit,
        tool_calls_limit=plan.budget.tool_calls_limit,
        total_tokens_limit=plan.budget.total_tokens_limit,
    )


def _text(output: Any) -> str:
    return output if isinstance(output, str) else json.dumps(output, sort_keys=True)


def _event(state: GraphState, kind: str, **fields: Any) -> dict[str, Any]:
    event = {"type": kind, "run_id": state.run_id, "step": state.step, **fields}
    host_ports.emit_run_event(event)
    return event


async def iterate_run(
    request: RunRequest,
    config: Mapping[str, Any] | None = None,
    *,
    plan: RunPlan | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run ``request`` step by step, yielding each event.

    Events: ``run_started``, one ``node_transition`` per step, then exactly
    one terminal ``graph_complete`` (completed or stopped) or ``error``. The
    terminal event carries ``result`` (a :class:`RunResult` as a dict).
    """
    cfg = dict(config or {})
    state = GraphState(run_id=request.run_id, query=request.task, mode=request.mode)
    try:
        run_plan = plan or await resolve_plan(request, cfg)
        if run_plan.harness.plan_scaffold == "on":
            state.scaffold = PlanState()
    except Exception as exc:  # noqa: BLE001 - a failed plan is a failed run
        state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
        yield _event(state, "error", error=state.error, result=_result(state))
        return
    deps = GraphDeps(state=state, plan=run_plan, config=cfg)
    state.status = "running"
    yield _event(state, "run_started", mode=state.mode, steps=len(run_plan.steps))
    history: list[Any] = []
    limits = _limits(run_plan)
    steps = run_plan.steps[: max(run_plan.stop.max_steps, 0)]
    try:
        async with host_ports.authority_keepalive_scope():
            async with host_ports.run_checkpoint_scope(request, run_plan) as port:
                checkpoint = (
                    RunCheckpoint(port, state, run_plan) if port is not None else None
                )
                if checkpoint is not None:
                    await checkpoint.restore()
                    history = checkpoint.history
                if checkpoint is None or not checkpoint.restored:
                    history = await read_conversation_history(request)
                    if history and checkpoint is None:
                        raise CheckpointUnavailable(
                            "conversation continuation requires durable checkpoints"
                        )
                    if checkpoint is not None:
                        checkpoint.history = history
                timeout = (
                    checkpoint.remaining_timeout()
                    if checkpoint
                    else run_plan.budget.timeout_seconds
                )
                if (
                    checkpoint is not None
                    and timeout <= 0
                    and state.status == "running"
                    and state.step < len(steps)
                ):
                    raise TimeoutError
                async with asyncio.timeout(timeout):
                    async with host_ports.scaffold_publication_scope(
                        request, state.scaffold
                    ) as publisher:
                        agent = build_agent(
                            run_plan,
                            state.run_id,
                            scaffold=state.scaffold,
                            publisher=publisher,
                        )
                        for prompt in steps[state.step :]:
                            if state.status != "running":
                                break
                            if checkpoint is not None:
                                await checkpoint.save("step_inflight")
                            state.step += 1
                            result = await agent.run(
                                prompt,
                                deps=deps,
                                message_history=history,
                                usage_limits=limits,
                                usage=checkpoint.usage()
                                if checkpoint is not None
                                else None,
                            )
                            history = list(result.all_messages())
                            if checkpoint is not None:
                                state.usage = {}
                            state.add_usage(result.usage)
                            text = _text(result.output)
                            state.outputs.append(text)
                            if any(m in text for m in run_plan.stop.stop_markers):
                                state.status = "stopped"
                            elif state.step == len(steps):
                                state.status = "completed"
                            if checkpoint is not None:
                                checkpoint.history = history
                                await checkpoint.save("completed")
                            yield _event(state, "node_transition", output=text)
                            if state.status != "running":
                                break
    except UsageLimitExceeded as exc:
        state.status, state.error = "stopped", f"budget: {exc}"
    except TimeoutError:
        state.status, state.error = "stopped", "budget: run timed out"
    except Exception as exc:  # noqa: BLE001 - surfaced as the terminal event
        state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
        yield _event(state, "error", error=state.error, result=_result(state))
        return
    if state.status == "running":
        state.status = "completed"
    yield _event(state, "graph_complete", status=state.status, result=_result(state))


def _result(state: GraphState) -> dict[str, Any]:
    return RunResult(
        run_id=state.run_id,
        output=state.outputs[-1] if state.outputs else "",
        status=state.status,
        steps=state.step,
        usage=dict(state.usage),
        error=state.error,
    ).as_dict()


async def run(
    request: RunRequest,
    config: Mapping[str, Any] | None = None,
    *,
    plan: RunPlan | None = None,
) -> RunResult:
    """Drive :func:`iterate_run` to its terminal event and return the result."""
    final: dict[str, Any] = {}
    async for event in iterate_run(request, config, plan=plan):
        final = event
    data = final["result"]
    return RunResult(**data)


class AgentRunExecutor:
    """The agent-execution port arbora composes into its control plane.

    ``execute_agent`` reads the fields of arbora's typed execution request
    (duck-typed, so the harness does not import arbora) and returns a
    :class:`RunResult`; arbora maps it onto its own result type.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    async def execute_agent(self, request: Any, *, session: Any = None) -> RunResult:
        run_request = RunRequest(
            task=str(request.task),
            agent_name=str(getattr(request, "agent_name", "") or ""),
            run_id=getattr(request, "run_id", None) or RunRequest(task="").run_id,
            context=getattr(request, "context", None),
            allowed_tools=getattr(request, "allowed_tools", None),
            max_steps=getattr(request, "max_steps", None),
            budget_tokens=getattr(request, "budget_tokens", None),
            response_format=getattr(request, "response_format", "text"),
            metadata={"session": session} if session is not None else {},
        )
        result = await run(run_request, self._config)
        if result.status == "failed":
            raise RuntimeError(result.error or "agent run failed")
        return result


__all__ = [
    "AgentRunExecutor",
    "build_agent",
    "iterate_run",
    "resolve_plan",
    "run",
]
