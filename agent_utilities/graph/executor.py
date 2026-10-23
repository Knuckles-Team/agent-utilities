"""The agent run loop: apply a received run plan with pydantic-ai.

The loop takes a :class:`RunPlan` from the host decision provider (or the
documented fallback in :func:`default_run_plan`), binds the model and tools
through ``host_ports``, runs serial steps on one history or applies a received
dependency DAG with bounded rolling slots, enforces the received limits, and
emits public transitions to the host run event sink. It chooses and stores
nothing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from agent_utilities import host_ports
from agent_utilities.harness.conversation_history import decode_completed_history
from agent_utilities.harness.plan import PlanState
from agent_utilities.harness.plan_runtime import PlanScaffold

from .budget import SharedBudget, initial_run_usage
from .dependency import iterate_dependency_plan, validate_dependency_plan
from .metrics import MeasuredModel, RunMetrics
from .state import (
    GraphDeps,
    GraphState,
    RunFailed,
    RunPlan,
    RunPlanFailure,
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
    metrics: RunMetrics | None = None,
    shared_budget: SharedBudget | None = None,
) -> Agent[GraphDeps, Any]:
    """One pydantic-ai agent bound to the plan's model, instructions and tools."""
    if plan.harness.plan_scaffold not in ("on", "off"):
        raise ValueError("plan_scaffold must be on or off")
    capabilities: list[Any] = (
        [PlanScaffold(scaffold if scaffold is not None else PlanState(), publisher)]
        if plan.harness.plan_scaffold == "on"
        else []
    )
    if shared_budget is not None:
        capabilities.append(shared_budget)
    model = (
        host_ports.resolve_model(plan.model)
        if isinstance(plan.model, str)
        else plan.model
    )
    if metrics is not None:
        model = MeasuredModel(model, metrics)
    toolsets = list(plan.toolsets)
    if plan.harness.client_tools:
        from agent_utilities.harness.client_capabilities import ClientCapabilityToolset

        toolsets.append(
            ClientCapabilityToolset(
                plan.client_transport,
                plan.harness.client_tools,
                limits=plan.harness.client_tool_limits,
                run_id=run_id,
            )
        )
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
    state.event_sequence += 1
    event = {
        "type": kind,
        "run_id": state.run_id,
        "step": state.step,
        **fields,
        "sequence": state.event_sequence,
        "event_id": uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"agent-utilities:run:{state.run_id}:event:{state.event_sequence}",
        ).hex,
    }
    host_ports.emit_run_event(event)
    return event


def _attachment_input(request: RunRequest, plan: RunPlan) -> list[str | BinaryContent]:
    """Enforce the received model input grant and limits without widening them."""
    if not request.attachments:
        return []
    for limit in (plan.budget.max_attachments, plan.budget.max_attachment_bytes):
        if type(limit) is not int or limit < 0:
            raise ValueError("invalid attachment budget")
    if len(request.attachments) > plan.budget.max_attachments:
        raise ValueError("attachment count exceeds the run budget")
    total = 0
    content: list[str | BinaryContent] = []
    for part in request.attachments:
        if not isinstance(part, BinaryContent) or not isinstance(part.data, bytes):
            raise ValueError("attachments must be inline binary content")
        if part.media_type not in plan.input_media_types:
            raise ValueError("attachment media type is not admitted for this model")
        if part.media_type not in (
            "text/plain",
            "image/png",
            "image/jpeg",
            "image/webp",
        ):
            raise ValueError("unsupported attachment media type")
        total += len(part.data)
        if not part.data or total > plan.budget.max_attachment_bytes:
            raise ValueError("attachment bytes exceed the run budget or are empty")
        content.append(
            part.data.decode("utf-8") if part.media_type == "text/plain" else part
        )
    return content


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
        supplied_usage = None
        if plan is not None:
            supplied_usage = initial_run_usage(plan)
            if supplied_usage.has_values() or plan.initial_elapsed_seconds:
                state.add_usage(supplied_usage)
        content = await host_ports.read_run_history(request)
        history = (
            decode_completed_history(content, run_id=state.run_id)
            if content is not None
            else []
        )
        try:
            run_plan = plan or await resolve_plan(request, cfg)
        except RunPlanFailure as failure:
            # A trusted provider can report measured work even when no plan
            # was admitted. It grants no execution and cannot replace limits.
            state.add_usage(initial_run_usage(failure))
            if isinstance(failure.error, (asyncio.CancelledError, GeneratorExit)):
                state.status = "stopped"
                _event(state, "run_cancelled", status="stopped", result=_result(state))
            raise failure.error from failure.error.__cause__
        initial_usage = (
            supplied_usage
            if supplied_usage is not None
            else initial_run_usage(run_plan)
        )
        has_initial_usage = initial_usage.has_values() or bool(
            run_plan.initial_elapsed_seconds
        )
        if has_initial_usage and supplied_usage is None:
            state.add_usage(initial_usage)
        if history and run_plan.dependency_plan is not None:
            raise ValueError("dependency-plan conversation history is unavailable")
        validate_dependency_plan(run_plan)
        attachments = _attachment_input(request, run_plan)
        if run_plan.harness.plan_scaffold == "on":
            state.scaffold = PlanState()
    except Exception as exc:  # noqa: BLE001 - a failed plan is a failed run
        state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
        yield _event(state, "error", error=state.error, result=_result(state))
        return
    deps = GraphDeps(state=state, plan=run_plan, config=cfg)
    state.status = "running"
    limits = _limits(run_plan)
    steps = run_plan.steps[: max(run_plan.stop.max_steps, 0)]
    shared_budget = (
        SharedBudget(
            run_plan.budget,
            max_concurrent_requests=run_plan.budget.max_concurrent_requests,
            initial_usage=initial_usage if has_initial_usage else None,
        )
        if run_plan.dependency_plan is not None or has_initial_usage
        else None
    )
    try:
        if has_initial_usage:
            assert shared_budget is not None
            shared_budget.check_initial_consumption()
            if run_plan.initial_elapsed_seconds >= run_plan.budget.timeout_seconds:
                raise TimeoutError
        yield _event(
            state,
            "run_started",
            mode=state.mode,
            steps=len(run_plan.dependency_plan.tasks)
            if run_plan.dependency_plan is not None
            else len(run_plan.steps),
        )
        async with host_ports.authority_keepalive_scope():
            async with asyncio.timeout(
                run_plan.budget.timeout_seconds - run_plan.initial_elapsed_seconds
            ) as timeout:
                async with host_ports.scaffold_publication_scope(
                    request, state.scaffold
                ) as publisher:
                    agent = build_agent(
                        run_plan,
                        state.run_id,
                        scaffold=state.scaffold,
                        publisher=publisher,
                        metrics=state.metrics,
                        shared_budget=shared_budget,
                    )
                    if run_plan.dependency_plan is not None:
                        assert shared_budget is not None
                        # Setup uses this same absolute deadline. Once setup
                        # finishes, the driver owns it, not the paused viewer.
                        deadline = timeout.when()
                        timeout.reschedule(None)
                        async with aclosing(
                            iterate_dependency_plan(
                                agent,
                                run_plan,
                                state,
                                cfg,
                                attachments,
                                shared_budget,
                                _event,
                                deadline=deadline,
                            )
                        ) as execution:
                            async for event in execution:
                                yield event
                    for prompt in steps:
                        state.step += 1
                        model_input: str | list[str | BinaryContent] = prompt
                        if state.step == 1 and attachments:
                            model_input = [prompt, *attachments]
                        try:
                            result = await agent.run(
                                model_input,
                                deps=deps,
                                message_history=history,
                                usage_limits=limits,
                            )
                        finally:
                            if shared_budget is not None:
                                state.usage.clear()
                                state.add_usage(shared_budget.usage)
                        history = list(result.all_messages())
                        if shared_budget is None:
                            state.add_usage(result.usage)
                        text = _text(result.output)
                        state.outputs.append(text)
                        yield _event(state, "node_transition", output=text)
                        if any(m in text for m in run_plan.stop.stop_markers):
                            state.status = "stopped"
                            break
    except UsageLimitExceeded as exc:
        state.status, state.error = "stopped", f"budget: {exc}"
    except TimeoutError:
        state.status, state.error = "stopped", "budget: run timed out"
    except (asyncio.CancelledError, GeneratorExit):
        state.status = "stopped"
        _event(state, "run_cancelled", status="stopped", result=_result(state))
        raise
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
        metrics=state.metrics.snapshot(),
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
    :class:`RunResult`; arbora maps it onto its own result type. A failed
    result raises :class:`RunFailed` with the same aggregate receipt.
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
            attachments=tuple(getattr(request, "attachments", ())),
        )
        result = await run(run_request, self._config)
        if result.status == "failed":
            raise RunFailed(result)
        return result


__all__ = [
    "AgentRunExecutor",
    "build_agent",
    "iterate_run",
    "resolve_plan",
    "run",
]
