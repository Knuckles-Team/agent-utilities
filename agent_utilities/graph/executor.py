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
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import Any

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.usage import UsageLimits

from agent_utilities import host_ports
from agent_utilities.harness.conversation_history import decode_completed_history
from agent_utilities.harness.plan import PlanState
from agent_utilities.harness.plan_runtime import PlanScaffold

from .budget import LiveSharedBudget, SharedBudget, initial_run_usage
from .checkpoint import RunCheckpoint
from .dependency import iterate_dependency_plan, validate_dependency_plan
from .live_events import run_step
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
from .text_only import (
    TextOnlyModel,
    TextOnlyResponseFence,
    no_host_scope,
    validate_text_plan,
)

ModelStreamObserver = Callable[[AgentStreamEvent], Awaitable[None]]


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
    deadline: float | None = None,
) -> Agent[GraphDeps, Any]:
    """One pydantic-ai agent bound to the plan's model, instructions and tools."""
    if plan.execution_constraint not in (None, "text-only"):
        raise ValueError("unknown received execution constraint")
    if plan.execution_constraint == "text-only":
        validate_text_plan(plan)
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
    if plan.execution_constraint == "text-only":
        model = TextOnlyModel(model, plan, deadline=deadline)
        capabilities.append(TextOnlyResponseFence(model))
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


def _event(
    state: GraphState, kind: str, *, publish: bool = True, **fields: Any
) -> dict[str, Any]:
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
    if publish:
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
    stream_observer: ModelStreamObserver | None = None,
    shared_execution: LiveSharedBudget | None = None,
) -> AsyncGenerator[dict[str, Any]]:
    """Run ``request`` step by step, yielding each event.

    Events: ``run_started``, one ``node_transition`` per step, then exactly
    one terminal ``graph_complete`` (completed or stopped) or ``error``. The
    terminal event carries ``result`` (a :class:`RunResult` as a dict).

    ``stream_observer`` is trusted host plumbing, not a public transcript. It
    receives native events (including private thinking/tool content) inline,
    under the same deadline. Only ordinary sequential plans support it.
    """

    async def observe(
        _ctx: RunContext[GraphDeps], events: AsyncIterable[AgentStreamEvent]
    ) -> None:
        assert stream_observer is not None
        async for event in events:
            try:
                await stream_observer(event)
            except Exception:
                raise RuntimeError("model stream observer failed") from None

    restricted = plan is not None and plan.execution_constraint == "text-only"

    def emit(state: GraphState, kind: str, **fields: Any) -> dict[str, Any]:
        return _event(state, kind, publish=not restricted, **fields)

    cfg = dict(config or {})
    state = GraphState(run_id=request.run_id, query=request.task, mode=request.mode)
    try:
        if shared_execution is not None and plan is None:
            raise ValueError("shared execution requires an explicit received plan")
        supplied_usage = None
        if plan is not None:
            supplied_usage = initial_run_usage(plan)
            if supplied_usage.has_values() or plan.initial_elapsed_seconds:
                state.add_usage(supplied_usage)
        if restricted:
            assert plan is not None
            validate_text_plan(plan)
            if (
                request.attachments
                or request.context
                or request.metadata
                or request.allowed_tools not in (None, ())
                or request.task != plan.steps[0]
            ):
                raise ValueError("text-only request cannot inherit context")
        # Read the existing prior-turn selection before planning.  If the
        # bound current-run transaction restores content, that content replaces
        # this history; the selection port still owns current-run precedence.
        content = None if restricted else await host_ports.read_run_history(request)
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
                emit(state, "run_cancelled", status="stopped", result=_result(state))
            raise failure.error from failure.error.__cause__
        if stream_observer is not None and (
            run_plan.dependency_plan is not None
            or run_plan.execution_constraint is not None
        ):
            raise ValueError("model streaming requires an ordinary sequential plan")
        if run_plan.execution_constraint is not None and not restricted:
            raise ValueError("restricted execution needs an explicit received plan")
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
        if shared_execution is not None:
            if not isinstance(shared_execution, LiveSharedBudget):
                raise TypeError("shared execution budget is invalid")
            if has_initial_usage or restricted:
                raise ValueError(
                    "shared execution cannot restore usage or use restricted execution"
                )
        if history and run_plan.dependency_plan is not None:
            raise ValueError("dependency-plan conversation history is unavailable")
        tool_caller = host_ports.tool_caller()
        validate_dependency_plan(run_plan, tool_caller=tool_caller)
        attachments = _attachment_input(request, run_plan)
        if run_plan.harness.plan_scaffold == "on":
            state.scaffold = PlanState()
    except Exception as exc:  # noqa: BLE001 - a failed plan is a failed run
        state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
        yield emit(state, "error", error=state.error, result=_result(state))
        return
    deps = GraphDeps(state=state, plan=run_plan, config=cfg)
    limits = _limits(run_plan)
    steps = run_plan.steps[: max(run_plan.stop.max_steps, 0)]
    try:
        async with host_ports.run_checkpoint_scope(
            request, run_plan
        ) as checkpoint_port:
            if checkpoint_port is not None and shared_execution is not None:
                raise ValueError("shared execution cannot use a durable checkpoint")
            checkpoint = (
                RunCheckpoint(checkpoint_port, state, run_plan, attachments)
                if checkpoint_port is not None
                else None
            )
            if shared_execution is not None:
                shared_execution.require(run_plan.budget)
            if checkpoint is not None:
                await checkpoint.restore()
                if checkpoint.restored:
                    # A current-run snapshot is authoritative.  It carries
                    # cumulative usage and elapsed time; plan-provided
                    # initial counters must not be charged a second time.
                    initial_usage = checkpoint.usage()
                    has_initial_usage = True
                    history = list(checkpoint.history)
                    if run_plan.dependency_plan is not None:
                        history = []
                elif run_plan.dependency_plan is None:
                    # Preserve an authorized prior-turn history if this
                    # transaction is new and only the current snapshot is
                    # absent; a later restart must retain the same messages.
                    checkpoint.history = list(history)
            if not (checkpoint is not None and checkpoint.restored):
                state.status = "running"
            shared_budget = (
                shared_execution.run_view() if shared_execution else None
            ) or (
                SharedBudget(
                    run_plan.budget,
                    max_concurrent_requests=run_plan.budget.max_concurrent_requests,
                    initial_usage=initial_usage if has_initial_usage else None,
                )
                if run_plan.dependency_plan is not None
                or has_initial_usage
                or restricted
                or checkpoint is not None
                else None
            )
            if has_initial_usage:
                assert shared_budget is not None
                elapsed = (
                    checkpoint.elapsed
                    if checkpoint is not None and checkpoint.restored
                    else run_plan.initial_elapsed_seconds
                )
                if state.status == "running":
                    shared_budget.check_initial_consumption()
                    if elapsed >= run_plan.budget.timeout_seconds:
                        raise TimeoutError
            else:
                elapsed = 0.0
            yield emit(
                state,
                "run_started",
                mode=state.mode,
                steps=len(run_plan.dependency_plan.tasks)
                if run_plan.dependency_plan is not None
                else len(run_plan.steps),
            )
            # A terminal completed/stopped checkpoint is already the run's
            # result.  Do not rebuild a model or bind tool effects on restore.
            if (
                checkpoint is not None
                and checkpoint.restored
                and state.status
                in (
                    "completed",
                    "stopped",
                )
            ):
                pass
            else:
                async with (
                    no_host_scope()
                    if restricted
                    else host_ports.authority_keepalive_scope()
                ):
                    scope_deadline = (
                        asyncio.get_running_loop().time()
                        + run_plan.budget.timeout_seconds
                        - elapsed
                    )
                    if shared_execution is not None:
                        scope_deadline = min(scope_deadline, shared_execution.deadline)
                    timeout_scope = (
                        asyncio.timeout_at(scope_deadline)
                        if shared_execution is not None
                        else asyncio.timeout(run_plan.budget.timeout_seconds - elapsed)
                    )
                    async with timeout_scope as timeout:
                        async with (
                            (
                                no_host_scope()
                                if restricted
                                else host_ports.scaffold_publication_scope(
                                    request, state.scaffold
                                )
                            ) as publisher,
                            (
                                no_host_scope()
                                if restricted
                                else host_ports.durable_run_event_scope(request)
                            ) as event_publisher,
                        ):
                            agent = build_agent(
                                run_plan,
                                state.run_id,
                                scaffold=state.scaffold,
                                publisher=publisher,
                                metrics=state.metrics,
                                shared_budget=shared_budget,
                                deadline=timeout.when(),
                            )
                            if run_plan.dependency_plan is not None:
                                assert shared_budget is not None
                                # Setup uses this same absolute deadline. Once
                                # setup finishes, the driver owns it, not the
                                # paused viewer.
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
                                        emit,
                                        deadline=deadline,
                                        tool_caller=tool_caller,
                                        checkpoint=checkpoint,
                                        event_publisher=event_publisher,
                                    )
                                ) as execution:
                                    async for event in execution:
                                        yield event
                            for index, prompt in enumerate(steps):
                                if index < state.step:
                                    continue
                                state.step = index + 1
                                marker = f"step:{index}"
                                if checkpoint is not None:
                                    await checkpoint.save(
                                        "step_inflight", task_id=marker
                                    )
                                model_input: str | list[str | BinaryContent] = prompt
                                if index == 0 and attachments:
                                    model_input = [prompt, *attachments]

                                async def after_result(result: Any) -> None:
                                    nonlocal history
                                    if shared_budget is not None:
                                        state.usage.clear()
                                        state.add_usage(shared_budget.usage)
                                    history = list(result.all_messages())
                                    if shared_budget is None:
                                        state.add_usage(result.usage)
                                    text = _text(result.output)
                                    state.outputs.append(text)
                                    if any(
                                        marker in text
                                        for marker in run_plan.stop.stop_markers
                                    ):
                                        state.status = "stopped"
                                    if checkpoint is not None:
                                        checkpoint.history = history
                                        checkpoint.inflight_tasks.discard(marker)
                                        await checkpoint.save("completed")

                                try:
                                    result = await run_step(
                                        agent,
                                        model_input,
                                        publisher=event_publisher,
                                        step=state.step,
                                        stream_observer=observe
                                        if stream_observer is not None
                                        else None,
                                        after_result=after_result,
                                        deps=deps,
                                        message_history=history,
                                        usage_limits=limits,
                                    )
                                finally:
                                    if shared_budget is not None:
                                        state.usage.clear()
                                        state.add_usage(shared_budget.usage)
                                text = _text(result.output)
                                yield emit(state, "node_transition", output=text)
                                if state.status == "stopped":
                                    break
    except UsageLimitExceeded as exc:
        state.status, state.error = "stopped", f"budget: {exc}"
    except TimeoutError:
        state.status, state.error = "stopped", "budget: run timed out"
    except (asyncio.CancelledError, GeneratorExit):
        state.status = "stopped"
        emit(state, "run_cancelled", status="stopped", result=_result(state))
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as the terminal event
        state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
        yield emit(state, "error", error=state.error, result=_result(state))
        return
    if state.status == "running":
        state.status = "completed"
    yield emit(state, "graph_complete", status=state.status, result=_result(state))


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
    stream_observer: ModelStreamObserver | None = None,
    shared_execution: LiveSharedBudget | None = None,
) -> RunResult:
    """Drive :func:`iterate_run` to its terminal event and return the result."""
    final: dict[str, Any] = {}
    async for event in iterate_run(
        request,
        config,
        plan=plan,
        stream_observer=stream_observer,
        shared_execution=shared_execution,
    ):
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

    async def execute_text_only(
        self, request: RunRequest, *, plan: RunPlan
    ) -> RunResult:
        """Apply an explicit constrained plan without inherited host authority.

        This creates no anonymous caller, admission or durable result. The host
        must supply those boundaries and its configured admission limits.
        The host owns the supplied client and closes it only after this call
        and cancellation drain finish. Cancellation returns no usage receipt;
        retain the full reservation when the outcome is unknown.
        """
        validate_text_plan(plan)
        if type(request) is not RunRequest:
            raise TypeError("text-only execution requires the public run request")
        return await run(request, self._config, plan=plan)

    async def execute_agent(
        self,
        request: Any,
        *,
        session: Any = None,
        stream_observer: ModelStreamObserver | None = None,
        shared_execution: LiveSharedBudget | None = None,
        plan: RunPlan | None = None,
    ) -> RunResult:
        """Run a request using trusted, per-call host capabilities."""
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
        result = await run(
            run_request,
            self._config,
            plan=plan,
            stream_observer=stream_observer,
            shared_execution=shared_execution,
        )
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
