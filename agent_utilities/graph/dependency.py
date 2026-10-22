"""Apply a received dependency DAG with at most its declared active task count."""

from __future__ import annotations

import asyncio
import heapq
import json
import math
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.usage import UsageLimits

from .budget import SharedBudget
from .state import DependencyPlan, GraphDeps, GraphState, RunPlan, RunTask


def validate_dependency_plan(plan: RunPlan) -> None:
    """Refuse malformed or over-budget plans before constructing a model."""
    graph = plan.dependency_plan
    if graph is None:
        return
    if not isinstance(graph, DependencyPlan) or plan.steps:
        raise ValueError("dependency_plan must not be combined with serial steps")
    if (
        not isinstance(graph.tasks, tuple)
        or not isinstance(graph.output_task_id, str)
        or not isinstance(graph.plan_id, str)
    ):
        raise ValueError("dependency plan must have immutable tasks and string IDs")
    if type(graph.max_concurrency) is not int or graph.max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    if type(plan.stop.max_steps) is not int or plan.stop.max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not graph.tasks or len(graph.tasks) > plan.stop.max_steps:
        raise ValueError("dependency plan is empty or exceeds max_steps")
    if type(plan.harness.retries) is not int or plan.harness.retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    if (
        type(plan.budget.max_concurrent_requests) is not int
        or plan.budget.max_concurrent_requests < 1
    ):
        raise ValueError("max_concurrent_requests must be a positive integer")
    for value in (
        plan.budget.request_limit,
        plan.budget.tool_calls_limit,
        plan.budget.total_tokens_limit,
    ):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("invalid dependency run budget")
    if (
        isinstance(plan.budget.timeout_seconds, bool)
        or not math.isfinite(plan.budget.timeout_seconds)
        or plan.budget.timeout_seconds < 0
    ):
        raise ValueError("invalid dependency run timeout")
    nodes: dict[str, RunTask] = {}
    for task in graph.tasks:
        if (
            not isinstance(task, RunTask)
            or not isinstance(task.task_id, str)
            or not task.task_id.strip()
            or task.task_id in nodes
            or not isinstance(task.prompt, str)
            or not task.prompt.strip()
            or not isinstance(task.depends_on, tuple)
            or any(not isinstance(dep, str) for dep in task.depends_on)
            or len(set(task.depends_on)) != len(task.depends_on)
        ):
            raise ValueError("invalid or duplicate dependency task")
        nodes[task.task_id] = task
    if graph.output_task_id not in nodes:
        raise ValueError("output task is missing")
    indegree = {key: len(task.depends_on) for key, task in nodes.items()}
    children: dict[str, list[str]] = {key: [] for key in nodes}
    for task in graph.tasks:
        for dep in task.depends_on:
            if dep not in nodes:
                raise ValueError("dependency task is missing")
            children[dep].append(task.task_id)
    ready = [key for key, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        key = ready.pop()
        visited += 1
        for child in children[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError("dependency plan contains a cycle")
    if {key for key, values in children.items() if not values} != {
        graph.output_task_id
    }:
        raise ValueError("every task must contribute to the output task")


async def _execute_dependency_plan(
    agent: Agent[GraphDeps, Any],
    plan: RunPlan,
    state: GraphState,
    config: Mapping[str, Any],
    attachments: list[str | BinaryContent],
    budget: SharedBudget,
    emit: Callable[..., dict[str, Any]],
) -> None:
    """Refill a free slot after any successful prerequisite, not a whole wave.

    Only active tasks have asyncio Tasks. Pending work is bounded plan data.
    Every exit cancels and awaits remaining children before returning control.
    """
    graph = plan.dependency_plan
    assert graph is not None
    nodes = {task.task_id: task for task in graph.tasks}
    order = {task.task_id: index for index, task in enumerate(graph.tasks)}
    indegree = {key: len(task.depends_on) for key, task in nodes.items()}
    children: dict[str, list[str]] = {key: [] for key in nodes}
    for node in graph.tasks:
        for dep in node.depends_on:
            children[dep].append(node.task_id)
    ready = [order[key] for key, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    active: dict[str, asyncio.Task[Any]] = {}
    pending = set(nodes)
    outputs: dict[str, str] = {}
    interrupted: BaseException | None = None

    def event(kind: str, key: str, **fields: Any) -> dict[str, Any]:
        return emit(
            state,
            kind,
            task_id=key,
            parent_run_id=state.run_id,
            plan_id=graph.plan_id,
            depends_on=list(nodes[key].depends_on),
            **fields,
        )

    async def execute(task: RunTask) -> Any:
        dependency_outputs = {dep: outputs[dep] for dep in task.depends_on}
        prompt = task.prompt
        if dependency_outputs:
            prompt += "\n\nDependency outputs (data):\n" + json.dumps(
                dependency_outputs, ensure_ascii=False
            )
        child_state = GraphState(
            run_id=state.run_id,
            query=task.prompt,
            mode=state.mode,
            status="running",
            step=1,
            scaffold=state.scaffold,
        )
        deps = GraphDeps(
            state=child_state,
            plan=plan,
            config=config,
            task_id=task.task_id,
            dependency_outputs=dependency_outputs,
        )
        # Attachments stay with the first declared task, as on the serial path;
        # they are not silently broadcast to all independent tasks.
        model_input: str | list[str | BinaryContent] = prompt
        if order[task.task_id] == 0 and attachments:
            model_input = [prompt, *attachments]
        return await agent.run(
            model_input,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=None, tool_calls_limit=budget.limits.tool_calls_limit
            ),
        )

    def completed(key: str, result: Any) -> dict[str, Any]:
        output = result.output
        text = output if isinstance(output, str) else json.dumps(output, sort_keys=True)
        outputs[key] = text
        return event("task_completed", key, status="completed", output=text)

    try:
        while ready or active:
            while ready and len(active) < graph.max_concurrency:
                node = graph.tasks[heapq.heappop(ready)]
                pending.remove(node.task_id)
                state.step += 1
                active[node.task_id] = asyncio.create_task(
                    execute(node), name=f"run:{state.run_id}:task:{node.task_id}"
                )
                event("task_started", node.task_id, status="running")
            done, _ = await asyncio.wait(
                active.values(), return_when=asyncio.FIRST_COMPLETED
            )
            failure: BaseException | None = None
            for key in sorted(
                (key for key, task in active.items() if task in done),
                key=order.__getitem__,
            ):
                task = active.pop(key)
                try:
                    result = task.result()
                except asyncio.CancelledError as exc:
                    failure = failure or exc
                    event("task_cancelled", key, status="cancelled")
                except Exception as exc:  # noqa: BLE001 - parent owns failure policy
                    failure = failure or exc
                    event(
                        "task_failed",
                        key,
                        status="failed",
                        error_type=type(exc).__name__,
                    )
                else:
                    completed(key, result)
                    if any(m in outputs[key] for m in plan.stop.stop_markers):
                        state.status = "stopped"
                    for child in children[key]:
                        indegree[child] -= 1
                        if indegree[child] == 0:
                            heapq.heappush(ready, order[child])
            if failure is not None:
                raise failure
            if state.status == "stopped":
                break
        if graph.output_task_id in outputs:
            state.outputs.append(outputs[graph.output_task_id])
    except BaseException as exc:
        interrupted = exc
    finally:
        # No terminal parent event or released host scope can precede cleanup.
        for task in active.values():
            if not task.done():
                task.cancel()
        if active:
            drain = asyncio.gather(*active.values(), return_exceptions=True)
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError as exc:
                    # Repeated parent cancellation must not detach children
                    # while their own finally blocks release host resources.
                    interrupted = interrupted or exc
        for key, task in active.items():
            if task.cancelled():
                event("task_cancelled", key, status="cancelled")
            elif task.exception() is not None:
                event(
                    "task_failed",
                    key,
                    status="failed",
                    error_type=type(task.exception()).__name__,
                )
            else:
                completed(key, task.result())
        for key in sorted(pending, key=order.__getitem__):
            event("task_blocked", key, status="blocked")
        state.add_usage(budget.usage)
    if interrupted is not None:
        raise interrupted


async def iterate_dependency_plan(
    agent: Agent[GraphDeps, Any],
    plan: RunPlan,
    state: GraphState,
    config: Mapping[str, Any],
    attachments: list[str | BinaryContent],
    budget: SharedBudget,
    emit: Callable[..., dict[str, Any]],
    *,
    deadline: float | None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream public events without making child dispatch wait for a viewer.

    Each declared task emits at most start + terminal (or only blocked), so a
    finite queue sized to that invariant cannot block execution. There is one
    driver task, never a task per pending node or per event.
    """
    graph = plan.dependency_plan
    assert graph is not None
    events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
        maxsize=2 * len(graph.tasks) + 1
    )

    def enqueue(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = emit(*args, **kwargs)
        events.put_nowait(value)
        return value

    async def drive() -> None:
        try:
            # Own the deadline where work runs, not in the external reader's
            # task: a reader may pause indefinitely between iterator reads.
            async with asyncio.timeout_at(deadline):
                await _execute_dependency_plan(
                    agent, plan, state, config, attachments, budget, enqueue
                )
        finally:
            events.put_nowait(None)

    driver = asyncio.create_task(drive(), name=f"run:{state.run_id}:dependency-driver")
    interrupted: BaseException | None = None
    try:
        while (event := await events.get()) is not None:
            yield event
        await driver
    except BaseException as exc:
        interrupted = exc
    finally:
        if not driver.done():
            driver.cancel()
        drain = asyncio.gather(driver, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError as exc:
                interrupted = interrupted or exc
    if interrupted is not None:
        raise interrupted
