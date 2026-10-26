"""Apply a received dependency DAG with at most its declared active task count."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import math
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from re import fullmatch
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from agent_utilities.host_ports import DurableRunEventPublisher

from .budget import SharedBudget
from .checkpoint import RunCheckpoint
from .live_events import run_step
from .state import (
    BindingUnresolved,
    DependencyPlan,
    GraphDeps,
    GraphState,
    RunPlan,
    RunTask,
    ToolCaller,
    ToolCallerUnavailable,
    ToolCallFailed,
    ToolCallRefused,
    ToolStep,
    ToolStepResult,
)

TOOL_ARGUMENT_LIMIT = 64 * 1024
TOOL_RESULT_LIMIT = 256 * 1024


@dataclass(frozen=True, slots=True)
class _ToolExecution:
    output: Any
    record: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    """Encode a structured value deterministically for size and digest checks."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolCallFailed("tool payload is not JSON-serializable") from exc


def _bounded(value: Any, limit: int) -> Any:
    """Keep a small value intact, or expose only its digest and byte count."""
    if value is None:
        return None
    encoded = _json_bytes(value)
    if len(encoded) <= limit:
        return value
    return {
        "digest": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def _path_tokens(path: str) -> tuple[str, ...]:
    """Parse the dotted JSON path form, accepting ``items[0]`` as a convenience."""
    if not path:
        return ()
    tokens: list[str] = []
    for component in path.split("."):
        if not component:
            raise ValueError("JSON path contains an empty component")
        remainder = component
        while "[" in remainder:
            prefix, suffix = remainder.split("[", 1)
            if prefix:
                tokens.append(prefix)
            if "]" not in suffix:
                raise ValueError("JSON path has an unterminated array index")
            index, remainder = suffix.split("]", 1)
            if not fullmatch(r"[0-9]+", index):
                raise ValueError("JSON path array indexes must be nonnegative integers")
            tokens.append(index)
        if remainder:
            tokens.append(remainder)
    return tuple(tokens)


def _reference_parts(
    reference: str, direct_prerequisites: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Split one ``steps.<id>.output[.<path>]`` reference."""
    if not isinstance(reference, str):
        raise ValueError("binding reference must be a string")
    for task_id in sorted(direct_prerequisites, key=len, reverse=True):
        prefix = f"steps.{task_id}.output"
        if reference == prefix:
            return task_id, ()
        if reference.startswith(f"{prefix}."):
            path = reference[len(prefix) + 1 :]
            if not path:
                raise ValueError("JSON path is empty")
            return task_id, _path_tokens(path)
    raise ValueError("binding must reference a direct prerequisite output")


def _validate_argument_value(
    value: Any,
    *,
    task_id: str,
    argument: str,
    direct_prerequisites: tuple[str, ...],
) -> None:
    if isinstance(value, Mapping):
        if "$ref" in value:
            if set(value) != {"$ref"}:
                raise ValueError("a binding object may contain only $ref")
            try:
                _reference_parts(value["$ref"], direct_prerequisites)
            except ValueError as exc:
                raise ValueError(
                    f"invalid binding at {argument or '/'} for task {task_id!r}: {exc}"
                ) from exc
            return
        if any(not isinstance(key, str) for key in value):
            raise ValueError("tool argument object keys must be strings")
        for key, child in value.items():
            _validate_argument_value(
                child,
                task_id=task_id,
                argument=f"{argument}/{key}",
                direct_prerequisites=direct_prerequisites,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_argument_value(
                child,
                task_id=task_id,
                argument=f"{argument}/{index}",
                direct_prerequisites=direct_prerequisites,
            )
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("tool arguments must contain finite JSON numbers")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("tool arguments must contain JSON-shaped literals or bindings")


def _lookup(value: Any, token: str) -> Any:
    if isinstance(value, Mapping):
        if token not in value:
            raise KeyError(token)
        return value[token]
    if isinstance(value, (list, tuple)):
        if not fullmatch(r"[0-9]+", token):
            raise TypeError(token)
        index = int(token)
        if index >= len(value):
            raise IndexError(index)
        return value[index]
    raise TypeError(token)


def _resolve_reference(
    reference: str,
    *,
    task_id: str,
    argument: str,
    outputs: Mapping[str, Any],
    direct_prerequisites: tuple[str, ...],
) -> Any:
    try:
        dependency, path = _reference_parts(reference, direct_prerequisites)
    except ValueError as exc:
        raise BindingUnresolved(task_id, argument, reference, str(exc)) from None
    try:
        value = outputs[dependency]
    except KeyError:
        raise BindingUnresolved(
            task_id, argument, reference, "prerequisite output is not completed"
        ) from None
    if path:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                raise BindingUnresolved(
                    task_id,
                    argument,
                    reference,
                    "prompt output is not JSON",
                ) from None
        for token in path:
            try:
                value = _lookup(value, token)
            except KeyError:
                raise BindingUnresolved(
                    task_id, argument, reference, f"missing path component {token!r}"
                ) from None
            except (IndexError, TypeError):
                raise BindingUnresolved(
                    task_id,
                    argument,
                    reference,
                    f"cannot traverse path component {token!r}",
                ) from None
    return value


def _resolve_value(
    value: Any,
    *,
    task_id: str,
    argument: str,
    outputs: Mapping[str, Any],
    direct_prerequisites: tuple[str, ...],
) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$ref"}:
            return _resolve_reference(
                value["$ref"],
                task_id=task_id,
                argument=argument or "/",
                outputs=outputs,
                direct_prerequisites=direct_prerequisites,
            )
        if "$ref" in value:
            raise BindingUnresolved(
                task_id,
                argument or "/",
                str(value.get("$ref", "")),
                "a binding object may contain only $ref",
            )
        return {
            key: _resolve_value(
                child,
                task_id=task_id,
                argument=f"{argument}/{key}",
                outputs=outputs,
                direct_prerequisites=direct_prerequisites,
            )
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _resolve_value(
                child,
                task_id=task_id,
                argument=f"{argument}/{index}",
                outputs=outputs,
                direct_prerequisites=direct_prerequisites,
            )
            for index, child in enumerate(value)
        ]
    return value


def resolve_tool_bindings(task: RunTask, outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a tool task's literals and completed-output references."""
    if task.call is None:
        raise ValueError("task has no tool call")
    resolved = _resolve_value(
        task.call.arguments,
        task_id=task.task_id,
        argument="",
        outputs=outputs,
        direct_prerequisites=task.depends_on,
    )
    if not isinstance(resolved, Mapping):
        raise BindingUnresolved(
            task.task_id,
            "/",
            "",
            "tool arguments must resolve to an object",
        )
    return dict(resolved)


def _record(
    task: RunTask,
    *,
    arguments: Any,
    outcome: str,
    result: Any = None,
    elapsed_seconds: float = 0.0,
    error: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "server": task.call.server if task.call is not None else "",
        "tool": task.call.tool if task.call is not None else "",
        "arguments": _bounded(arguments, TOOL_ARGUMENT_LIMIT),
        "outcome": outcome,
        "result": _bounded(result, TOOL_RESULT_LIMIT),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
    }
    if error is not None:
        record["error"] = error
    if reference is not None:
        record["reference"] = reference
    return record


def _attach_record(error: BaseException, record: dict[str, Any]) -> BaseException:
    error.tool_step = record  # type: ignore[attr-defined]
    return error


def _tool_result(value: Any) -> ToolStepResult:
    return value if isinstance(value, ToolStepResult) else ToolStepResult(result=value)


def _output_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def validate_dependency_plan(
    plan: RunPlan, *, tool_caller: ToolCaller | None = None
) -> None:
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
    tool_tasks: list[RunTask] = []
    for task in graph.tasks:
        if (
            not isinstance(task, RunTask)
            or not isinstance(task.task_id, str)
            or not task.task_id.strip()
            or task.task_id in nodes
            or not isinstance(task.prompt, str)
            or not isinstance(task.depends_on, tuple)
            or any(not isinstance(dep, str) for dep in task.depends_on)
            or len(set(task.depends_on)) != len(task.depends_on)
            or task.call is not None
            and not isinstance(task.call, ToolStep)
        ):
            raise ValueError("invalid or duplicate dependency task")
        if task.call is None and not task.prompt.strip():
            raise ValueError("prompt task must have a nonempty prompt")
        nodes[task.task_id] = task
        if task.call is not None:
            tool_tasks.append(task)
    if graph.output_task_id not in nodes:
        raise ValueError("output task is missing")
    indegree = {key: len(task.depends_on) for key, task in nodes.items()}
    children: dict[str, list[str]] = {key: [] for key in nodes}
    for task in graph.tasks:
        for dep in task.depends_on:
            if dep not in nodes:
                raise ValueError("dependency task is missing")
            children[dep].append(task.task_id)
    for task in tool_tasks:
        assert task.call is not None
        if (
            not isinstance(task.call.server, str)
            or not task.call.server.strip()
            or not isinstance(task.call.tool, str)
            or not task.call.tool.strip()
            or not isinstance(task.call.arguments, Mapping)
        ):
            raise ValueError("invalid tool step")
        _validate_argument_value(
            task.call.arguments,
            task_id=task.task_id,
            argument="",
            direct_prerequisites=task.depends_on,
        )
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
    if tool_tasks and tool_caller is None:
        raise ToolCallerUnavailable(
            "dependency plan contains tool steps but no host ToolCaller is bound"
        )


async def _execute_dependency_plan(
    agent: Agent[GraphDeps, Any],
    plan: RunPlan,
    state: GraphState,
    config: Mapping[str, Any],
    attachments: list[str | BinaryContent],
    budget: SharedBudget,
    emit: Callable[..., dict[str, Any]],
    tool_caller: ToolCaller | None = None,
    checkpoint: RunCheckpoint | None = None,
    event_publisher: DurableRunEventPublisher | None = None,
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
    active: dict[str, asyncio.Task[Any]] = {}
    completed_tasks = (
        set(checkpoint.completed_tasks)
        if checkpoint is not None and checkpoint.restored
        else set()
    )
    outputs: dict[str, Any] = (
        dict(checkpoint.dependency_outputs)
        if checkpoint is not None and checkpoint.restored
        else {}
    )
    for key in completed_tasks:
        for child in children[key]:
            indegree[child] -= 1
    ready = [
        order[key]
        for key, count in indegree.items()
        if count == 0 and key not in completed_tasks
    ]
    heapq.heapify(ready)
    pending = set(nodes) - completed_tasks
    interrupted: BaseException | None = None
    tool_order = [task.task_id for task in graph.tasks if task.call is not None]
    tool_terminal_buffer: dict[str, tuple[str, dict[str, Any]]] = {}
    # A tool effect may finish out of declaration order when the graph has
    # more than one active slot.  Keep its bounded result here until the
    # declaration terminal cursor reaches it.  This preserves parallel host
    # effects while making checkpoint ACKs, durable tool_result records, and
    # terminal run events one ordered publication sequence.
    tool_completion_buffer: dict[str, _ToolExecution] = {}
    tool_record_buffer: dict[str, tuple[RunTask, dict[str, Any], int]] = {}
    tool_terminal_cursor = 0
    step_by_task: dict[str, int] = {}
    checkpoint_acked: set[str] = set()
    publication_failed = False
    while (
        tool_terminal_cursor < len(tool_order)
        and tool_order[tool_terminal_cursor] in completed_tasks
    ):
        tool_terminal_cursor += 1

    def sync_usage() -> None:
        state.usage.clear()
        state.add_usage(budget.usage)

    async def publish_durable(kind: str, content: Mapping[str, Any]) -> None:
        nonlocal publication_failed
        if event_publisher is None:
            return
        try:
            await event_publisher.publish(kind, content)
        except Exception as exc:  # noqa: BLE001 - sanitize host details
            publication_failed = True
            raise RuntimeError("durable run event publication failed") from exc

    async def publish_tool_record(
        task: RunTask, record: Mapping[str, Any], *, step: int
    ) -> None:
        """Publish the same bounded record that the terminal run event carries."""
        content: dict[str, Any] = {
            "step": step,
            "task_id": task.task_id,
            "tool_call_id": task.task_id,
            "name": record["tool"],
            "server": record["server"],
            "arguments": record["arguments"],
            "result": record["result"],
            "outcome": record["outcome"],
            "elapsed_seconds": record["elapsed_seconds"],
        }
        for key in ("error", "reference"):
            if key in record:
                content[key] = record[key]
        await publish_durable("tool_result", content)

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

    def terminal_event(kind: str, key: str, **fields: Any) -> dict[str, Any] | None:
        """Emit prompt terminals immediately and tool records in declaration order."""
        if nodes[key].call is None:
            return event(kind, key, **fields)
        tool_terminal_buffer[key] = (kind, fields)
        return None

    def tool_event_fields(record: dict[str, Any] | None) -> dict[str, Any]:
        """Expose the bounded record both nested and as stable event fields."""
        if record is None:
            return {}
        fields = {
            "tool_step": record,
            "server": record["server"],
            "tool": record["tool"],
            "arguments": record["arguments"],
            "outcome": record["outcome"],
            "result": record["result"],
            "elapsed_seconds": record["elapsed_seconds"],
        }
        for key in ("error", "reference"):
            if key in record:
                fields[key] = record[key]
        return fields

    async def flush_tool_terminals(*, publish_records: bool = True) -> tuple[str, ...]:
        """Publish buffered tool terminals in declaration order.

        Effects run concurrently, but a completed result is only exposed once
        every earlier declared tool has reached its own terminal publication.
        The checkpoint ACK intentionally precedes the durable result and the
        public terminal event for each step.
        """
        nonlocal tool_terminal_cursor
        released: list[str] = []
        while tool_terminal_cursor < len(tool_order):
            key = tool_order[tool_terminal_cursor]
            result = tool_completion_buffer.get(key)
            buffered = tool_terminal_buffer.get(key)
            record_info = tool_record_buffer.get(key)
            if result is None and buffered is None:
                break

            if result is not None:
                tool_completion_buffer.pop(key, None)
                record = result.record
                if publish_records:
                    try:
                        if checkpoint is not None and key not in checkpoint_acked:
                            sync_usage()
                            checkpoint.record_completed(key, result.output)
                            await checkpoint.save("completed")
                            checkpoint_acked.add(key)
                        output_text = _output_text(result.output)
                        if key == graph.output_task_id:
                            state.outputs = [output_text]
                            if any(
                                marker in output_text
                                for marker in plan.stop.stop_markers
                            ):
                                state.status = "stopped"
                        await publish_tool_record(
                            nodes[key], record, step=step_by_task[key]
                        )
                    except BaseException:
                        # Do not expose a terminal event after a durable
                        # publication failure; the failed record remains
                        # uncertain and must be retried from its checkpoint.
                        tool_terminal_buffer.pop(key, None)
                        raise
                if buffered is None:
                    buffered = (
                        "task_completed",
                        {
                            "status": "completed",
                            "output": record["result"],
                            "tool_step": record,
                            "server": record["server"],
                            "tool": record["tool"],
                            "arguments": record["arguments"],
                            "outcome": record["outcome"],
                            "result": record["result"],
                            "elapsed_seconds": record["elapsed_seconds"],
                        },
                    )
                tool_terminal_buffer.pop(key, None)
                if publish_records:
                    released.append(key)
            else:
                assert buffered is not None
                tool_terminal_buffer.pop(key, None)
                if record_info is not None:
                    tool_record_buffer.pop(key, None)
                    if publish_records:
                        record_task, record, record_step = record_info
                        await publish_tool_record(record_task, record, step=record_step)

            buffered_kind, buffered_fields = buffered
            event(buffered_kind, key, **buffered_fields)
            tool_terminal_cursor += 1
        return tuple(released)

    async def execute(task: RunTask) -> Any:
        dependency_outputs = {dep: outputs[dep] for dep in task.depends_on}
        task_step = step_by_task[task.task_id]
        if task.call is not None:
            assert tool_caller is not None
            started = asyncio.get_running_loop().time()
            try:
                arguments = resolve_tool_bindings(task, outputs)
            except BindingUnresolved as exc:
                record = _record(
                    task,
                    arguments=None,
                    outcome="failed",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error=str(exc),
                    reference=exc.reference,
                )
                tool_record_buffer[task.task_id] = (task, record, task_step)
                raise _attach_record(exc, record) from None
            try:
                budget.reserve_tool_call()
                await publish_durable(
                    "tool_call",
                    {
                        "step": task_step,
                        "task_id": task.task_id,
                        "tool_call_id": task.task_id,
                        "name": task.call.tool,
                        "server": task.call.server,
                        "arguments": _bounded(arguments, TOOL_ARGUMENT_LIMIT),
                        "args_valid": True,
                    },
                )
                raw_result = await tool_caller(
                    task.call.server,
                    task.call.tool,
                    arguments,
                    task.task_id,
                )
                result = _tool_result(raw_result)
                if result.outcome == "refused":
                    raise ToolCallRefused(result.error or "tool call refused")
                if result.outcome == "failed":
                    raise ToolCallFailed(result.error or "tool call failed")
            except asyncio.CancelledError as exc:
                record = _record(
                    task,
                    arguments=locals().get("arguments"),
                    outcome="failed",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error="tool call cancelled",
                )
                raise _attach_record(exc, record) from None
            except UsageLimitExceeded as exc:
                record = _record(
                    task,
                    arguments=locals().get("arguments"),
                    outcome="refused",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error=str(exc),
                )
                tool_record_buffer[task.task_id] = (task, record, task_step)
                raise _attach_record(exc, record) from None
            except ToolCallRefused as exc:
                record = _record(
                    task,
                    arguments=locals().get("arguments"),
                    outcome="refused",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error=str(exc),
                )
                tool_record_buffer[task.task_id] = (task, record, task_step)
                raise _attach_record(exc, record) from None
            except ToolCallFailed as exc:
                record = _record(
                    task,
                    arguments=locals().get("arguments"),
                    outcome="failed",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error=str(exc),
                )
                tool_record_buffer[task.task_id] = (task, record, task_step)
                raise _attach_record(exc, record) from None
            except RuntimeError as exc:
                if str(exc) == "durable run event publication failed":
                    raise
                failure = ToolCallFailed("tool caller failed")
                record = _record(
                    task,
                    arguments=locals().get("arguments"),
                    outcome="failed",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error=str(failure),
                )
                tool_record_buffer[task.task_id] = (task, record, task_step)
                raise _attach_record(failure, record) from exc
            except Exception as exc:  # noqa: BLE001 - host port owns failure details
                failure = ToolCallFailed("tool caller failed")
                record = _record(
                    task,
                    arguments=locals().get("arguments"),
                    outcome="failed",
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    error=str(failure),
                )
                tool_record_buffer[task.task_id] = (task, record, task_step)
                raise _attach_record(failure, record) from exc
            record = _record(
                task,
                arguments=arguments,
                outcome="ok",
                result=result.result,
                elapsed_seconds=asyncio.get_running_loop().time() - started,
            )
            return _ToolExecution(result.result, record)
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

        async def after_prompt(result: Any) -> None:
            if checkpoint is None:
                return
            sync_usage()
            text = _output_text(result.output)
            outputs[task.task_id] = text
            if task.task_id == graph.output_task_id:
                state.outputs = [text]
            if task.task_id == graph.output_task_id and any(
                marker in text for marker in plan.stop.stop_markers
            ):
                state.status = "stopped"
            checkpoint.record_completed(task.task_id, text, list(result.all_messages()))
            await checkpoint.save("completed")
            checkpoint_acked.add(task.task_id)

        return await run_step(
            agent,
            model_input,
            publisher=event_publisher,
            step=task_step,
            event_fields={"task_id": task.task_id},
            after_result=after_prompt,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=None, tool_calls_limit=budget.limits.tool_calls_limit
            ),
        )

    async def completed(key: str, result: Any) -> tuple[str, ...]:
        if isinstance(result, _ToolExecution):
            outputs[key] = result.output
            record = result.record
            output_text = _output_text(result.output)
            if key == graph.output_task_id:
                state.outputs = [output_text]
            if (
                event_publisher is None
                and key == graph.output_task_id
                and any(marker in output_text for marker in plan.stop.stop_markers)
            ):
                state.status = "stopped"
            if event_publisher is not None:
                tool_completion_buffer[key] = result
                terminal_event(
                    "task_completed",
                    key,
                    status="completed",
                    output=record["result"],
                    tool_step=record,
                    server=record["server"],
                    tool=record["tool"],
                    arguments=record["arguments"],
                    outcome=record["outcome"],
                    result=record["result"],
                    elapsed_seconds=record["elapsed_seconds"],
                )
                return await flush_tool_terminals()
            if checkpoint is not None and key not in checkpoint_acked:
                sync_usage()
                checkpoint.record_completed(key, result.output)
                await checkpoint.save("completed")
                checkpoint_acked.add(key)
            await publish_tool_record(nodes[key], record, step=step_by_task[key])
            terminal_event(
                "task_completed",
                key,
                status="completed",
                output=record["result"],
                tool_step=record,
                server=record["server"],
                tool=record["tool"],
                arguments=record["arguments"],
                outcome=record["outcome"],
                result=record["result"],
                elapsed_seconds=record["elapsed_seconds"],
            )
            return await flush_tool_terminals()
        output = result.output
        text = output if isinstance(output, str) else json.dumps(output, sort_keys=True)
        outputs[key] = text
        if key == graph.output_task_id:
            state.outputs = [text]
        if key == graph.output_task_id and any(
            marker in text for marker in plan.stop.stop_markers
        ):
            state.status = "stopped"
        if checkpoint is not None and key not in checkpoint_acked:
            sync_usage()
            checkpoint.record_completed(key, text, list(result.all_messages()))
            await checkpoint.save("completed")
            checkpoint_acked.add(key)
        event("task_completed", key, status="completed", output=text)
        return (key,)

    try:
        while ready or active:
            while ready and len(active) < graph.max_concurrency:
                node = graph.tasks[heapq.heappop(ready)]
                pending.remove(node.task_id)
                state.step += 1
                step_by_task[node.task_id] = state.step
                if checkpoint is not None:
                    sync_usage()
                    await checkpoint.save("step_inflight", task_id=node.task_id)
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
                    record = getattr(exc, "tool_step", None)
                    terminal_event(
                        "task_cancelled",
                        key,
                        status="cancelled",
                        **tool_event_fields(record),
                    )
                except Exception as exc:  # noqa: BLE001 - parent owns failure policy
                    failure = failure or exc
                    if str(exc) == "durable run event publication failed":
                        publication_failed = True
                    record = getattr(exc, "tool_step", None)
                    terminal_event(
                        "task_failed",
                        key,
                        status="failed",
                        error_type=type(exc).__name__,
                        **tool_event_fields(record),
                    )
                else:
                    released = await completed(key, result)
                    for released_key in released:
                        if any(
                            m in _output_text(outputs[released_key])
                            for m in plan.stop.stop_markers
                        ):
                            state.status = "stopped"
                        for child in children[released_key]:
                            indegree[child] -= 1
                            if indegree[child] == 0:
                                heapq.heappush(ready, order[child])
            if failure is not None:
                raise failure
            if state.status == "stopped":
                break
        if graph.output_task_id in outputs and not state.outputs:
            state.outputs.append(_output_text(outputs[graph.output_task_id]))
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
                terminal_event("task_cancelled", key, status="cancelled")
            elif task.exception() is not None:
                record = getattr(task.exception(), "tool_step", None)
                terminal_event(
                    "task_failed",
                    key,
                    status="failed",
                    error_type=type(task.exception()).__name__,
                    **tool_event_fields(record),
                )
            elif not publication_failed:
                await completed(key, task.result())
        await flush_tool_terminals(
            publish_records=not publication_failed
            and not isinstance(interrupted, asyncio.CancelledError)
        )
        for key in sorted(pending, key=order.__getitem__):
            event("task_blocked", key, status="blocked")
        # This seeded ledger is the whole run, not a delta to add again.
        state.usage.clear()
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
    tool_caller: ToolCaller | None = None,
    checkpoint: RunCheckpoint | None = None,
    event_publisher: DurableRunEventPublisher | None = None,
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
                    agent,
                    plan,
                    state,
                    config,
                    attachments,
                    budget,
                    enqueue,
                    tool_caller,
                    checkpoint,
                    event_publisher,
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
