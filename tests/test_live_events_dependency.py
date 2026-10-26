"""R043.10 dependency publisher coverage on the composed executor path."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports
from agent_utilities.graph import (
    DependencyPlan,
    RunBudget,
    RunPlan,
    RunRequest,
    RunTask,
    ToolStep,
    ToolStepResult,
    run,
)


class Checkpoint:
    admission_identity = "admission/r04310"

    def __init__(self, timeline: list[tuple[str, str, str]]) -> None:
        self.content: str | None = None
        self.timeline = timeline

    async def restore(self) -> str | None:
        return self.content

    async def save(self, content: str) -> None:
        self.content = content
        payload = json.loads(content)
        completed = ",".join(payload["completed_tasks"])
        self.timeline.append(("checkpoint", payload["phase"], completed))


class Publisher:
    def __init__(
        self,
        timeline: list[tuple[str, str, str]],
        *,
        fail_kind: str | None = None,
    ) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.attempts: list[tuple[str, dict[str, Any]]] = []
        self.timeline = timeline
        self.fail_kind = fail_kind

    async def publish(self, kind: str, content: dict[str, Any]) -> None:
        self.attempts.append((kind, content))
        if kind == self.fail_kind:
            raise OSError("private durable backend detail")
        self.events.append((kind, content))
        self.timeline.append(("durable", kind, str(content.get("task_id", ""))))


def _bind(checkpoint: Checkpoint, publisher: Publisher) -> None:
    @asynccontextmanager
    async def checkpoint_scope(request: Any, plan: Any):
        yield checkpoint

    @asynccontextmanager
    async def publisher_scope(request: Any):
        yield publisher

    host_ports.register_run_checkpoint_factory(checkpoint_scope)
    host_ports.register_durable_run_event_factory(publisher_scope)


def _plan(model: Any) -> RunPlan:
    return RunPlan(
        model=model,
        dependency_plan=DependencyPlan(
            tasks=(
                RunTask("prompt", "prepare"),
                RunTask(
                    "tool",
                    depends_on=("prompt",),
                    call=ToolStep(
                        "host", "lookup", {"value": {"$ref": "steps.prompt.output"}}
                    ),
                ),
                RunTask("final", "finish", depends_on=("tool",)),
            ),
            output_task_id="final",
            max_concurrency=1,
            plan_id="r04310-dependency-events",
        ),
        budget=RunBudget(request_limit=8, tool_calls_limit=1),
    )


def _at(
    timeline: list[tuple[str, str, str]],
    source: str,
    kind: str,
    value: str,
) -> int:
    return next(
        index for index, item in enumerate(timeline) if item == (source, kind, value)
    )


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_dependency_events_follow_checkpoint_ack_and_carry_tool_record() -> None:
    timeline: list[tuple[str, str, str]] = []
    checkpoint = Checkpoint(timeline)
    publisher = Publisher(timeline)
    _bind(checkpoint, publisher)
    host_ports.register_run_event_sink(
        lambda event: timeline.append(
            ("run_event", event["type"], str(event.get("task_id", "")))
        )
    )

    async def stream(messages, info):
        prompt = str(messages[-1].parts[-1].content)
        yield "prepared" if prompt.startswith("prepare") else "finished"

    async def caller(server, tool, arguments, task_id):
        assert (server, tool, task_id) == ("host", "lookup", "tool")
        return ToolStepResult(result={"value": arguments["value"]})

    host_ports.register_tool_caller(caller)
    result = await run(
        RunRequest(task="dag", run_id="r04310-dependency-run"),
        plan=_plan(FunctionModel(stream_function=stream)),
    )

    assert result.status == "completed"
    assert result.output == "finished"
    assert [kind for kind, _ in publisher.events] == [
        "text_delta",
        "state",
        "tool_call",
        "tool_result",
        "text_delta",
        "state",
    ]
    tool_result = next(
        content for kind, content in publisher.events if kind == "tool_result"
    )
    assert tool_result["task_id"] == "tool"
    assert tool_result["arguments"] == {"value": "prepared"}
    assert tool_result["result"] == {"value": "prepared"}
    assert tool_result["outcome"] == "ok"

    prompt_state = _at(timeline, "durable", "state", "prompt")
    prompt_ack = next(
        index
        for index, item in enumerate(timeline)
        if item[0:2] == ("checkpoint", "completed") and "prompt" in item[2]
    )
    assert prompt_ack < prompt_state

    tool_call = _at(timeline, "durable", "tool_call", "tool")
    tool_result_index = _at(timeline, "durable", "tool_result", "tool")
    tool_started = _at(timeline, "run_event", "task_started", "tool")
    tool_inflight = next(
        index
        for index, item in enumerate(timeline)
        if item[0:2] == ("checkpoint", "step_inflight") and index < tool_started
    )
    tool_ack = next(
        index
        for index, item in enumerate(timeline)
        if item[0:2] == ("checkpoint", "completed") and "tool" in item[2]
    )
    tool_terminal = _at(timeline, "run_event", "task_completed", "tool")
    assert tool_inflight < tool_started < tool_call < tool_ack < tool_result_index
    assert tool_result_index < tool_terminal


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_dependency_failed_result_publication_effect_runs_once_and_restart_skips() -> (
    None
):
    timeline: list[tuple[str, str, str]] = []
    checkpoint = Checkpoint(timeline)
    publisher = Publisher(timeline, fail_kind="tool_result")
    _bind(checkpoint, publisher)
    effects: list[str] = []

    async def caller(server, tool, arguments, task_id):
        effects.append(task_id)
        return ToolStepResult(result="done")

    host_ports.register_tool_caller(caller)
    plan = RunPlan(
        model=FunctionModel(lambda messages, info: "unused"),
        dependency_plan=DependencyPlan(
            tasks=(RunTask("tool", call=ToolStep("host", "effect")),),
            output_task_id="tool",
            max_concurrency=1,
            plan_id="r04310-failed-result",
        ),
        budget=RunBudget(request_limit=2, tool_calls_limit=1),
    )
    request = RunRequest(task="tool", run_id="r04310-failed-result-run")

    first = await run(request, plan=plan)
    assert first.status == "failed"
    assert first.error == "RuntimeError: durable run event publication failed"
    assert effects == ["tool"]
    assert [kind for kind, _ in publisher.attempts] == ["tool_call", "tool_result"]
    payload = json.loads(checkpoint.content or "null")
    assert payload["status"] == "completed"
    assert payload["completed_tasks"] == ["tool"]

    second = await run(request, plan=plan)
    assert second.status == "completed"
    assert effects == ["tool"]
    assert [kind for kind, _ in publisher.attempts] == ["tool_call", "tool_result"]


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_publisher_scope_closes_when_run_is_cancelled() -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    publisher = Publisher([])

    @asynccontextmanager
    async def scope(request):
        entered.set()
        try:
            yield publisher
        finally:
            closed.set()

    host_ports.register_durable_run_event_factory(scope)

    async def stream(messages, info):
        await asyncio.Event().wait()
        yield "unreachable"

    task = asyncio.create_task(
        run(
            RunRequest(task="wait", run_id="r04310-cancel-run"),
            plan=RunPlan(model=FunctionModel(stream_function=stream), steps=("wait",)),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert closed.is_set()
    assert publisher.events == []
