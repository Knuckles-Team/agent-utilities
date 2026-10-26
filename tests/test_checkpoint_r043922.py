"""R043.9.2.2 acceptance checks for host checkpoint restore and continuation."""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from agent_utilities import host_ports
from agent_utilities.graph import (
    DependencyPlan,
    RunBudget,
    RunPlan,
    RunRequest,
    RunTask,
    run,
)

pytestmark = pytest.mark.spec("AU-BOUNDARY-R043.9.2.2")


class MemoryCheckpoint:
    """Fake host transaction; durability and admission fencing stay external."""

    admission_identity = "admission/checkpoint-run"

    def __init__(self) -> None:
        self.content: str | None = None
        self.saves: list[dict[str, Any]] = []
        self.timeline: list[tuple[str, str, str | None]] = []

    async def restore(self) -> str | None:
        return self.content

    async def save(self, content: str) -> None:
        payload = json.loads(content)
        self.content = content
        self.saves.append(payload)
        self.timeline.append(
            (
                "save",
                payload["phase"],
                next(iter(payload["inflight_tasks"]), None),
            )
        )


class RecordingModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

        def respond(messages, info):
            prompt = str(messages[-1].parts[-1].content)
            self.prompts.append(prompt)
            return ModelResponse(
                parts=[TextPart(f"reply:{prompt.splitlines()[0]}")],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
            )

        self.model = FunctionModel(respond)


def _bind(port: MemoryCheckpoint) -> None:
    @contextlib.asynccontextmanager
    async def factory(request, plan):
        yield port

    host_ports.register_run_checkpoint_factory(factory)


def _plan(model: Any, *, request_limit: int = 8, timeout_seconds: float = 60.0):
    return RunPlan(
        model=model,
        dependency_plan=DependencyPlan(
            tasks=(
                RunTask("A", "alpha"),
                RunTask("B", "bravo", ("A",)),
            ),
            output_task_id="B",
            max_concurrency=1,
            plan_id="checkpoint-dag-v1",
        ),
        budget=RunBudget(
            request_limit=request_limit,
            timeout_seconds=timeout_seconds,
        ),
    )


async def _seed_partial(
    port: MemoryCheckpoint,
    recorder: RecordingModel,
    *,
    request_limit: int = 8,
    timeout_seconds: float = 60.0,
    usage: dict[str, int] | None = None,
    elapsed_seconds: float = 0.0,
) -> tuple[RunRequest, RunPlan]:
    """Create a real completed checkpoint, then seed a valid partial DAG state."""
    _bind(port)
    request = RunRequest(task="dag", run_id="checkpoint-run")
    plan = _plan(
        recorder.model,
        request_limit=request_limit,
        timeout_seconds=timeout_seconds,
    )
    first = await run(request, plan=plan)
    assert first.status == "completed"
    assert port.content is not None

    payload = json.loads(port.content)
    payload.update(
        {
            "phase": "completed",
            "inflight_tasks": [],
            "next_step": 1,
            "messages": [],
            "completed_tasks": ["A"],
            "dependency_outputs": {"A": "seeded-A"},
            "outputs": [],
            "status": "running",
            "usage": usage
            or {
                "requests": 1,
                "tool_calls": 0,
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
            "elapsed_seconds": elapsed_seconds,
        }
    )
    payload["task_messages"] = {
        key: value
        for key, value in payload.get("task_messages", {}).items()
        if key == "A"
    }
    port.content = json.dumps(payload)
    port.saves.clear()
    port.timeline.clear()
    recorder.prompts.clear()
    return request, plan


async def _seed_full(
    port: MemoryCheckpoint, recorder: RecordingModel
) -> tuple[RunRequest, RunPlan, dict[str, Any]]:
    _bind(port)
    request = RunRequest(task="dag", run_id="checkpoint-run")
    plan = _plan(recorder.model)
    result = await run(request, plan=plan)
    assert result.status == "completed"
    assert port.content is not None
    return request, plan, json.loads(port.content)


async def test_dependency_restore_skips_completed_tasks_and_keeps_outputs():
    port, recorder = MemoryCheckpoint(), RecordingModel()
    request, plan = await _seed_partial(port, recorder)

    resumed = await run(request, plan=plan)

    assert resumed.status == "completed"
    assert resumed.output == "reply:bravo"
    assert [prompt.splitlines()[0] for prompt in recorder.prompts] == ["bravo"]
    assert "seeded-A" in recorder.prompts[0]
    saved = json.loads(port.content or "null")
    assert set(saved["completed_tasks"]) == {"A", "B"}
    assert saved["dependency_outputs"]["A"] == "seeded-A"

    recorder.prompts.clear()
    repeated = await run(request, plan=plan)
    assert repeated.status == "completed"
    assert repeated.output == "reply:bravo"
    assert recorder.prompts == []


async def test_dependency_checkpoint_ack_precedes_effect_and_output_events():
    port, recorder = MemoryCheckpoint(), RecordingModel()
    _bind(port)
    events: list[dict[str, Any]] = []

    def capture(event: dict[str, Any]) -> None:
        events.append(dict(event))
        port.timeline.append(("event", event["type"], event.get("task_id")))

    host_ports.register_run_event_sink(capture)
    result = await run(
        RunRequest(task="dag", run_id="checkpoint-run"),
        plan=_plan(recorder.model),
    )

    assert result.status == "completed"
    for task_id in ("A", "B"):
        started = next(
            index
            for index, item in enumerate(port.timeline)
            if item == ("event", "task_started", task_id)
        )
        completed = next(
            index
            for index, item in enumerate(port.timeline)
            if item == ("event", "task_completed", task_id)
        )
        before_effect = [
            index
            for index, item in enumerate(port.timeline[:started])
            if item[0] == "save" and item[1] == "step_inflight"
        ]
        before_output = [
            index
            for index, item in enumerate(port.timeline[:completed])
            if item[0] == "save" and item[1] == "completed"
        ]
        assert before_effect and before_effect[-1] < started
        assert before_output and before_output[-1] < completed


@pytest.mark.parametrize("field", ["run_id", "admission_identity"])
async def test_foreign_checkpoint_binding_refuses_before_any_model_effect(field):
    port, recorder = MemoryCheckpoint(), RecordingModel()
    request, plan, payload = await _seed_full(port, recorder)
    payload[field] = "foreign-context"
    port.content = json.dumps(payload)
    port.saves.clear()
    recorder.prompts.clear()

    result = await run(request, plan=plan)

    assert result.status == "failed"
    assert recorder.prompts == []
    assert port.saves == []


async def test_changed_received_plan_refuses_before_any_model_effect():
    port, recorder = MemoryCheckpoint(), RecordingModel()
    request, plan, _ = await _seed_full(port, recorder)
    assert plan.dependency_plan is not None
    changed = replace(
        plan,
        dependency_plan=replace(plan.dependency_plan, plan_id="checkpoint-dag-v2"),
    )
    port.saves.clear()
    recorder.prompts.clear()

    result = await run(request, plan=changed)

    assert result.status == "failed"
    assert recorder.prompts == []
    assert port.saves == []


async def test_inflight_dependency_checkpoint_refuses_without_replay():
    port, recorder = MemoryCheckpoint(), RecordingModel()
    request, plan, payload = await _seed_full(port, recorder)
    payload.update(
        {
            "phase": "step_inflight",
            "inflight_tasks": ["B"],
            "next_step": 1,
            "messages": [],
            "completed_tasks": ["A"],
            "dependency_outputs": {"A": "seeded-A"},
            "outputs": [],
            "status": "running",
        }
    )
    payload["task_messages"] = {}
    port.content = json.dumps(payload)
    port.saves.clear()
    recorder.prompts.clear()

    result = await run(request, plan=plan)

    assert result.status == "failed"
    assert "uncertain" in (result.error or "")
    assert recorder.prompts == []
    assert port.saves == []


async def test_restored_request_budget_is_not_reset():
    port, recorder = MemoryCheckpoint(), RecordingModel()
    request, plan = await _seed_partial(
        port,
        recorder,
        request_limit=2,
        usage={
            "requests": 2,
            "tool_calls": 0,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )

    result = await run(request, plan=plan)

    assert result.status == "stopped"
    assert result.usage["requests"] == 2
    assert recorder.prompts == []


async def test_restored_elapsed_budget_is_not_reset():
    port, recorder = MemoryCheckpoint(), RecordingModel()
    request, plan = await _seed_partial(
        port,
        recorder,
        timeout_seconds=60.0,
        elapsed_seconds=60.0,
    )

    result = await run(request, plan=plan)

    assert result.status == "stopped"
    assert "timed out" in (result.error or "")
    assert recorder.prompts == []
