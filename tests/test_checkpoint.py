"""Actual harness restart behavior over an injected host transaction boundary."""

import asyncio
import contextlib
import json

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
    ToolStep,
    ToolStepResult,
    run,
)
from agent_utilities.graph.executor import iterate_run

pytestmark = pytest.mark.spec("AU-BOUNDARY-R043.8")


class HostCheckpoint:
    admission_identity = "admitted-work-item/run"

    def __init__(self):
        self.content = None
        self.phases = []
        self.snapshots = []

    async def restore(self):
        return self.content

    async def save(self, content):
        self.content = content
        self.snapshots.append(content)
        self.phases.append(json.loads(content)["phase"])


def bind(port):
    @contextlib.asynccontextmanager
    async def scope(request, plan):
        yield port

    host_ports.register_run_checkpoint_factory(scope)


def setup(port, *, limit=50):
    prompts = []

    def respond(messages, info):
        prompt = messages[-1].parts[-1].content
        prompts.append(prompt)
        return ModelResponse(parts=[TextPart(f"reply:{prompt}")])

    bind(port)
    plan = RunPlan(
        model=FunctionModel(respond),
        steps=("first", "second", "third"),
        budget=RunBudget(request_limit=limit),
    )
    return RunRequest(task="task", run_id="same-run"), plan, prompts


async def stop_after_first(request, plan):
    events = iterate_run(request, plan=plan)
    async for event in events:
        if event["type"] == "node_transition":
            break
    await events.aclose()


async def test_restart_restores_messages_and_skips_completed_model_step():
    port = HostCheckpoint()
    request, plan, prompts = setup(port)
    await stop_after_first(request, plan)
    saved = json.loads(port.content)
    assert saved["next_step"] == 1 and saved["messages"]
    assert saved["usage"]["requests"] == 1
    result = await run(request, plan=plan)
    assert (result.status, result.steps, result.output) == (
        "completed",
        3,
        "reply:third",
    )
    assert prompts == ["first", "second", "third"]
    assert result.usage["requests"] == 3
    assert port.phases == ["step_inflight", "completed"] * 3
    again = await run(request, plan=plan)
    assert again.output == result.output and prompts == ["first", "second", "third"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"phase": "step_inflight"},
        {"plan_digest": "foreign"},
        {"run_id": "foreign"},
        {"admission_identity": "foreign"},
        {"next_step": 999},
        {"messages": [{"secret": "DO-NOT-ECHO"}]},
        {"usage": {"requests": -1}},
    ],
)
async def test_invalid_or_uncertain_restore_never_calls_model(mutation):
    port = HostCheckpoint()
    request, plan, prompts = setup(port)
    await stop_after_first(request, plan)
    raw = json.loads(port.content)
    raw.update(mutation)
    port.content = json.dumps(raw)
    result = await run(request, plan=plan)
    assert result.status == "failed"
    assert prompts == ["first"]
    assert "DO-NOT-ECHO" not in (result.error or "")


async def test_restart_preserves_cumulative_request_budget():
    port = HostCheckpoint()
    request, plan, prompts = setup(port, limit=2)
    await stop_after_first(request, plan)
    result = await run(request, plan=plan)
    assert result.status == "stopped"
    assert prompts == ["first", "second"]
    assert json.loads(port.content)["phase"] == "step_inflight"
    assert (await run(request, plan=plan)).status == "failed"
    assert prompts == ["first", "second"]


async def test_completed_restore_skips_model_when_request_limit_is_exhausted():
    port = HostCheckpoint()
    request, plan, prompts = setup(port, limit=3)
    result = await run(request, plan=plan)
    assert result.status == "completed" and result.usage["requests"] == 3

    again = await run(request, plan=plan)

    assert again.status == "completed"
    assert again.output == result.output
    assert again.usage == result.usage
    assert prompts == ["first", "second", "third"]


async def test_host_save_failure_stops_before_first_model_call():
    class Refused(HostCheckpoint):
        async def save(self, content):
            raise PermissionError("native fence changed")

    request, plan, prompts = setup(Refused())
    result = await run(request, plan=plan)
    assert result.status == "failed" and prompts == []


async def test_restart_does_not_repeat_completed_native_tool_effect():
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.toolsets import FunctionToolset

    effects = []
    tools = FunctionToolset()

    @tools.tool_plain
    def effect() -> str:
        effects.append("effect")
        return "done"

    def respond(messages, info):
        last = messages[-1].parts[-1]
        if getattr(last, "part_kind", "") == "tool-return":
            return ModelResponse(parts=[TextPart(str(last.content))])
        return ModelResponse(parts=[ToolCallPart("effect", {})])

    host_ports.register_native_toolset_provider(
        host_ports.NativeToolsetProvider(
            available_tools=lambda: ["effect"], build_toolset=lambda names, key: tools
        )
    )
    port = HostCheckpoint()
    bind(port)
    request = RunRequest(task="t", run_id="native-tool-run")
    plan = RunPlan(
        model=FunctionModel(respond),
        steps=("first", "second"),
        tool_names=("effect",),
        budget=RunBudget(tool_calls_limit=1),
    )
    await stop_after_first(request, plan)
    assert effects == ["effect"]
    result = await run(request, plan=plan)
    assert result.status == "stopped" and effects == ["effect"]
    assert result.usage["tool_calls"] == 1


async def test_changed_plan_stops_before_effects():
    from dataclasses import replace

    port = HostCheckpoint()
    request, plan, prompts = setup(port)
    await stop_after_first(request, plan)
    result = await run(request, plan=replace(plan, steps=("first", "different")))
    assert result.status == "failed" and prompts == ["first"]


async def test_exhausted_restored_wall_budget_never_calls_model():
    port = HostCheckpoint()
    request, plan, prompts = setup(port)
    await stop_after_first(request, plan)
    raw = json.loads(port.content)
    raw["elapsed_seconds"] = plan.budget.timeout_seconds + 1.0
    port.content = json.dumps(raw)
    result = await run(request, plan=plan)
    assert result.status == "stopped" and prompts == ["first"]
    assert port.phases == ["step_inflight", "completed"]


async def test_empty_plan_has_no_steps_or_checkpoint_writes():
    from dataclasses import replace

    port = HostCheckpoint()
    request, plan, prompts = setup(port)
    result = await run(request, plan=replace(plan, steps=()))
    assert result.status == "completed" and result.steps == 0
    assert prompts == [] and port.phases == []


async def test_dependency_checkpoint_restores_completed_tasks_and_shared_budget():
    """A completed prompt and tool are never replayed on a DAG restart."""
    port = HostCheckpoint()
    bind(port)
    prompts: list[str] = []
    tool_calls: list[tuple[str, dict]] = []
    tool_gate = asyncio.Event()

    def respond(messages, info):
        prompt = messages[-1].parts[-1].content
        prompts.append(prompt.split("\n", 1)[0])
        return ModelResponse(
            parts=[TextPart(prompt.split("\n", 1)[0].lower())],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    async def caller(server, tool, arguments, task_id):
        await tool_gate.wait()
        tool_calls.append((task_id, dict(arguments)))
        return ToolStepResult(result="b")

    host_ports.register_tool_caller(caller)
    plan = RunPlan(
        model=FunctionModel(respond),
        budget=RunBudget(request_limit=3, tool_calls_limit=1),
        dependency_plan=DependencyPlan(
            tasks=(
                RunTask("A", "A"),
                RunTask(
                    "B",
                    depends_on=("A",),
                    call=ToolStep(
                        "server", "tool", {"value": {"$ref": "steps.A.output"}}
                    ),
                ),
                RunTask("C", "C", depends_on=("B",)),
                RunTask("D", "D", depends_on=("C",)),
            ),
            output_task_id="D",
            plan_id="checkpoint-dag",
        ),
    )
    request = RunRequest(task="dag", run_id="checkpoint-dag-run")

    events = iterate_run(request, plan=plan)
    async for event in events:
        if event["type"] == "task_completed" and event["task_id"] == "A":
            break
    await events.aclose()
    # The driver may have published the next task's in-flight marker before
    # the consumer closes its event iterator.  Restore the last acknowledged
    # completed snapshot, which is the host's normal CAS choice after restart.
    saved_content = next(
        content
        for content in reversed(port.snapshots)
        if json.loads(content)["completed_tasks"] == ["A"]
        and not json.loads(content)["inflight_tasks"]
    )
    port.content = saved_content
    saved = json.loads(saved_content)
    assert saved["completed_tasks"] == ["A"]
    assert saved["dependency_outputs"] == {"A": "a"}
    assert saved["usage"]["requests"] == 1
    tool_gate.set()

    result = await run(request, plan=plan)

    assert result.status == "completed"
    assert result.output == "d"
    assert result.steps == 4
    assert result.usage["requests"] == 3
    assert result.usage["tool_calls"] == 1
    assert prompts == ["A", "C", "D"]
    assert tool_calls == [("B", {"value": "a"})]
    finished = json.loads(port.content)
    assert finished["status"] == "completed"
    assert finished["outputs"] == ["d"]
    again = await run(request, plan=plan)
    assert again.output == "d" and prompts == ["A", "C", "D"]
