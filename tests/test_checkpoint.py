"""Actual harness restart behavior over an injected host transaction boundary."""

import contextlib
import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports
from agent_utilities.graph import RunBudget, RunPlan, RunRequest, run
from agent_utilities.graph.executor import iterate_run


class HostCheckpoint:
    admission_identity = "admitted-work-item/run"

    def __init__(self):
        self.content = None
        self.phases = []

    async def restore(self):
        return self.content

    async def save(self, content):
        self.content = content
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
