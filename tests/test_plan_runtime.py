"""Scaffold behavior through the real run loop and scripted model requests."""

from copy import deepcopy
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from agent_utilities.graph import HarnessConfig, RunPlan, RunRequest, run


def injected(messages):
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and part.content.startswith("Current run plan:")
    ]


@pytest.mark.spec("AU-BOUNDARY-R043.4.2")
@pytest.mark.parametrize("mode", ["on", "off"])
async def test_received_decision_gates_tool_and_injection(mode):
    seen = []

    def model(messages, info):
        seen.append(([tool.name for tool in info.function_tools], injected(messages)))
        return ModelResponse(parts=[TextPart("done")])

    result = await run(
        RunRequest(task="task"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            harness=HarnessConfig(plan_scaffold=mode),
        ),
    )
    assert result.status == "completed"
    assert (
        seen == [(["update_plan"], ["Current run plan: []"])]
        if mode == "on"
        else seen == [([], [])]
    )


@pytest.mark.spec("AU-BOUNDARY-R043.4.2")
@pytest.mark.spec("AU-BOUNDARY-R043.4.3")
@pytest.mark.spec("AU-BOUNDARY-R043.4.5")
@pytest.mark.spec("AU-BOUNDARY-R043.4")
async def test_two_updates_reach_each_request_but_never_stored_history(monkeypatch):
    captured = []
    histories = []
    original = Agent.run

    async def recording_run(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        histories.append(deepcopy(result.all_messages()))
        return result

    monkeypatch.setattr(Agent, "run", recording_run)

    def model(messages, info):
        captured.append(deepcopy(messages))
        turn = len(captured)
        if turn <= 2:
            items = [{"text": f"private-plan-{turn}", "status": "pending"}]
        elif turn == 3:
            items = [{"text": "private-invalid", "status": "invalid"}]
        else:
            return ModelResponse(parts=[TextPart("finished")])
        return ModelResponse(parts=[ToolCallPart("update_plan", {"items": items})])

    result = await run(
        RunRequest(task="task"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("first prompt", "second prompt"),
            harness=HarnessConfig(plan_scaffold="on"),
        ),
    )
    assert result.status == "completed"
    assert len(captured) == 5
    plans = [injected(messages) for messages in captured]
    assert all(len(plan) == 1 for plan in plans)
    assert plans[0] == ["Current run plan: []"]
    assert "private-plan-1" in plans[1][0]
    assert all("private-plan-2" in plan[0] for plan in plans[2:])
    assert "invalid_plan" in repr(captured[3])
    assert "updated" in repr(captured[1])
    assert len(histories) == 2
    for history in histories:
        assert "private-" not in repr(history)
        assert not injected(history)
        calls = [
            part
            for msg in history
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert len(calls) == 3
        assert all(call.args == {} for call in calls)
    assert "first prompt" in repr(histories[-1])
    assert "second prompt" in repr(histories[-1])


@pytest.mark.spec("AU-BOUNDARY-R043.4.1")
@pytest.mark.spec("AU-BOUNDARY-R043.4.3")
async def test_separate_executions_start_with_empty_scaffolds():
    initial = []

    def model(messages, info):
        initial.append(injected(messages))
        return ModelResponse(parts=[TextPart("done")])

    plan = RunPlan(
        model=FunctionModel(model),
        steps=("task",),
        harness=HarnessConfig(plan_scaffold="on"),
    )
    for _ in range(2):
        await run(RunRequest(task="task"), plan=plan)
    assert initial == [["Current run plan: []"], ["Current run plan: []"]]


@pytest.mark.spec("AU-BOUNDARY-R043.4.3")
async def test_budget_exit_cleans_tool_payload_without_another_request(monkeypatch):
    from agent_utilities.graph import RunBudget
    from agent_utilities.harness.plan_runtime import PlanScaffold

    retained = []
    original = PlanScaffold.before_model_request

    async def recording_request(self, ctx, request_context):
        retained.append(ctx.messages)
        return await original(self, ctx, request_context)

    monkeypatch.setattr(PlanScaffold, "before_model_request", recording_request)

    def model(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "update_plan",
                    {"items": [{"text": "terminal-private", "status": "done"}]},
                )
            ]
        )

    result = await run(
        RunRequest(task="task"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            harness=HarnessConfig(plan_scaffold="on"),
            budget=RunBudget(request_limit=1),
        ),
    )
    assert result.status == "stopped"
    assert len(retained) == 1
    assert "terminal-private" not in repr(retained)
    assert not injected(retained[0])


@pytest.mark.spec("AU-BOUNDARY-R043.4.4")
async def test_scaffold_surface_and_imports_have_no_selection_authority():
    import ast
    import inspect

    from agent_utilities.harness import plan, plan_runtime

    for module in (plan, plan_runtime):
        tree = ast.parse(inspect.getsource(module))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(name.startswith("agent_utilities.decide") for name in imports)

    schemas: list[dict[str, Any]] = []

    def model(messages, info):
        schemas.extend(tool.parameters_json_schema for tool in info.function_tools)
        return ModelResponse(parts=[TextPart("done")])

    await run(
        RunRequest(task="task"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            harness=HarnessConfig(plan_scaffold="on"),
        ),
    )
    assert len(schemas) == 1
    assert set(schemas[0]["properties"]) == {"items"}


@pytest.mark.spec("AU-BOUNDARY-R043.4.2")
@pytest.mark.parametrize("mode", ["unknown", "", None])
async def test_invalid_received_scaffold_decision_fails_before_model_call(mode):
    calls = []

    def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    result = await run(
        RunRequest(task="task"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            harness=HarnessConfig(plan_scaffold=mode),
        ),
    )
    assert result.status == "failed"
    assert result.error is not None
    assert "plan_scaffold must be on or off" in result.error
    assert calls == []


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
async def test_host_receives_only_actual_accepted_revisions():
    from contextlib import asynccontextmanager

    from agent_utilities import host_ports

    published = []
    lifecycle = []
    turns = []

    class Publisher:
        async def publish(self, revision, items):
            published.append((revision, tuple(item.model_dump() for item in items)))

    @asynccontextmanager
    async def factory(request):
        lifecycle.append(request.run_id)
        try:
            yield Publisher()
        finally:
            lifecycle.append("closed")

    host_ports.register_scaffold_publication_factory(factory)

    def model(messages, info):
        turns.append(len(published))
        turn = len(turns)
        if turn == 4:
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "update_plan",
                    {
                        "items": [
                            {
                                "text": f"private-{turn}",
                                "status": "invalid" if turn == 2 else "done",
                            }
                        ]
                    },
                )
            ]
        )

    result = await run(
        RunRequest(task="task", run_id="bound-run"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            harness=HarnessConfig(plan_scaffold="on"),
        ),
    )
    assert result.status == "completed"
    assert turns == [1, 2, 2, 3]
    assert published == [
        (0, ()),
        (1, ({"text": "private-1", "status": "done"},)),
        (2, ({"text": "private-3", "status": "done"},)),
    ]
    assert lifecycle == ["bound-run", "closed"]


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
@pytest.mark.parametrize("failure_at", ["enter", "initial", "update", "close"])
async def test_publication_failure_stops_run_without_disclosing_payload(failure_at):
    from contextlib import asynccontextmanager

    from agent_utilities import host_ports

    calls = []
    closed = []

    class Publisher:
        async def publish(self, revision, items):
            if (failure_at == "initial" and revision == 0) or (
                failure_at == "update" and revision == 1
            ):
                raise ValueError("private-plan-payload")

    @asynccontextmanager
    async def factory(request):
        if failure_at == "enter":
            raise ValueError("private-plan-payload")
        try:
            yield Publisher()
        finally:
            closed.append(True)
            if failure_at == "close":
                raise ValueError("private-plan-payload")

    host_ports.register_scaffold_publication_factory(factory)

    def model(messages, info):
        calls.append(True)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "update_plan",
                        {
                            "items": [
                                {"text": "private-plan-payload", "status": "pending"}
                            ]
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = await run(
        RunRequest(task="task"),
        plan=RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            harness=HarnessConfig(plan_scaffold="on"),
        ),
    )
    assert result.status == "failed"
    assert "private-plan-payload" not in repr(result)
    assert "scaffold publication" in result.error
    assert len(calls) == {"enter": 0, "initial": 0, "update": 1, "close": 2}[failure_at]
    assert closed == ([] if failure_at == "enter" else [True])


@pytest.mark.spec("AU-BOUNDARY-R043.5.2")
async def test_concurrent_runs_publish_to_their_own_host_binding():
    import asyncio
    from contextlib import asynccontextmanager

    from agent_utilities import host_ports

    observed = {}
    ready = asyncio.Event()

    class Publisher:
        def __init__(self, run_id):
            self.run_id = run_id

        async def publish(self, revision, items):
            observed.setdefault(self.run_id, []).append((revision, items))
            if len(observed) == 2:
                ready.set()
            await ready.wait()

    @asynccontextmanager
    async def factory(request):
        yield Publisher(request.run_id)

    host_ports.register_scaffold_publication_factory(factory)

    async def execute(run_id):
        calls = []

        def model(messages, info):
            calls.append(True)
            if len(calls) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "update_plan",
                            {"items": [{"text": run_id, "status": "done"}]},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("done")])

        return await run(
            RunRequest(task="task", run_id=run_id),
            plan=RunPlan(
                model=FunctionModel(model),
                steps=("task",),
                harness=HarnessConfig(plan_scaffold="on"),
            ),
        )

    results = await asyncio.gather(execute("first"), execute("second"))
    assert all(result.status == "completed" for result in results)
    for run_id, snapshots in observed.items():
        assert snapshots[0] == (0, ())
        assert snapshots[1][0] == 1
        assert snapshots[1][1][0].text == run_id
        assert len(snapshots) == 2
