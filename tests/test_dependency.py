"""Received DAG execution through the public run API, without sleeps or network."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage

from agent_utilities import host_ports
from agent_utilities.graph import (
    DependencyPlan,
    RunBudget,
    RunPlan,
    RunRequest,
    RunTask,
    StopRule,
    iterate_run,
    run,
)

pytestmark = pytest.mark.spec("AU-BOUNDARY-R043.6")


class ControlledModel:
    """Explicit gates control completion; model calls record the real overlap."""

    def __init__(self, *names: str, failed: str | None = None):
        self.started = {key: asyncio.Event() for key in names}
        self.release = {key: asyncio.Event() for key in names}
        self.ended = {key: asyncio.Event() for key in names}
        self.failed = failed
        self.timeline: list[tuple[str, str]] = []
        self.active = 0
        self.peak = 0
        self.inputs: dict[str, str] = {}
        self.history_lengths: list[int] = []
        self.model = FunctionModel(self.respond)

    async def respond(self, messages, info):
        prompt = messages[-1].parts[-1].content
        key = prompt.split("\n", 1)[0]
        self.inputs[key] = prompt
        self.history_lengths.append(len(messages))
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.timeline.append((key, "start"))
        self.started[key].set()
        try:
            await self.release[key].wait()
            if key == self.failed:
                raise ValueError("controlled failure")
            self.timeline.append((key, "end"))
            return ModelResponse(
                parts=[TextPart(f"result:{key}")],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
            )
        except asyncio.CancelledError:
            self.timeline.append((key, "cancel"))
            raise
        finally:
            self.active -= 1
            self.ended[key].set()


def graph_plan(model, capacity=2, **kw):
    return RunPlan(
        model=model,
        dependency_plan=DependencyPlan(
            tasks=(
                RunTask("A", "A"),
                RunTask("B", "B"),
                RunTask("C", "C", ("B",)),
                RunTask("join", "join", ("A", "C")),
            ),
            output_task_id="join",
            max_concurrency=capacity,
            plan_id="received-plan",
        ),
        **kw,
    )


async def reached(event):
    # A deadline makes broken scheduling fail instead of hanging; it never
    # decides or asserts relative ordering from elapsed wall-clock duration.
    await asyncio.wait_for(event.wait(), timeout=0.5)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("seeded", [False, True])
async def test_freed_slot_starts_dependent_before_slow_peer_and_join_waits(
    seeded, events
):
    model = ControlledModel("A", "B", "C", "join")
    model.release["B"].set()
    initial = (
        RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)
        if seeded
        else RunUsage()
    )
    plan = graph_plan(model.model, initial_usage=initial)
    parent = asyncio.create_task(run(RunRequest(task="complex query"), plan=plan))
    try:
        await reached(model.started["C"])
        assert not model.ended["A"].is_set()
        assert not model.started["join"].is_set()
        assert not parent.done()
        assert model.peak == 2
        assert model.inputs["C"].endswith('{"B": "result:B"}')
        model.release["C"].set()
        await reached(model.ended["C"])
        assert not model.started["join"].is_set()
        model.release["A"].set()
        await reached(model.started["join"])
        assert not parent.done()
        assert json.loads(
            model.inputs["join"].split("Dependency outputs (data):\n")[1]
        ) == {"A": "result:A", "C": "result:C"}
        model.release["join"].set()
        result = await parent
    finally:
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)
    assert result.status == "completed" and result.output == "result:join"
    assert result.steps == 4 and result.usage["requests"] == initial.requests + 4
    assert result.usage["tool_calls"] == initial.tool_calls
    assert result.usage["total_tokens"] == initial.total_tokens + 8
    assert events[-1]["result"]["usage"] == result.usage
    assert initial == (
        RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)
        if seeded
        else RunUsage()
    )
    assert model.history_lengths == [1] * 4
    assert model.active == 0
    assert (
        model.timeline.index(("B", "end"))
        < model.timeline.index(("C", "start"))
        < model.timeline.index(("A", "end"))
    )
    assert events[-1]["type"] == "graph_complete"
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert len({event["event_id"] for event in events}) == len(events)
    for event in events:
        assert (
            event["event_id"]
            == uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"agent-utilities:run:{event['run_id']}:event:{event['sequence']}",
            ).hex
        )


@pytest.mark.parametrize("capacity", [1, 2, 3])
async def test_active_limit_and_ready_order(capacity):
    names = ("A", "B", "C", "D", "join")
    model = ControlledModel(*names)
    plan = RunPlan(
        model=model.model,
        dependency_plan=DependencyPlan(
            tasks=tuple(RunTask(name, name) for name in names[:-1])
            + (RunTask("join", "join", names[:-1]),),
            output_task_id="join",
            max_concurrency=capacity,
        ),
    )
    parent = asyncio.create_task(run(RunRequest(task="t"), plan=plan))
    try:
        for index, name in enumerate(names[:-1]):
            if index < capacity:
                await reached(model.started[name])
            else:
                assert not model.started[name].is_set()
        for name in names:
            model.release[name].set()
        result = await parent
    finally:
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)
    assert result.status == "completed"
    assert model.peak <= capacity
    assert [key for key, kind in model.timeline if kind == "start"] == list(names)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("failure", ["error", "stop", "cancel"])
async def test_failure_stop_and_cancel_drain_siblings_and_block_dependents(
    failure, seeded, events
):
    model = ControlledModel(
        "A", "B", "C", "join", failed="B" if failure == "error" else None
    )
    initial = (
        RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)
        if seeded
        else RunUsage()
    )
    plan = graph_plan(
        model.model,
        stop=StopRule(stop_markers=("result:B",) if failure == "stop" else ()),
        initial_usage=initial,
    )
    parent = asyncio.create_task(run(RunRequest(task="t"), plan=plan))
    await reached(model.started["A"])
    await reached(model.started["B"])
    if failure == "cancel":
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
    else:
        model.release["B"].set()
        result = await parent
        assert result.status == ("failed" if failure == "error" else "stopped")
    assert model.active == 0 and model.ended["A"].is_set()
    assert events[-1]["result"]["usage"] == {
        "requests": initial.requests + 2,
        "tool_calls": initial.tool_calls,
        "input_tokens": initial.input_tokens + (failure == "stop"),
        "output_tokens": initial.output_tokens + (failure == "stop"),
        "total_tokens": initial.total_tokens + (2 if failure == "stop" else 0),
    }
    if failure != "cancel":
        assert events[-1]["result"]["usage"] == result.usage
    assert initial == (
        RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)
        if seeded
        else RunUsage()
    )
    assert not model.started["C"].is_set()
    assert not model.started["join"].is_set()
    assert {e["task_id"] for e in events if e["type"] == "task_blocked"} == {
        "C",
        "join",
    }
    assert {e["task_id"] for e in events if e["type"] == "task_cancelled"} >= {"A"}
    assert (
        events[-1]["type"]
        == {"error": "error", "stop": "graph_complete", "cancel": "run_cancelled"}[
            failure
        ]
    )
    assert (
        len(
            [
                e
                for e in events
                if e["type"] in ("error", "graph_complete", "run_cancelled")
            ]
        )
        == 1
    )


async def test_iterator_close_drains_children(events):
    model = ControlledModel("A", "B", "C", "join")
    stream = iterate_run(RunRequest(task="t"), plan=graph_plan(model.model))
    assert (await anext(stream))["type"] == "run_started"
    assert (await anext(stream))["type"] == "task_started"
    await reached(model.started["A"])
    await stream.aclose()
    assert model.active == 0 and model.ended["A"].is_set()
    assert not model.started["C"].is_set()
    assert events[-1]["type"] == "run_cancelled"


@pytest.mark.parametrize(
    "change",
    [
        {"tasks": ()},
        {"tasks": (RunTask("A", "A"), RunTask("A", "A"))},
        {"tasks": (RunTask("A", "A", ("missing",)),)},
        {"tasks": (RunTask("A", "A", ("A",)),)},
        {"tasks": (RunTask("A", "A", ("B",)), RunTask("B", "B", ("A",)))},
        {"tasks": (RunTask("A", "A"), RunTask("B", "B", ("A", "A")))},
        {"output_task_id": "missing"},
        {"output_task_id": "A"},
        {"max_concurrency": 0},
        {"max_concurrency": True},
        {"max_concurrency": 1.5},
    ],
)
async def test_invalid_plan_fails_before_model_binding(change, events):
    host_ports.register_model_provider(lambda _: pytest.fail("must not bind model"))
    base = graph_plan("unbound")
    plan = replace(base, dependency_plan=replace(base.dependency_plan, **change))
    result = await run(RunRequest(task="t"), plan=plan)
    assert result.status == "failed"
    assert [e["type"] for e in events] == ["error"]


@pytest.mark.parametrize(
    "change",
    [
        {"stop": StopRule(max_steps=3)},
        {"steps": ("serial",)},
        {"budget": RunBudget(request_limit=-1)},
        {"budget": RunBudget(timeout_seconds=float("nan"))},
    ],
)
async def test_over_budget_or_ambiguous_plan_fails_before_binding(change, events):
    host_ports.register_model_provider(lambda _: pytest.fail("must not bind model"))
    result = await run(
        RunRequest(task="t"), plan=replace(graph_plan("unbound"), **change)
    )
    assert result.status == "failed"
    assert [e["type"] for e in events] == ["error"]


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
async def test_shared_request_limit_never_overshoots(limit, events):
    model = ControlledModel("A", "B", "C", "join")
    for gate in model.release.values():
        gate.set()
    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(model.model, budget=RunBudget(request_limit=limit)),
    )
    assert result.status == "stopped"
    assert len(model.inputs) == limit
    assert result.usage["requests"] == limit
    assert model.active == 0


async def test_shared_token_limit_stops_after_reported_usage():
    model = ControlledModel("A", "B", "C", "join")
    for gate in model.release.values():
        gate.set()
    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(
            model.model, capacity=1, budget=RunBudget(total_tokens_limit=3)
        ),
    )
    assert result.status == "stopped"
    assert list(model.inputs) == ["A", "B"]
    assert result.usage["total_tokens"] == 4


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("initial_tool_calls", [0, 1])
async def test_tool_reservations_are_shared_across_parallel_children(
    initial_tool_calls,
):
    effects: list[str] = []
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect() -> str:
        effects.append("effect")
        return "done"

    async def model(messages, info):
        if getattr(messages[-1].parts[-1], "part_kind", "") == "tool-return":
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("effect", {})])

    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(
            FunctionModel(model),
            toolsets=(toolset,),
            budget=RunBudget(tool_calls_limit=initial_tool_calls + 1),
            initial_usage=RunUsage(tool_calls=initial_tool_calls),
        ),
    )
    assert result.status == "stopped"
    assert effects == ["effect"]
    assert result.usage["tool_calls"] == initial_tool_calls + 1


@pytest.mark.parametrize("mode", ["timeout", "cancel", "fail"])
async def test_host_scope_stays_live_until_children_are_drained(mode, events):
    from contextlib import asynccontextmanager
    from contextvars import ContextVar

    identity = ContextVar("identity", default="absent")
    entered = asyncio.Event()
    scope_log = []
    active = 0

    @asynccontextmanager
    async def scope():
        assert identity.get() == "admitted-caller"
        scope_log.append("open")
        try:
            yield
        finally:
            assert active == 0
            scope_log.append("close")

    async def model(messages, info):
        nonlocal active
        assert identity.get() == "admitted-caller"
        active += 1
        entered.set()
        try:
            if mode == "fail":
                raise ValueError("fail")
            await asyncio.Event().wait()
        finally:
            active -= 1

    token = identity.set("admitted-caller")
    host_ports.register_authority_keepalive(scope)
    try:
        plan = graph_plan(
            FunctionModel(model),
            budget=RunBudget(timeout_seconds=0 if mode == "timeout" else 600),
        )
        parent = asyncio.create_task(run(RunRequest(task="t"), plan=plan))
        if mode == "cancel":
            await reached(entered)
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parent
        else:
            result = await parent
            assert result.status == ("failed" if mode == "fail" else "stopped")
        assert scope_log == ["open", "close"] and active == 0
        assert events[-1]["type"] in ("graph_complete", "run_cancelled", "error")
    finally:
        host_ports.register_authority_keepalive(None)
        identity.reset(token)


async def test_normal_error_iteration_returns_same_cleanup_events_as_sink(events):
    async def fail(messages, info):
        raise ValueError("failure")

    yielded = [
        event
        async for event in iterate_run(
            RunRequest(task="t"), plan=graph_plan(FunctionModel(fail))
        )
    ]
    assert yielded == events
    assert yielded[-1]["type"] == "error"
    assert len([event for event in yielded if event["type"] == "task_blocked"]) == 2


async def test_independent_runs_do_not_share_budget_or_task_state():
    async def reply(messages, info):
        return ModelResponse(parts=[TextPart("ok")])

    plans = [
        RunPlan(
            model=FunctionModel(reply),
            dependency_plan=DependencyPlan(
                tasks=(RunTask("task", "task"),),
                output_task_id="task",
                max_concurrency=1,
            ),
            budget=RunBudget(request_limit=1),
        )
        for _ in range(2)
    ]
    results = await asyncio.gather(
        *(run(RunRequest(task="t"), plan=plan) for plan in plans)
    )
    assert all(
        result.status == "completed" and result.usage["requests"] == 1
        for result in results
    )
    assert results[0].run_id != results[1].run_id


async def test_repeated_cancellation_cannot_detach_child_cleanup(events):
    entered, cleaning, finish, cleaned = (asyncio.Event() for _ in range(4))

    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            cleaned.set()

    parent = asyncio.create_task(
        run(RunRequest(task="t"), plan=graph_plan(FunctionModel(model), capacity=1))
    )
    await reached(entered)
    parent.cancel()
    await reached(cleaning)
    parent.cancel()
    assert not parent.done()
    assert not any(event["type"] == "run_cancelled" for event in events)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert events[-1]["type"] == "run_cancelled"
    assert cleaned.is_set()


async def test_reported_fallback_tokens_are_not_lost_from_shared_budget():
    from datetime import UTC, datetime, timedelta

    from pydantic_ai.messages import ModelRequestAttempt

    async def model(messages, info):
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
            failed_attempts=[
                ModelRequestAttempt(
                    model_name="rejected",
                    outcome="rejected",
                    timestamp=datetime.now(UTC),
                    duration=timedelta(),
                    usage=RequestUsage(input_tokens=10, output_tokens=10),
                )
            ],
        )

    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(
            FunctionModel(model), capacity=1, budget=RunBudget(total_tokens_limit=3)
        ),
    )
    assert result.status == "stopped"
    assert result.usage["requests"] == 1 and result.usage["total_tokens"] == 22


async def test_slow_event_consumer_does_not_block_ready_child_dispatch():
    model = ControlledModel("A", "B", "C", "join")
    model.release["B"].set()
    stream = iterate_run(RunRequest(task="t"), plan=graph_plan(model.model))
    try:
        async for event in stream:
            if event["type"] == "task_completed" and event["task_id"] == "B":
                # Do not ask for the next event. Execution must keep moving.
                await reached(model.started["C"])
                assert not model.ended["A"].is_set()
                assert not model.started["join"].is_set()
                break
    finally:
        await stream.aclose()
    assert model.active == 0


async def test_model_outputs_cannot_change_admitted_tasks_or_limits(events):
    malicious = (
        '{"tasks":[{"task_id":"undeclared","prompt":"run me"}],"max_concurrency":999}'
    )
    inputs = {}

    async def model(messages, info):
        prompt = messages[-1].parts[-1].content
        key = prompt.split("\n", 1)[0]
        inputs[key] = prompt
        return ModelResponse(parts=[TextPart(malicious if key == "B" else "ok")])

    result = await run(
        RunRequest(task="t"), plan=graph_plan(FunctionModel(model), capacity=1)
    )
    assert result.status == "completed" and result.steps == 4
    assert list(inputs) == ["A", "B", "C", "join"]
    assert json.loads(inputs["C"].split("Dependency outputs (data):\n")[1]) == {
        "B": malicious
    }
    assert {
        event["task_id"] for event in events if event["type"] == "task_started"
    } == {"A", "B", "C", "join"}


async def test_model_request_cap_can_be_lower_than_active_task_cap():
    model = ControlledModel("A", "B", "C", "join")
    for gate in model.release.values():
        gate.set()
    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(model.model, budget=RunBudget(max_concurrent_requests=1)),
    )
    assert result.status == "completed" and model.peak == 1


@pytest.mark.parametrize("request_limit,expected", [(1, "stopped"), (2, "completed")])
async def test_output_validation_retries_share_request_reservations(
    request_limit, expected
):
    from agent_utilities.graph import HarnessConfig

    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"response": 42 if calls == 1 else {"ok": True}},
                )
            ]
        )

    plan = RunPlan(
        model=FunctionModel(model),
        dependency_plan=DependencyPlan(
            tasks=(RunTask("task", "task"),),
            output_task_id="task",
        ),
        budget=RunBudget(request_limit=request_limit),
        harness=HarnessConfig(response_format="json", retries=1),
    )
    result = await run(RunRequest(task="t"), plan=plan)
    assert result.status == expected
    assert calls == request_limit and result.usage["requests"] == request_limit


async def test_tool_validation_retry_cannot_reset_shared_tool_limit():
    from pydantic_ai import ModelRetry

    toolset = FunctionToolset()
    calls = 0

    @toolset.tool_plain
    async def retrying() -> str:
        nonlocal calls
        calls += 1
        raise ModelRetry("try again")

    async def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("retrying", {})])

    plan = RunPlan(
        model=FunctionModel(model),
        dependency_plan=DependencyPlan(
            tasks=(RunTask("task", "task"),),
            output_task_id="task",
        ),
        budget=RunBudget(tool_calls_limit=1),
        toolsets=(toolset,),
    )
    result = await run(RunRequest(task="t"), plan=plan)
    assert result.status == "stopped" and calls == 1
    assert result.usage["tool_calls"] == 1


@pytest.mark.parametrize("dag", [False, True])
async def test_close_immediately_after_run_started_emits_cancelled(dag):
    events = []
    host_ports.register_run_event_sink(events.append)
    plan = RunPlan(
        model=FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")])),
        dependency_plan=DependencyPlan(
            tasks=(RunTask("task", "task"),), output_task_id="task"
        ),
    )
    if not dag:
        plan = replace(plan, dependency_plan=None, steps=("serial",))
    stream = iterate_run(RunRequest(task="t"), plan=plan)
    assert (await anext(stream))["type"] == "run_started"
    await stream.aclose()
    assert [event["type"] for event in events] == ["run_started", "run_cancelled"]


async def test_unspecified_parallel_tool_limit_refuses_oversized_batch_before_fanout():
    calls = 0
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect() -> str:
        nonlocal calls
        calls += 1
        return "effect"

    async def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("effect", {}) for _ in range(51)])

    result = await run(
        RunRequest(task="t"),
        plan=RunPlan(
            model=FunctionModel(model),
            toolsets=(toolset,),
            dependency_plan=DependencyPlan(
                tasks=(RunTask("task", "task"),), output_task_id="task"
            ),
        ),
    )
    assert result.status == "stopped" and calls == 0
    assert result.usage["tool_calls"] == 0


async def test_dependency_deadline_drains_work_while_reader_is_paused(monkeypatch):
    entered, drained = asyncio.Event(), asyncio.Event()
    deadlines = []
    setup_deadlines = []
    timeout_factory, at_factory = asyncio.timeout, asyncio.timeout_at

    def setup_timeout(delay):
        scope = timeout_factory(delay)
        setup_deadlines.append(scope.when())
        return scope

    def driver_timeout(deadline):
        scope = at_factory(None)
        deadlines.append((deadline, scope))
        return scope

    monkeypatch.setattr(asyncio, "timeout", setup_timeout)
    monkeypatch.setattr(asyncio, "timeout_at", driver_timeout)
    events = []
    host_ports.register_run_event_sink(events.append)

    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    plan = RunPlan(
        model=FunctionModel(model),
        dependency_plan=DependencyPlan(
            tasks=(RunTask("task", "task"),),
            output_task_id="task",
        ),
        budget=RunBudget(timeout_seconds=600),
    )
    stream = iterate_run(RunRequest(task="t"), plan=plan)
    try:
        assert (await anext(stream))["type"] == "run_started"
        assert (await anext(stream))["type"] == "task_started"
        await asyncio.wait_for(entered.wait(), 0.5)
        assert len(deadlines) == 1 and deadlines[0][0] == setup_deadlines[0]
        deadlines[0][1].reschedule(asyncio.get_running_loop().time())
        # No further iterator reads: the reader's unrelated await must survive.
        await asyncio.wait_for(drained.wait(), 0.5)
        remaining = [event async for event in stream]
    finally:
        await stream.aclose()
    assert remaining[-1]["type"] == "graph_complete"
    assert remaining[-1]["result"]["status"] == "stopped"
    assert "timed out" in remaining[-1]["result"]["error"]
    assert [event["type"] for event in events] == [
        "run_started",
        "task_started",
        "task_cancelled",
        "graph_complete",
    ]


@pytest.mark.parametrize("stage", ["bind", "publish"])
async def test_dependency_deadline_includes_scaffold_setup(stage, monkeypatch):
    from contextlib import asynccontextmanager

    from agent_utilities.graph import HarnessConfig

    entered, closed = asyncio.Event(), asyncio.Event()
    scopes = []
    timeout_factory = asyncio.timeout

    def capture_timeout(delay):
        scope = timeout_factory(None)
        scopes.append(scope)
        return scope

    monkeypatch.setattr(asyncio, "timeout", capture_timeout)
    calls = 0

    class Publisher:
        async def publish(self, revision, items):
            entered.set()
            await asyncio.Event().wait()

    @asynccontextmanager
    async def scope(request):
        try:
            if stage == "bind":
                entered.set()
                await asyncio.Event().wait()
            yield Publisher()
        finally:
            closed.set()

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("never")])

    host_ports.register_scaffold_publication_factory(scope)
    parent = asyncio.create_task(
        run(
            RunRequest(task="t"),
            plan=RunPlan(
                model=FunctionModel(model),
                dependency_plan=DependencyPlan(
                    tasks=(RunTask("task", "task"),),
                    output_task_id="task",
                ),
                budget=RunBudget(timeout_seconds=600),
                harness=HarnessConfig(plan_scaffold="on"),
            ),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 0.5)
        assert len(scopes) == 1
        scopes[0].reschedule(asyncio.get_running_loop().time())
        result = await asyncio.wait_for(parent, 0.5)
        assert result.status == "stopped" and "timed out" in result.error
        assert calls == 0 and closed.is_set()
    finally:
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    ("initial", "budget", "elapsed"),
    [
        (RunUsage(requests=3), RunBudget(request_limit=3), 0),
        (RunUsage(input_tokens=2, output_tokens=3), RunBudget(total_tokens_limit=5), 0),
        (RunUsage(tool_calls=1), RunBudget(tool_calls_limit=0), 0),
        (RunUsage(tool_calls=3), RunBudget(tool_calls_limit=2), 0),
        (RunUsage(), RunBudget(timeout_seconds=5), 5),
    ],
)
async def test_initial_dependency_budget_refuses_before_binding_or_child_dispatch(
    initial, budget, elapsed, events
):
    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    host_ports.register_native_toolset_provider(
        host_ports.NativeToolsetProvider(
            available_tools=lambda: pytest.fail("tools discovered"),
            build_toolset=lambda *_: pytest.fail("tools bound"),
        )
    )
    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(
            "unbound",
            tool_names=("effect",),
            initial_usage=initial,
            initial_elapsed_seconds=elapsed,
            budget=budget,
        ),
    )
    assert result.status == "stopped" and result.steps == 0
    assert (result.error or "").startswith("budget:")
    assert result.usage["requests"] == initial.requests
    assert result.usage["tool_calls"] == initial.tool_calls
    assert result.usage["total_tokens"] == initial.total_tokens
    assert events[-1]["result"]["usage"] == result.usage
    assert not any(event["type"].startswith("task_") for event in events)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("attempt_tool", [False, True])
async def test_exact_initial_dependency_tool_limit_allows_only_model_work(
    limit, attempt_tool, events
):
    effects = []
    prompts = []
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect() -> str:
        effects.append("effect")
        return "done"

    async def reply(messages, info):
        prompts.append(messages[-1].parts[-1].content.split("\n", 1)[0])
        return ModelResponse(
            parts=[ToolCallPart("effect", {}) if attempt_tool else TextPart("ok")],
            usage=RequestUsage(input_tokens=2, output_tokens=1),
        )

    initial = RunUsage(requests=1, tool_calls=limit, input_tokens=3, output_tokens=4)
    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(
            FunctionModel(reply),
            capacity=1 if attempt_tool else 2,
            toolsets=(toolset,),
            initial_usage=initial,
            budget=RunBudget(tool_calls_limit=limit),
        ),
    )
    calls = 1 if attempt_tool else 4
    assert prompts == ["A", "B", "C", "join"][:calls]
    assert (result.status, result.steps, result.output) == (
        "stopped" if attempt_tool else "completed",
        calls,
        "" if attempt_tool else "ok",
    )
    assert effects == []
    assert result.usage == {
        "requests": 1 + calls,
        "tool_calls": limit,
        "input_tokens": 3 + 2 * calls,
        "output_tokens": 4 + calls,
        "total_tokens": 7 + 3 * calls,
    }
    assert initial == RunUsage(
        requests=1, tool_calls=limit, input_tokens=3, output_tokens=4
    )
    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == "graph_complete"
    assert events[-1]["result"] == result.as_dict()
    lifecycle = [(event["type"], event["task_id"]) for event in events[1:-1]]
    if attempt_tool:
        assert (result.error or "").startswith("budget:")
        assert "tool_calls_limit" in result.error
        assert lifecycle == [
            ("task_started", "A"),
            ("task_failed", "A"),
            ("task_blocked", "B"),
            ("task_blocked", "C"),
            ("task_blocked", "join"),
        ]
    else:
        assert result.error is None
        assert len(lifecycle) == 8
        assert set(lifecycle) == {
            (kind, task)
            for kind in ("task_started", "task_completed")
            for task in ("A", "B", "C", "join")
        }


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    ("initial", "budget", "calls", "tokens"),
    [
        (RunUsage(requests=1), RunBudget(request_limit=4), 3, 6),
        (RunUsage(input_tokens=3), RunBudget(total_tokens_limit=5), 1, 5),
    ],
)
async def test_initial_dependency_usage_leaves_only_the_aggregate_remainder(
    initial, budget, calls, tokens, events
):
    model = ControlledModel("A", "B", "C", "join")
    for gate in model.release.values():
        gate.set()
    result = await run(
        RunRequest(task="t"),
        plan=graph_plan(model.model, capacity=1, initial_usage=initial, budget=budget),
    )
    assert result.status == "stopped"
    assert len(model.inputs) == calls and model.active == 0
    assert result.usage["requests"] == initial.requests + calls
    assert result.usage["total_tokens"] == tokens
    assert events[-1]["result"]["usage"] == result.usage


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_reusing_one_seeded_dependency_plan_does_not_share_mutable_usage():
    async def reply(messages, info):
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    initial = RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)
    plan = graph_plan(FunctionModel(reply), initial_usage=initial)
    results = await asyncio.gather(
        *(run(RunRequest(task="t"), plan=plan) for _ in range(2))
    )
    expected = {
        "requests": 6,
        "tool_calls": 1,
        "input_tokens": 7,
        "output_tokens": 8,
        "total_tokens": 15,
    }
    assert all(
        result.status == "completed" and result.usage == expected for result in results
    )
    assert initial == RunUsage(
        requests=2, tool_calls=1, input_tokens=3, output_tokens=4
    )
    assert results[0].run_id != results[1].run_id
