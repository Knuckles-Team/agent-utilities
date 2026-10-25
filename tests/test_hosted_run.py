"""The public client-run seam preserves received requests and run semantics."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities import host_ports
from agent_utilities.graph import RunBudget, RunPlan, RunRequest
from agent_utilities.graph.state import HarnessConfig
from agent_utilities.harness.hosted_run import iterate_client_run


class Transport:
    def __init__(self):
        self.calls = []

    async def call(self, method, params):
        self.calls.append((method, params))
        return "client reply"


async def collect(request, transport):
    return [event async for event in iterate_client_run(request, transport)]


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_public_hosted_run_binds_transport_without_replacing_request(events):
    transport = Transport()
    request = RunRequest(task="inspect", run_id="admitted-run", context="bound")
    observed = []

    def model(messages, info):
        last = messages[-1].parts[-1]
        if last.part_kind == "tool-return":
            return ModelResponse(parts=[TextPart(str(last.content))])
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "file"})])

    plan = RunPlan(
        model=FunctionModel(model),
        steps=("inspect",),
        harness=HarnessConfig(client_tools=("read_file",)),
    )

    def resolve(value):
        observed.append(value)
        return plan

    host_ports.register_run_decision_provider(resolve)
    emitted = await collect(request, transport)
    assert observed == [request] and observed[0] is request
    assert emitted[-1]["type"] == "graph_complete"
    assert all(event["run_id"] == request.run_id for event in emitted)
    assert transport.calls == [("fs/read_text_file", {"path": "file"})]
    assert any(event.get("output") == "client reply" for event in emitted)
    assert plan.client_transport is None


@pytest.mark.parametrize("server_tools", ["names", "toolsets"])
@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_public_hosted_run_rejects_server_tools(server_tools):
    transport = Transport()
    plan = RunPlan(
        model="unused",
        tool_names=("server_tool",) if server_tools == "names" else (),
        toolsets=(FunctionToolset(),) if server_tools == "toolsets" else (),
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    with pytest.raises(PermissionError, match="client-only"):
        await collect(RunRequest("inspect"), transport)
    assert transport.calls == []


@pytest.mark.parametrize("invalid", ["prompt without authority", None, {}])
@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_non_request_refuses_before_host_provider(invalid):
    calls = []
    host_ports.register_run_decision_provider(lambda value: calls.append(value))
    with pytest.raises(TypeError, match="admitted RunRequest"):
        await collect(invalid, Transport())
    assert calls == []


@pytest.mark.spec("AU-BOUNDARY-R043.9.1")
@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_original_request_reaches_history_selection(valid):
    request = RunRequest(
        task="next",
        run_id="current",
        context="conversation",
        metadata={"predecessor": "prior"},
    )
    seen, calls = [], []

    async def history(value):
        seen.append(value)
        if not valid:
            return "malformed private history"
        return json.dumps(
            {
                "version": 1,
                "phase": "completed",
                "run_id": "prior",
                "messages": [
                    {"role": "user", "text": "earlier"},
                    {"role": "assistant", "text": "answer"},
                ],
            }
        )

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("fresh")])

    host_ports.register_run_history_provider(history)
    host_ports.register_run_decision_provider(
        lambda value: RunPlan(model=FunctionModel(respond), steps=(value.task,))
    )
    emitted = await collect(request, Transport())
    assert seen == [request] and seen[0] is request
    if valid:
        assert emitted[-1]["result"]["status"] == "completed"
        assert [message.parts[0].content for message in calls[0]] == [
            "earlier",
            "answer",
            "next",
        ]
    else:
        assert calls == []
        assert emitted[-1]["result"]["status"] == "failed"
        assert "malformed private" not in json.dumps(emitted)


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_received_budget_is_not_widened():
    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(respond),
            steps=("inspect",),
            budget=RunBudget(request_limit=0),
        )
    )
    emitted = await collect(RunRequest("inspect"), Transport())
    assert emitted[-1]["result"]["status"] == "stopped"
    assert calls == []


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_closing_consumer_closes_delegated_run_before_return(events):
    host_ports.register_run_decision_provider(lambda request: RunPlan(model="unused"))
    execution = iterate_client_run(RunRequest("inspect", run_id="closed"), Transport())
    assert (await anext(execution))["type"] == "run_started"
    await execution.aclose()
    assert events[-1]["type"] == "run_cancelled"
    assert events[-1]["run_id"] == "closed"


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_cancellation_reaches_active_model_and_run_events(events):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def respond(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(respond),
            steps=("inspect",),
        )
    )
    task = asyncio.create_task(collect(RunRequest("inspect"), Transport()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert events[-1]["type"] == "run_cancelled"


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_native_stream_text_arrives_before_run_completion(events):
    from pydantic_ai.messages import PartDeltaEvent, PartStartEvent

    order, native = [], []

    async def stream(messages, info):
        yield "hello"
        yield " world"

    async def observe(event):
        native.append(event)
        order.append("native")

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("inspect",),
        )
    )
    emitted = []
    async for event in iterate_client_run(
        RunRequest("inspect", run_id="streamed"),
        Transport(),
        stream_observer=observe,
    ):
        emitted.append(event)
        order.append(event["type"])
    assert any(isinstance(event, PartStartEvent) for event in native)
    assert any(isinstance(event, PartDeltaEvent) for event in native)
    assert emitted[-1]["result"]["output"] == "hello world"
    assert (
        order.index("native")
        < order.index("node_transition")
        < order.index("graph_complete")
    )
    assert all(event["run_id"] == "streamed" for event in emitted)
    assert not any(event["type"] == "native" for event in events)


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_native_stream_tool_calls_use_same_bound_transport():
    from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
    from pydantic_ai.models.function import DeltaToolCall

    transport, native = Transport(), []

    async def stream(messages, info):
        if messages[-1].parts[-1].part_kind == "tool-return":
            yield "done"
        else:
            yield {0: DeltaToolCall(name="read_file", json_args='{"path":"file"}')}

    async def observe(event):
        native.append(event)

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("inspect",),
            harness=HarnessConfig(client_tools=("read_file",)),
        )
    )
    emitted = [
        event
        async for event in iterate_client_run(
            RunRequest("inspect"),
            transport,
            stream_observer=observe,
        )
    ]
    assert emitted[-1]["result"]["output"] == "done"
    assert transport.calls == [("fs/read_text_file", {"path": "file"})]
    assert any(isinstance(event, FunctionToolCallEvent) for event in native)
    assert any(isinstance(event, FunctionToolResultEvent) for event in native)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_native_observer_failure_or_cancel_never_retries(cancel, events):
    calls, received = [], []

    async def stream(messages, info):
        calls.append(messages)
        yield "first"
        yield "not retried"

    async def observe(event):
        received.append(event)
        if cancel:
            raise asyncio.CancelledError
        raise RuntimeError("private transport failure")

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("inspect",),
        )
    )

    async def execute():
        return [
            event
            async for event in iterate_client_run(
                RunRequest("inspect"),
                Transport(),
                stream_observer=observe,
            )
        ]

    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await execute()
        assert events[-1]["type"] == "run_cancelled"
    else:
        emitted = await execute()
        assert emitted[-1]["result"]["status"] == "failed"
        assert "model stream observer failed" in emitted[-1]["error"]
        assert "private transport" not in json.dumps(emitted)
    assert len(calls) == len(received) == 1


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_native_observer_budget_refuses_before_streaming():
    calls = []

    async def stream(messages, info):
        calls.append("model")
        yield "unexpected"

    async def observe(event):
        calls.append("observer")

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("inspect",),
            budget=RunBudget(request_limit=0),
        )
    )
    emitted = [
        event
        async for event in iterate_client_run(
            RunRequest("inspect"),
            Transport(),
            stream_observer=observe,
        )
    ]
    assert calls == []
    assert emitted[-1]["result"]["status"] == "stopped"


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_native_observer_dependency_plan_refuses_before_model():
    from agent_utilities.graph import DependencyPlan, RunTask

    calls = []

    async def stream(messages, info):
        calls.append("model")
        yield "unexpected"

    async def observe(event):
        calls.append("observer")

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            dependency_plan=DependencyPlan((RunTask("step", "inspect"),), "step"),
        )
    )
    emitted = [
        event
        async for event in iterate_client_run(
            RunRequest("inspect"),
            Transport(),
            stream_observer=observe,
        )
    ]
    assert calls == []
    assert emitted[-1]["result"]["status"] == "failed"
    assert "ordinary sequential plan" in emitted[-1]["error"]


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_private_thinking_stays_out_of_public_run_events(events):
    from pydantic_ai.messages import PartStartEvent, ThinkingPart
    from pydantic_ai.models.function import DeltaThinkingPart

    native = []

    async def stream(messages, info):
        yield {0: DeltaThinkingPart(content="private model reasoning")}
        yield "public answer"

    async def observe(event):
        native.append(event)

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("inspect",),
        )
    )
    emitted = [
        event
        async for event in iterate_client_run(
            RunRequest("inspect"),
            Transport(),
            stream_observer=observe,
        )
    ]
    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart)
        for event in native
    )
    assert emitted[-1]["result"]["output"] == "public answer"
    assert "private model reasoning" not in json.dumps(events + emitted)


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_observer_backpressure_uses_existing_deadline(events):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def stream(messages, info):
        yield "answer"

    async def observe(event):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("inspect",),
            budget=RunBudget(timeout_seconds=0.05),
        )
    )
    emitted = [
        event
        async for event in iterate_client_run(
            RunRequest("inspect"),
            Transport(),
            stream_observer=observe,
        )
    ]
    assert entered.is_set() and cancelled.is_set()
    assert emitted[-1]["result"]["status"] == "stopped"
    assert emitted[-1]["result"]["error"] == "budget: run timed out"
    assert events[-1]["type"] == "graph_complete"
