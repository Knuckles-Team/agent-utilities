"""Actual streamed model and tool observations over an injected durable host port."""

from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, run


class Publisher:
    def __init__(self, fail_kind=None):
        self.events = []
        self.fail_kind = fail_kind

    async def publish(self, kind, content):
        if kind == self.fail_kind:
            raise OSError("sensitive backend detail")
        self.events.append((kind, content))


def bind(publisher):
    @asynccontextmanager
    async def scope(request):
        assert request.run_id == "run"
        yield publisher

    host_ports.register_durable_run_event_factory(scope)


async def test_actual_text_chunks_preserve_output_and_usage():
    calls = []

    async def stream(messages, info):
        calls.append(messages)
        for text in ("hello", " ", "world"):
            yield text

    publisher = Publisher()
    bind(publisher)
    result = await run(
        RunRequest(task="hello", run_id="run"),
        plan=RunPlan(model=FunctionModel(stream_function=stream), steps=("hello",)),
    )
    assert result.status == "completed"
    assert result.output == "hello world"
    assert len(calls) == 1
    assert result.usage["requests"] == 1
    assert (
        "".join(
            content["delta"]
            for kind, content in publisher.events
            if kind == "text_delta"
        )
        == result.output
    )
    assert publisher.events[-1] == (
        "state",
        {"step": 1, "state": "model_step_completed"},
    )
    assert not any(
        content.get("state") == "completed" for _, content in publisher.events
    )


@pytest.mark.parametrize("fail_kind", [None, "tool_call", "tool_result"])
async def test_actual_tool_events_are_ordered_and_never_repeat_effects(fail_kind):
    effects = []
    model_calls = []
    tools = FunctionToolset()

    @tools.tool_plain
    async def effect(value: int) -> str:
        effects.append(value)
        return f"observed:{value}"

    async def stream(messages, info):
        model_calls.append(messages)
        if any(
            getattr(part, "part_kind", "") == "tool-return"
            for part in messages[-1].parts
        ):
            yield "done"
        else:
            yield {
                0: DeltaToolCall(
                    name="effect", json_args='{"value":7}', tool_call_id="call-one"
                )
            }

    publisher = Publisher(fail_kind)
    bind(publisher)
    result = await run(
        RunRequest(task="do", run_id="run"),
        plan=RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("do",),
            toolsets=(tools,),
        ),
    )
    assert len(effects) <= 1
    if fail_kind is None:
        assert result.status == "completed"
        assert effects == [7]
        assert [kind for kind, _ in publisher.events] == [
            "tool_call",
            "tool_result",
            "text_delta",
            "state",
        ]
        call, returned = publisher.events[:2]
        assert call[1]["tool_call_id"] == returned[1]["tool_call_id"] == "call-one"
        assert call[1]["name"] == "effect"
        assert returned[1]["result"] == "observed:7"
        assert len(model_calls) == 2
    else:
        assert result.status == "failed"
        assert "durable run event publication failed" in result.error
        assert "sensitive backend detail" not in result.error
        assert not any(kind == "state" for kind, _ in publisher.events)
        assert len(model_calls) == 1
        assert effects == ([] if fail_kind == "tool_call" else [7])


async def test_unbound_execution_preserves_non_streaming_model_path():
    calls = []

    def ordinary(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("ordinary")])

    result = await run(
        RunRequest(task="hello"),
        plan=RunPlan(model=FunctionModel(ordinary), steps=("hello",)),
    )
    assert result.status == "completed" and result.output == "ordinary"
    assert len(calls) == 1


async def test_multi_step_publisher_scope_binds_once_and_preserves_history():
    bindings = []
    histories = []
    publisher = Publisher()

    @asynccontextmanager
    async def scope(request):
        bindings.append(request.run_id)
        yield publisher

    async def stream(messages, info):
        histories.append(len(messages))
        yield "next"

    host_ports.register_durable_run_event_factory(scope)
    result = await run(
        RunRequest(task="run", run_id="run"),
        plan=RunPlan(
            model=FunctionModel(stream_function=stream), steps=("first", "second")
        ),
    )
    assert result.status == "completed" and result.steps == 2
    assert bindings == ["run"]
    assert histories[0] < histories[1]
    assert [body["step"] for kind, body in publisher.events if kind == "state"] == [
        1,
        2,
    ]
