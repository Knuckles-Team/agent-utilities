"""R043.10 durable model/tool event projection over an injected publisher."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, run
from agent_utilities.graph.live_events import (
    LineageBinding,
    LineageIdentityError,
    lineage_relationship,
    run_step,
)


class Publisher:
    def __init__(self, fail_kind: str | None = None) -> None:
        self.events: list[tuple[str, dict]] = []
        self.fail_kind = fail_kind

    async def publish(self, kind: str, content: dict) -> None:
        if kind == self.fail_kind:
            raise OSError("private durable backend detail")
        self.events.append((kind, content))


@pytest.mark.spec("AU-BOUNDARY-R043.14")
def test_lineage_relationships_use_explicit_keys_and_fail_closed() -> None:
    root = LineageBinding("run-root", "run", "same-name", plan_id="plan")
    child = LineageBinding(
        "run-child",
        "run",
        "same-name",
        plan_id="plan",
        parent_run_id="run-root",
        parent_node="same-name",
    )
    grandchild = LineageBinding(
        "run-grandchild",
        "run",
        "same-name",
        plan_id="plan",
        parent_run_id="run-child",
        parent_node="same-name",
    )
    spawn = lineage_relationship("spawn", root, child)
    nested_spawn = lineage_relationship("spawn", child, grandchild)
    dependency = lineage_relationship(
        "dependency",
        LineageBinding("run-root", "step", "same-name", plan_id="plan"),
        LineageBinding("run-root", "step", "join", plan_id="plan"),
    )

    assert root.key == ("run-root", "run", "same-name")
    assert child.key == ("run-child", "run", "same-name")
    assert spawn["type"] == nested_spawn["type"] == "spawn"
    assert lineage_relationship("spawn", root, child)["edge_id"] == spawn["edge_id"]
    assert spawn["edge_id"] != nested_spawn["edge_id"]
    assert dependency["type"] == "dependency"
    assert dependency["edge_id"] != spawn["edge_id"]

    with pytest.raises(LineageIdentityError):
        lineage_relationship(
            "spawn",
            root,
            LineageBinding("unbound", "run", "same-name"),
        )


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_executor_binds_one_publisher_before_exposing_model_output() -> None:
    bindings: list[str] = []
    publisher = Publisher()

    @asynccontextmanager
    async def scope(request):
        bindings.append(request.run_id)
        yield publisher

    host_ports.register_durable_run_event_factory(scope)

    async def stream(messages, info):
        yield "answer"

    result = await run(
        RunRequest(task="hello", run_id="run"),
        plan=RunPlan(model=FunctionModel(stream_function=stream), steps=("hello",)),
    )

    assert result.status == "completed"
    assert result.output == "answer"
    assert bindings == ["run"]
    assert [kind for kind, _ in publisher.events] == ["text_delta", "state"]
    assert publisher.events[-1] == (
        "state",
        {"step": 1, "state": "model_step_completed"},
    )


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_executor_stops_before_tool_effect_on_call_publication_failure() -> None:
    effects: list[int] = []
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect(value: int) -> str:
        effects.append(value)
        return f"observed:{value}"

    async def stream(messages, info):
        yield {
            0: DeltaToolCall(
                name="effect", json_args='{"value":7}', tool_call_id="call-one"
            )
        }

    publisher = Publisher("tool_call")

    @asynccontextmanager
    async def scope(request):
        yield publisher

    host_ports.register_durable_run_event_factory(scope)
    result = await run(
        RunRequest(task="do", run_id="run"),
        plan=RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=("do",),
            toolsets=(toolset,),
        ),
    )

    assert result.status == "failed"
    assert result.error == "RuntimeError: durable run event publication failed"
    assert effects == []


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_executor_without_host_publisher_keeps_ordinary_path() -> None:
    calls: list[list] = []

    def ordinary(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("ordinary")])

    result = await run(
        RunRequest(task="hello", run_id="run"),
        plan=RunPlan(model=FunctionModel(ordinary), steps=("hello",)),
    )

    assert result.status == "completed"
    assert result.output == "ordinary"
    assert len(calls) == 1


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_text_deltas_are_published_in_order_and_keep_model_output() -> None:
    async def stream(messages, info):
        for text in ("hello", " ", "world"):
            yield text

    publisher = Publisher()
    agent = Agent(FunctionModel(stream_function=stream))
    result = await run_step(agent, "hello", publisher=publisher, step=1)

    assert result.output == "hello world"
    assert (
        "".join(
            body["delta"] for kind, body in publisher.events if kind == "text_delta"
        )
        == result.output
    )
    assert publisher.events[-1] == (
        "state",
        {"step": 1, "state": "model_step_completed"},
    )


@pytest.mark.spec("AU-BOUNDARY-R043.10")
@pytest.mark.parametrize("fail_kind", [None, "tool_call", "tool_result"])
async def test_tool_publication_orders_effects_and_never_retries(
    fail_kind: str | None,
) -> None:
    effects: list[int] = []
    calls: list[list] = []
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect(value: int) -> str:
        effects.append(value)
        return f"observed:{value}"

    async def stream(messages, info):
        calls.append(messages)
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
    agent = Agent(FunctionModel(stream_function=stream), toolsets=(toolset,))
    if fail_kind is None:
        result = await run_step(agent, "do", publisher=publisher, step=1)
    else:
        with pytest.raises(RuntimeError, match="durable run event publication failed"):
            await run_step(agent, "do", publisher=publisher, step=1)
        result = None

    assert len(effects) <= 1
    if fail_kind is None:
        assert result is not None and result.output == "done"
        assert effects == [7]
        assert [kind for kind, _ in publisher.events] == [
            "tool_call",
            "tool_result",
            "text_delta",
            "state",
        ]
        tool_call, tool_result = publisher.events[:2]
        assert tool_call[1]["tool_call_id"] == tool_result[1]["tool_call_id"]
        assert tool_call[1]["args_valid"] is True
        assert tool_result[1]["result"] == "observed:7"
        assert len(calls) == 2
    elif fail_kind == "tool_call":
        assert effects == []
        assert calls and not any(kind == "tool_result" for kind, _ in publisher.events)
    else:
        assert effects == [7]
        assert len(calls) == 1
        assert not any(kind == "state" for kind, _ in publisher.events)


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_unbound_step_keeps_ordinary_model_path() -> None:
    calls: list[list] = []

    def ordinary(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("ordinary")])

    agent = Agent(FunctionModel(ordinary))
    result = await run_step(agent, "hello", publisher=None, step=1)
    assert result.output == "ordinary"
    assert len(calls) == 1
