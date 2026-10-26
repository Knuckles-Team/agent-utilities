"""Full public-run durable event failure and cancellation contracts for R043.10."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, iterate_run, run


class Publisher:
    def __init__(self, fail_kind: str | None = None) -> None:
        self.events: list[tuple[str, dict]] = []
        self.fail_kind = fail_kind

    async def publish(self, kind: str, content: dict) -> None:
        if kind == self.fail_kind:
            raise OSError("private durable backend detail")
        self.events.append((kind, content))


def bind(publisher: Publisher, closed: asyncio.Event | None = None) -> None:
    @asynccontextmanager
    async def scope(request):
        try:
            yield publisher
        finally:
            if closed is not None:
                closed.set()

    host_ports.register_durable_run_event_factory(scope)


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_full_executor_stops_after_failed_result_publication_without_retry() -> (
    None
):
    effects: list[int] = []
    model_calls: list[list] = []
    toolset = FunctionToolset()

    @toolset.tool_plain
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

    publisher = Publisher("tool_result")
    bind(publisher)
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
    assert effects == [7]
    assert len(model_calls) == 1
    assert not any(kind == "state" for kind, _ in publisher.events)
    assert "private durable backend detail" not in (result.error or "")


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_full_executor_propagates_publisher_cancellation_and_closes_scope() -> (
    None
):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    closed = asyncio.Event()
    model_calls: list[list] = []
    publisher = Publisher()

    async def publish(kind: str, content: dict) -> None:
        publisher.events.append((kind, content))
        if kind == "text_delta":
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    publisher.publish = publish  # type: ignore[method-assign]
    bind(publisher, closed)

    async def stream(messages, info):
        model_calls.append(messages)
        yield "answer"

    task = asyncio.create_task(
        run(
            RunRequest(task="hello", run_id="run"),
            plan=RunPlan(model=FunctionModel(stream_function=stream), steps=("hello",)),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    assert closed.is_set()
    assert len(model_calls) == 1
    assert not any(kind == "state" for kind, _ in publisher.events)


@pytest.mark.spec("AU-BOUNDARY-R043.10")
async def test_public_state_ack_precedes_node_transition() -> None:
    order: list[str] = []
    publisher = Publisher()

    async def publish(kind: str, content: dict) -> None:
        publisher.events.append((kind, content))
        order.append(f"publisher:{kind}")

    publisher.publish = publish  # type: ignore[method-assign]
    bind(publisher)

    async def stream(messages, info):
        yield "answer"

    emitted = []
    async for event in iterate_run(
        RunRequest(task="hello", run_id="run"),
        plan=RunPlan(model=FunctionModel(stream_function=stream), steps=("hello",)),
    ):
        emitted.append(event)
        order.append(f"run:{event['type']}")

    assert emitted[-1]["type"] == "graph_complete"
    assert order.index("publisher:state") < order.index("run:node_transition")
