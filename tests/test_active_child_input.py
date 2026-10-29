"""The host owns admission and durable message receipts."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports
from agent_utilities.graph import RunBudget, RunPlan, RunRequest, run


class Port:
    def __init__(self) -> None:
        self.binding = host_ports.ActiveChildBinding("child", "parent", "admission", 1)
        self.pending: list[host_ports.ActiveChildMessage] = []
        self.claimed: set[str] = set()
        self.receipts: list[str] = []
        self.fail = False

    async def take_ready(self) -> tuple[host_ports.ActiveChildMessage, ...]:
        # This fake represents the host's durable claim/deduplication owner.
        batch = []
        for item in self.pending:
            if item.message_id not in self.claimed:
                self.claimed.add(item.message_id)
                batch.append(item)
        self.pending.clear()
        return tuple(batch)

    async def report_applied(self, message: host_ports.ActiveChildMessage) -> None:
        if self.fail:
            raise OSError("receipt unavailable")
        self.receipts.append(message.event_id)

    def message(self, content: str = "steer") -> host_ports.ActiveChildMessage:
        return host_ports.ActiveChildMessage(self.binding, "message", "event", content)


def bind(port: Port) -> None:
    @asynccontextmanager
    async def factory(request):
        yield port

    host_ports.register_active_child_input_factory(factory)


@pytest.fixture(autouse=True)
def reset_port():
    yield
    host_ports.register_active_child_input_factory(None)


async def test_input_waits_for_next_model_request_and_preserves_budget() -> None:
    port = Port()
    bind(port)
    started, release = asyncio.Event(), asyncio.Event()
    observations: list[list[str]] = []

    async def model(messages, info):
        observations.append(
            [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            ]
        )
        if len(observations) == 1:
            started.set()
            await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    task = asyncio.create_task(
        run(
            RunRequest(task="start", run_id="child"),
            plan=RunPlan(
                model=FunctionModel(model),
                steps=("one", "two"),
                budget=RunBudget(request_limit=2),
            ),
        )
    )
    await started.wait()
    port.pending.extend([port.message(), port.message()])
    assert port.receipts == []
    release.set()
    result = await task
    assert result.status == "completed"
    assert "steer" not in observations[0]
    assert observations[1].count("steer") == 1
    assert port.receipts == ["event"]
    assert result.usage["requests"] == 2


async def test_wrong_child_or_failed_receipt_refuses_before_model() -> None:
    for wrong_binding, fail_receipt in [(True, False), (False, True)]:
        port = Port()
        if wrong_binding:
            port.binding = host_ports.ActiveChildBinding(
                "other", "parent", "admission", 1
            )
        else:
            port.pending.append(port.message())
            port.fail = fail_receipt
        bind(port)
        calls = 0

        def model(messages, info):
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart("unexpected")])

        result = await run(
            RunRequest(task="start", run_id="child"),
            plan=RunPlan(model=FunctionModel(model), steps=("one",)),
        )
        assert result.status == "failed"
        assert calls == 0
        assert port.receipts == []


async def test_input_is_applied_between_internal_tool_and_model_turns() -> None:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.toolsets import FunctionToolset

    port = Port()
    bind(port)
    tool_started, release = asyncio.Event(), asyncio.Event()
    tools = FunctionToolset()
    prompts: list[list[str]] = []

    @tools.tool_plain
    async def effect() -> str:
        tool_started.set()
        await release.wait()
        return "done"

    def model(messages, info):
        prompts.append(
            [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            ]
        )
        return ModelResponse(
            parts=[ToolCallPart("effect", {})]
            if len(prompts) == 1
            else [TextPart("done")]
        )

    task = asyncio.create_task(
        run(
            RunRequest(task="start", run_id="child"),
            plan=RunPlan(
                model=FunctionModel(model),
                steps=("one",),
                toolsets=(tools,),
                budget=RunBudget(request_limit=2, tool_calls_limit=1),
            ),
        )
    )
    await tool_started.wait()
    port.pending.append(port.message())
    assert not port.receipts
    release.set()
    result = await task
    assert result.status == "completed"
    assert "steer" not in prompts[0]
    assert prompts[1].count("steer") == 1
    assert port.receipts == ["event"]
    assert result.usage["requests"] == 2
    assert result.usage["tool_calls"] == 1


async def test_cancellation_does_not_apply_queued_input() -> None:
    port = Port()
    closed = asyncio.Event()
    started = asyncio.Event()
    blocker = asyncio.Event()

    @asynccontextmanager
    async def factory(request):
        try:
            yield port
        finally:
            closed.set()

    host_ports.register_active_child_input_factory(factory)

    async def model(messages, info):
        started.set()
        await blocker.wait()
        return ModelResponse(parts=[TextPart("unexpected")])

    task = asyncio.create_task(
        run(
            RunRequest(task="start", run_id="child"),
            plan=RunPlan(model=FunctionModel(model), steps=("one", "two")),
        )
    )
    await started.wait()
    port.pending.append(port.message())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert not port.receipts
    assert len(port.pending) == 1


async def test_checkpointed_input_refuses_before_model() -> None:
    port = Port()
    bind(port)
    calls: list[str] = []

    class Checkpoint:
        admission_identity = "admission"

        async def restore(self):
            return None

        async def save(self, value):
            raise AssertionError("must not write checkpoint")

    @asynccontextmanager
    async def checkpoint(request, plan):
        yield Checkpoint()

    host_ports.register_run_checkpoint_factory(checkpoint)
    try:

        def model(messages, info):
            calls.append("model")
            return ModelResponse(parts=[TextPart("unexpected")])

        result = await run(
            RunRequest(task="start", run_id="child"),
            plan=RunPlan(model=FunctionModel(model), steps=("one",)),
        )
        assert result.status == "failed"
        assert calls == [] and port.receipts == []
    finally:
        host_ports.register_run_checkpoint_factory(None)


async def test_failed_receipt_after_tool_never_repeats_effect() -> None:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.toolsets import FunctionToolset

    port = Port()
    bind(port)
    effects: list[str] = []
    tools = FunctionToolset()

    @tools.tool_plain
    def effect() -> str:
        effects.append("effect")
        port.pending.append(port.message())
        port.fail = True
        return "done"

    calls = 0

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("effect", {})])

    result = await run(
        RunRequest(task="start", run_id="child"),
        plan=RunPlan(model=FunctionModel(model), steps=("one",), toolsets=(tools,)),
    )
    assert result.status == "failed"
    assert "receipt uncertain" in result.error
    assert effects == ["effect"] and calls == 1


async def test_entire_invalid_batch_refuses_before_receipts_or_model() -> None:
    from dataclasses import replace

    port = Port()
    bind(port)
    port.pending.extend(
        [
            port.message("valid"),
            replace(
                port.message("foreign"),
                message_id="other",
                event_id="other",
                binding=replace(port.binding, lifecycle_revision=2),
            ),
        ]
    )
    calls = 0

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("unexpected")])

    result = await run(
        RunRequest(task="start", run_id="child"),
        plan=RunPlan(model=FunctionModel(model), steps=("one",)),
    )
    assert result.status == "failed"
    assert port.receipts == [] and calls == 0


async def test_host_order_is_preserved_and_completion_does_not_reopen_port() -> None:
    from dataclasses import replace

    port = Port()
    port.pending.extend(
        [
            port.message("first"),
            replace(port.message("second"), message_id="message-2", event_id="event-2"),
        ]
    )
    bind(port)
    observations: list[str] = []

    def model(messages, info):
        observations.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        return ModelResponse(parts=[TextPart("done")])

    result = await run(
        RunRequest(task="start", run_id="child"),
        plan=RunPlan(model=FunctionModel(model), steps=("one",)),
    )
    assert result.status == "completed"
    assert observations[-2:] == ["first", "second"]
    assert port.receipts == ["event", "event-2"]
    port.pending.append(
        replace(port.message("late"), message_id="late", event_id="late")
    )
    assert port.receipts == ["event", "event-2"]
