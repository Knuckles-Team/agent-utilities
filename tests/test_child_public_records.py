"""R043.17 bounded public child records with provenance."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities.graph.child_public_records import (
    ChildIdentity,
    ChildPublicationError,
    ChildPublicRecordPublisher,
)
from agent_utilities.graph.live_events import run_step

CHILD = ChildIdentity("child-1", "conv-1")


class Publisher:
    def __init__(self, fail_kind: str | None = None) -> None:
        self.records: list[tuple[str, dict]] = []
        self.fail_kind = fail_kind

    async def publish(self, kind: str, content: dict) -> None:
        if kind == self.fail_kind:
            raise OSError("private durable backend detail")
        self.records.append((kind, content))


def _agent(effects: list[int], *, secret_args: bool = True) -> Agent:
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect(value: int) -> str:
        effects.append(value)
        return "RAW-RESULT-SECRET"

    state = {"called": False}

    async def stream(messages, info):
        if not state["called"]:
            state["called"] = True
            yield {0: DeltaThinkingPart(content="PRIVATE-REASONING")}
            yield {
                1: DeltaToolCall(
                    name="effect", json_args='{"value":7}', tool_call_id="c1"
                )
            }
        else:
            yield "public answer"

    return Agent(FunctionModel(stream_function=stream), toolsets=[toolset])


async def _drive(
    publisher: Publisher, effects: list[int]
) -> ChildPublicRecordPublisher:
    records = ChildPublicRecordPublisher(publisher, CHILD)
    await run_step(_agent(effects), "go", publisher=None, step=1, child_records=records)
    return records


async def test_records_ordered_with_child_identity_and_nothing_private() -> None:
    publisher, effects = Publisher(), []
    records = await _drive(publisher, effects)
    kinds = [kind for kind, _ in publisher.records]
    assert kinds == ["tool_activity", "tool_activity", "public_text"]
    assert [c["status"] for k, c in publisher.records if k == "tool_activity"] == [
        "called",
        "returned",
    ]
    events = [c["source_event"] for _, c in publisher.records]
    assert events == sorted(events, key=lambda e: int(e.rsplit(":", 1)[1]))
    for _, content in publisher.records:
        assert content["child_id"] == "child-1"
        assert content["conversation_id"] == "conv-1"
        assert content["visibility"] == "public"
    blob = repr(publisher.records)
    assert "PRIVATE-REASONING" not in blob
    assert "RAW-RESULT-SECRET" not in blob
    assert '"value"' not in blob and "'value'" not in blob
    assert effects == [7]
    assert records.transcript_status == "public_records"


async def test_publication_failure_does_not_repeat_effect_or_claim_transcript() -> None:
    # A rejected call record stops the run before the effect.
    publisher, effects = Publisher("tool_activity"), []
    records = ChildPublicRecordPublisher(publisher, CHILD)
    with pytest.raises(ChildPublicationError):
        await run_step(
            _agent(effects), "go", publisher=None, step=1, child_records=records
        )
    assert effects == []
    assert records.transcript_status == "interrupted"
    with pytest.raises(ChildPublicationError):
        await records.publish_text("later", source_event="step:1:event:9")

    # A rejected result record fails after the effect and never repeats it.
    class AfterEffect(Publisher):
        async def publish(self, kind, content):
            if content.get("status") == "returned":
                raise OSError("down")
            await super().publish(kind, content)

    effects = []
    records = ChildPublicRecordPublisher(AfterEffect(), CHILD)
    with pytest.raises(ChildPublicationError):
        await run_step(
            _agent(effects), "go", publisher=None, step=1, child_records=records
        )
    assert effects == [7]
    assert records.transcript_status == "interrupted"


async def test_explicit_summary_and_unbound_run() -> None:
    publisher = Publisher()
    records = ChildPublicRecordPublisher(publisher, CHILD)
    await records.publish_summary("done", source_event="step:1:event:3")
    assert publisher.records[0][1]["kind"] == "public_summary"
    with pytest.raises(ValueError):
        await records._publish("reasoning", "x", {"text": "private"})

    unbound = ChildPublicRecordPublisher(None, CHILD)
    effects: list[int] = []
    await run_step(_agent(effects), "go", publisher=None, step=1, child_records=unbound)
    assert effects == [7]
    assert unbound.transcript_status == "unavailable"


@pytest.mark.spec("MERISTEM-BOUNDARY-R043.17")
async def test_run_loop_builds_child_publisher_from_host_child_identity() -> None:
    from contextlib import asynccontextmanager

    from agent_utilities import host_ports
    from agent_utilities.graph import RunPlan, RunRequest, run

    publisher = Publisher()
    effects: list[int] = []

    @asynccontextmanager
    async def scope(request):
        yield publisher

    host_ports.register_durable_run_event_factory(scope)
    try:
        agent = _agent(effects)
        result = await run(
            RunRequest(
                task="go",
                run_id="run",
                metadata={"child_id": "child-1", "conversation_id": "conv-1"},
            ),
            plan=RunPlan(
                model=agent.model,
                steps=("go",),
                toolsets=tuple(agent.toolsets[1:]),
            ),
        )
    finally:
        host_ports.register_durable_run_event_factory(None)

    assert result.status == "completed"
    assert effects == [7]
    kinds = {kind for kind, _ in publisher.records}
    assert kinds <= {"public_text", "tool_activity", "state"}
    assert "tool_activity" in kinds
    child = [c for k, c in publisher.records if k != "state"]
    assert child and all(c["child_id"] == "child-1" for c in child)
    assert all(c["visibility"] == "public" for c in child)
    dump = repr(publisher.records)
    assert "PRIVATE-REASONING" not in dump and "RAW-RESULT-SECRET" not in dump


@pytest.mark.spec("MERISTEM-BOUNDARY-R043.17")
async def test_run_loop_without_child_identity_keeps_ordinary_projection() -> None:
    from contextlib import asynccontextmanager

    from agent_utilities import host_ports
    from agent_utilities.graph import RunPlan, RunRequest, run

    publisher = Publisher()

    @asynccontextmanager
    async def scope(request):
        yield publisher

    async def stream(messages, info):
        yield "answer"

    host_ports.register_durable_run_event_factory(scope)
    try:
        await run(
            RunRequest(task="go", run_id="run"),
            plan=RunPlan(model=FunctionModel(stream_function=stream), steps=("go",)),
        )
    finally:
        host_ports.register_durable_run_event_factory(None)
    assert [k for k, _ in publisher.records] == ["text_delta", "state"]
