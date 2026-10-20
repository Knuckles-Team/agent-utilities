"""Project actual model/tool stream events through an awaited host publisher."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_core import to_jsonable_python

from agent_utilities.host_ports import DurableRunEventPublisher


def _projection(event: Any) -> tuple[str, dict[str, Any]] | None:
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return "text_delta", {
            "part_index": event.index,
            "delta": event.part.content,
            "reset": True,
        }
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return "text_delta", {
            "part_index": event.index,
            "delta": event.delta.content_delta,
            "reset": False,
        }
    if isinstance(event, FunctionToolCallEvent):
        return "tool_call", {
            "tool_call_id": event.tool_call_id,
            "name": event.part.tool_name,
            "arguments": to_jsonable_python(event.part.args),
            "args_valid": event.args_valid,
        }
    if isinstance(event, FunctionToolResultEvent):
        return "tool_result", {
            "tool_call_id": event.tool_call_id,
            "name": event.part.tool_name,
            "result": to_jsonable_python(event.part.content),
            "retry": event.part.part_kind == "retry-prompt",
        }
    return None


async def run_step(
    agent: Any,
    prompt: str,
    *,
    publisher: DurableRunEventPublisher | None,
    step: int,
    **kwargs: Any,
) -> Any:
    """Run exactly once, awaiting durable publication of each observed event."""
    if publisher is None:
        return await agent.run(prompt, **kwargs)

    async def receive(_context: Any, events: AsyncIterable[Any]) -> None:
        async for event in events:
            projected = _projection(event)
            if projected is not None:
                kind, content = projected
                try:
                    await publisher.publish(kind, {"step": step, **content})
                except Exception as exc:
                    raise RuntimeError("durable run event publication failed") from exc

    result = await agent.run(prompt, event_stream_handler=receive, **kwargs)
    # This is the model step's observation, not a committed WorkItem terminal event.
    try:
        await publisher.publish(
            "state", {"step": step, "state": "model_step_completed"}
        )
    except Exception as exc:
        raise RuntimeError("durable run event publication failed") from exc
    return result
