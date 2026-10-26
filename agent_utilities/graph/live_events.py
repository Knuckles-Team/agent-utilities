"""Project model and tool stream events through an awaited host publisher.

The host owns the durable event record.  This module only validates the small
public projection emitted by the pydantic-ai stream and awaits the host write
before allowing the model run to continue.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from typing import Any, Protocol

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_core import to_jsonable_python


class DurableRunEventPublisher(Protocol):
    """Host-owned durable writer injected into one run step."""

    async def publish(self, kind: str, content: Mapping[str, Any]) -> None:
        """Persist one validated public model/tool observation."""
        ...


def _projection(event: Any) -> tuple[str, dict[str, Any]] | None:
    """Return the safe public projection for one native model event."""
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
    prompt: str | list[Any],
    *,
    publisher: DurableRunEventPublisher | None,
    step: int,
    stream_observer: Any = None,
    **kwargs: Any,
) -> Any:
    """Run exactly once, awaiting durable publication of each public event.

    ``stream_observer`` is the existing trusted native observer.  When both
    ports are bound, each native event is published first and then forwarded to
    the observer through the same async stream.  A publisher failure aborts the
    model call before a tool effect for a rejected tool-call event; a result
    publication failure aborts after that effect and never retries it.
    """
    if publisher is None and stream_observer is None:
        return await agent.run(prompt, **kwargs)

    async def receive(context: Any, events: AsyncIterable[Any]) -> None:
        async def projected_events() -> AsyncIterable[Any]:
            async for event in events:
                projected = _projection(event)
                if publisher is not None and projected is not None:
                    kind, content = projected
                    try:
                        await publisher.publish(kind, {"step": step, **content})
                    except Exception as exc:
                        raise RuntimeError(
                            "durable run event publication failed"
                        ) from exc
                yield event

        if stream_observer is not None:
            await stream_observer(context, projected_events())
        else:
            async for _ in projected_events():
                pass

    result = await agent.run(
        prompt,
        event_stream_handler=receive,
        **kwargs,
    )
    if publisher is not None:
        try:
            # This is a model-step observation, not a committed WorkItem
            # terminal event; only the host's native commit establishes that.
            await publisher.publish(
                "state", {"step": step, "state": "model_step_completed"}
            )
        except Exception as exc:
            raise RuntimeError("durable run event publication failed") from exc
    return result


__all__ = ["DurableRunEventPublisher", "run_step"]
