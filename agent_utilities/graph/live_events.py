"""Project model and tool stream events through an awaited host publisher.

The host owns the durable event record.  This module only validates the small
public projection emitted by the pydantic-ai stream and awaits the host write
before allowing the model run to continue.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

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


class LineageIdentityError(ValueError):
    """A nested event lacks an admitted identity or has mismatched parents."""


@dataclass(frozen=True, slots=True)
class LineageBinding:
    """The producer-declared identity needed to address one local node.

    ``(run_id, node_kind, local_id)`` is the stable local key.  Parent fields
    are deliberately optional for roots and are required together for nested
    nodes.  AU validates a binding it receives; it never derives one from
    depth, an environment object, or a local task name.
    """

    run_id: str
    node_kind: str
    local_id: str
    plan_id: str | None = None
    parent_run_id: str | None = None
    parent_node: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "node_kind", "local_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise LineageIdentityError(f"{field_name} must be a nonempty string")
        if self.plan_id is not None and (
            type(self.plan_id) is not str or not self.plan_id.strip()
        ):
            raise LineageIdentityError(
                "plan_id must be a nonempty string when supplied"
            )
        parent_fields = (self.parent_run_id, self.parent_node)
        if any(value is not None for value in parent_fields) and not all(
            type(value) is str and bool(value.strip()) for value in parent_fields
        ):
            raise LineageIdentityError(
                "parent_run_id and parent_node must be supplied together"
            )

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the stable ``(run_id, node_kind, local_id)`` key."""

        return (self.run_id, self.node_kind, self.local_id)

    def as_dict(self) -> dict[str, str]:
        """Return only producer-declared identity fields for public events."""

        value = {
            "run_id": self.run_id,
            "node_kind": self.node_kind,
            "local_id": self.local_id,
        }
        if self.plan_id is not None:
            value["plan_id"] = self.plan_id
        if self.parent_run_id is not None:
            assert self.parent_node is not None
            value["parent_run_id"] = self.parent_run_id
            value["parent_node"] = self.parent_node
        return value


def lineage_relationship(
    relationship: str,
    source: LineageBinding,
    target: LineageBinding,
) -> dict[str, Any]:
    """Build one explicit, replay-stable spawn or dependency relationship."""

    if relationship not in {"spawn", "dependency"}:
        raise ValueError("relationship must be spawn or dependency")
    if relationship == "spawn":
        if (
            target.run_id == source.run_id
            or target.parent_run_id != source.run_id
            or target.parent_node != source.local_id
        ):
            raise LineageIdentityError(
                "spawn target is not explicitly bound to its source node"
            )
    elif target.run_id != source.run_id or source.plan_id != target.plan_id:
        raise LineageIdentityError(
            "dependency endpoints must belong to the same admitted run and plan"
        )
    edge_key = "agent-utilities:lineage:{relationship}:{source}:{target}".format(
        relationship=relationship,
        source="/".join(source.key),
        target="/".join(target.key),
    )
    return {
        "type": relationship,
        "relationship": relationship,
        "edge_id": uuid5(NAMESPACE_URL, edge_key).hex,
        "source": source.as_dict(),
        "target": target.as_dict(),
    }


def lineage_event_fields(
    binding: LineageBinding,
    relationships: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    """Project explicit identity and relationships into an existing event."""

    fields: dict[str, Any] = {
        "node_kind": binding.node_kind,
        "local_id": binding.local_id,
        "lineage": binding.as_dict(),
    }
    if binding.plan_id is not None:
        fields["lineage_plan_id"] = binding.plan_id
    if binding.parent_run_id is not None:
        fields["lineage_parent_run_id"] = binding.parent_run_id
        fields["lineage_parent_node"] = binding.parent_node
    if relationships:
        fields["lineage_relationships"] = [dict(item) for item in relationships]
    return fields


def emit_lineage_event(
    kind: str,
    binding: LineageBinding,
    sequence: int,
    *,
    relationships: tuple[Mapping[str, Any], ...] = (),
    publish: bool = True,
    **fields: Any,
) -> dict[str, Any]:
    """Create and optionally publish a stable event for an explicit binding."""

    if type(sequence) is not int or sequence < 1:
        raise ValueError("sequence must be a positive integer")
    event = {
        "type": kind,
        "run_id": binding.run_id,
        "sequence": sequence,
        "event_id": uuid5(
            NAMESPACE_URL,
            f"agent-utilities:run:{binding.run_id}:event:{sequence}",
        ).hex,
        **({"plan_id": binding.plan_id} if binding.plan_id is not None else {}),
        **(
            {
                "parent_run_id": binding.parent_run_id,
                "parent_node": binding.parent_node,
            }
            if binding.parent_run_id is not None
            else {}
        ),
        **lineage_event_fields(binding, relationships),
        **fields,
    }
    if publish:
        from agent_utilities import host_ports

        host_ports.emit_run_event(event)
    return event


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
    after_result: Callable[[Any], Awaitable[None]] | None = None,
    event_fields: Mapping[str, Any] | None = None,
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
        result = await agent.run(prompt, **kwargs)
        if after_result is not None:
            await after_result(result)
        return result

    fields = dict(event_fields or {})

    async def receive(context: Any, events: AsyncIterable[Any]) -> None:
        async def projected_events() -> AsyncIterable[Any]:
            async for event in events:
                projected = _projection(event)
                if publisher is not None and projected is not None:
                    kind, content = projected
                    try:
                        await publisher.publish(
                            kind, {"step": step, **fields, **content}
                        )
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
    if after_result is not None:
        await after_result(result)
    if publisher is not None:
        try:
            # This is a model-step observation, not a committed WorkItem
            # terminal event; only the host's native commit establishes that.
            await publisher.publish(
                "state",
                {"step": step, **fields, "state": "model_step_completed"},
            )
        except Exception as exc:
            raise RuntimeError("durable run event publication failed") from exc
    return result


__all__ = [
    "DurableRunEventPublisher",
    "LineageBinding",
    "LineageIdentityError",
    "emit_lineage_event",
    "lineage_event_fields",
    "lineage_relationship",
    "run_step",
]
