"""Bounded public child records with provenance (MERISTEM-BOUNDARY-R043.17).

A child run may publish three kinds of public record through the existing
:class:`~agent_utilities.host_ports.DurableRunEventPublisher`: public text,
tool activity and an explicit public summary.  Every record carries its
``kind``, the ``source_event`` it came from, its ``visibility`` and the
child identity (child id and conversation id).

Nothing private is public by default: model reasoning is never projected, a
summary is published only when the caller supplies it explicitly, and tool
activity carries the tool name, call id and status but never raw arguments or
results.  Publication failure follows R043.10: the failure propagates (before
a tool effect for a call record, after it for a result record), nothing is
retried, and no complete transcript is claimed.  With no publisher bound the
run advertises no durable transcript.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)

from agent_utilities.host_ports import DurableRunEventPublisher

PUBLIC_KINDS = frozenset({"public_text", "tool_activity", "public_summary"})
TranscriptStatus = Literal["unavailable", "public_records", "interrupted"]


class ChildPublicationError(RuntimeError):
    """A public child record could not be durably published."""


@dataclass(frozen=True, slots=True)
class ChildIdentity:
    """Child identity supplied by the host after admission."""

    child_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        for name in ("child_id", "conversation_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


class ChildPublicRecordPublisher:
    """Publish public records for one child, in order, once each."""

    def __init__(
        self, publisher: DurableRunEventPublisher | None, child: ChildIdentity
    ) -> None:
        self._publisher = publisher
        self._child = child
        self._failed = False

    @property
    def transcript_status(self) -> TranscriptStatus:
        """What may be advertised: never a complete transcript."""
        if self._publisher is None:
            return "unavailable"
        return "interrupted" if self._failed else "public_records"

    async def _publish(
        self, kind: str, source_event: str, content: Mapping[str, Any]
    ) -> None:
        if kind not in PUBLIC_KINDS:
            raise ValueError(f"not a public record kind: {kind!r}")
        if type(source_event) is not str or not source_event.strip():
            raise ValueError("source_event must be a nonempty string")
        if self._publisher is None:
            return
        if self._failed:
            raise ChildPublicationError("child public publication interrupted")
        record = {
            "kind": kind,
            "source_event": source_event,
            "visibility": "public",
            "child_id": self._child.child_id,
            "conversation_id": self._child.conversation_id,
            **content,
        }
        try:
            await self._publisher.publish(kind, record)
        except Exception as exc:
            self._failed = True
            raise ChildPublicationError(
                "durable child record publication failed"
            ) from exc

    async def publish_text(self, text: str, *, source_event: str) -> None:
        await self._publish("public_text", source_event, {"text": text})

    async def publish_tool_activity(
        self, *, name: str, tool_call_id: str, status: str, source_event: str
    ) -> None:
        await self._publish(
            "tool_activity",
            source_event,
            {"name": name, "tool_call_id": tool_call_id, "status": status},
        )

    async def publish_summary(self, text: str, *, source_event: str) -> None:
        """Publish a summary the caller explicitly marked public."""
        await self._publish("public_summary", source_event, {"summary": text})

    async def publish_native(self, event: Any, *, source_event: str) -> None:
        """Project one native model event; private events publish nothing."""
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            await self.publish_text(event.part.content, source_event=source_event)
        elif isinstance(event, PartDeltaEvent) and isinstance(
            event.delta, TextPartDelta
        ):
            await self.publish_text(
                event.delta.content_delta, source_event=source_event
            )
        elif isinstance(event, FunctionToolCallEvent):
            await self.publish_tool_activity(
                name=event.part.tool_name,
                tool_call_id=event.tool_call_id,
                status="called",
                source_event=source_event,
            )
        elif isinstance(event, FunctionToolResultEvent):
            await self.publish_tool_activity(
                name=event.part.tool_name,
                tool_call_id=event.tool_call_id,
                status="retry"
                if event.part.part_kind == "retry-prompt"
                else "returned",
                source_event=source_event,
            )


def child_records_for(
    request: Any, publisher: DurableRunEventPublisher | None
) -> ChildPublicRecordPublisher | None:
    """Build the run's child publisher when the host marked it a child run.

    The host supplies ``child_id`` and ``conversation_id`` in the request
    metadata after admission.  A run without both is not a child run and keeps
    the ordinary durable event projection.
    """
    metadata = getattr(request, "metadata", None) or {}
    child_id = metadata.get("child_id")
    conversation_id = metadata.get("conversation_id")
    if not isinstance(child_id, str) or not isinstance(conversation_id, str):
        return None
    return ChildPublicRecordPublisher(
        publisher, ChildIdentity(child_id, conversation_id)
    )


__all__ = [
    "PUBLIC_KINDS",
    "child_records_for",
    "ChildIdentity",
    "ChildPublicRecordPublisher",
    "ChildPublicationError",
]
