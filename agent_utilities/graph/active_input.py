"""Apply host-claimed child input at the model-request boundary."""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext

from agent_utilities.host_ports import (
    ActiveChildBinding,
    ActiveChildInputPort,
    ActiveChildMessage,
)

_MAX_BATCH = 16
_MAX_TEXT_BYTES = 16_384
_MAX_ID_BYTES = 512


def _text(value: Any, limit: int) -> bool:
    return (
        type(value) is str
        and bool(value.strip())
        and len(value.encode("utf-8")) <= limit
    )


def _binding(value: Any, run_id: str) -> bool:
    return (
        isinstance(value, ActiveChildBinding)
        and value.child_run_id == run_id
        and _text(value.child_run_id, _MAX_ID_BYTES)
        and _text(value.parent_run_id, _MAX_ID_BYTES)
        and value.parent_run_id != value.child_run_id
        and _text(value.admission_id, _MAX_ID_BYTES)
        and type(value.lifecycle_revision) is int
        and value.lifecycle_revision > 0
    )


class ActiveChildInput(AbstractCapability[Any]):
    """The host owns claims, deduplication, terminal refusal, and receipts."""

    def __init__(self, port: ActiveChildInputPort, run_id: str) -> None:
        if not _binding(port.binding, run_id):
            raise ValueError("active child input binding refused")
        self._port = port
        self._binding = port.binding

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        try:
            batch = await self._port.take_ready()
        except Exception:
            raise RuntimeError("active child input claim unavailable") from None
        if type(batch) is not tuple or len(batch) > _MAX_BATCH:
            raise ValueError("active child input batch refused")
        ids: set[str] = set()
        events: set[str] = set()
        for item in batch:
            if (
                not isinstance(item, ActiveChildMessage)
                or item.binding != self._binding
                or not _text(item.message_id, _MAX_ID_BYTES)
                or not _text(item.event_id, _MAX_ID_BYTES)
                or not _text(item.content, _MAX_TEXT_BYTES)
                or item.message_id in ids
                or item.event_id in events
            ):
                raise ValueError("active child input message refused")
            ids.add(item.message_id)
            events.add(item.event_id)
        for item in batch:
            message = ModelRequest(parts=[UserPromptPart(content=item.content)])
            request_context.messages = [*request_context.messages, message]
            ctx.messages.append(message)
            try:
                await self._port.report_applied(item)
            except Exception:
                raise RuntimeError("active child input receipt uncertain") from None
        return request_context
