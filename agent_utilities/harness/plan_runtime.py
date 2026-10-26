"""Run-local scaffold tool and request-only injection into the model boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities.host_ports import ScaffoldPublisher

from .plan import PlanItem, PlanState

_ITEMS = TypeAdapter(list[PlanItem])


def _without_request_payload(part: ModelRequestPart) -> ModelRequestPart:
    if isinstance(part, ToolReturnPart) and part.tool_name == "update_plan":
        return replace(part, content={"status": "scaffold_observation"})
    if isinstance(part, RetryPromptPart) and part.tool_name == "update_plan":
        return replace(part, content="Invalid scaffold update.")
    return part


def _without_scaffold_payload(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Retain tool pairing while dropping scaffold arguments and observations."""
    cleaned: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [
                _without_request_payload(part)
                for part in message.parts
                if not (
                    isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content.startswith("Current run plan:")
                )
            ]
            # A request containing only the prior ephemeral rendering has no
            # durable meaning. Drop it instead of leaving an empty history turn.
            if not parts:
                continue
            cleaned.append(
                replace(
                    message,
                    parts=parts,
                )
            )
        else:
            parts = [
                replace(part, args={})
                if isinstance(part, ToolCallPart) and part.tool_name == "update_plan"
                else part
                for part in message.parts
            ]
            cleaned.append(replace(message, parts=parts))
    return cleaned


def _without_plan_injections(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove only prior ephemeral renderings from an outgoing request.

    The current tool return is still part of the model's immediate request and
    must remain visible. ``_without_scaffold_payload`` is reserved for the
    durable run history, where both the plan arguments and observations are
    redacted.
    """
    cleaned: list[ModelMessage] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            cleaned.append(message)
            continue
        parts = [
            part
            for part in message.parts
            if not (
                isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and part.content.startswith("Current run plan:")
            )
        ]
        if parts:
            cleaned.append(replace(message, parts=parts))
    return cleaned


class PlanScaffold(AbstractCapability[Any]):
    """One scaffold shared by a run's update tool and outgoing request hook."""

    def __init__(
        self, state: PlanState, publisher: ScaffoldPublisher | None = None
    ) -> None:
        self.state = state
        self.publisher = publisher
        self._history: list[ModelMessage] | None = None

    async def for_run(self, ctx: RunContext[Any]) -> PlanScaffold:
        """Keep cleanup tracking local to each agent call, sharing its run scaffold."""
        return PlanScaffold(self.state, self.publisher)

    def get_toolset(self) -> FunctionToolset[Any]:
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.update_plan, sequential=True)
        return toolset

    async def update_plan(self, items: Any) -> dict[str, Any]:
        """Replace the whole plan with ordered text/status items.

        Status must be pending, in_progress or done. This tool does not select
        agents, skills, tools or models.
        """
        try:
            values = _ITEMS.validate_python(items, strict=True)
            revision, current = self.state.replace_versioned(values)
        except ValidationError:
            return {"status": "invalid_plan"}
        if self.publisher is not None:
            await self.publisher.publish(revision, current)
        return {"status": "updated", "items": [item.model_dump() for item in current]}

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        # Keep the real tool observation in this outgoing request only. Replacing
        # parts instead of mutating them leaves that request's shallow copy intact.
        self._history = ctx.messages
        ctx.messages[:] = _without_scaffold_payload(ctx.messages)
        rendered = json.dumps([item.model_dump() for item in self.state.items])
        injection = ModelRequest(
            parts=[UserPromptPart(f"Current run plan: {rendered}")]
        )
        # Assignment through replace is request-only. Appending to the framework's
        # special request list would also append to persistent history.
        request_messages = _without_plan_injections(list(request_context.messages))
        return replace(request_context, messages=[*request_messages, injection])

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        try:
            return await handler()
        finally:
            if self._history is not None:
                self._history[:] = _without_scaffold_payload(self._history)
                self._history = None
