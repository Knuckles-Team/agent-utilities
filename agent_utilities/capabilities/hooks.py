#!/usr/bin/python
from __future__ import annotations

"""Lifecycle hooks capability.

CONCEPT:AU-ORCH.adapter.kg-graph-materialization

Provides PRE_TOOL_USE, POST_TOOL_USE, BEFORE_RUN, and AFTER_RUN hooks for
in-process auditing and safety policy. Durable execution tracing is owned by
``agent_utilities.observability.trace_ontology`` and its orchestration writer;
hooks never persist tool payloads.
"""


import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

logger = logging.getLogger(__name__)


class HookEvent(enum.Enum):
    BEFORE_RUN = "before_run"
    AFTER_RUN = "after_run"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"


@dataclass
class HookInput:
    event: HookEvent
    ctx: RunContext[Any]
    tool_def: ToolDefinition | None = None
    call: ToolCallPart | None = None
    args: dict[str, Any] | None = None
    result: Any | None = None
    error: Exception | None = None
    start_time: float | None = None


@dataclass
class HookResult:
    modify_args: dict[str, Any] | None = None
    modify_result: Any | None = None
    cancel: bool = False
    cancel_reason: str | None = None


Hook = Callable[[HookInput], HookResult | None]


@dataclass
class HooksCapability(AbstractCapability[Any]):
    """Execute registered, in-process hooks during the agent lifecycle."""

    hooks: list[Hook] = field(default_factory=list)

    _tool_sessions: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    async def for_run(self, ctx: RunContext[Any]) -> HooksCapability:
        return replace(self)

    async def _run_hooks(self, input: HookInput) -> HookResult:
        import inspect

        final_result = HookResult()
        for hook in self.hooks:
            try:
                if inspect.iscoroutinefunction(hook):
                    res = await hook(input)
                else:
                    res = hook(input)
                if res:
                    if res.modify_args:
                        final_result.modify_args = res.modify_args
                    if res.modify_result is not None:
                        final_result.modify_result = res.modify_result
                    if res.cancel:
                        final_result.cancel = True
                        final_result.cancel_reason = res.cancel_reason
            except Exception as e:
                logger.error(f"Error in hook {hook}: {e}")
        return final_result

    async def before_run(self, ctx: RunContext[Any]) -> None:
        await self._run_hooks(HookInput(event=HookEvent.BEFORE_RUN, ctx=ctx))

    async def after_run(
        self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        await self._run_hooks(
            HookInput(event=HookEvent.AFTER_RUN, ctx=ctx, result=result)
        )
        return result

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self._tool_sessions[call.tool_call_id] = time.time()

        res = await self._run_hooks(
            HookInput(
                event=HookEvent.PRE_TOOL_USE,
                ctx=ctx,
                tool_def=tool_def,
                call=call,
                args=args,
            )
        )

        if res.cancel:
            # We can't easily "cancel" from here without raising,
            # but we could modify args to be invalid or similar.
            # pydantic-ai usually handles this via ApprovalRequired.
            pass

        return res.modify_args or args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        start_time = self._tool_sessions.pop(call.tool_call_id, time.time())

        res = await self._run_hooks(
            HookInput(
                event=HookEvent.POST_TOOL_USE,
                ctx=ctx,
                tool_def=tool_def,
                call=call,
                args=args,
                result=result,
                start_time=start_time,
            )
        )

        return res.modify_result if res.modify_result is not None else result
