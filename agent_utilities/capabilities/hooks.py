#!/usr/bin/python
# coding: utf-8
"""Lifecycle hooks capability with knowledge graph integration.

Provides PRE_TOOL_USE, POST_TOOL_USE, BEFORE_RUN, and AFTER_RUN hooks
for auditing, safety, and automatic graph tracing.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from ..models.knowledge_graph import RegistryNodeType, ToolCallNode

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
    tool_def: Optional[ToolDefinition] = None
    call: Optional[ToolCallPart] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None


@dataclass
class HookResult:
    modify_args: Optional[Dict[str, Any]] = None
    modify_result: Optional[Any] = None
    cancel: bool = False
    cancel_reason: Optional[str] = None


Hook = Callable[[HookInput], Union[HookResult, None]]


@dataclass
class HooksCapability(AbstractCapability[Any]):
    """Capability that executes registered hooks during the agent lifecycle.

    If auto_graph_trace is True, it automatically records tool calls as
    ToolCallNode entities in the knowledge graph.
    """

    hooks: List[Hook] = field(default_factory=list)
    auto_graph_trace: bool = True

    _tool_sessions: Dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    async def for_run(self, ctx: RunContext[Any]) -> HooksCapability:
        return replace(self)

    async def _run_hooks(self, input: HookInput) -> HookResult:
        final_result = HookResult()
        for hook in self.hooks:
            try:
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

    async def after_run(self, ctx: RunContext[Any]) -> None:
        await self._run_hooks(HookInput(event=HookEvent.AFTER_RUN, ctx=ctx))

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._tool_sessions[call.tool_call_id] = time.time()

        # Auto-trace to graph
        if self.auto_graph_trace:
            engine = getattr(ctx.deps, "graph_engine", None)
            if engine:
                node = ToolCallNode(
                    id=call.tool_call_id,
                    type=RegistryNodeType.TOOL_CALL,
                    name=f"Tool Call: {call.tool_name}",
                    tool_name=call.tool_name,
                    args=args,
                    importance_score=0.3,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                try:
                    engine.graph.add_node(node.id, **node.model_dump())
                    # Edge to episode if available
                    episode_id = getattr(ctx.deps, "episode_id", None)
                    if episode_id:
                        engine.graph.add_edge(episode_id, node.id, type="USED_TOOL")
                except Exception:
                    pass

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
        args: Dict[str, Any],
        result: Any,
    ) -> Any:
        start_time = self._tool_sessions.pop(call.tool_call_id, time.time())
        duration = time.time() - start_time

        # Update auto-trace in graph
        if self.auto_graph_trace:
            engine = getattr(ctx.deps, "graph_engine", None)
            if engine:
                try:
                    if call.tool_call_id in engine.graph:
                        engine.graph.nodes[call.tool_call_id]["result"] = str(result)[
                            :1000
                        ]
                        engine.graph.nodes[call.tool_call_id]["duration"] = duration
                        # Persistence would happen in the background or at the end of the run
                except Exception:
                    pass

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
