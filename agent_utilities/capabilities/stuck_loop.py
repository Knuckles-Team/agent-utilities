#!/usr/bin/python
"""Stuck loop detection capability with knowledge graph integration.

Detects repetitive agent behavior and intervenes, recording the event
as a SelfEvaluation node in the knowledge graph.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

# Guarded import for pydantic_ai.capabilities (not available in all versions)
try:
    from pydantic_ai.capabilities import AbstractCapability

    _CAPABILITIES_AVAILABLE = True
except ImportError:
    _CAPABILITIES_AVAILABLE = False

    # Type stub for when module is missing
    class AbstractCapability:  # type: ignore
        def __init__(self, **kwargs):
            pass


from ..models.knowledge_graph import RegistryNodeType, SelfEvaluationNode


class StuckLoopError(Exception):
    """Raised when the agent is stuck in a loop and action is "error"."""

    def __init__(self, pattern: str, message: str) -> None:
        self.pattern = pattern
        super().__init__(message)


def _hash_args(args: dict[str, Any]) -> str:
    """Create a stable hash of tool arguments for comparison."""
    try:
        serialized = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(args)
    return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()


def _hash_result(result: Any) -> str:
    """Create a stable hash of a tool result for comparison."""
    try:
        if isinstance(result, str):
            serialized = result
        else:
            serialized = json.dumps(result, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(result)
    return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()


@dataclass
class StuckLoopDetection(AbstractCapability[Any]):
    """Capability that detects and breaks repetitive agent loops.

    Integrated with IntelligenceGraphEngine to record stuck events.
    """

    max_repeated: int = 3
    action: Literal["warn", "error"] = "warn"
    detect_repeated: bool = True
    detect_alternating: bool = True
    detect_noop: bool = True

    _call_history: list[tuple[str, str]] = field(
        default_factory=list, init=False, repr=False
    )
    _result_history: list[tuple[str, str]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_repeated < 2:
            raise ValueError("max_repeated must be at least 2.")

    async def for_run(self, ctx: RunContext[Any]) -> StuckLoopDetection:
        """Return a fresh instance with isolated per-run state."""
        return replace(self)

    async def _react(
        self, ctx: RunContext[Any], pattern: str, message: str, tool_name: str
    ) -> None:
        """Record the event in the graph and raise the appropriate exception."""
        # Graph integration: Write SelfEvaluation node
        engine = getattr(ctx.deps, "graph_engine", None)
        if engine:
            eval_node = SelfEvaluationNode(
                id=f"stuck:{int(time.time())}:{hashlib.md5(message.encode(), usedforsecurity=False).hexdigest()[:8]}",
                type=RegistryNodeType.SELF_EVALUATION,
                name="Stuck Loop Detection",
                evaluation=message,
                confidence_calibration=0.0,
                task_difficulty=1.0,
                importance_score=0.8,
                metadata={
                    "pattern": pattern,
                    "message": message,
                    "tool_name": tool_name,
                    "event_type": "stuck_loop",
                },
            )
            with contextlib.suppress(Exception):
                # Silent fail for graph write in capability
                engine.graph.add_node(eval_node.id, **eval_node.model_dump())
                if engine.backend:
                    await engine.backend.upsert_node(
                        RegistryNodeType.SELF_EVALUATION,
                        eval_node.id,
                        eval_node.model_dump(),
                    )

        if self.action == "error":
            raise StuckLoopError(pattern, message)
        raise ModelRetry(message)

    def _check_repeated(self) -> str | None:
        history = self._call_history
        n = self.max_repeated
        if len(history) < n:
            return None
        tail = history[-n:]
        if all(entry == tail[0] for entry in tail):
            tool_name = tail[0][0]
            return f"You called `{tool_name}` with identical arguments {n} times in a row. Try a different approach."
        return None

    def _check_alternating(self) -> str | None:
        history = self._call_history
        n = self.max_repeated * 2
        if len(history) < n:
            return None
        tail = history[-n:]
        a, b = tail[0], tail[1]
        if a == b:
            return None
        if all(tail[i] == (a if i % 2 == 0 else b) for i in range(n)):
            return f"You're alternating between `{a[0]}` and `{b[0]}` in a loop ({n // 2} cycles). Step back and try a different strategy."
        return None

    def _check_noop(self) -> str | None:
        history = self._result_history
        n = self.max_repeated
        if len(history) < n:
            return None
        tail = history[-n:]
        if all(entry == tail[0] for entry in tail):
            tool_name = tail[0][0]
            return f"`{tool_name}` returned the same result {n} times in a row. The operation has no effect — try something different."
        return None

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        call_key = (call.tool_name, _hash_args(args))
        self._call_history.append(call_key)

        result_key = (call.tool_name, _hash_result(result))
        self._result_history.append(result_key)

        if self.detect_repeated:
            msg = self._check_repeated()
            if msg:
                await self._react(ctx, "repeated", msg, call.tool_name)

        if self.detect_alternating:
            msg = self._check_alternating()
            if msg:
                await self._react(ctx, "alternating", msg, call.tool_name)

        if self.detect_noop:
            msg = self._check_noop()
            if msg:
                await self._react(ctx, "noop", msg, call.tool_name)

        return result
