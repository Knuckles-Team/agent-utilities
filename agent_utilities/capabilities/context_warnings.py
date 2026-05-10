#!/usr/bin/python
from __future__ import annotations

"""Context limit warning capability with graph integration.

Monitors token usage and injects warnings into the model context as
the limit is approached. Records context pressure events in the graph.
"""


import contextlib
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, SystemPromptPart

from agent_utilities.protocols.capability import CapabilityContext

from ..models.knowledge_graph import RegistryNodeType, SelfEvaluationNode

logger = logging.getLogger(__name__)


@dataclass
class ContextLimitWarner(AbstractCapability[Any]):
    """Capability that warns the agent when context token limits are near.

    Warns at 70% (URGENT) and 90% (CRITICAL), recording the latter in the graph.
    """

    warn_at: float = 0.70
    critical_at: float = 0.90
    max_tokens: int | None = None

    _has_warned: bool = field(default=False, init=False, repr=False)
    _has_criticized: bool = field(default=False, init=False, repr=False)

    @property
    def capability_name(self) -> str:
        return "context_limit_warner"

    def can_handle(self, context: CapabilityContext) -> bool:
        return context.trigger_data.get("event") == "before_model_run"

    async def execute(self, context: CapabilityContext) -> dict[str, Any]:
        return {"status": "success", "action": "warn_context"}

    async def for_run(self, ctx: RunContext[Any]) -> ContextLimitWarner:
        return replace(self)

    async def before_model_run(
        self, ctx: RunContext[Any], request: ModelRequest
    ) -> ModelRequest:
        usage = getattr(ctx, "usage", None)
        if not usage or not usage.total_tokens:
            return request

        limit = self.max_tokens or getattr(ctx.model, "max_input_tokens", None)
        if not limit:
            return request

        ratio = usage.total_tokens / limit

        if ratio >= self.critical_at and not self._has_criticized:
            self._has_criticized = True
            msg = f"CRITICAL: Context usage is at {ratio:.1%}. Limit is {limit} tokens. You MUST wrap up immediately."
            request.parts = [SystemPromptPart(content=msg)] + list(request.parts)

            # Record event in knowledge graph if configured
            engine = getattr(ctx.deps, "graph_engine", None)
            if engine:
                eval_node = SelfEvaluationNode(
                    id=f"ctx_crit:{int(time.time())}",
                    type=RegistryNodeType.SELF_EVALUATION,
                    name="Context Limit Critical",
                    evaluation=msg,
                    confidence_calibration=1.0,
                    task_difficulty=1.0,
                    importance_score=0.9,
                    metadata={
                        "ratio": ratio,
                        "limit": limit,
                        "tokens": ctx.usage.total_tokens,
                    },
                )
                with contextlib.suppress(Exception):
                    engine.graph.add_node(eval_node.id, **eval_node.model_dump())

        elif ratio >= self.warn_at and not self._has_warned:
            self._has_warned = True
            msg = f"URGENT: Context usage is at {ratio:.1%}. Limit is {limit} tokens. Be concise."
            request.parts = [SystemPromptPart(content=msg)] + list(request.parts)

        return request
