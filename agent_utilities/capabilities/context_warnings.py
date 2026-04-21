#!/usr/bin/python
"""Context limit warning capability with graph integration.

Monitors token usage and injects warnings into the model context as
the limit is approached. Records context pressure events in the graph.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, SystemPromptPart

from ..models.knowledge_graph import MemoryNode, RegistryNodeType

logger = logging.getLogger(__name__)


@dataclass
class ContextLimitWarner(AbstractCapability[Any]):
    """Capability that warns the agent when context token limits are near.

    Warns at 70% (URGENT) and 90% (CRITICAL), recording the latter in the graph.
    """

    warn_at: float = 0.70
    critical_at: float = 0.90
    max_tokens: int | None = None

    _has_warned: bool = False
    _has_criticized: bool = False

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
            msg = f"CRITICAL: Context usage is at {ratio:.1%}. You are very close to the {limit} token limit. Prune your context or conclude immediately."

            # Graph integration
            engine = getattr(ctx.deps, "graph_engine", None)
            if engine:
                node = MemoryNode(
                    id=f"ctx_pressure_{int(time.time())}",
                    type=RegistryNodeType.MEMORY,
                    name="Context Limit Warning",
                    content=msg,
                    importance_score=0.9,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    metadata={
                        "ratio": ratio,
                        "limit": limit,
                        "type": "context_pressure",
                    },
                )
                try:
                    engine.graph.add_node(node.id, **node.model_dump())
                except Exception:
                    pass

            request.parts.insert(0, SystemPromptPart(content=msg))

        elif ratio >= self.warn_at and not self._has_warned:
            self._has_warned = True
            msg = f"URGENT: Context usage is at {ratio:.1%}. Limit is {limit} tokens. Be concise."
            request.parts.insert(0, SystemPromptPart(content=msg))

        return request
