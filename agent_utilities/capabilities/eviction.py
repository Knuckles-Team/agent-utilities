#!/usr/bin/python
# coding: utf-8
"""Tool output eviction capability with knowledge base integration.

Intercepts large tool outputs and moves them to the Knowledge Base
as RawSource nodes, leaving a small preview in the conversation history.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from ..models.knowledge_graph import RegistryNodeType, RawSourceNode

logger = logging.getLogger(__name__)


@dataclass
class ToolOutputEviction(AbstractCapability[Any]):
    """Capability that evicts large tool outputs to the Knowledge Base.

    Prevents token bloat by replacing massive results with previews.
    The full content is persisted in the KB graph for later retrieval.
    """

    threshold_chars: int = 80_000  # ~20k tokens
    store_in_graph: bool = True

    async def for_run(self, ctx: RunContext[Any]) -> ToolOutputEviction:
        return replace(self)

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        content = str(result)
        if len(content) <= self.threshold_chars:
            return result

        preview_len = 1000
        preview = f"{content[:preview_len]}\n... [EVICTED {len(content) - preview_len} chars. Content stored in KB graph] ..."

        if self.store_in_graph:
            engine = getattr(ctx.deps, "graph_engine", None)
            if engine:
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                node = RawSourceNode(
                    id=f"evicted_{content_hash[:12]}",
                    type=RegistryNodeType.RAW_SOURCE,
                    name=f"Evicted result from {call.tool_name}",
                    file_path=f"memory://evicted/{call.tool_call_id}",
                    source_type="tool_eviction",
                    content_hash=content_hash,
                    file_size=len(content),
                    status="evicted",
                    importance_score=0.5,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    metadata={
                        "tool_name": call.tool_name,
                        "tool_call_id": call.tool_call_id,
                        "full_content": content,  # In-memory graph node stores full content
                    },
                )
                try:
                    engine.graph.add_node(node.id, **node.model_dump())
                    # In a real backend, we might write to a blob store or dedicated KB table
                    if engine.backend:
                        await engine.backend.upsert_node(
                            RegistryNodeType.RAW_SOURCE, node.id, node.model_dump()
                        )
                except Exception as e:
                    logger.error(f"Failed to evict content to graph: {e}")

        return preview
