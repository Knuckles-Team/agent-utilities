#!/usr/bin/python
from __future__ import annotations

"""Live Memento context compaction capability.

CONCEPT:KG-2.20 — Mementified Context Management (MEM-1, the live sawtooth).

Assimilated from *Memento: Teaching LLMs to Manage Their Own Context* (Kontonis et al., MSR AI
Frontiers, 2026). The paper teaches a *model* to segment its chain-of-thought into blocks, compress
each into a dense memento, and evict the raw block from attention — yielding a sawtooth KV-cache
profile (~2–2.5× peak reduction). The authors flag agents as the prime next application: *"Terminal
and CLI agents are naturally multi-turn, where each action-observation cycle is laid out as a natural
block."* This capability implements that pattern at the **orchestration layer**: before each model
request, when the running history exceeds budget, it segments the history into semantic blocks
(``segment_into_blocks``), compresses the oldest *completed* blocks into mementos
(``compress_to_memento`` with the judge-refine loop), and **evicts** the raw blocks from the message
list actually sent to the model — keeping ``mementos + the current block``.

**Honest limitation.** We run hosted/API models with no KV-cache control, so this is the paper's
"restart mode": it cannot reproduce the implicit dual-channel KV side-information (−15pp in the
paper). Eviction is made *lossless* instead (MEM-4): each evicted block is persisted with a
``SUMMARIZES`` pointer and is recoverable on demand via
``memento_compressor.recover_evicted_block``.

Wiring: registered in ``agent/factory.py`` alongside ``ContextLimitWarner``/``ToolOutputEviction`` and
fired from pydantic-ai's ``before_model_request`` hook, which receives the full
``ModelRequestContext.messages`` (``list[ModelMessage]``) — the list sent to the model. Default ON.
"""

import dataclasses
import logging
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
)

logger = logging.getLogger(__name__)


def _message_to_dict(msg: ModelMessage) -> dict[str, Any]:
    """Flatten a pydantic-ai ModelMessage into a ``{role, content}`` dict for segmentation."""
    if isinstance(msg, ModelResponse):
        role = "assistant"
    else:  # ModelRequest — classify by the parts it carries
        kinds = {getattr(p, "part_kind", "") for p in getattr(msg, "parts", [])}
        if "tool-return" in kinds or "retry-prompt" in kinds:
            role = "tool"
        elif "system-prompt" in kinds and "user-prompt" not in kinds:
            role = "system"
        else:
            role = "user"
    texts: list[str] = []
    for part in getattr(msg, "parts", []):
        val = getattr(part, "content", None)
        if val is None:
            val = getattr(part, "text", None)
        if val is not None:
            texts.append(str(val))
    return {"role": role, "content": "\n".join(texts)}


@dataclass
class MementoCompaction(AbstractCapability[Any]):
    """Evicts old completed blocks from the live context, replacing them with dense mementos.

    CONCEPT:KG-2.20 — the live block-compress-evict sawtooth.

    Args:
        max_tokens: Token budget for the message history. If ``None``, derived from the model's
            ``max_input_tokens`` when available.
        auto_compaction_ratio: Trigger compaction once usage exceeds this fraction of the budget.
        keep_recent_blocks: Number of most-recent blocks never evicted (the block being reasoned
            through). Paper keeps the current block uncompressed.
        keep_head: Leading messages always preserved (system prompt etc.).
        min_block_tokens: Minimum block size for segmentation (paper floor: 200).
        enabled: Master switch (default True — integration is ON by default).
        source: Memento ``source`` tag for persistence/recall grouping.
    """

    max_tokens: int | None = None
    auto_compaction_ratio: float = 0.8
    keep_recent_blocks: int = 1
    keep_head: int = 1
    min_block_tokens: int = 200
    enabled: bool = True
    source: str = "agent_runner"

    async def for_run(self, ctx: RunContext[Any]) -> MementoCompaction:
        return replace(self)

    def _budget(self, ctx: RunContext[Any] | None) -> int | None:
        if self.max_tokens:
            return self.max_tokens
        model = getattr(ctx, "model", None) if ctx is not None else None
        limit = getattr(model, "max_input_tokens", None)
        return int(limit) if limit else None

    def mementoize_messages(
        self,
        messages: list[ModelMessage],
        *,
        budget_tokens: int,
        engine: Any = None,
    ) -> tuple[list[ModelMessage], int]:
        """Pure transform: segment → compress completed blocks → evict. Returns (new_messages, n_evicted).

        Compresses each evicted block into a memento (persisted + recoverable when an engine is given;
        dry-run otherwise) and replaces the block's raw messages with one ``ModelRequest`` carrying a
        ``SystemPromptPart`` memento. This is the unit-testable core of the capability.
        """
        from agent_utilities.knowledge_graph.memory.agent_context import (
            estimate_message_tokens,
        )
        from agent_utilities.knowledge_graph.memory.memento_compressor import (
            compress_to_memento,
            plan_block_eviction,
        )

        dicts = [_message_to_dict(m) for m in messages]
        if estimate_message_tokens(dicts) <= budget_tokens:
            return messages, 0

        evicted_groups, _kept = plan_block_eviction(
            dicts,
            budget_tokens=budget_tokens,
            keep_recent_blocks=self.keep_recent_blocks,
            keep_head=self.keep_head,
            min_block_tokens=self.min_block_tokens,
        )
        if not evicted_groups:
            return messages, 0

        group_of = {i: gi for gi, grp in enumerate(evicted_groups) for i in grp}
        evicted_idx = set(group_of)
        new_messages: list[ModelMessage] = []
        inserted: set[int] = set()
        n_evicted = 0
        for i, msg in enumerate(messages):
            if i in evicted_idx:
                gi = group_of[i]
                if gi not in inserted:
                    inserted.add(gi)
                    block_dicts = [dicts[j] for j in evicted_groups[gi]]
                    memento = (
                        compress_to_memento(
                            engine,
                            block_dicts,
                            source=self.source,
                            dry_run=engine is None,
                        )
                        or "[block evicted to fit context budget]"
                    )
                    new_messages.append(
                        ModelRequest(
                            parts=[
                                SystemPromptPart(
                                    content=(
                                        "PRIOR CONTEXT MEMENTO (compressed; reason forward from "
                                        f"this, recoverable on demand):\n{memento}"
                                    )
                                )
                            ]
                        )
                    )
                    n_evicted += len(evicted_groups[gi])
            else:
                new_messages.append(msg)
        return new_messages, n_evicted

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: Any
    ) -> Any:
        """Evict old blocks from the outgoing message list when over budget."""
        if not self.enabled:
            return request_context
        messages = getattr(request_context, "messages", None)
        if not messages or len(messages) < 4:
            return request_context
        budget = self._budget(ctx)
        if not budget:
            return request_context
        trigger = int(budget * self.auto_compaction_ratio)
        engine = getattr(getattr(ctx, "deps", None), "graph_engine", None)
        try:
            new_messages, n_evicted = self.mementoize_messages(
                list(messages), budget_tokens=trigger, engine=engine
            )
        except Exception as e:  # noqa: BLE001 - compaction must never break the run
            logger.warning("Memento compaction failed, sending full history: %s", e)
            return request_context
        if n_evicted:
            logger.info(
                "[KG-2.20] Memento-evicted %d messages from live context (budget %d)",
                n_evicted,
                trigger,
            )
            return dataclasses.replace(request_context, messages=new_messages)
        return request_context
