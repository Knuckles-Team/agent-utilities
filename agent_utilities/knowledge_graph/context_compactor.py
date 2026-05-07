#!/usr/bin/python
"""Token-Aware Context Compaction (CONCEPT:KG-2.10).

Intelligent context window management that replaces naive truncation
with strategy-based compaction.  Adapted from Goose's
``context_mgmt/mod.rs`` with Python-native implementations and KG
integration for episodic memory persistence.

Three compaction strategies are available:

1. **summarize_tools** — replaces large tool outputs with summaries,
   keeping the conversation structure intact (default).
2. **drop_middle** — retains the first and last N messages, dropping
   the middle of the conversation.
3. **progressive** — iteratively summarizes the oldest messages until
   the total token count falls under budget.

Key differences from Goose:

* **KG-native episode persistence** — compaction summaries are stored
  as ``EpisodeNode`` snapshots in the Knowledge Graph, enabling
  cross-session context continuity via ``MemoryRetriever``
  (CONCEPT:KG-2.1).
* **Backward compatibility** — ``prune_large_messages()`` in
  ``chat_persistence.py`` is kept as an alias.
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using the 1.33 tokens/word heuristic.

    This matches Goose's ``token_counter.rs`` default estimation for
    models without a specific tokenizer.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (integer).
    """
    if not text:
        return 0
    word_count = len(text.split())
    return int(word_count * 1.33)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total token count across a list of messages.

    Args:
        messages: List of message dicts with ``content`` keys.

    Returns:
        Total estimated tokens.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multi-part content (e.g., tool responses)
            for part in content:
                if isinstance(part, str):
                    total += estimate_tokens(part)
                elif isinstance(part, dict):
                    text = part.get("text", "") or str(part.get("content", ""))
                    total += estimate_tokens(text)
        # Add overhead per message (role, formatting)
        total += 4  # ~4 tokens per message wrapper
    return total


# ---------------------------------------------------------------------------
# Compaction strategy
# ---------------------------------------------------------------------------


class CompactionStrategy(str, Enum):
    """Strategy for reducing context window usage."""

    SUMMARIZE_TOOLS = "summarize_tools"
    DROP_MIDDLE = "drop_middle"
    PROGRESSIVE = "progressive"


# ---------------------------------------------------------------------------
# Compacted result model
# ---------------------------------------------------------------------------


class CompactedResult(BaseModel):
    """Result of a context compaction operation.

    Attributes:
        messages: The compacted message list.
        tokens_before: Token count before compaction.
        tokens_after: Token count after compaction.
        strategy_used: Which compaction strategy was applied.
        summary_text: A text summary of what was compacted.
        messages_removed: Number of messages removed or summarized.
        compaction_id: Unique identifier for this compaction event.
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    strategy_used: str = ""
    summary_text: str = ""
    messages_removed: int = 0
    compaction_id: str = Field(
        default_factory=lambda: f"compact:{uuid.uuid4().hex[:8]}"
    )


# ---------------------------------------------------------------------------
# ContextCompactor
# ---------------------------------------------------------------------------


class ContextCompactor:
    """Intelligent context window compaction engine.

    CONCEPT:KG-2.10 — Token-Aware Context Compaction

    Adapted from Goose's ``compact_messages()`` (Rust) with three
    strategies for reducing token usage while preserving semantic
    content.

    Example::

        compactor = ContextCompactor(max_tokens=8000)
        messages = [{"role": "user", "content": "..."}, ...]
        result = compactor.compact(messages)
        print(f"Reduced {result.tokens_before} → {result.tokens_after} tokens")
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        auto_compaction_ratio: float = 0.8,
        tool_summary_max_length: int = 500,
    ) -> None:
        """Initialize the compactor.

        Args:
            max_tokens: Target maximum token count after compaction.
            auto_compaction_ratio: Trigger compaction when usage exceeds
                this fraction of max_tokens (default 0.8).
            tool_summary_max_length: Max characters for tool output summaries
                in ``summarize_tools`` strategy.
        """
        self.max_tokens = max_tokens
        self.auto_compaction_ratio = auto_compaction_ratio
        self.tool_summary_max_length = tool_summary_max_length

    def should_compact(self, messages: list[dict[str, Any]]) -> bool:
        """Check if messages exceed the auto-compaction threshold.

        Args:
            messages: Current message list.

        Returns:
            True if compaction is recommended.
        """
        current_tokens = estimate_message_tokens(messages)
        threshold = int(self.max_tokens * self.auto_compaction_ratio)
        return current_tokens > threshold

    def compact(
        self,
        messages: list[dict[str, Any]],
        strategy: CompactionStrategy | str = CompactionStrategy.SUMMARIZE_TOOLS,
    ) -> CompactedResult:
        """Compact messages using the specified strategy.

        Args:
            messages: The message list to compact.
            strategy: Compaction strategy to apply.

        Returns:
            CompactedResult with compacted messages and metadata.
        """
        if isinstance(strategy, str):
            strategy = CompactionStrategy(strategy)

        tokens_before = estimate_message_tokens(messages)

        # If already under budget, return as-is
        if tokens_before <= self.max_tokens:
            return CompactedResult(
                messages=list(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                strategy_used=strategy.value,
                summary_text="No compaction needed",
                messages_removed=0,
            )

        if strategy == CompactionStrategy.SUMMARIZE_TOOLS:
            return self._compact_summarize_tools(messages, tokens_before)
        elif strategy == CompactionStrategy.DROP_MIDDLE:
            return self._compact_drop_middle(messages, tokens_before)
        elif strategy == CompactionStrategy.PROGRESSIVE:
            return self._compact_progressive(messages, tokens_before)
        else:
            # Fallback
            return self._compact_summarize_tools(messages, tokens_before)

    def _compact_summarize_tools(
        self,
        messages: list[dict[str, Any]],
        tokens_before: int,
    ) -> CompactedResult:
        """Summarize large tool outputs to reduce token usage.

        Adapted from Goose's tool result summarization logic.
        Replaces tool response content exceeding ``tool_summary_max_length``
        with a truncated summary preserving the first and last portions.
        """
        compacted = []
        removed = 0
        max_len = self.tool_summary_max_length

        for msg in messages:
            new_msg = deepcopy(msg)
            content = new_msg.get("content", "")

            # Check if this is a tool response with large output
            role = new_msg.get("role", "")
            is_tool = role in ("tool", "function") or new_msg.get("tool_call_id")

            if is_tool and isinstance(content, str) and len(content) > max_len:
                # Summarize: keep first/last portions
                head = content[: max_len // 2]
                tail = content[-(max_len // 2) :]
                new_msg["content"] = (
                    f"{head}\n\n"
                    f"[... tool output summarized: {len(content)} chars → {max_len} chars ...]\n\n"
                    f"{tail}"
                )
                removed += 1
            elif (
                not is_tool and isinstance(content, str) and len(content) > max_len * 4
            ):
                # Also summarize very large non-tool messages
                head = content[:max_len]
                tail = content[-max_len:]
                new_msg["content"] = (
                    f"{head}\n\n"
                    f"[... content summarized: {len(content)} chars ...]\n\n"
                    f"{tail}"
                )
                removed += 1

            compacted.append(new_msg)

        tokens_after = estimate_message_tokens(compacted)

        return CompactedResult(
            messages=compacted,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy_used=CompactionStrategy.SUMMARIZE_TOOLS.value,
            summary_text=f"Summarized {removed} large tool outputs",
            messages_removed=removed,
        )

    def _compact_drop_middle(
        self,
        messages: list[dict[str, Any]],
        tokens_before: int,
    ) -> CompactedResult:
        """Keep first and last N messages, drop the middle.

        Ensures the system prompt and most recent context are preserved.
        """
        if len(messages) <= 4:
            # Too few messages to drop
            return CompactedResult(
                messages=list(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                strategy_used=CompactionStrategy.DROP_MIDDLE.value,
                summary_text="Too few messages to drop",
                messages_removed=0,
            )

        # Keep first 2 and last 2 messages, expand outward from both ends
        # until we hit the budget
        keep_first = 2
        keep_last = 2
        total = len(messages)

        while keep_first + keep_last < total:
            candidate = messages[:keep_first] + messages[-keep_last:]
            if estimate_message_tokens(candidate) <= self.max_tokens:
                # Add one more from each end
                if keep_first + keep_last + 2 <= total:
                    keep_first += 1
                    keep_last += 1
                else:
                    break
            else:
                break

        compacted = messages[:keep_first] + messages[-keep_last:]
        dropped = total - len(compacted)

        # Insert a system note about dropped messages
        if dropped > 0:
            note = {
                "role": "system",
                "content": (
                    f"[Context note: {dropped} messages from the middle of this "
                    f"conversation were removed to fit the context window. "
                    f"The earliest and most recent messages are preserved.]"
                ),
            }
            compacted.insert(keep_first, note)

        tokens_after = estimate_message_tokens(compacted)

        return CompactedResult(
            messages=compacted,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy_used=CompactionStrategy.DROP_MIDDLE.value,
            summary_text=f"Dropped {dropped} middle messages",
            messages_removed=dropped,
        )

    def _compact_progressive(
        self,
        messages: list[dict[str, Any]],
        tokens_before: int,
    ) -> CompactedResult:
        """Iteratively summarize oldest messages until under budget.

        Always preserves the first message (system prompt) and the last
        3 messages (recent context).  Works inward from the second
        message, progressively replacing older messages with compact
        summaries.
        """
        if len(messages) <= 4:
            return CompactedResult(
                messages=list(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                strategy_used=CompactionStrategy.PROGRESSIVE.value,
                summary_text="Too few messages for progressive compaction",
                messages_removed=0,
            )

        compacted = deepcopy(messages)
        removed = 0
        preserve_last = 3
        max_summary = self.tool_summary_max_length

        # Start from index 1 (preserve system prompt at 0)
        idx = 1
        while (
            estimate_message_tokens(compacted) > self.max_tokens
            and idx < len(compacted) - preserve_last
        ):
            msg = compacted[idx]
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_summary:
                role = msg.get("role", "unknown")
                summary = content[: max_summary // 3]
                compacted[idx] = {
                    "role": role,
                    "content": (
                        f"[Summarized {role} message: {len(content)} chars → "
                        f"{len(summary)} chars]\n{summary}..."
                    ),
                }
                removed += 1
            idx += 1

        # If still over budget, start dropping summarized messages entirely
        idx = 1
        while (
            estimate_message_tokens(compacted) > self.max_tokens
            and idx < len(compacted) - preserve_last
        ):
            if "[Summarized" in str(compacted[idx].get("content", "")):
                compacted.pop(idx)
                removed += 1
            else:
                idx += 1

        tokens_after = estimate_message_tokens(compacted)

        return CompactedResult(
            messages=compacted,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy_used=CompactionStrategy.PROGRESSIVE.value,
            summary_text=f"Progressively compacted {removed} messages",
            messages_removed=removed,
        )

    def create_episode_node(
        self,
        result: CompactedResult,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Create a KG-persistable EpisodeNode from a compaction result.

        CONCEPT:KG-2.1 — Tiered Memory

        Compaction summaries are stored as episodic snapshots to enable
        cross-session context recall via ``MemoryRetriever``.

        Returns:
            Dict with EpisodeNode data for KG ingestion.
        """
        return {
            "id": f"episode:{result.compaction_id}",
            "type": "episode",
            "summary": result.summary_text,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "strategy": result.strategy_used,
            "messages_removed": result.messages_removed,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Elastic Context Operators (LongSeeker-inspired)
# ---------------------------------------------------------------------------


class ContextOperator(str, Enum):
    """Atomic context operations for elastic context orchestration.

    CONCEPT:KG-2.10 — Derived from LongSeeker (arXiv:2605.05191v1).

    Five atomic operators for reshaping working context:
    - SKIP: Mark a message as irrelevant, exclude from future processing
    - COMPRESS: Replace message(s) with a compact summary
    - ROLLBACK: Revert to a previous context checkpoint
    - SNIPPET: Extract focused evidence from verbose content
    - DELETE: Permanently remove a message from context

    Compress is expressively complete (any operation can be expressed as
    a compression), but specialized operators reduce generation cost and
    hallucination risk.
    """

    SKIP = "skip"
    COMPRESS = "compress"
    ROLLBACK = "rollback"
    SNIPPET = "snippet"
    DELETE = "delete"


class ContextCheckpoint(BaseModel):
    """A snapshot of context state for rollback support.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        messages: Deep copy of messages at checkpoint time.
        token_count: Token count at checkpoint time.
        timestamp: When the checkpoint was created.
    """

    checkpoint_id: str = Field(default_factory=lambda: f"ckpt:{uuid.uuid4().hex[:8]}")
    messages: list[dict[str, Any]] = Field(default_factory=list)
    token_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class OperatorResult(BaseModel):
    """Result of applying a context operator.

    Attributes:
        operator: Which operator was applied.
        messages: The resulting message list.
        tokens_before: Token count before the operation.
        tokens_after: Token count after the operation.
        affected_indices: Which message indices were affected.
        metadata: Additional operator-specific metadata.
    """

    operator: ContextOperator
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    affected_indices: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ElasticContextManager:
    """Elastic context orchestration with 5 atomic operators.

    CONCEPT:KG-2.10 — Derived from LongSeeker's Context-ReAct paradigm.

    Provides fine-grained control over working context using atomic
    operations that preserve important evidence, summarize resolved
    information, discard unhelpful branches, and control context size.

    Example::

        ecm = ElasticContextManager(max_tokens=8000)
        messages = [...]

        # Create a checkpoint before risky operations
        ecm.checkpoint(messages)

        # Skip irrelevant messages
        result = ecm.apply(ContextOperator.SKIP, messages, indices=[3, 5])

        # Extract a snippet from a verbose message
        result = ecm.apply(ContextOperator.SNIPPET, messages,
                          indices=[7], snippet_query="key findings")

        # Rollback if the approach didn't work
        result = ecm.rollback()
    """

    def __init__(self, max_tokens: int = 8000) -> None:
        """Initialize the elastic context manager.

        Args:
            max_tokens: Target maximum token count.
        """
        self.max_tokens = max_tokens
        self._checkpoints: list[ContextCheckpoint] = []
        self._skip_set: set[int] = set()

    def checkpoint(self, messages: list[dict[str, Any]]) -> ContextCheckpoint:
        """Create a context checkpoint for potential rollback.

        Args:
            messages: Current message list to snapshot.

        Returns:
            The created checkpoint.
        """
        ckpt = ContextCheckpoint(
            messages=deepcopy(messages),
            token_count=estimate_message_tokens(messages),
        )
        self._checkpoints.append(ckpt)
        logger.debug(
            "Created checkpoint %s (%d tokens)", ckpt.checkpoint_id, ckpt.token_count
        )
        return ckpt

    def apply(
        self,
        operator: ContextOperator,
        messages: list[dict[str, Any]],
        *,
        indices: list[int] | None = None,
        snippet_query: str = "",
        snippet_max_length: int = 200,
    ) -> OperatorResult:
        """Apply a context operator to the message list.

        Args:
            operator: The operator to apply.
            messages: Current message list.
            indices: Message indices to operate on (required for SKIP,
                COMPRESS, SNIPPET, DELETE).
            snippet_query: Query string for SNIPPET operator.
            snippet_max_length: Max length for extracted snippets.

        Returns:
            OperatorResult with the transformed messages.
        """
        tokens_before = estimate_message_tokens(messages)

        if operator == ContextOperator.SKIP:
            return self._apply_skip(messages, indices or [], tokens_before)
        elif operator == ContextOperator.COMPRESS:
            return self._apply_compress(messages, indices or [], tokens_before)
        elif operator == ContextOperator.ROLLBACK:
            return self._apply_rollback(tokens_before)
        elif operator == ContextOperator.SNIPPET:
            return self._apply_snippet(
                messages,
                indices or [],
                snippet_query,
                snippet_max_length,
                tokens_before,
            )
        elif operator == ContextOperator.DELETE:
            return self._apply_delete(messages, indices or [], tokens_before)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def rollback(self) -> OperatorResult:
        """Rollback to the most recent checkpoint.

        Returns:
            OperatorResult with the restored messages.

        Raises:
            ValueError: If no checkpoints exist.
        """
        return self._apply_rollback(0)

    # ── Operator implementations ──────────────────────────────────────────

    def _apply_skip(
        self,
        messages: list[dict[str, Any]],
        indices: list[int],
        tokens_before: int,
    ) -> OperatorResult:
        """Mark messages as skipped — excluded from future processing."""
        self._skip_set.update(indices)
        result = [msg for i, msg in enumerate(messages) if i not in self._skip_set]
        return OperatorResult(
            operator=ContextOperator.SKIP,
            messages=result,
            tokens_before=tokens_before,
            tokens_after=estimate_message_tokens(result),
            affected_indices=indices,
            metadata={"total_skipped": len(self._skip_set)},
        )

    def _apply_compress(
        self,
        messages: list[dict[str, Any]],
        indices: list[int],
        tokens_before: int,
    ) -> OperatorResult:
        """Compress specified messages into a single summary message."""
        if not indices:
            indices = list(range(len(messages)))

        to_compress = [messages[i] for i in indices if i < len(messages)]
        if not to_compress:
            return OperatorResult(
                operator=ContextOperator.COMPRESS,
                messages=list(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                affected_indices=[],
            )

        # Build summary from compressed messages
        summaries = []
        for msg in to_compress:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:150]
            summaries.append(f"[{role}]: {content}")

        compressed_content = f"[Compressed {len(to_compress)} messages]\n" + "\n".join(
            summaries
        )

        # Replace the range with the compressed message
        result = []
        indices_set = set(indices)
        inserted = False
        for i, msg in enumerate(messages):
            if i in indices_set:
                if not inserted:
                    result.append(
                        {
                            "role": "system",
                            "content": compressed_content,
                        }
                    )
                    inserted = True
                # Skip original message
            else:
                result.append(msg)

        return OperatorResult(
            operator=ContextOperator.COMPRESS,
            messages=result,
            tokens_before=tokens_before,
            tokens_after=estimate_message_tokens(result),
            affected_indices=indices,
            metadata={"messages_compressed": len(to_compress)},
        )

    def _apply_rollback(self, tokens_before: int) -> OperatorResult:
        """Restore messages from the most recent checkpoint."""
        if not self._checkpoints:
            raise ValueError("No checkpoints available for rollback")

        ckpt = self._checkpoints.pop()
        restored = deepcopy(ckpt.messages)
        self._skip_set.clear()

        return OperatorResult(
            operator=ContextOperator.ROLLBACK,
            messages=restored,
            tokens_before=tokens_before,
            tokens_after=estimate_message_tokens(restored),
            affected_indices=[],
            metadata={
                "checkpoint_id": ckpt.checkpoint_id,
                "checkpoint_tokens": ckpt.token_count,
            },
        )

    def _apply_snippet(
        self,
        messages: list[dict[str, Any]],
        indices: list[int],
        query: str,
        max_length: int,
        tokens_before: int,
    ) -> OperatorResult:
        """Extract focused evidence snippets from verbose messages."""
        result = deepcopy(messages)
        query_lower = query.lower()

        for idx in indices:
            if idx >= len(result):
                continue
            content = str(result[idx].get("content", ""))
            if len(content) <= max_length:
                continue

            # Find the most relevant passage containing query terms
            sentences = content.replace("\n", ". ").split(". ")
            scored = []
            for sent in sentences:
                relevance = sum(
                    1 for term in query_lower.split() if term in sent.lower()
                )
                scored.append((relevance, sent))

            scored.sort(key=lambda x: x[0], reverse=True)
            snippet = ". ".join(s for _, s in scored[:3])[:max_length]

            result[idx] = {
                **result[idx],
                "content": (f"[Snippet extracted for: '{query}']\n{snippet}"),
            }

        return OperatorResult(
            operator=ContextOperator.SNIPPET,
            messages=result,
            tokens_before=tokens_before,
            tokens_after=estimate_message_tokens(result),
            affected_indices=indices,
            metadata={"query": query},
        )

    def _apply_delete(
        self,
        messages: list[dict[str, Any]],
        indices: list[int],
        tokens_before: int,
    ) -> OperatorResult:
        """Permanently remove messages from context."""
        indices_set = set(indices)
        result = [msg for i, msg in enumerate(messages) if i not in indices_set]

        return OperatorResult(
            operator=ContextOperator.DELETE,
            messages=result,
            tokens_before=tokens_before,
            tokens_after=estimate_message_tokens(result),
            affected_indices=indices,
            metadata={"messages_deleted": len(indices_set)},
        )
