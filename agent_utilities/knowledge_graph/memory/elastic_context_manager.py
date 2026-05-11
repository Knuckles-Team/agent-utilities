from __future__ import annotations

import hashlib
import logging
import math
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# --- Merged from elastic_context_manager.py ---

#!/usr/bin/python
"""Token-Aware Context Compaction (CONCEPT:KG-2.1).

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


class CompactionStrategy(StrEnum):
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

    CONCEPT:KG-2.1 — Token-Aware Context Compaction

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


class ContextOperator(StrEnum):
    """Atomic context operations for elastic context orchestration.

    CONCEPT:KG-2.1 — Derived from LongSeeker (arXiv:2605.05191v1).

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

    CONCEPT:KG-2.1 — Derived from LongSeeker's Context-ReAct paradigm.

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


# --- Merged from elastic_context_manager.py ---

#!/usr/bin/python
"""Multi-Timescale Memory Dynamics (CONCEPT:KG-2.1 Enhancement).

Derived from: Continual Knowledge Updating (arXiv:2605.05097v1, Score 11.2)

Three memory tiers with exponential decay, consolidation, and pruning:
- WORKING: 5min half-life, promotes at 3+ accesses
- EPISODIC: 4hr half-life, promotes at 5+ accesses
- SEMANTIC: 30-day half-life, permanent
"""


logger = logging.getLogger(__name__)


class MemoryTimescale(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


DEFAULT_HALF_LIVES = {
    MemoryTimescale.WORKING: 300.0,
    MemoryTimescale.EPISODIC: 14400.0,
    MemoryTimescale.SEMANTIC: 2592000.0,
}
CONSOLIDATION_THRESHOLDS = {
    MemoryTimescale.WORKING: 3.0,
    MemoryTimescale.EPISODIC: 5.0,
    MemoryTimescale.SEMANTIC: float("inf"),
}


class MemoryEntry(BaseModel):
    """A memory entry with timescale-aware decay (CONCEPT:KG-2.1)."""

    memory_id: str
    content: str
    timescale: MemoryTimescale = MemoryTimescale.WORKING
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_accessed: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    access_count: int = 1
    activation: float = 1.0
    relevance_score: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_session: str = ""

    def compute_current_activation(self, half_lives: dict | None = None) -> float:
        lives = half_lives or DEFAULT_HALF_LIVES
        half_life = lives.get(self.timescale, 3600.0)
        last = datetime.fromisoformat(self.last_accessed)
        elapsed = (datetime.now(UTC) - last).total_seconds()
        return self.activation * math.pow(2, -elapsed / half_life)

    def access(self) -> None:
        self.access_count += 1
        self.activation = min(self.activation + 0.5, 10.0)
        self.last_accessed = datetime.now(UTC).isoformat()


class TimescaleMemoryStore:
    """Multi-tier memory with consolidation (CONCEPT:KG-2.1)."""

    def __init__(self, half_lives: dict | None = None, decay_floor: float = 0.01):
        self.half_lives = half_lives or dict(DEFAULT_HALF_LIVES)
        self.decay_floor = decay_floor
        self._memories: dict[str, MemoryEntry] = {}

    def store(
        self,
        content: str,
        *,
        timescale: MemoryTimescale = MemoryTimescale.WORKING,
        tags: list[str] | None = None,
        relevance_score: float = 0.5,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        memory_id = f"mem:{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        if memory_id in self._memories:
            self._memories[memory_id].access()
            return self._memories[memory_id]
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            timescale=timescale,
            tags=list(tags or []),
            relevance_score=relevance_score,
            source_session=session_id,
            metadata=dict(metadata or {}),
        )
        self._memories[memory_id] = entry
        return entry

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        timescale: MemoryTimescale | None = None,
        min_activation: float = 0.0,
    ) -> list[MemoryEntry]:
        query_words = set(query.lower().split())
        candidates = []
        for entry in self._memories.values():
            if timescale and entry.timescale != timescale:
                continue
            activation = entry.compute_current_activation(self.half_lives)
            if activation < min_activation:
                continue
            content_words = set(entry.content.lower().split())
            tag_words = set(t.lower() for t in entry.tags)
            overlap = len(query_words & (content_words | tag_words))
            score = (
                (overlap / max(len(query_words), 1))
                * activation
                * entry.relevance_score
            )
            if score > 0:
                candidates.append((score, entry))
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, entry in candidates[:top_k]:
            entry.access()
            results.append(entry)
        return results

    def consolidate(self) -> list[tuple[str, MemoryTimescale, MemoryTimescale]]:
        promotions = []
        for entry in list(self._memories.values()):
            threshold = CONSOLIDATION_THRESHOLDS.get(entry.timescale, float("inf"))
            if entry.access_count >= threshold:
                old = entry.timescale
                new = self._next_timescale(old)
                if new and new != old:
                    entry.timescale = new
                    entry.activation = 1.0
                    entry.access_count = 0
                    promotions.append((entry.memory_id, old, new))
        return promotions

    def prune(self) -> int:
        to_prune = [
            mid
            for mid, e in self._memories.items()
            if e.compute_current_activation(self.half_lives) < self.decay_floor
        ]
        for mid in to_prune:
            del self._memories[mid]
        return len(to_prune)

    def get_stats(self) -> dict[str, Any]:
        by_tier: dict[str, int] = defaultdict(int)
        for e in self._memories.values():
            by_tier[e.timescale.value] += 1
        return {
            "total_memories": len(self._memories),
            "by_timescale": dict(by_tier),
            "total_accesses": sum(e.access_count for e in self._memories.values()),
        }

    @staticmethod
    def _next_timescale(current: MemoryTimescale) -> MemoryTimescale | None:
        return {
            MemoryTimescale.WORKING: MemoryTimescale.EPISODIC,
            MemoryTimescale.EPISODIC: MemoryTimescale.SEMANTIC,
        }.get(current)


# --- Merged from elastic_context_manager.py ---

"""Vectorized Context-Window Filtering (CONCEPT:KG-2.6).

This module implements token-aware context compaction by semantically pruning
non-relevant subgraph context before swapping models on token overflow.
"""


logger = logging.getLogger(__name__)


def prune_context_by_semantic_distance(
    context_nodes: list[dict[str, Any]], query: str, max_tokens: int
) -> list[dict[str, Any]]:
    """Prune graph nodes from context if they exceed the token budget.

    Instead of hard-truncation, it drops the most semantically distant nodes
    from the query.
    """
    if not context_nodes:
        return []

    # Naive token estimation: ~4 chars per token
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    total_tokens = sum(estimate_tokens(str(n)) for n in context_nodes)
    if total_tokens <= max_tokens:
        return context_nodes

    logger.info(
        f"Context overflow detected ({total_tokens} > {max_tokens}). Applying topological pruning."
    )

    # Sort nodes by semantic relevance (assuming 'relevance_score' or 'topological_distance' exists)
    # If not, we fall back to trimming the longest nodes first as a heuristic, but
    # ideally we rely on the KG embeddings.
    try:
        sorted_nodes = sorted(
            context_nodes,
            key=lambda n: n.get(
                "topological_distance", n.get("distance", float("inf"))
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to sort context nodes by distance: {e}")
        sorted_nodes = context_nodes

    pruned_nodes = []
    current_tokens = 0
    for node in sorted_nodes:
        node_tokens = estimate_tokens(str(node))
        if current_tokens + node_tokens <= max_tokens:
            pruned_nodes.append(node)
            current_tokens += node_tokens
        else:
            logger.debug(
                f"Pruned node {node.get('id', 'unknown')} to save {node_tokens} tokens."
            )

    logger.info(f"Topological pruning complete. Final tokens: {current_tokens}.")
    return pruned_nodes
