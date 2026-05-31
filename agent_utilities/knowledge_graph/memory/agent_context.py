from __future__ import annotations

# --- FROM elastic_context_manager.py ---
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

    # ── KG-Persistent Compaction (LCM Integration) ────────────────────

    def persist_compaction(
        self,
        result: CompactedResult,
        thread_id: str,
        source_message_ids: list[str],
        engine: Any = None,
        level: int = 1,
        partition: str = "",
    ) -> str | None:
        """Persist a compaction result as a Summary node in the KG.

        CONCEPT:KG-2.1 — LCM DAG-Based Summarization

        Creates a Summary node linked to the source messages via SUMMARIZES
        edges. The original messages are preserved (lossless). This is the
        single entry point for all KG-persistent compaction — both manual
        and daemon-triggered compaction flows through here.

        Args:
            result: CompactedResult from any compaction strategy.
            thread_id: The Thread node ID this summary belongs to.
            source_message_ids: IDs of the Message nodes being summarized.
            engine: IntelligenceGraphEngine instance for KG writes.
            level: Summary DAG level (1=first-pass, 2=escalated, etc.).
            partition: Optional partition tag for filtering.

        Returns:
            Summary node ID, or None if no engine available.
        """
        if not engine or not getattr(engine, "backend", None):
            return None

        summary_id = f"summary:{result.compaction_id}"
        timestamp = datetime.now(UTC).isoformat()

        # Create Summary node
        engine.add_node(
            summary_id,
            "Summary",
            properties={
                "content": result.summary_text,
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "strategy": result.strategy_used,
                "messages_summarized": result.messages_removed,
                "level": level,
                "thread_id": thread_id,
                "partition": partition,
                "timestamp": timestamp,
            },
        )

        # Link Thread → Summary
        engine.link_nodes(
            thread_id,
            summary_id,
            "HAS_SUMMARY",
            properties={
                "level": level,
                "created_at": timestamp,
            },
        )

        # Link Summary → source Messages (lossless pointers)
        for msg_id in source_message_ids:
            engine.link_nodes(
                summary_id,
                msg_id,
                "SUMMARIZES",
                properties={
                    "level": level,
                    "compaction_id": result.compaction_id,
                },
            )

        logger.info(
            "Persisted Summary %s (L%d) for thread %s — %d messages → %d tokens",
            summary_id,
            level,
            thread_id,
            len(source_message_ids),
            result.tokens_after,
        )
        return summary_id

    def escalate(
        self,
        thread_id: str,
        engine: Any = None,
        batch_size: int = 5,
    ) -> str | None:
        """Build the next level of the summary DAG by summarizing summaries.

        CONCEPT:KG-2.1 — LCM Escalated Summarization

        Finds un-escalated Summary nodes at the current highest level,
        groups them into batches, and creates a higher-level Summary that
        SUMMARIZES the batch. This builds the DAG's vertical depth.

        Args:
            thread_id: Thread to escalate summaries for.
            engine: IntelligenceGraphEngine instance.
            batch_size: Number of summaries to group per escalation.

        Returns:
            New escalated Summary ID, or None if nothing to escalate.
        """
        if not engine or not getattr(engine, "backend", None):
            return None

        # Find the current max level for this thread
        level_results = engine.query_cypher(
            "MATCH (t {id: $tid})-[:HAS_SUMMARY]->(s:Summary) "
            "RETURN max(s.level) AS max_level",
            {"tid": thread_id},
        )
        max_level = 1
        if level_results and level_results[0].get("max_level"):
            max_level = int(level_results[0]["max_level"])

        # Find summaries at max level that haven't been escalated
        summaries = engine.query_cypher(
            "MATCH (t {id: $tid})-[:HAS_SUMMARY]->(s:Summary) "
            "WHERE s.level = $level "
            "AND NOT exists { MATCH (:Summary)-[:SUMMARIZES]->(s) } "
            "RETURN s.id AS id, s.content AS content "
            "ORDER BY s.timestamp "
            "LIMIT $batch",
            {"tid": thread_id, "level": max_level, "batch": batch_size},
        )

        if len(summaries) < 2:
            return None  # Need at least 2 summaries to escalate

        # Combine summary texts
        combined_text = "\n".join(
            f"[Summary L{max_level}]: {s.get('content', '')[:200]}" for s in summaries
        )
        summary_ids = [s["id"] for s in summaries]

        # Create escalated compaction result
        escalated_result = CompactedResult(
            messages=[{"role": "system", "content": combined_text}],
            tokens_before=estimate_tokens(combined_text),
            tokens_after=estimate_tokens(combined_text),
            strategy_used="escalated",
            summary_text=f"Escalated {len(summaries)} L{max_level} summaries",
            messages_removed=len(summaries),
        )

        # Persist at next level, linking to child summaries
        return self.persist_compaction(
            escalated_result,
            thread_id=thread_id,
            source_message_ids=summary_ids,
            engine=engine,
            level=max_level + 1,
        )

    @staticmethod
    def get_summary_dag(thread_id: str, engine: Any = None) -> dict[str, Any]:
        """Traverse the summary DAG for a thread.

        CONCEPT:KG-2.1 — LCM Summary Chain Retrieval

        Returns a structured view of all summaries with their source links,
        organized by level. This is the read-path counterpart to
        persist_compaction/escalate.

        Args:
            thread_id: Thread to get summary DAG for.
            engine: IntelligenceGraphEngine instance.

        Returns:
            Dict with levels, summaries, and source message IDs.
        """
        if not engine or not getattr(engine, "backend", None):
            return {"thread_id": thread_id, "levels": {}, "total_summaries": 0}

        summaries = engine.query_cypher(
            "MATCH (t {id: $tid})-[:HAS_SUMMARY]->(s:Summary) "
            "RETURN s.id AS id, s.content AS content, s.level AS level, "
            "s.tokens_before AS tokens_before, s.tokens_after AS tokens_after, "
            "s.strategy AS strategy, s.timestamp AS timestamp "
            "ORDER BY s.level, s.timestamp",
            {"tid": thread_id},
        )

        levels: dict[int, list[dict]] = {}
        for s in summaries:
            lvl = int(s.get("level", 1))
            if lvl not in levels:
                levels[lvl] = []

            # Get source IDs for this summary
            sources = engine.query_cypher(
                "MATCH (s {id: $sid})-[:SUMMARIZES]->(child) RETURN child.id AS id",
                {"sid": s["id"]},
            )
            source_ids = [src["id"] for src in sources]

            levels[lvl].append(
                {
                    "id": s["id"],
                    "content": s.get("content", ""),
                    "tokens_before": s.get("tokens_before", 0),
                    "tokens_after": s.get("tokens_after", 0),
                    "strategy": s.get("strategy", ""),
                    "timestamp": s.get("timestamp", ""),
                    "source_ids": source_ids,
                }
            )

        return {
            "thread_id": thread_id,
            "levels": {str(k): v for k, v in sorted(levels.items())},
            "total_summaries": len(summaries),
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


class AgentContextManager:
    """Elastic context orchestration with 5 atomic operators.

    CONCEPT:KG-2.1 — Derived from LongSeeker's Context-ReAct paradigm.

    Provides fine-grained control over working context using atomic
    operations that preserve important evidence, summarize resolved
    information, discard unhelpful branches, and control context size.

    Example::

        ecm = AgentContextManager(max_tokens=8000)
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

    # ── Unified LCM Operations ────────────────────────────────────────
    # These delegate to ContextCompactor for the actual work and use the
    # KG for persistence. Single entry points — no parallel patterns.

    def compact_thread(
        self,
        thread_id: str,
        engine: Any = None,
        strategy: str = "progressive",
        compaction_threshold: int = 30,
    ) -> dict[str, Any]:
        """Compact a conversation thread's messages into summary DAG nodes.

        CONCEPT:KG-2.1 — LCM Thread Compaction (Unified Entry Point)

        This is the single entry point for all thread compaction. It:
        1. Retrieves messages from the KG for the thread
        2. Delegates to ContextCompactor.compact() for the actual work
        3. Persists results via ContextCompactor.persist_compaction()
        4. Escalates summaries if enough L1 summaries exist

        Args:
            thread_id: Thread node ID to compact.
            engine: IntelligenceGraphEngine instance.
            strategy: Compaction strategy (progressive, drop_middle, summarize_tools).
            compaction_threshold: Min messages before compaction triggers.

        Returns:
            Dict with compaction results and summary IDs.
        """
        if not engine or not getattr(engine, "backend", None):
            return {"error": "No KG engine available"}

        # Get messages for this thread
        messages_data = engine.query_cypher(
            "MATCH (t {id: $tid})-[:CONTAINS]->(m:Message) "
            "RETURN m.id AS id, m.role AS role, m.content AS content "
            "ORDER BY m.timestamp",
            {"tid": thread_id},
        )

        if len(messages_data) < compaction_threshold:
            return {
                "status": "below_threshold",
                "message_count": len(messages_data),
                "threshold": compaction_threshold,
            }

        # Get partition from thread
        thread_data = engine.query_cypher(
            "MATCH (t {id: $tid}) RETURN t.partition AS partition",
            {"tid": thread_id},
        )
        partition = ""
        if thread_data:
            partition = str(thread_data[0].get("partition", "") or "")

        # Convert to message dicts for ContextCompactor
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages_data
        ]
        message_ids = [m["id"] for m in messages_data]

        # Use the existing ContextCompactor for actual compaction work
        compactor = ContextCompactor(max_tokens=self.max_tokens)
        result = compactor.compact(messages, strategy=strategy)

        # Persist via the compactor's KG persistence
        summary_id = compactor.persist_compaction(
            result,
            thread_id=thread_id,
            source_message_ids=message_ids,
            engine=engine,
            level=1,
            partition=partition,
        )

        # Update thread with compaction timestamp
        try:
            engine.backend.execute(
                "MATCH (t {id: $tid}) SET t.last_compacted = $ts",
                {"tid": thread_id, "ts": datetime.now(UTC).isoformat()},
            )
        except Exception as e:
            logger.debug("Failed to update thread compaction timestamp: %s", e)

        # Try escalation if enough L1 summaries exist
        escalated_id = compactor.escalate(thread_id, engine=engine)

        return {
            "status": "compacted",
            "thread_id": thread_id,
            "summary_id": summary_id,
            "escalated_id": escalated_id,
            "messages_compacted": len(messages_data),
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "strategy": strategy,
        }

    @staticmethod
    def expand_summary(
        summary_id: str, engine: Any = None, max_depth: int = 3
    ) -> dict[str, Any]:
        """Drill down from a Summary node to recover original messages.

        CONCEPT:KG-2.1 — LCM Expand (maps to lossless-claw's lcm_expand)

        Traverses the SUMMARIZES edges up to max_depth levels to recover
        the original raw messages. Works at any level of the DAG.

        Args:
            summary_id: Summary node ID to expand.
            engine: IntelligenceGraphEngine instance.
            max_depth: Max DAG levels to traverse.

        Returns:
            Dict with summary info and recovered messages.
        """
        if not engine or not getattr(engine, "backend", None):
            return {"error": "No KG engine available"}

        # Get the summary itself
        summary_data = engine.query_cypher(
            "MATCH (s {id: $sid}) RETURN s.content AS content, s.level AS level, "
            "s.thread_id AS thread_id, s.strategy AS strategy",
            {"sid": summary_id},
        )
        if not summary_data:
            return {"error": f"Summary not found: {summary_id}"}

        # Traverse SUMMARIZES edges to find original messages
        # Use variable-length path traversal
        children = engine.query_cypher(
            f"MATCH (s {{id: $sid}})-[:SUMMARIZES*1..{max_depth}]->(child) "
            "WHERE NOT exists { MATCH (child)-[:SUMMARIZES]->() } "
            "RETURN child.id AS id, child.role AS role, child.content AS content, "
            "child.timestamp AS timestamp "
            "ORDER BY child.timestamp",
            {"sid": summary_id},
        )

        return {
            "summary_id": summary_id,
            "summary_content": summary_data[0].get("content", ""),
            "summary_level": summary_data[0].get("level", 1),
            "thread_id": summary_data[0].get("thread_id", ""),
            "original_messages": children,
            "message_count": len(children),
        }

    @staticmethod
    def grep_memories(
        query: str,
        engine: Any = None,
        partition: str = "",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Full-text search across Summary and Message nodes.

        CONCEPT:KG-2.1 — LCM Grep (maps to lossless-claw's lcm_grep)

        Searches content across both raw messages and their summaries,
        with optional partition filtering. Returns results with context
        (thread ID, partition, summary level).

        Args:
            query: Search query string.
            engine: IntelligenceGraphEngine instance.
            partition: Optional partition filter.
            top_k: Maximum results.

        Returns:
            List of matching nodes with context metadata.
        """
        if not engine or not getattr(engine, "backend", None):
            return []

        partition_filter = ""
        params: dict[str, Any] = {"q": query, "k": top_k}
        if partition:
            partition_filter = "AND n.partition = $partition "
            params["partition"] = partition

        results = engine.query_cypher(
            "MATCH (n) WHERE (n.content CONTAINS $q OR n.summary CONTAINS $q) "
            f"{partition_filter}"
            "RETURN n.id AS id, n.content AS content, n.partition AS partition, "
            "n.level AS level, n.thread_id AS thread_id, n.role AS role "
            f"LIMIT $k",
            params,
        )

        return results

    @staticmethod
    def describe_summary(summary_id: str, engine: Any = None) -> dict[str, Any]:
        """Show a Summary node's metadata and direct children.

        CONCEPT:KG-2.1 — LCM Describe (maps to lossless-claw's lcm_describe)

        Returns the summary's content, level, strategy, and a list of
        its direct children (messages or lower-level summaries).

        Args:
            summary_id: Summary node ID to describe.
            engine: IntelligenceGraphEngine instance.

        Returns:
            Dict with summary metadata and child list.
        """
        if not engine or not getattr(engine, "backend", None):
            return {"error": "No KG engine available"}

        summary = engine.query_cypher(
            "MATCH (s {id: $sid}) "
            "RETURN s.content AS content, s.level AS level, "
            "s.thread_id AS thread_id, s.strategy AS strategy, "
            "s.tokens_before AS tokens_before, s.tokens_after AS tokens_after, "
            "s.timestamp AS timestamp, s.partition AS partition, "
            "s.messages_summarized AS messages_summarized",
            {"sid": summary_id},
        )
        if not summary:
            return {"error": f"Summary not found: {summary_id}"}

        children = engine.query_cypher(
            "MATCH (s {id: $sid})-[:SUMMARIZES]->(child) "
            "RETURN child.id AS id, child.content AS content, "
            "child.role AS role, child.level AS level",
            {"sid": summary_id},
        )

        return {
            "summary_id": summary_id,
            **summary[0],
            "children": children,
            "child_count": len(children),
        }


# For backward compatibility with Goose and older test files
ElasticContextManager = AgentContextManager

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


# --- FROM preemptive_caching.py ---
"""Preemptive Caching Engine.

Combines Markov Transition Forecasting (KG-2.49) with Vectorized
Context Filtering (KG-2.50) to predict and preload KG context.

Configurable and disabled by default.
"""

import logging

logger = logging.getLogger(__name__)


class PreemptiveCacheEngine:
    """Predicts next tool calls and pre-loads required context."""

    def __init__(
        self, markov_forecaster: Any, context_manager: Any, enabled: bool = False
    ):
        self.enabled = enabled
        self.markov_forecaster = markov_forecaster
        self.context_manager = context_manager

    def predict_and_preload(self, current_state: str) -> None:
        """Forecast the next probable states and preload memory."""
        if not self.enabled:
            return

        logger.debug(f"Running Preemptive Cache prediction for state: {current_state}")

        if hasattr(self.markov_forecaster, "predict_next_states"):
            # Predict top 3 likely next steps
            likely_states = self.markov_forecaster.predict_next_states(
                current_state, k=3
            )
            logger.info(f"Predicted likely next states: {likely_states}")

            # Preload and vector-filter the necessary context for those states
            for state in likely_states:
                self._preload_context_for_state(state)

    def _preload_context_for_state(self, target_state: str) -> None:
        """Fetch and filter context into the fast memory layer."""
        # Simulated context retrieval
        predicted_context = {"state": target_state, "data": "preloaded_vectors"}

        # Inject into the context manager's working memory
        if hasattr(self.context_manager, "add_event"):
            self.context_manager.add_event(
                {"type": "cache_preload", "context": predicted_context}
            )


# --- FROM memory_compaction.py ---
#!/usr/bin/python
"""Semantic Compaction (CONCEPT:KG-2.20).

Compacts low-level trace/episodic memory nodes into consolidated,
high-level declarative knowledge representations to prevent graph explosion.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SemanticCompactor:
    """Semantic Compactor for Knowledge Graph trace nodes (CONCEPT:KG-2.20)."""

    def __init__(self, engine: Any, compute_engine: Any = None) -> None:
        self.engine = engine
        self._compute_engine = compute_engine

    def compact_traces(self, agent_id: str, threshold: int = 10) -> int:
        """Find trace/process nodes for a given agent and compact them.

        Traces represent highly verbose execution steps. When they exceed
        the threshold, we compress them into high-level summary edges/nodes
        and prune the verbose details.

        If a Rust-backed GraphComputeEngine was provided, uses the compiled
        ``compact_nodes_by_type`` FFI for zero round-trip compaction.
        """
        if not self.engine:
            return 0

        # Fast path: compiled Rust compaction (avoids N+2 Cypher round-trips)
        if self._compute_engine is not None:
            rust_graph = getattr(self._compute_engine, "_rust_graph", None)
            if rust_graph is not None and hasattr(rust_graph, "compact_nodes_by_type"):
                try:
                    removed = rust_graph.compact_nodes_by_type(
                        "AgentProcess", threshold
                    )
                    if removed:
                        logger.info(
                            f"SemanticCompactor (Rust): Compacted {len(removed)} "
                            f"AgentProcess nodes via compiled FFI"
                        )
                        return len(removed)
                except Exception as e:
                    logger.warning(
                        f"Rust compaction failed, falling back to Cypher: {e}"
                    )

        # Try to find all trace nodes for the agent from the database
        try:
            # We fetch all AgentProcess/Trace nodes linked to this agent
            query = (
                "MATCH (a:Agent {id: $agent_id})-[:HAS_PROCESS]->(p:AgentProcess) "
                "RETURN p.id, p.state, p.tokens_used"
            )
            res = self.engine.backend.execute(query, {"agent_id": agent_id})

            trace_nodes = []
            if res and hasattr(res, "rows"):
                trace_nodes = res.rows
            elif isinstance(res, list):
                trace_nodes = res

            if len(trace_nodes) < threshold:
                return 0

            # Aggregate stats
            total_tokens = 0
            states_summary: dict[str, int] = {}
            for row in trace_nodes:
                # row can be a dict, a list/tuple, or an object
                state = "unknown"
                tokens = 0
                pid = None
                if isinstance(row, dict):
                    pid = row.get("p.id") or row.get("id")
                    state = row.get("p.state") or row.get("state") or "unknown"
                    tokens = row.get("p.tokens_used") or row.get("tokens_used") or 0
                elif isinstance(row, list | tuple) and len(row) >= 3:
                    pid, state, tokens = row[0], row[1], row[2]

                if pid:
                    total_tokens += int(tokens or 0)
                    states_summary[state] = states_summary.get(state, 0) + 1

            summary_node_id = f"summary:agent:{agent_id}:{len(trace_nodes)}_compacted"

            # 1. Create consolidated summary node
            query_summary = (
                "MERGE (s:SemanticSummary {id: $summary_id}) "
                "SET s.name = $name, "
                "    s.compacted_count = $compacted_count, "
                "    s.total_tokens_consumed = $total_tokens, "
                "    s.agent_id = $agent_id"
            )
            self.engine.backend.execute(
                query_summary,
                {
                    "summary_id": summary_node_id,
                    "name": f"Compacted Trace Summary for Agent {agent_id}",
                    "compacted_count": len(trace_nodes),
                    "total_tokens": total_tokens,
                    "agent_id": agent_id,
                },
            )

            # 2. Link summary to Agent
            query_link = (
                "MATCH (a:Agent {id: $agent_id}) "
                "MATCH (s:SemanticSummary {id: $summary_id}) "
                "MERGE (a)-[:HAS_COMPACTED_HISTORY]->(s)"
            )
            self.engine.backend.execute(
                query_link,
                {"agent_id": agent_id, "summary_id": summary_node_id},
            )

            # 3. Delete verbose process/trace nodes to prune database
            deleted = 0
            for row in trace_nodes:
                pid = None
                if isinstance(row, dict):
                    pid = row.get("p.id") or row.get("id")
                elif isinstance(row, list | tuple):
                    pid = row[0]

                if pid:
                    query_delete = "MATCH (p:AgentProcess {id: $pid}) DETACH DELETE p"
                    self.engine.backend.execute(query_delete, {"pid": pid})
                    deleted += 1

            logger.info(
                f"SemanticCompactor: Compacted {deleted} traces for agent '{agent_id}' "
                f"into summary node '{summary_node_id}'"
            )
            return deleted

        except Exception as e:
            logger.error(f"SemanticCompactor failed during compaction: {e}")
            return 0


# --- FROM memento_compressor.py ---
#!/usr/bin/python

"""Generalized Memento Context Compressor.

CONCEPT:KG-2.1 -- Observational Memory Bridge (Extension)

Provides generalized LLM-based state compression for long-running
agents. Takes a block of conversation history and generates a dense
memento preserving formulas, intermediate values, and strategic decisions.
"""

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

MEMENTO_SYSTEM_PROMPT = """You are a state-compression Memento generator for an autonomous agent.
Your task is to take a block of reasoning and conversation history and compress it into a dense Memento.

## Strict Rules:
1. You are NOT summarizing for a human. You are compressing state for an LLM to reason forward from.
2. You MUST extract exact formulas, key intermediate values, commands executed, and their precise outcomes.
3. Keep the strategic decisions and the current execution state (what succeeded, what failed, what is next).
4. Do NOT hallucinate or add outside knowledge.
5. Provide a terse, information-dense output that can act as a drop-in replacement for the raw block.
6. Output ONLY the memento text.
"""


def compress_to_memento(
    engine: IntelligenceGraphEngine,
    messages: list[dict[str, str]],
    *,
    source: str = "agent_runner",
    dry_run: bool = False,
) -> str | None:
    """Compress a block of messages into a dense memento and persist it.

    Args:
        engine: IntelligenceGraphEngine instance.
        messages: The block of raw messages to compress.
        source: The source agent or component name.
        dry_run: If True, do not persist to the KG.

    Returns:
        The generated memento string, or None if compression failed.
    """
    if not messages:
        return None

    # Format block for compression
    transcript_lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        transcript_lines.append(f"[{role}]: {content}")
    block_text = "\n\n".join(transcript_lines)

    try:
        from pydantic_ai import Agent

        from agent_utilities.core.config import (
            DEFAULT_KG_MODEL_ID,
            DEFAULT_LLM_PROVIDER,
        )
        from agent_utilities.core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = Agent(model, system_prompt=MEMENTO_SYSTEM_PROMPT)

        import nest_asyncio

        nest_asyncio.apply()

        user_content = (
            f"## Compress the following block into a Memento:\n\n{block_text}"
        )
        result = agent.run_sync(user_content)
        memento_text = str(getattr(result, "data", result)).strip()
    except Exception as e:
        logger.warning("Memento compression failed: %s", e)
        return None

    if dry_run:
        return memento_text

    _persist_memento(engine, memento_text, source=source)
    return memento_text


def _persist_memento(
    engine: IntelligenceGraphEngine,
    memento_text: str,
    *,
    source: str = "unknown",
) -> None:
    """Persist the generated memento to the Knowledge Graph."""
    if not engine or not engine.backend:
        return

    memento_id = f"mem_{hashlib.md5(memento_text.encode(), usedforsecurity=False).hexdigest()[:10]}"
    current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    props: dict[str, Any] = {
        "name": f"Memento: {current_time}",
        "content": memento_text,
        "source": source,
        "timestamp": current_time,
        "type": "MementoBlock",
    }

    try:
        engine.add_node(memento_id, "Memento", properties=props)
        logger.info("[KG-2.10] Persisted Memento context block (%s)", memento_id)
    except Exception as e:
        logger.debug("Failed to persist Memento: %s", e)


def get_recent_mementos(
    engine: IntelligenceGraphEngine,
    source: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve the most recent mementos for a given source."""
    if not engine or not engine.backend:
        return []

    try:
        rows = engine.backend.execute(
            "MATCH (m:Memento {source: $source}) "
            "RETURN m.content AS content "
            "ORDER BY m.timestamp ASC LIMIT $limit",
            {"source": source, "limit": limit},
        )
        return [r.get("content", "") for r in rows if r.get("content")]
    except Exception as e:
        logger.debug("Failed to retrieve Mementos: %s", e)
        return []
