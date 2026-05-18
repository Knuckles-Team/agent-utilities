#!/usr/bin/python
from __future__ import annotations

"""Tests for Token-Aware Context Compaction (CONCEPT:KG-2.1)."""


import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compactor():
    from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
        ContextCompactor,
    )

    return ContextCompactor(max_tokens=100, tool_summary_max_length=50)


@pytest.fixture
def large_compactor():
    from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
        ContextCompactor,
    )

    return ContextCompactor(max_tokens=500, tool_summary_max_length=200)


def make_messages(count: int, content_size: int = 50) -> list[dict]:
    """Helper to create test messages with real words for token estimation."""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        # Use real words so token estimation (word-based) works correctly
        words = " ".join([f"word{j}" for j in range(content_size)])
        content = f"Message {i}: {words}"
        msgs.append({"role": role, "content": content})
    return msgs


def make_tool_messages(count: int, tool_size: int = 200) -> list[dict]:
    """Helper to create messages with large tool outputs."""
    msgs = [{"role": "user", "content": "Please analyze the code."}]
    for i in range(count):
        msgs.append({"role": "assistant", "content": f"Calling tool {i}..."})
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "content": f"Tool output {i}: {'data ' * (tool_size // 5)}",
            }
        )
    msgs.append({"role": "assistant", "content": "Here is my analysis."})
    return msgs


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    """Tests for token estimation accuracy."""

    def test_empty_string(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            estimate_tokens,
        )

        assert estimate_tokens("") == 0

    def test_single_word(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            estimate_tokens,
        )

        result = estimate_tokens("hello")
        assert result == 1  # 1 word * 1.33 ≈ 1

    def test_sentence(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            estimate_tokens,
        )

        result = estimate_tokens("The quick brown fox jumps over the lazy dog")
        assert 9 <= result <= 15  # 9 words * 1.33 ≈ 12

    def test_message_estimation(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            estimate_message_tokens,
        )

        msgs = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there, how can I help?"},
        ]
        result = estimate_message_tokens(msgs)
        assert result > 0
        assert result < 100  # Reasonable for short messages


# ---------------------------------------------------------------------------
# Auto-compaction threshold
# ---------------------------------------------------------------------------


class TestAutoCompaction:
    """Tests for should_compact threshold detection."""

    def test_under_threshold(self, compactor):
        msgs = [{"role": "user", "content": "hi"}]
        assert not compactor.should_compact(msgs)

    def test_over_threshold(self, compactor):
        msgs = make_messages(20, content_size=100)
        assert compactor.should_compact(msgs)


# ---------------------------------------------------------------------------
# Summarize tools strategy
# ---------------------------------------------------------------------------


class TestSummarizeToolsStrategy:
    """Tests for the summarize_tools compaction strategy."""

    def test_summarizes_large_tool_outputs(self, compactor):
        msgs = make_tool_messages(3, tool_size=200)
        result = compactor.compact(msgs, strategy="summarize_tools")
        assert result.tokens_after <= result.tokens_before
        assert result.messages_removed > 0
        assert "summarized" in result.summary_text.lower()

    def test_preserves_small_messages(self, large_compactor):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = large_compactor.compact(msgs, strategy="summarize_tools")
        assert result.messages_removed == 0

    def test_no_compaction_when_under_budget(self, large_compactor):
        msgs = [{"role": "user", "content": "Hello"}]
        result = large_compactor.compact(msgs)
        assert result.summary_text == "No compaction needed"
        assert result.tokens_before == result.tokens_after


# ---------------------------------------------------------------------------
# Drop middle strategy
# ---------------------------------------------------------------------------


class TestDropMiddleStrategy:
    """Tests for the drop_middle compaction strategy."""

    def test_drops_middle_messages(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ContextCompactor,
        )

        c = ContextCompactor(max_tokens=50, tool_summary_max_length=50)
        msgs = make_messages(10, content_size=30)
        result = c.compact(msgs, strategy="drop_middle")
        # Should have fewer messages
        assert len(result.messages) < len(msgs)
        assert result.messages_removed > 0
        # First and last messages should be preserved
        assert result.messages[0]["content"] == msgs[0]["content"]
        assert result.messages[-1]["content"] == msgs[-1]["content"]

    def test_inserts_context_note(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ContextCompactor,
        )

        c = ContextCompactor(max_tokens=50, tool_summary_max_length=50)
        msgs = make_messages(10, content_size=30)
        result = c.compact(msgs, strategy="drop_middle")
        # Should have a system note about dropped messages
        context_notes = [
            m
            for m in result.messages
            if m.get("role") == "system" and "Context note" in m.get("content", "")
        ]
        assert len(context_notes) > 0

    def test_too_few_messages(self, compactor):
        msgs = make_messages(3, content_size=10)
        result = compactor.compact(msgs, strategy="drop_middle")
        # Too few to drop - should return as-is (or no compaction needed)
        assert result.messages_removed == 0


# ---------------------------------------------------------------------------
# Progressive strategy
# ---------------------------------------------------------------------------


class TestProgressiveStrategy:
    """Tests for the progressive compaction strategy."""

    def test_progressive_reduces_tokens(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ContextCompactor,
        )

        c = ContextCompactor(max_tokens=50, tool_summary_max_length=50)
        msgs = make_messages(10, content_size=100)
        result = c.compact(msgs, strategy="progressive")
        assert result.tokens_after <= result.tokens_before
        assert result.messages_removed > 0

    def test_preserves_last_messages(self, compactor):
        msgs = make_messages(10, content_size=100)
        result = compactor.compact(msgs, strategy="progressive")
        # Last message should be preserved
        assert result.messages[-1]["content"] == msgs[-1]["content"]

    def test_preserves_first_message(self, compactor):
        msgs = make_messages(10, content_size=100)
        result = compactor.compact(msgs, strategy="progressive")
        # First (system) message should be preserved
        assert result.messages[0]["content"] == msgs[0]["content"]


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class TestCompactionStrategy:
    """Tests for CompactionStrategy enum."""

    def test_string_strategy(self, compactor):
        msgs = make_messages(10, content_size=100)
        result = compactor.compact(msgs, strategy="summarize_tools")
        assert result.strategy_used == "summarize_tools"

    def test_enum_strategy(self, compactor):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            CompactionStrategy,
        )

        msgs = make_messages(10, content_size=100)
        result = compactor.compact(msgs, strategy=CompactionStrategy.DROP_MIDDLE)
        assert result.strategy_used == "drop_middle"


# ---------------------------------------------------------------------------
# Episode node creation
# ---------------------------------------------------------------------------


class TestEpisodeNode:
    """Tests for KG episode node creation."""

    def test_create_episode_node(self, compactor):
        msgs = make_messages(10, content_size=100)
        result = compactor.compact(msgs)
        episode = compactor.create_episode_node(result, session_id="s1")
        assert episode["type"] == "episode"
        assert episode["session_id"] == "s1"
        assert episode["tokens_before"] == result.tokens_before
        assert episode["tokens_after"] == result.tokens_after
        assert "compact:" in episode["id"]

    def test_episode_node_timestamp(self, compactor):
        msgs = make_messages(10, content_size=100)
        result = compactor.compact(msgs)
        episode = compactor.create_episode_node(result)
        assert "timestamp" in episode
        assert len(episode["timestamp"]) > 10


# ---------------------------------------------------------------------------
# CompactedResult model
# ---------------------------------------------------------------------------


class TestCompactedResult:
    """Tests for CompactedResult model."""

    def test_default_values(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            CompactedResult,
        )

        result = CompactedResult()
        assert result.messages == []
        assert result.tokens_before == 0
        assert result.tokens_after == 0
        assert result.compaction_id.startswith("compact:")

    def test_compaction_id_unique(self):
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            CompactedResult,
        )

        r1 = CompactedResult()
        r2 = CompactedResult()
        assert r1.compaction_id != r2.compaction_id


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests that compact_messages in chat_persistence works."""

    def test_compact_messages_wrapper(self):
        from agent_utilities.core.chat_persistence import compact_messages

        msgs = make_messages(5, content_size=50)
        result = compact_messages(msgs, max_tokens=8000)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_prune_large_messages_still_works(self):
        from agent_utilities.core.chat_persistence import prune_large_messages

        msgs = [
            {"role": "user", "content": "short message"},
            {"role": "tool", "content": "x" * 10000},
        ]
        result = prune_large_messages(msgs, max_length=500)
        assert len(result) == 2
        assert len(result[1]["content"]) < 10000
