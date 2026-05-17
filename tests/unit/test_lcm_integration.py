"""Tests for Lossless Context Management (LCM) integration.

Validates the KG-persistent compaction pipeline, Summary DAG construction,
escalation logic, and the unified ElasticContextManager LCM operations.

CONCEPT:KG-2.1 — Lossless Context Management
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_engine():
    """Create a mock IntelligenceGraphEngine with KG methods stubbed."""
    engine = MagicMock()
    engine.query_cypher = MagicMock(return_value=[])
    engine.add_node = MagicMock()
    engine.link_nodes = MagicMock()
    engine.search = MagicMock(return_value=[])
    engine.backend = MagicMock()  # Ensure backend check passes
    return engine


@pytest.fixture()
def compactor():
    """Create a ContextCompactor instance."""
    from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
        ContextCompactor,
    )

    return ContextCompactor(max_tokens=16000)


@pytest.fixture()
def compacted_result():
    """Create a sample CompactedResult for testing persistence."""
    from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
        CompactedResult,
    )

    return CompactedResult(
        messages=[{"role": "assistant", "content": "Summary of conversation."}],
        tokens_before=5000,
        tokens_after=200,
        strategy_used="progressive",
        summary_text="Summary of 5 messages about architecture decisions.",
        messages_removed=5,
    )


# ---------------------------------------------------------------------------
# ContextCompactor — KG Persistence Tests
# ---------------------------------------------------------------------------


class TestContextCompactorPersistence:
    """Tests for persist_compaction() and escalate() methods."""

    def test_persist_compaction_creates_summary_node(
        self, compactor, compacted_result, mock_engine
    ):
        """persist_compaction() should call engine.add_node to create a Summary."""
        source_ids = ["msg_001", "msg_002", "msg_003", "msg_004", "msg_005"]

        result_id = compactor.persist_compaction(
            result=compacted_result,
            thread_id="thread_abc",
            source_message_ids=source_ids,
            engine=mock_engine,
        )

        # Should return a summary ID (or None if engine issues)
        # The method calls add_node internally
        mock_engine.add_node.assert_called()

    def test_persist_compaction_links_source_messages(
        self, compactor, compacted_result, mock_engine
    ):
        """persist_compaction() should create SUMMARIZES edges to source messages."""
        source_ids = ["msg_a", "msg_b", "msg_c"]

        compactor.persist_compaction(
            result=compacted_result,
            thread_id="thread_123",
            source_message_ids=source_ids,
            engine=mock_engine,
        )

        # link_nodes should be called for SUMMARIZES edges
        assert mock_engine.link_nodes.call_count >= len(source_ids)

    def test_escalate_returns_none_without_engine(self, compactor):
        """escalate() should return None when no engine is provided."""
        result = compactor.escalate(thread_id="thread_solo", engine=None)
        assert result is None

    def test_escalate_returns_none_for_insufficient_summaries(
        self, compactor, mock_engine
    ):
        """escalate() should return None when fewer than batch_size summaries exist."""
        # Mock level query returning only 1 summary at current level
        mock_engine.query_cypher.side_effect = [
            [{"max_level": 1}],  # max level query
            [{"id": "sum_only", "content": "Solo summary.", "level": 1}],  # summaries
        ]

        result = compactor.escalate(
            thread_id="thread_solo",
            engine=mock_engine,
            batch_size=5,
        )

        # Should return None since we have fewer summaries than batch_size
        assert result is None

    def test_get_summary_dag_returns_structured_result(self, mock_engine):
        """get_summary_dag() should return a dict with levels and total_summaries."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ContextCompactor,
        )

        mock_engine.query_cypher.return_value = [
            {"id": "s1", "content": "L1 summary", "level": 1,
             "tokens_before": 1000, "tokens_after": 100,
             "strategy": "progressive", "timestamp": "2026-01-01"},
        ]

        dag = ContextCompactor.get_summary_dag(
            thread_id="thread_dag",
            engine=mock_engine,
        )

        assert isinstance(dag, dict)
        assert "thread_id" in dag
        assert dag["thread_id"] == "thread_dag"

    def test_get_summary_dag_empty_without_engine(self):
        """get_summary_dag() should return empty levels without an engine."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ContextCompactor,
        )

        dag = ContextCompactor.get_summary_dag(
            thread_id="thread_none",
            engine=None,
        )

        assert dag["total_summaries"] == 0
        assert dag["levels"] == {}


# ---------------------------------------------------------------------------
# ElasticContextManager — LCM Operations Tests
# ---------------------------------------------------------------------------


class TestElasticContextManagerLCM:
    """Tests for compact_thread, expand_summary, grep_memories, describe_summary."""

    def test_compact_thread_below_threshold(self, mock_engine):
        """compact_thread() should return 'below_threshold' for small threads."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        ecm = ElasticContextManager(max_tokens=16000)

        # Mock a thread with only 5 messages (below default threshold of 30)
        mock_engine.query_cypher.return_value = [
            {"id": f"msg_{i}", "role": "user", "content": f"Message {i}"}
            for i in range(5)
        ]

        result = ecm.compact_thread(
            thread_id="thread_small",
            engine=mock_engine,
            strategy="progressive",
            compaction_threshold=30,
        )

        assert isinstance(result, dict)
        assert result["status"] == "below_threshold"
        assert result["message_count"] == 5

    def test_compact_thread_no_engine_returns_error(self):
        """compact_thread() should return error when no engine provided."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        ecm = ElasticContextManager(max_tokens=16000)

        result = ecm.compact_thread(
            thread_id="thread_no_engine",
            engine=None,
        )

        assert isinstance(result, dict)
        assert "error" in result

    def test_expand_summary_no_engine_returns_error(self):
        """expand_summary() should return error when no engine provided."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        result = ElasticContextManager.expand_summary(
            summary_id="sum_001",
            engine=None,
        )

        assert isinstance(result, dict)
        assert "error" in result

    def test_expand_summary_returns_messages(self, mock_engine):
        """expand_summary() should traverse SUMMARIZES edges to recover messages."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        # First call: summary itself, Second call: children
        mock_engine.query_cypher.side_effect = [
            [{"content": "Summary text", "level": 1,
              "thread_id": "t1", "strategy": "progressive"}],
            [
                {"id": "msg_1", "role": "user", "content": "Original msg 1",
                 "timestamp": "2026-01-01"},
                {"id": "msg_2", "role": "assistant", "content": "Original msg 2",
                 "timestamp": "2026-01-02"},
            ],
        ]

        result = ElasticContextManager.expand_summary(
            summary_id="sum_001",
            engine=mock_engine,
        )

        assert isinstance(result, dict)
        assert result["message_count"] == 2

    def test_grep_memories_empty_without_engine(self):
        """grep_memories() should return empty list without engine."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        result = ElasticContextManager.grep_memories(
            query="compaction",
            engine=None,
        )

        assert result == []

    def test_grep_memories_with_partition_filter(self, mock_engine):
        """grep_memories() should support partition filtering."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        mock_engine.query_cypher.return_value = [
            {"id": "msg_1", "content": "Matching content",
             "partition": "antigravity", "level": None,
             "thread_id": "t1", "role": "user"},
        ]

        result = ElasticContextManager.grep_memories(
            query="compaction",
            engine=mock_engine,
            partition="antigravity",
        )

        assert isinstance(result, list)
        assert len(result) == 1

    def test_describe_summary_no_engine_returns_error(self):
        """describe_summary() should return error without engine."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        result = ElasticContextManager.describe_summary(
            summary_id="sum_002",
            engine=None,
        )

        assert isinstance(result, dict)
        assert "error" in result

    def test_describe_summary_returns_metadata(self, mock_engine):
        """describe_summary() should return summary metadata and child count."""
        from agent_utilities.knowledge_graph.memory.elastic_context_manager import (
            ElasticContextManager,
        )

        # First call: summary metadata, Second call: children
        mock_engine.query_cypher.side_effect = [
            [{"content": "Summary of 5 msgs", "level": 1,
              "thread_id": "t1", "strategy": "progressive",
              "tokens_before": 5000, "tokens_after": 200,
              "timestamp": "2026-01-01", "partition": "antigravity",
              "messages_summarized": 5}],
            [
                {"id": "msg_1", "content": "Child 1", "role": "user", "level": None},
                {"id": "msg_2", "content": "Child 2", "role": "assistant", "level": None},
            ],
        ]

        result = ElasticContextManager.describe_summary(
            summary_id="sum_002",
            engine=mock_engine,
        )

        assert isinstance(result, dict)
        assert "error" not in result
