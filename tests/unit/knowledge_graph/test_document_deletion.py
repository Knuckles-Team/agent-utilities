"""
Tests for Document Deletion and Cleanup.
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.id_management.unified_id import (
    UnifiedIDManager,
    UnifiedIDRegistry,
)
from agent_utilities.knowledge_graph.maintenance.document_cleanup import DocumentCleanup
from agent_utilities.knowledge_graph.pipeline.document_deletion import (
    DocumentDeletionPipeline,
)


class TestDocumentDeletionPipeline:
    """Tests for DocumentDeletionPipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        knowledge_graph = MagicMock()

        pipeline = DocumentDeletionPipeline(knowledge_graph=knowledge_graph)

        assert pipeline.knowledge_graph == knowledge_graph
        assert isinstance(pipeline.id_manager, UnifiedIDManager)
        assert isinstance(pipeline.id_registry, UnifiedIDRegistry)

    @pytest.mark.asyncio
    async def test_delete_document_not_in_registry(self):
        """Test deleting document not in registry."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Test content",
        }

        pipeline = DocumentDeletionPipeline(knowledge_graph=knowledge_graph)

        # Should still work even if not in registry
        result = await pipeline.delete_document("doc_123", hard_delete=True)

        assert result["status"] == "completed"
        assert result["unified_id"] == "doc_123"

    @pytest.mark.asyncio
    async def test_delete_document_soft_delete(self):
        """Test soft deleting a document."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Test content",
            "is_deleted": False,
        }

        # Register document
        id_registry = UnifiedIDRegistry()
        id_registry.register_document("doc_123")

        pipeline = DocumentDeletionPipeline(
            knowledge_graph=knowledge_graph, id_registry=id_registry
        )

        result = await pipeline.delete_document("doc_123", hard_delete=False)

        assert result["status"] == "completed"
        assert result["hard_delete"] is False
        assert "doc_123" in id_registry.document_ids  # Still in registry

    @pytest.mark.asyncio
    async def test_delete_document_hard_delete(self):
        """Test hard deleting a document."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Test content",
            "is_deleted": False,
        }

        # Register document
        id_registry = UnifiedIDRegistry()
        id_registry.register_document("doc_123")

        pipeline = DocumentDeletionPipeline(
            knowledge_graph=knowledge_graph, id_registry=id_registry
        )

        result = await pipeline.delete_document("doc_123", hard_delete=True)

        assert result["status"] == "completed"
        assert result["hard_delete"] is True
        assert "doc_123" not in id_registry.document_ids  # Removed from registry

    @pytest.mark.asyncio
    async def test_restore_document(self):
        """Test restoring a soft-deleted document."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Test content",
            "is_deleted": True,
            "deleted_at": "2025-01-01T00:00:00",
        }

        pipeline = DocumentDeletionPipeline(knowledge_graph=knowledge_graph)

        result = await pipeline.restore_document("doc_123")

        assert result["status"] == "restored"
        assert result["unified_id"] == "doc_123"

    @pytest.mark.asyncio
    async def test_restore_document_not_soft_deleted(self):
        """Test restoring document that is not soft-deleted."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Test content",
            "is_deleted": False,
        }

        pipeline = DocumentDeletionPipeline(knowledge_graph=knowledge_graph)

        with pytest.raises(ValueError, match="is not soft-deleted"):
            await pipeline.restore_document("doc_123")

    @pytest.mark.asyncio
    async def test_batch_delete_documents(self):
        """Test batch deleting multiple documents."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Test content",
            "is_deleted": False,
        }

        pipeline = DocumentDeletionPipeline(knowledge_graph=knowledge_graph)

        result = await pipeline.batch_delete_documents(
            ["doc_1", "doc_2", "doc_3"], hard_delete=True
        )

        assert result["total"] == 3
        assert len(result["successful"]) == 3
        assert len(result["failed"]) == 0


class TestDocumentCleanup:
    """Tests for DocumentCleanup class."""

    def test_initialization(self):
        """Test cleanup initialization."""
        knowledge_graph = MagicMock()

        cleanup = DocumentCleanup(knowledge_graph=knowledge_graph)

        assert cleanup.knowledge_graph == knowledge_graph
        assert isinstance(cleanup.id_manager, UnifiedIDManager)
        assert isinstance(cleanup.id_registry, UnifiedIDRegistry)

    @pytest.mark.asyncio
    async def test_cleanup_old_documents(self):
        """Test cleaning up old documents."""
        knowledge_graph = MagicMock()

        # Add some old documents to registry
        id_registry = UnifiedIDRegistry()
        id_registry.register_document(
            "doc_old_1", {"created_at": "2024-01-01T00:00:00"}
        )
        id_registry.register_document(
            "doc_old_2", {"created_at": "2024-01-01T00:00:00"}
        )
        id_registry.register_document("doc_new", {"created_at": "2025-01-01T00:00:00"})

        cleanup = DocumentCleanup(
            knowledge_graph=knowledge_graph, id_registry=id_registry
        )

        result = await cleanup.cleanup_old_documents(age_days=30)

        assert "soft_deleted_cleaned" in result
        assert "hard_deleted_cleaned" in result
        assert "total_cleaned" in result

    @pytest.mark.asyncio
    async def test_cleanup_orphan_embeddings(self):
        """Test cleaning up orphan embeddings."""
        knowledge_graph = MagicMock()

        cleanup = DocumentCleanup(knowledge_graph=knowledge_graph)

        result = await cleanup.cleanup_orphan_embeddings()

        assert "embeddings_deleted" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_cleanup_orphan_graph_nodes(self):
        """Test cleaning up orphan graph nodes."""
        import networkx as nx

        # Use actual NetworkX graph for this test
        knowledge_graph = MagicMock()
        knowledge_graph.graph = nx.MultiDiGraph()

        # Add some orphan chunk nodes to the graph
        # Use proper unified ID format
        knowledge_graph.graph.add_node("doc_orphan1234567890abcdef_chunk_0000")
        knowledge_graph.graph.add_node("doc_orphan1234567890abcdef_chunk_0001")

        cleanup = DocumentCleanup(knowledge_graph=knowledge_graph)

        result = await cleanup.cleanup_orphan_graph_nodes()

        assert result["nodes_deleted"] == 2
        assert not knowledge_graph.graph.has_node(
            "doc_orphan1234567890abcdef_chunk_0000"
        )
        assert not knowledge_graph.graph.has_node(
            "doc_orphan1234567890abcdef_chunk_0001"
        )

    @pytest.mark.asyncio
    async def test_cleanup_soft_deleted_documents(self):
        """Test cleaning up soft-deleted documents."""
        knowledge_graph = MagicMock()

        cleanup = DocumentCleanup(knowledge_graph=knowledge_graph)

        result = await cleanup.cleanup_soft_deleted_documents(age_days=7)

        assert "documents_hard_deleted" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_run_all_cleanup_operations(self):
        """Test running all cleanup operations."""
        knowledge_graph = MagicMock()

        cleanup = DocumentCleanup(knowledge_graph=knowledge_graph)

        result = await cleanup.run_all_cleanup_operations()

        assert "old_documents" in result
        assert "orphan_embeddings" in result
        assert "orphan_graph_nodes" in result
        assert "soft_deleted_documents" in result

    def test_get_cleanup_statistics(self):
        """Test getting cleanup statistics."""
        knowledge_graph = MagicMock()

        # Add some documents to registry
        id_registry = UnifiedIDRegistry()
        id_registry.register_document("doc_1")
        id_registry.register_document("doc_2")

        cleanup = DocumentCleanup(
            knowledge_graph=knowledge_graph, id_registry=id_registry
        )

        stats = cleanup.get_cleanup_statistics()

        assert "total_documents" in stats
        assert "cleanup_ready" in stats
        assert stats["total_documents"] == 2
