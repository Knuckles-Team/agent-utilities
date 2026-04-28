"""
Tests for Document Update Pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_utilities.knowledge_graph.pipeline.document_update import DocumentUpdatePipeline
from agent_utilities.knowledge_graph.id_management.unified_id import UnifiedIDManager, UnifiedIDRegistry


class TestDocumentUpdatePipeline:
    """Tests for DocumentUpdatePipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        document_db = AsyncMock()
        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph
        )

        assert pipeline.document_db == document_db
        assert pipeline.vector_db == vector_db
        assert pipeline.knowledge_graph == knowledge_graph
        assert isinstance(pipeline.id_manager, UnifiedIDManager)
        assert isinstance(pipeline.id_registry, UnifiedIDRegistry)

    def test_chunk_document(self):
        """Test document chunking."""
        document_db = AsyncMock()
        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph
        )

        # Test with paragraphs
        content = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = pipeline._chunk_document(content)
        assert len(chunks) == 3

        # Test with single paragraph
        content = "Single paragraph"
        chunks = pipeline._chunk_document(content)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_update_document_not_found(self):
        """Test updating non-existent document."""
        document_db = AsyncMock()
        document_db.find_document.return_value = None
        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph
        )

        with pytest.raises(ValueError, match="Document .* not found"):
            await pipeline.update_document(
                unified_id="doc_nonexistent",
                new_content="New content"
            )

    @pytest.mark.asyncio
    async def test_update_document_soft_deleted(self):
        """Test updating soft-deleted document."""
        document_db = AsyncMock()
        document_db.find_document.return_value = {
            "id": "doc_123",
            "content": "Old content",
            "is_deleted": True
        }
        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph
        )

        with pytest.raises(ValueError, match="Document .* is soft-deleted"):
            await pipeline.update_document(
                unified_id="doc_123",
                new_content="New content"
            )

    @pytest.mark.asyncio
    async def test_update_document_metadata_only(self):
        """Test updating document metadata only (no content change)."""
        document_db = AsyncMock()
        document_db.find_document.return_value = {
            "id": "doc_123",
            "content": "Existing content",
            "metadata": {"title": "Old Title"},
            "is_deleted": False
        }
        document_db.update_document.return_value = True

        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        # Register document in registry
        id_registry = UnifiedIDRegistry()
        id_registry.register_document("doc_123", {"title": "Old Title"})

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        result = await pipeline.update_document(
            unified_id="doc_123",
            metadata_updates={"title": "New Title"},
            regenerate_embeddings=False
        )

        assert result["status"] == "completed"
        assert result["content_changed"] is False
        assert result["metadata_updated"] is True
        assert result["embeddings_regenerated"] is False

        # Verify document was updated
        document_db.update_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_chunks(self):
        """Test getting document chunks."""
        document_db = AsyncMock()
        document_db.find_documents.return_value = [
            {"id": "doc_123_chunk_0000", "chunk_index": 0, "content": "Chunk 1"},
            {"id": "doc_123_chunk_0001", "chunk_index": 1, "content": "Chunk 2"},
            {"id": "doc_123_chunk_0002", "chunk_index": 2, "content": "Chunk 3"}
        ]

        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph
        )

        chunks = await pipeline._get_document_chunks("doc_123")

        assert len(chunks) == 3
        assert chunks[0] == "Chunk 1"
        assert chunks[1] == "Chunk 2"
        assert chunks[2] == "Chunk 3"

    @pytest.mark.asyncio
    async def test_generate_embeddings(self):
        """Test embedding generation (currently uses dummy embeddings)."""
        document_db = AsyncMock()
        vector_db = MagicMock()
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph
        )

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        embeddings = await pipeline._generate_embeddings(chunks)

        assert len(embeddings) == 3
        assert len(embeddings[0]) == 768  # Dummy embedding dimension
