"""

CONCEPT:KG-2.0
Tests for Document Update Pipeline.
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.id_management.ontological_identifier import (
    OntologicalIdentifierManager,
    OntologicalIdentifierRegistry,
)
from agent_utilities.knowledge_graph.pipeline.document_update import (
    DocumentUpdatePipeline,
)


class TestDocumentUpdatePipeline:
    """Tests for DocumentUpdatePipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        knowledge_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(knowledge_graph=knowledge_graph)

        assert pipeline.knowledge_graph == knowledge_graph
        assert isinstance(pipeline.id_manager, OntologicalIdentifierManager)
        assert isinstance(pipeline.id_registry, OntologicalIdentifierRegistry)

    def test_chunk_document(self):
        """Test document chunking."""
        knowledge_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(knowledge_graph=knowledge_graph)

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
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = None

        pipeline = DocumentUpdatePipeline(knowledge_graph=knowledge_graph)

        with pytest.raises(ValueError, match="Document .* not found"):
            await pipeline.update_document(
                ontological_identifier="doc_nonexistent", new_content="New content"
            )

    @pytest.mark.asyncio
    async def test_update_document_soft_deleted(self):
        """Test updating soft-deleted document."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Old content",
            "status": "ARCHIVED",
        }

        pipeline = DocumentUpdatePipeline(knowledge_graph=knowledge_graph)

        with pytest.raises(ValueError, match="archived"):
            await pipeline.update_document(
                ontological_identifier="doc_123", new_content="New content"
            )

    @pytest.mark.asyncio
    async def test_update_document_metadata_only(self):
        """Test updating document metadata only (no content change)."""
        knowledge_graph = MagicMock()
        knowledge_graph.graph.nodes.get.return_value = {
            "id": "doc_123",
            "content": "Existing content",
            "metadata": {"title": "Old Title"},
            "status": "ACTIVE",
        }

        # Register document in registry
        id_registry = OntologicalIdentifierRegistry()
        id_registry.register_document("doc_123", {"title": "Old Title"})

        pipeline = DocumentUpdatePipeline(
            knowledge_graph=knowledge_graph, id_registry=id_registry
        )

        result = await pipeline.update_document(
            ontological_identifier="doc_123",
            metadata_updates={"title": "New Title"},
            regenerate_embeddings=False,
        )

        assert result["status"] == "completed"
        assert result["content_changed"] is False
        assert result["metadata_updated"] is True
        assert result["embeddings_regenerated"] is False

        # Verify document was updated in graph
        knowledge_graph.graph.add_node.assert_called()

    @pytest.mark.asyncio
    async def test_get_document_chunks(self):
        """Test getting document chunks."""
        knowledge_graph = MagicMock()
        # Mock edges to return 3 chunks
        knowledge_graph.graph.edges.return_value = [
            ("doc_123", "doc_123_chunk_0000", {"relationship_type": "HAS_CHUNK"}),
            ("doc_123", "doc_123_chunk_0001", {"relationship_type": "HAS_CHUNK"}),
            ("doc_123", "doc_123_chunk_0002", {"relationship_type": "HAS_CHUNK"}),
        ]
        knowledge_graph.graph.nodes.get.side_effect = lambda x: {
            "doc_123_chunk_0000": {
                "id": "doc_123_chunk_0000",
                "chunk_index": 0,
                "content": "Chunk 1",
            },
            "doc_123_chunk_0001": {
                "id": "doc_123_chunk_0001",
                "chunk_index": 1,
                "content": "Chunk 2",
            },
            "doc_123_chunk_0002": {
                "id": "doc_123_chunk_0002",
                "chunk_index": 2,
                "content": "Chunk 3",
            },
        }.get(x)

        pipeline = DocumentUpdatePipeline(knowledge_graph=knowledge_graph)

        chunks = await pipeline._get_document_chunks("doc_123")

        assert len(chunks) == 3
        assert chunks[0] == "Chunk 1"
        assert chunks[1] == "Chunk 2"
        assert chunks[2] == "Chunk 3"

    @pytest.mark.asyncio
    async def test_generate_embeddings(self):
        """Test embedding generation (currently uses dummy embeddings)."""
        knowledge_graph = MagicMock()

        pipeline = DocumentUpdatePipeline(knowledge_graph=knowledge_graph)

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        embeddings = await pipeline._generate_embeddings(chunks)

        assert len(embeddings) == 3
        assert len(embeddings[0]) == 768  # Dummy embedding dimension
