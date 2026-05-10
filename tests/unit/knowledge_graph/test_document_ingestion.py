"""
Tests for Document Ingestion Pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.id_management.ontological_identifier import (
    OntologicalIdentifierManager,
    OntologicalIdentifierRegistry,
)
from agent_utilities.knowledge_graph.pipeline.document_ingestion import (
    DocumentIngestionPipeline,
)


class TestDocumentIngestionPipeline:
    """Tests for DocumentIngestionPipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        # Create mock backends
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        # Create pipeline
        pipeline = DocumentIngestionPipeline(knowledge_graph=knowledge_graph)

        assert pipeline.knowledge_graph == knowledge_graph
        assert isinstance(pipeline.id_manager, OntologicalIdentifierManager)
        assert isinstance(pipeline.id_registry, OntologicalIdentifierRegistry)
        assert pipeline.get_ingested_documents() == []

    def test_initialization_with_custom_id_manager(self):
        """Test pipeline initialization with custom ID manager."""
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        id_manager = OntologicalIdentifierManager()
        id_registry = OntologicalIdentifierRegistry()

        pipeline = DocumentIngestionPipeline(
            knowledge_graph=knowledge_graph,
            id_manager=id_manager,
            id_registry=id_registry,
        )

        assert pipeline.id_manager == id_manager
        assert pipeline.id_registry == id_registry

    @patch("agent_utilities.knowledge_graph.distillation.distillation_engine.chunk_text")
    def test_chunk_document(self, mock_chunk_text):
        """Test document chunking."""
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentIngestionPipeline(knowledge_graph=knowledge_graph)

        # Test with paragraphs
        content = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        mock_chunk_text.return_value = ["Paragraph 1", "Paragraph 2", "Paragraph 3"]
        chunks = pipeline._chunk_document(content)
        assert len(chunks) == 3
        assert chunks[0] == "Paragraph 1"
        assert chunks[1] == "Paragraph 2"
        assert chunks[2] == "Paragraph 3"

        # Test with single paragraph
        content = "Single paragraph"
        mock_chunk_text.return_value = ["Single paragraph"]
        chunks = pipeline._chunk_document(content)
        assert len(chunks) == 1
        assert chunks[0] == "Single paragraph"

        # Test with empty content
        content = ""
        mock_chunk_text.return_value = [""]
        chunks = pipeline._chunk_document(content)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_get_ingested_documents(self):
        """Test getting ingested documents list."""
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentIngestionPipeline(knowledge_graph=knowledge_graph)

        # Initially empty
        assert pipeline.get_ingested_documents() == []

        # Simulate adding documents (would be done in ingest_document)
        pipeline._ingested_docs.append("doc_1")
        pipeline._ingested_docs.append("doc_2")

        docs = pipeline.get_ingested_documents()
        assert len(docs) == 2
        assert "doc_1" in docs
        assert "doc_2" in docs

    def test_get_registry_statistics(self):
        """Test getting registry statistics."""
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentIngestionPipeline(knowledge_graph=knowledge_graph)

        stats = pipeline.get_registry_statistics()
        assert "total_documents" in stats
        assert "fully_synced" in stats
        assert "partially_synced" in stats
        assert "system_sync_counts" in stats

        # Initially should be empty
        assert stats["total_documents"] == 0

    @pytest.mark.asyncio
    @patch("agent_utilities.core.embedding_utilities.create_embedding_model")
    async def test_generate_embeddings(self, mock_create):
        """Test embedding generation (currently uses dummy embeddings)."""
        mock_create.side_effect = Exception("Not available")
        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = MagicMock()

        pipeline = DocumentIngestionPipeline(knowledge_graph=knowledge_graph)

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        embeddings = await pipeline._generate_embeddings(chunks)

        # Should return dummy embeddings for now
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 768  # Dummy embedding dimension
        assert all(e == 0.0 for e in embeddings[0])  # All zeros for dummy
