"""
Performance Tests for Document Pipeline.

Benchmarks document operations across different backends:
- Document ingestion speed
- Document update speed
- Document deletion speed
- Memory usage profiling
- Backend comparison
"""

import time
import tracemalloc
from unittest.mock import MagicMock

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.backends.document_storage import (
    SQLiteBackend,
    SQLiteMemoryBackend,
)
from agent_utilities.knowledge_graph.id_management.unified_id import UnifiedIDRegistry
from agent_utilities.knowledge_graph.pipeline.document_deletion import (
    DocumentDeletionPipeline,
)
from agent_utilities.knowledge_graph.pipeline.document_ingestion import (
    DocumentIngestionPipeline,
)
from agent_utilities.knowledge_graph.pipeline.document_update import (
    DocumentUpdatePipeline,
)


class TestDocumentIngestionPerformance:
    """Performance tests for document ingestion."""

    @pytest.fixture
    def setup_components(self):
        """Set up test components."""
        document_db = SQLiteMemoryBackend()
        document_db.create_collection("documents")
        document_db.create_collection("chunks")

        vector_db = MagicMock()
        vector_db.insert_documents = MagicMock()
        vector_db.delete_documents = MagicMock()
        vector_db.semantic_search = MagicMock(return_value=[])
        vector_db.get_documents_by_ids = MagicMock(return_value=[])

        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = nx.MultiDiGraph()

        id_registry = UnifiedIDRegistry()

        return {
            "document_db": document_db,
            "vector_db": vector_db,
            "knowledge_graph": knowledge_graph,
            "id_registry": id_registry
        }

    @pytest.mark.asyncio
    async def test_ingestion_small_document(self, setup_components):
        """Test ingestion performance for small document."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        # Small document (~100 words)
        content = " ".join(["word"] * 100)

        start_time = time.time()
        result = await pipeline.ingest_document(
            file_path="/path/to/small.pdf",
            content=content,
            metadata={"title": "Small Document"}
        )
        end_time = time.time()

        ingestion_time = end_time - start_time

        print(f"\nSmall document ingestion time: {ingestion_time:.4f}s")

        # Should complete in less than 1 second
        assert ingestion_time < 1.0
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_ingestion_medium_document(self, setup_components):
        """Test ingestion performance for medium document."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        # Medium document (~1000 words)
        content = " ".join(["word"] * 1000)

        start_time = time.time()
        result = await pipeline.ingest_document(
            file_path="/path/to/medium.pdf",
            content=content,
            metadata={"title": "Medium Document"}
        )
        end_time = time.time()

        ingestion_time = end_time - start_time

        print(f"\nMedium document ingestion time: {ingestion_time:.4f}s")

        # Should complete in less than 5 seconds
        assert ingestion_time < 5.0
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_ingestion_large_document(self, setup_components):
        """Test ingestion performance for large document."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        # Large document (~10000 words)
        content = " ".join(["word"] * 10000)

        start_time = time.time()
        result = await pipeline.ingest_document(
            file_path="/path/to/large.pdf",
            content=content,
            metadata={"title": "Large Document"}
        )
        end_time = time.time()

        ingestion_time = end_time - start_time

        print(f"\nLarge document ingestion time: {ingestion_time:.4f}s")

        # Should complete in less than 30 seconds
        assert ingestion_time < 30.0
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_batch_ingestion(self, setup_components):
        """Test batch ingestion performance."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        # Ingest 10 documents
        start_time = time.time()
        for i in range(10):
            await pipeline.ingest_document(
                file_path=f"/path/to/doc{i}.pdf",
                content=f"Content for document {i}",
                metadata={"title": f"Document {i}"}
            )
        end_time = time.time()

        batch_time = end_time - start_time
        avg_time = batch_time / 10

        print(f"\nBatch ingestion time (10 docs): {batch_time:.4f}s")
        print(f"Average time per document: {avg_time:.4f}s")

        # Average should be less than 1 second per document
        assert avg_time < 1.0


class TestDocumentUpdatePerformance:
    """Performance tests for document updates."""

    @pytest.fixture
    def setup_components(self):
        """Set up test components."""
        document_db = SQLiteMemoryBackend()
        document_db.create_collection("documents")
        document_db.create_collection("chunks")

        vector_db = MagicMock()
        vector_db.insert_documents = MagicMock()
        vector_db.delete_documents = MagicMock()
        vector_db.semantic_search = MagicMock(return_value=[])
        vector_db.get_documents_by_ids = MagicMock(return_value=[])

        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = nx.MultiDiGraph()

        id_registry = UnifiedIDRegistry()

        return {
            "document_db": document_db,
            "vector_db": vector_db,
            "knowledge_graph": knowledge_graph,
            "id_registry": id_registry
        }

    @pytest.mark.asyncio
    async def test_update_performance(self, setup_components):
        """Test document update performance."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        # First ingest a document
        ingestion_pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        ingest_result = await ingestion_pipeline.ingest_document(
            file_path="/path/to/document.pdf",
            content="Original content",
            metadata={"title": "Test Document"}
        )

        unified_id = ingest_result["unified_id"]

        # Now update it
        update_pipeline = DocumentUpdatePipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        start_time = time.time()
        update_result = await update_pipeline.update_document(
            unified_id=unified_id,
            new_content="Updated content with new text",
            metadata_updates={"title": "Updated Document"}
        )
        end_time = time.time()

        update_time = end_time - start_time

        print(f"\nDocument update time: {update_time:.4f}s")

        # Should complete in less than 1 second
        assert update_time < 1.0
        assert update_result["status"] == "completed"


class TestDocumentDeletionPerformance:
    """Performance tests for document deletion."""

    @pytest.fixture
    def setup_components(self):
        """Set up test components."""
        document_db = SQLiteMemoryBackend()
        document_db.create_collection("documents")
        document_db.create_collection("chunks")

        vector_db = MagicMock()
        vector_db.insert_documents = MagicMock()
        vector_db.delete_documents = MagicMock()
        vector_db.semantic_search = MagicMock(return_value=[])
        vector_db.get_documents_by_ids = MagicMock(return_value=[])

        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = nx.MultiDiGraph()

        id_registry = UnifiedIDRegistry()

        return {
            "document_db": document_db,
            "vector_db": vector_db,
            "knowledge_graph": knowledge_graph,
            "id_registry": id_registry
        }

    @pytest.mark.asyncio
    async def test_soft_delete_performance(self, setup_components):
        """Test soft delete performance."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        # First ingest a document
        ingestion_pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        ingest_result = await ingestion_pipeline.ingest_document(
            file_path="/path/to/document.pdf",
            content="Test content",
            metadata={"title": "Test Document"}
        )

        unified_id = ingest_result["unified_id"]

        # Now soft delete it
        deletion_pipeline = DocumentDeletionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        start_time = time.time()
        delete_result = await deletion_pipeline.delete_document(
            unified_id=unified_id,
            hard_delete=False
        )
        end_time = time.time()

        delete_time = end_time - start_time

        print(f"\nSoft delete time: {delete_time:.4f}s")

        # Should complete in less than 1 second
        assert delete_time < 1.0
        assert delete_result["hard_delete"] is False

    @pytest.mark.asyncio
    async def test_hard_delete_performance(self, setup_components):
        """Test hard delete performance."""
        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        # First ingest a document
        ingestion_pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        ingest_result = await ingestion_pipeline.ingest_document(
            file_path="/path/to/document.pdf",
            content="Test content",
            metadata={"title": "Test Document"}
        )

        unified_id = ingest_result["unified_id"]

        # Now hard delete it
        deletion_pipeline = DocumentDeletionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        start_time = time.time()
        delete_result = await deletion_pipeline.delete_document(
            unified_id=unified_id,
            hard_delete=True
        )
        end_time = time.time()

        delete_time = end_time - start_time

        print(f"\nHard delete time: {delete_time:.4f}s")

        # Should complete in less than 1 second
        assert delete_time < 1.0
        assert delete_result["hard_delete"] is True


class TestMemoryProfiling:
    """Memory usage profiling tests."""

    @pytest.fixture
    def setup_components(self):
        """Set up test components."""
        document_db = SQLiteMemoryBackend()
        document_db.create_collection("documents")
        document_db.create_collection("chunks")

        vector_db = MagicMock()
        vector_db.insert_documents = MagicMock()
        vector_db.delete_documents = MagicMock()
        vector_db.semantic_search = MagicMock(return_value=[])
        vector_db.get_documents_by_ids = MagicMock(return_value=[])

        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = nx.MultiDiGraph()

        id_registry = UnifiedIDRegistry()

        return {
            "document_db": document_db,
            "vector_db": vector_db,
            "knowledge_graph": knowledge_graph,
            "id_registry": id_registry
        }

    @pytest.mark.asyncio
    async def test_memory_usage_ingestion(self, setup_components):
        """Test memory usage during document ingestion."""
        tracemalloc.start()

        document_db = setup_components["document_db"]
        vector_db = setup_components["vector_db"]
        knowledge_graph = setup_components["knowledge_graph"]
        id_registry = setup_components["id_registry"]

        pipeline = DocumentIngestionPipeline(
            document_db=document_db,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry
        )

        # Ingest a medium document
        content = " ".join(["word"] * 1000)

        snapshot1 = tracemalloc.take_snapshot()

        await pipeline.ingest_document(
            file_path="/path/to/document.pdf",
            content=content,
            metadata={"title": "Test Document"}
        )

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print("\nMemory usage statistics:")
        for stat in top_stats[:5]:
            print(stat)

        tracemalloc.stop()

        # Memory increase should be reasonable (< 10 MB for a 1000-word document)
        total_increase = sum(stat.size_diff for stat in top_stats)
        print(f"\nTotal memory increase: {total_increase / 1024 / 1024:.2f} MB")
        assert total_increase < 10 * 1024 * 1024  # 10 MB


class TestBackendComparison:
    """Performance comparison across different backends."""

    @pytest.mark.asyncio
    async def test_sqlite_vs_memory_backend(self):
        """Compare SQLite file backend vs in-memory backend."""
        # In-memory backend
        memory_backend = SQLiteMemoryBackend()
        memory_backend.create_collection("documents")
        memory_backend.create_collection("chunks")

        # File-based backend (using temp file)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            file_backend = SQLiteBackend(db_path=f.name)
            file_backend.create_collection("documents")
            file_backend.create_collection("chunks")

        vector_db = MagicMock()
        vector_db.insert_documents = MagicMock()
        vector_db.delete_documents = MagicMock()
        vector_db.semantic_search = MagicMock(return_value=[])
        vector_db.get_documents_by_ids = MagicMock(return_value=[])

        knowledge_graph = MagicMock()
        knowledge_graph.nx_graph = nx.MultiDiGraph()

        id_registry_memory = UnifiedIDRegistry()
        id_registry_file = UnifiedIDRegistry()

        # Test in-memory backend
        pipeline_memory = DocumentIngestionPipeline(
            document_db=memory_backend,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry_memory
        )

        content = " ".join(["word"] * 1000)

        start_time = time.time()
        await pipeline_memory.ingest_document(
            file_path="/path/to/document.pdf",
            content=content,
            metadata={"title": "Test Document"}
        )
        memory_time = time.time() - start_time

        # Test file-based backend
        pipeline_file = DocumentIngestionPipeline(
            document_db=file_backend,
            vector_db=vector_db,
            knowledge_graph=knowledge_graph,
            id_registry=id_registry_file
        )

        start_time = time.time()
        await pipeline_file.ingest_document(
            file_path="/path/to/document.pdf",
            content=content,
            metadata={"title": "Test Document"}
        )
        file_time = time.time() - start_time

        print(f"\nIn-memory backend time: {memory_time:.4f}s")
        print(f"File-based backend time: {file_time:.4f}s")
        print(f"Speedup: {file_time / memory_time:.2f}x")

        # In-memory should be faster (or at least comparable)
        assert memory_time <= file_time * 5  # Allow some tolerance for flaky CI environments

        # Cleanup
        import os
        os.unlink(f.name)
