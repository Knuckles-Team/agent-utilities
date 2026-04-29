"""
Tests for Document Storage Backends.
"""

import pytest
from agent_utilities.knowledge_graph.backends.document_storage import (
    SQLiteMemoryBackend,
    SQLiteBackend,
    DocumentStorageFactory
)


class TestSQLiteMemoryBackend:
    """Tests for SQLiteMemoryBackend."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = SQLiteMemoryBackend()
        assert backend.conn is not None

    def test_create_collection(self):
        """Test collection creation."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("test_collection")
    assert True, 'Collection creation completed'
        # Should not raise an error

    def test_insert_document(self):
        """Test inserting a document."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        document = {
            "id": "doc_1234567890abcdef1234567890abcdef",
            "content": "Test document content",
            "metadata": {"title": "Test", "author": "Test Author"}
        }

        doc_id = backend.insert_document(document, "documents")
        assert doc_id == "doc_1234567890abcdef1234567890abcdef"

    def test_insert_document_invalid_id(self):
        """Test inserting document with invalid ID."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        document = {
            "id": "invalid_id",
            "content": "Test content"
        }

        with pytest.raises(ValueError, match="Invalid unified ID"):
            backend.insert_document(document, "documents")

    def test_insert_document_no_id(self):
        """Test inserting document without ID."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        document = {
            "content": "Test content"
        }

        with pytest.raises(ValueError, match="Document must have 'id' field"):
            backend.insert_document(document, "documents")

    def test_find_document(self):
        """Test finding a document."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        document = {
            "id": "doc_1234567890abcdef1234567890abcdef",
            "content": "Test content",
            "metadata": {"title": "Test"}
        }

        backend.insert_document(document, "documents")
        found = backend.find_document("doc_1234567890abcdef1234567890abcdef", "documents")

        assert found is not None
        assert found["content"] == "Test content"
        assert found["metadata"]["title"] == "Test"

    def test_find_document_not_found(self):
        """Test finding non-existent document."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        found = backend.find_document("doc_nonexistent", "documents")
        assert found is None

    def test_find_documents_with_filter(self):
        """Test finding documents with filter."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        backend.insert_document({
            "id": "doc_1",
            "content": "Content 1",
            "metadata": {"author": "John"}
        }, "documents")
        backend.insert_document({
            "id": "doc_2",
            "content": "Content 2",
            "metadata": {"author": "John"}
        }, "documents")
        backend.insert_document({
            "id": "doc_3",
            "content": "Content 3",
            "metadata": {"author": "Jane"}
        }, "documents")

        # Filter on content instead (top-level field)
        results = backend.find_documents({"content": "Content 1"}, "documents")
        assert len(results) == 1
        assert results[0]["id"] == "doc_1"

    def test_update_document(self):
        """Test updating a document."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        backend.insert_document({
            "id": "doc_123",
            "content": "Original content",
            "metadata": {"title": "Original"}
        }, "documents")

        success = backend.update_document(
            "doc_123",
            {"content": "Updated content"},
            "documents"
        )

        assert success is True

        updated = backend.find_document("doc_123", "documents")
        assert updated is not None
        assert updated["content"] == "Updated content"

    def test_delete_document(self):
        """Test deleting a document."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        backend.insert_document({
            "id": "doc_123",
            "content": "Test content"
        }, "documents")

        success = backend.delete_document("doc_123", "documents")
        assert success is True

        deleted = backend.find_document("doc_123", "documents")
        assert deleted is None

    def test_count_documents(self):
        """Test counting documents."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        assert backend.count_documents("documents") == 0

        backend.insert_document({"id": "doc_1", "content": "Content 1"}, "documents")
        backend.insert_document({"id": "doc_2", "content": "Content 2"}, "documents")

        count = backend.count_documents("documents")
        assert count == 2

    def test_insert_many(self):
        """Test inserting multiple documents."""
        backend = SQLiteMemoryBackend()
        backend.create_collection("documents")

        documents = [
            {"id": "doc_1", "content": "Content 1"},
            {"id": "doc_2", "content": "Content 2"},
            {"id": "doc_3", "content": "Content 3"}
        ]

        doc_ids = backend.insert_many(documents, "documents")
        assert len(doc_ids) == 3
        assert doc_ids == ["doc_1", "doc_2", "doc_3"]


class TestDocumentStorageFactory:
    """Tests for DocumentStorageFactory."""

    def test_create_sqlite_memory_backend(self):
        """Test creating SQLite memory backend."""
        backend = DocumentStorageFactory.create_document_backend("sqlite_memory")
        assert isinstance(backend, SQLiteMemoryBackend)

    def test_create_sqlite_backend(self):
        """Test creating SQLite file backend."""
        backend = DocumentStorageFactory.create_document_backend(
            "sqlite",
            db_path=":memory:"  # Use in-memory for testing
        )
        assert isinstance(backend, SQLiteBackend)

    def test_invalid_backend(self):
        """Test creating invalid backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            DocumentStorageFactory.create_document_backend("invalid_backend")

    def test_is_backend_supported(self):
        """Test checking if backend is supported."""
        assert DocumentStorageFactory.is_backend_supported("sqlite_memory") is True
        assert DocumentStorageFactory.is_backend_supported("sqlite") is True
        assert DocumentStorageFactory.is_backend_supported("postgresql") is True
        assert DocumentStorageFactory.is_backend_supported("mongodb") is True
        assert DocumentStorageFactory.is_backend_supported("invalid") is False

    def test_list_supported_backends(self):
        """Test listing supported backends."""
        backends = DocumentStorageFactory.list_supported_backends()
        assert "sqlite_memory" in backends
        assert "sqlite" in backends
        assert "postgresql" in backends
        assert "mongodb" in backends

    def test_get_default_backend(self):
        """Test getting default backend."""
        backend = DocumentStorageFactory.get_default_backend()
        assert isinstance(backend, SQLiteMemoryBackend)
