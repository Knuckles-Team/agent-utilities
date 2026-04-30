"""
Tests for Unified ID System.
"""

from agent_utilities.knowledge_graph.id_management.unified_id import (
    UnifiedIDManager,
    UnifiedIDRegistry,
)


class TestUnifiedIDManager:
    """Tests for UnifiedIDManager class."""

    def test_generate_document_id(self):
        """Test document ID generation."""
        doc_id = UnifiedIDManager.generate_document_id()
        assert doc_id.startswith("doc_")
        assert len(doc_id) == 36  # "doc_" + 32 char UUID hex
        assert UnifiedIDManager.is_unified_id(doc_id) is True

    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        chunk_id = UnifiedIDManager.generate_chunk_id(doc_id, 0)
        assert chunk_id == f"{doc_id}_chunk_0000"

        chunk_id_5 = UnifiedIDManager.generate_chunk_id(doc_id, 5)
        assert chunk_id_5 == f"{doc_id}_chunk_0005"

        chunk_id_100 = UnifiedIDManager.generate_chunk_id(doc_id, 100)
        assert chunk_id_100 == f"{doc_id}_chunk_0100"

    def test_generate_entity_id(self):
        """Test entity ID generation."""
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        entity_id = UnifiedIDManager.generate_entity_id(doc_id, "PERSON", 0)
        assert entity_id == f"{doc_id}_entity_PERSON_0000"

        entity_id_5 = UnifiedIDManager.generate_entity_id(doc_id, "ORGANIZATION", 5)
        assert entity_id_5 == f"{doc_id}_entity_ORGANIZATION_0005"

    def test_is_unified_id(self):
        """Test unified ID validation."""
        assert UnifiedIDManager.is_unified_id("doc_1234567890abcdef") is True
        assert UnifiedIDManager.is_unified_id("doc_1234567890abcdef_chunk_0000") is True
        assert UnifiedIDManager.is_unified_id("doc_1234567890abcdef_entity_PERSON_0000") is True
        assert UnifiedIDManager.is_unified_id("not_a_doc_id") is False
        assert UnifiedIDManager.is_unified_id("") is False
        assert UnifiedIDManager.is_unified_id("12345") is False

    def test_extract_doc_id(self):
        """Test extracting document ID from unified ID."""
        doc_id = "doc_1234567890abcdef1234567890abcdef"

        # Extract from document ID
        assert UnifiedIDManager.extract_doc_id(doc_id) == doc_id

        # Extract from chunk ID
        chunk_id = f"{doc_id}_chunk_0005"
        assert UnifiedIDManager.extract_doc_id(chunk_id) == doc_id

        # Extract from entity ID
        entity_id = f"{doc_id}_entity_PERSON_0000"
        assert UnifiedIDManager.extract_doc_id(entity_id) == doc_id

        # Invalid ID
        assert UnifiedIDManager.extract_doc_id("invalid_id") is None

    def test_get_id_type(self):
        """Test getting ID type."""
        doc_id = "doc_1234567890abcdef1234567890abcdef"

        assert UnifiedIDManager.get_id_type(doc_id) == "document"
        assert UnifiedIDManager.get_id_type(f"{doc_id}_chunk_0000") == "chunk"
        assert UnifiedIDManager.get_id_type(f"{doc_id}_entity_PERSON_0000") == "entity"
        assert UnifiedIDManager.get_id_type("invalid") is None

    def test_parse_chunk_id(self):
        """Test parsing chunk ID."""
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        chunk_id = f"{doc_id}_chunk_0042"

        result = UnifiedIDManager.parse_chunk_id(chunk_id)
        assert result is not None
        assert result["doc_id"] == doc_id
        assert result["chunk_index"] == 42

        # Invalid chunk ID
        assert UnifiedIDManager.parse_chunk_id("invalid") is None
        assert UnifiedIDManager.parse_chunk_id(doc_id) is None

    def test_parse_entity_id(self):
        """Test parsing entity ID."""
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        entity_id = f"{doc_id}_entity_PERSON_0015"

        result = UnifiedIDManager.parse_entity_id(entity_id)
        assert result is not None
        assert result["doc_id"] == doc_id
        assert result["entity_type"] == "PERSON"
        assert result["entity_index"] == 15

        # Invalid entity ID
        assert UnifiedIDManager.parse_entity_id("invalid") is None
        assert UnifiedIDManager.parse_entity_id(doc_id) is None


class TestUnifiedIDRegistry:
    """Tests for UnifiedIDRegistry class."""

    def test_register_document(self):
        """Test registering a document."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        metadata = {"title": "Test Document", "author": "Test Author"}

        registry.register_document(doc_id, metadata)

        assert doc_id in registry.document_ids
        assert registry.document_ids[doc_id]["metadata"] == metadata
        assert registry.document_ids[doc_id]["status"] == "registered"
        assert registry.document_ids[doc_id]["systems"]["document_db"] is False
        assert registry.document_ids[doc_id]["systems"]["vector_db"] is False
        assert registry.document_ids[doc_id]["systems"]["knowledge_graph"] is False

    def test_mark_system_synced(self):
        """Test marking document as synced to system."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        registry.register_document(doc_id)

        # Mark document_db as synced
        registry.mark_system_synced(doc_id, "document_db")
        assert registry.is_system_synced(doc_id, "document_db") is True
        assert registry.document_ids[doc_id]["status"] == "registered"  # Not fully synced yet

        # Mark vector_db as synced
        registry.mark_system_synced(doc_id, "vector_db")
        assert registry.is_system_synced(doc_id, "vector_db") is True

        # Mark knowledge_graph as synced
        registry.mark_system_synced(doc_id, "knowledge_graph")
        assert registry.is_system_synced(doc_id, "knowledge_graph") is True
        assert registry.document_ids[doc_id]["status"] == "fully_synced"

    def test_is_fully_synced(self):
        """Test checking if document is fully synced."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        registry.register_document(doc_id)

        assert registry.is_fully_synced(doc_id) is False

        registry.mark_system_synced(doc_id, "document_db")
        assert registry.is_fully_synced(doc_id) is False

        registry.mark_system_synced(doc_id, "vector_db")
        assert registry.is_fully_synced(doc_id) is False

        registry.mark_system_synced(doc_id, "knowledge_graph")
        assert registry.is_fully_synced(doc_id) is True

    def test_is_system_synced(self):
        """Test checking if document is synced to specific system."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        registry.register_document(doc_id)

        assert registry.is_system_synced(doc_id, "document_db") is False

        registry.mark_system_synced(doc_id, "document_db")
        assert registry.is_system_synced(doc_id, "document_db") is True
        assert registry.is_system_synced(doc_id, "vector_db") is False

    def test_get_document_info(self):
        """Test getting document information."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        metadata = {"title": "Test"}
        registry.register_document(doc_id, metadata)

        info = registry.get_document_info(doc_id)
        assert info is not None
        assert info["metadata"] == metadata

        # Non-existent document
        assert registry.get_document_info("nonexistent") is None

    def test_unregister_document(self):
        """Test unregistering a document."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        registry.register_document(doc_id)

        assert registry.unregister_document(doc_id) is True
        assert doc_id not in registry.document_ids

        # Try to unregister again
        assert registry.unregister_document(doc_id) is False

    def test_get_all_documents(self):
        """Test getting all document IDs."""
        registry = UnifiedIDRegistry()

        assert registry.get_all_documents() == []

        registry.register_document("doc_1")
        registry.register_document("doc_2")
        registry.register_document("doc_3")

        all_docs = registry.get_all_documents()
        assert len(all_docs) == 3
        assert "doc_1" in all_docs
        assert "doc_2" in all_docs
        assert "doc_3" in all_docs

    def test_get_documents_by_status(self):
        """Test getting documents by status."""
        registry = UnifiedIDRegistry()

        registry.register_document("doc_1")
        registry.register_document("doc_2")
        registry.mark_system_synced("doc_1", "document_db")
        registry.mark_system_synced("doc_1", "vector_db")
        registry.mark_system_synced("doc_1", "knowledge_graph")

        registered = registry.get_documents_by_status("registered")
        fully_synced = registry.get_documents_by_status("fully_synced")

        assert len(registered) == 1
        assert "doc_2" in registered
        assert len(fully_synced) == 1
        assert "doc_1" in fully_synced

    def test_get_unsynced_documents(self):
        """Test getting documents not synced to specific system."""
        registry = UnifiedIDRegistry()

        registry.register_document("doc_1")
        registry.register_document("doc_2")
        registry.mark_system_synced("doc_1", "document_db")

        unsynced = registry.get_unsynced_documents("document_db")
        assert len(unsynced) == 1
        assert "doc_2" in unsynced

        unsynced_vector = registry.get_unsynced_documents("vector_db")
        assert len(unsynced_vector) == 2

    def test_update_document_metadata(self):
        """Test updating document metadata."""
        registry = UnifiedIDRegistry()
        doc_id = "doc_1234567890abcdef1234567890abcdef"
        registry.register_document(doc_id, {"title": "Original"})

        result = registry.update_document_metadata(doc_id, {"author": "Test Author"})
        assert result is True
        assert registry.document_ids[doc_id]["metadata"] == {
            "title": "Original",
            "author": "Test Author"
        }

        # Non-existent document
        assert registry.update_document_metadata("nonexistent", {}) is False

    def test_get_statistics(self):
        """Test getting registry statistics."""
        registry = UnifiedIDRegistry()

        stats = registry.get_statistics()
        assert stats["total_documents"] == 0
        assert stats["fully_synced"] == 0
        assert stats["partially_synced"] == 0

        # Add some documents
        registry.register_document("doc_1")
        registry.register_document("doc_2")
        registry.mark_system_synced("doc_1", "document_db")
        registry.mark_system_synced("doc_1", "vector_db")
        registry.mark_system_synced("doc_1", "knowledge_graph")

        stats = registry.get_statistics()
        assert stats["total_documents"] == 2
        assert stats["fully_synced"] == 1
        assert stats["partially_synced"] == 1
        assert stats["system_sync_counts"]["document_db"] == 1
        assert stats["system_sync_counts"]["vector_db"] == 1
        assert stats["system_sync_counts"]["knowledge_graph"] == 1
