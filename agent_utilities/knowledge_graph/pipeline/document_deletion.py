"""
Document Deletion Pipeline for Knowledge Graph.

Handle document deletion with cascading cleanup across all storage layers.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ..backends.document_storage.base import DocumentDB
from ..engine import IntelligenceGraphEngine
from ..id_management.unified_id import UnifiedIDManager, UnifiedIDRegistry

logger = logging.getLogger(__name__)


class DocumentDeletionPipeline:
    """
    Handle document deletion with cascading cleanup.

    This pipeline deletes from:
    1. Knowledge graph (nodes + relationships)
    2. Vector database (embeddings)
    3. Document database (document + chunks)
    4. Unified ID registry
    """

    def __init__(
        self,
        document_db: DocumentDB,
        vector_db: Any,  # Will be vector-mcp VectorDB
        knowledge_graph: IntelligenceGraphEngine,
        id_manager: UnifiedIDManager | None = None,
        id_registry: UnifiedIDRegistry | None = None,
    ):
        """
        Initialize the document deletion pipeline.

        Args:
            document_db: Document database backend
            vector_db: Vector database backend (vector-mcp)
            knowledge_graph: Knowledge graph engine
            id_manager: Optional unified ID manager
            id_registry: Optional unified ID registry
        """
        self.document_db = document_db
        self.vector_db = vector_db
        self.knowledge_graph = knowledge_graph
        self.id_manager = id_manager or UnifiedIDManager()
        self.id_registry = id_registry or UnifiedIDRegistry()

    async def delete_document(
        self,
        unified_id: str,
        hard_delete: bool = False,
        delete_from_registry: bool = True,
    ) -> dict[str, Any]:
        """
        Delete document from all storage systems.

        This deletes from:
        1. Knowledge graph (nodes + relationships)
        2. Vector database (embeddings)
        3. Document database (document + chunks)
        4. Unified ID registry (if hard_delete and delete_from_registry)

        Args:
            unified_id: Unified document ID
            hard_delete: If True, permanently delete. If False, soft delete.
            delete_from_registry: If True, remove from registry

        Returns:
            Dict with deletion results

        Raises:
            ValueError: If document not found
            Exception: If deletion fails
        """
        # Step 1: Get document information
        if asyncio.iscoroutinefunction(self.document_db.find_document):
            existing_doc = await self.document_db.find_document(unified_id, "documents")
        else:
            existing_doc = self.document_db.find_document(unified_id, "documents")

        if not existing_doc:
            logger.warning(
                f"Document {unified_id} not found in registry, attempting deletion anyway"
            )

        # Step 2: Get related chunk IDs
        chunk_ids = await self._get_document_chunk_ids(unified_id)
        entity_ids = await self._get_document_entity_ids(unified_id)

        logger.info(
            f"Deleting document {unified_id} with {len(chunk_ids)} chunks and {len(entity_ids)} entities"
        )

        try:
            # Step 3: Delete from knowledge graph
            await self._delete_from_knowledge_graph(unified_id, chunk_ids, entity_ids)
            logger.info(f"Deleted from knowledge graph: {unified_id}")

            # Step 4: Delete embeddings from vector database
            await self._delete_from_vector_db(chunk_ids)
            logger.info(
                f"Deleted embeddings from vector database: {len(chunk_ids)} embeddings"
            )

            # Step 5: Delete from document database
            if hard_delete:
                await self._hard_delete_from_document_db(unified_id, chunk_ids)
                logger.info(f"Hard deleted from document database: {unified_id}")
            else:
                await self._soft_delete_from_document_db(unified_id)
                logger.info(f"Soft deleted from document database: {unified_id}")

            # Step 6: Remove from registry
            if delete_from_registry and hard_delete:
                self.id_registry.unregister_document(unified_id)
                logger.info(f"Removed from registry: {unified_id}")

            return {
                "unified_id": unified_id,
                "chunks_deleted": len(chunk_ids),
                "entities_deleted": len(entity_ids),
                "hard_delete": hard_delete,
                "status": "completed",
                "deleted_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Document deletion failed for {unified_id}: {e}")
            raise Exception(f"Document deletion failed: {e}") from e

    async def _get_document_chunk_ids(self, unified_id: str) -> list[str]:
        """
        Get chunk IDs for a document.

        Args:
            unified_id: Unified document ID

        Returns:
            List[str]: List of chunk IDs
        """
        # Handle both sync and async document DB
        if asyncio.iscoroutinefunction(self.document_db.find_documents):
            chunks = await self.document_db.find_documents(
                {"parent_doc_id": unified_id}, "chunks"
            )
        else:
            chunks = self.document_db.find_documents(
                {"parent_doc_id": unified_id}, "chunks"
            )

        return [chunk.get("id") for chunk in chunks]

    async def _get_document_entity_ids(self, unified_id: str) -> list[str]:
        """
        Get entity IDs for a document.

        Args:
            unified_id: Unified document ID

        Returns:
            List[str]: List of entity IDs
        """
        # For now, return empty list
        # Entity extraction will be implemented in Phase 3
        return []

    async def _delete_from_knowledge_graph(
        self, unified_id: str, chunk_ids: list[str], entity_ids: list[str]
    ):
        """
        Delete document from knowledge graph.

        Args:
            unified_id: Unified document ID
            chunk_ids: List of chunk IDs
            entity_ids: List of entity IDs
        """
        # Delete chunk nodes
        for chunk_id in chunk_ids:
            if self.knowledge_graph.graph.has_node(chunk_id):
                self.knowledge_graph.graph.remove_node(chunk_id)

        # Delete entity nodes
        for entity_id in entity_ids:
            if self.knowledge_graph.graph.has_node(entity_id):
                self.knowledge_graph.graph.remove_node(entity_id)

        # Delete document node
        if self.knowledge_graph.graph.has_node(unified_id):
            self.knowledge_graph.graph.remove_node(unified_id)

    async def _delete_from_vector_db(self, chunk_ids: list[str]):
        """
        Delete embeddings from vector database.

        Args:
            chunk_ids: List of chunk IDs
        """
        if chunk_ids:
            self.vector_db.delete_documents(
                chunk_ids, collection_name="knowledge_graph"
            )

    async def _hard_delete_from_document_db(
        self, unified_id: str, chunk_ids: list[str]
    ):
        """
        Hard delete document from document database.

        Args:
            unified_id: Unified document ID
            chunk_ids: List of chunk IDs
        """
        # Delete chunks
        for chunk_id in chunk_ids:
            if asyncio.iscoroutinefunction(self.document_db.delete_document):
                await self.document_db.delete_document(chunk_id, "chunks")
            else:
                self.document_db.delete_document(chunk_id, "chunks")

        # Delete document
        if asyncio.iscoroutinefunction(self.document_db.delete_document):
            await self.document_db.delete_document(unified_id, "documents")
        else:
            self.document_db.delete_document(unified_id, "documents")

    async def _soft_delete_from_document_db(self, unified_id: str):
        """
        Soft delete document from document database.

        Args:
            unified_id: Unified document ID
        """
        # Mark document as deleted
        if asyncio.iscoroutinefunction(self.document_db.update_document):
            await self.document_db.update_document(
                unified_id,
                {"is_deleted": True, "deleted_at": datetime.now().isoformat()},
                "documents",
            )
        else:
            self.document_db.update_document(
                unified_id,
                {"is_deleted": True, "deleted_at": datetime.now().isoformat()},
                "documents",
            )

    async def restore_document(self, unified_id: str) -> dict[str, Any]:
        """
        Restore a soft-deleted document.

        Args:
            unified_id: Unified document ID

        Returns:
            Dict with restoration results

        Raises:
            ValueError: If document not found or not soft-deleted
        """
        # Get document
        if asyncio.iscoroutinefunction(self.document_db.find_document):
            doc = await self.document_db.find_document(unified_id, "documents")
        else:
            doc = self.document_db.find_document(unified_id, "documents")

        if not doc:
            raise ValueError(f"Document {unified_id} not found")

        if not doc.get("is_deleted"):
            raise ValueError(f"Document {unified_id} is not soft-deleted")

        # Restore document
        if asyncio.iscoroutinefunction(self.document_db.update_document):
            await self.document_db.update_document(
                unified_id,
                {
                    "is_deleted": False,
                    "deleted_at": None,
                    "updated_at": datetime.now().isoformat(),
                },
                "documents",
            )
        else:
            self.document_db.update_document(
                unified_id,
                {
                    "is_deleted": False,
                    "deleted_at": None,
                    "updated_at": datetime.now().isoformat(),
                },
                "documents",
            )

        logger.info(f"Restored document: {unified_id}")

        return {
            "unified_id": unified_id,
            "status": "restored",
            "restored_at": datetime.now().isoformat(),
        }

    async def batch_delete_documents(
        self, unified_ids: list[str], hard_delete: bool = False
    ) -> dict[str, Any]:
        """
        Delete multiple documents.

        Args:
            unified_ids: List of unified document IDs
            hard_delete: Whether to hard delete

        Returns:
            Dict with batch deletion results
        """
        results: dict[str, Any] = {
            "successful": [],
            "failed": [],
            "total": len(unified_ids),
        }

        for unified_id in unified_ids:
            try:
                await self.delete_document(unified_id, hard_delete=hard_delete)
                results["successful"].append(unified_id)
            except Exception as e:
                logger.error(f"Failed to delete {unified_id}: {e}")
                results["failed"].append({"id": unified_id, "error": str(e)})

        logger.info(
            f"Batch delete completed: {len(results['successful'])} successful, {len(results['failed'])} failed"
        )

        return results
