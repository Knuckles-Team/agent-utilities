"""
Automated Cleanup Operations for Knowledge Graph.

Handle automated cleanup of old documents, orphan embeddings, etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from ..backends.document_storage.base import DocumentDB
from ..engine import IntelligenceGraphEngine
from ..id_management.unified_id import UnifiedIDManager, UnifiedIDRegistry

logger = logging.getLogger(__name__)


class DocumentCleanup:
    """
    Automated cleanup operations for document maintenance.

    This handles:
    1. Age-based document cleanup
    2. Orphan embedding cleanup
    3. Orphan graph node cleanup
    4. Soft-deleted document cleanup
    """

    def __init__(
        self,
        document_db: DocumentDB,
        vector_db: Any,  # Interface from vector-mcp if available
        knowledge_graph: IntelligenceGraphEngine,
        id_registry: UnifiedIDRegistry | None = None,
    ):
        """
        Initialize the document cleanup operations.

        Args:
            document_db: Document database backend
            vector_db: Vector database backend (vector-mcp)
            knowledge_graph: Knowledge graph engine
            id_registry: Optional unified ID registry
        """
        self.document_db = document_db
        self.vector_db = vector_db
        self.knowledge_graph = knowledge_graph
        self.id_manager = UnifiedIDManager()
        self.id_registry = id_registry or UnifiedIDRegistry()

    async def cleanup_old_documents(
        self, age_days: int = 30, hard_delete_soft_deleted: bool = True
    ) -> dict[str, Any]:
        """
        Clean up documents older than specified days.

        Args:
            age_days: Age threshold in days
            hard_delete_soft_deleted: Whether to hard delete soft-deleted documents

        Returns:
            Dict with cleanup results
        """
        cutoff_date = datetime.now() - timedelta(days=age_days)

        logger.info(
            f"Starting cleanup of documents older than {age_days} days (cutoff: {cutoff_date})"
        )

        results: dict[str, Any] = {
            "soft_deleted_cleaned": 0,
            "hard_deleted_cleaned": 0,
            "total_cleaned": 0,
            "errors": [],
        }

        # Get old documents from document database
        # This depends on the document DB implementation
        # For now, we'll use the registry to get old documents
        old_doc_ids = self.id_registry.get_documents_older_than(age_days)

        logger.info(
            f"Found {len(old_doc_ids)} documents older than {age_days} days in registry"
        )

        from ..pipeline.document_deletion import DocumentDeletionPipeline

        deletion_pipeline = DocumentDeletionPipeline(
            document_db=self.document_db,
            vector_db=self.vector_db,
            knowledge_graph=self.knowledge_graph,
            id_registry=self.id_registry,
        )

        for doc_id in old_doc_ids:
            try:
                # Check if document is soft-deleted
                import asyncio

                if asyncio.iscoroutinefunction(self.document_db.find_document):
                    doc = await self.document_db.find_document(doc_id, "documents")
                else:
                    doc = self.document_db.find_document(doc_id, "documents")
                if doc and doc.get("is_deleted"):
                    if hard_delete_soft_deleted:
                        await deletion_pipeline.delete_document(
                            doc_id, hard_delete=True
                        )
                        results["hard_deleted_cleaned"] += 1
                        results["total_cleaned"] += 1
                else:
                    # Soft delete the document
                    await deletion_pipeline.delete_document(doc_id, hard_delete=False)
                    results["soft_deleted_cleaned"] += 1
                    results["total_cleaned"] += 1

            except Exception as e:
                logger.error(f"Failed to cleanup document {doc_id}: {e}")
                results["errors"].append({"id": doc_id, "error": str(e)})

        logger.info(
            f"Cleanup completed: {results['total_cleaned']} documents cleaned, {len(results['errors'])} errors"
        )

        return results

    async def cleanup_orphan_embeddings(self) -> dict[str, Any]:
        """
        Clean up embeddings without corresponding documents.

        Returns:
            Dict with cleanup results
        """
        logger.info("Starting orphan embedding cleanup")

        results: dict[str, Any] = {"embeddings_deleted": 0, "errors": []}

        # Get all embeddings from vector database
        # This depends on vector-mcp API
        # For now, we'll implement a placeholder
        try:
            # Placeholder: Get all documents from vector-mcp
            # all_embeddings = await self.vector_db.get_all_documents(collection_name="knowledge_graph")

            # For each embedding, check if parent document exists in registry
            # for embedding_doc in all_embeddings:
            #     unified_id = embedding_doc.get("metadata", {}).get("parent_doc_id")
            #     if unified_id and unified_id not in self.id_registry.document_ids:
            #         # Orphan embedding - delete it
            #         self.vector_db.delete_documents([embedding_doc["id"]], collection_name="knowledge_graph")
            #         results["embeddings_deleted"] += 1

            logger.warning(
                "Orphan embedding cleanup not fully implemented - requires vector-mcp API integration"
            )

        except Exception as e:
            logger.error(f"Failed to cleanup orphan embeddings: {e}")
            results["errors"].append(str(e))

        logger.info(
            f"Orphan embedding cleanup completed: {results['embeddings_deleted']} embeddings deleted"
        )

        return results

    async def cleanup_orphan_graph_nodes(self) -> dict[str, Any]:
        """
        Clean up graph nodes without corresponding documents.

        Returns:
            Dict with cleanup results
        """
        logger.info("Starting orphan graph node cleanup")

        results: dict[str, Any] = {"nodes_deleted": 0, "errors": []}

        try:
            # Get all chunk nodes
            chunk_nodes = [
                node for node in self.knowledge_graph.graph.nodes() if "_chunk_" in node
            ]

            for node_id in chunk_nodes:
                # Extract parent document ID
                doc_id = self.id_manager.extract_doc_id(node_id)

                if doc_id and doc_id not in self.id_registry.document_ids:
                    # Orphan node - delete it
                    if self.knowledge_graph.graph.has_node(node_id):
                        self.knowledge_graph.graph.remove_node(node_id)
                        results["nodes_deleted"] += 1

            logger.info(
                f"Orphan graph node cleanup completed: {results['nodes_deleted']} nodes deleted"
            )

        except Exception as e:
            logger.error(f"Failed to cleanup orphan graph nodes: {e}")
            results["errors"].append(str(e))

        return results

    async def cleanup_soft_deleted_documents(self, age_days: int = 7) -> dict[str, Any]:
        """
        Hard delete soft-deleted documents older than specified days.

        Args:
            age_days: Age threshold in days for soft-deleted documents

        Returns:
            Dict with cleanup results
        """
        logger.info(
            f"Starting cleanup of soft-deleted documents older than {age_days} days"
        )

        results: dict[str, Any] = {"documents_hard_deleted": 0, "errors": []}

        # cutoff_date = datetime.now() - timedelta(days=age_days)

        # Get all documents and filter for soft-deleted
        # This depends on document DB implementation
        # For now, we'll implement a placeholder

        try:
            # Placeholder: Get all soft-deleted documents older than threshold
            # soft_deleted_docs = await self.document_db.find_documents({
            #     "is_deleted": True,
            #     "deleted_at": {"$lt": cutoff_date.isoformat()}
            # }, "documents")

            # for doc in soft_deleted_docs:
            #     from .document_deletion import DocumentDeletionPipeline
            #     deletion_pipeline = DocumentDeletionPipeline(...)
            #     await deletion_pipeline.delete_document(doc["id"], hard_delete=True)
            #     results["documents_hard_deleted"] += 1

            logger.warning(
                "Soft-deleted document cleanup not fully implemented - requires document DB API integration"
            )

        except Exception as e:
            logger.error(f"Failed to cleanup soft-deleted documents: {e}")
            results["errors"].append(str(e))

        logger.info(
            f"Soft-deleted document cleanup completed: {results['documents_hard_deleted']} documents hard deleted"
        )

        return results

    async def run_all_cleanup_operations(
        self, age_days: int = 30, soft_delete_age_days: int = 7
    ) -> dict[str, Any]:
        """
        Run all cleanup operations.

        Args:
            age_days: Age threshold for old documents
            soft_delete_age_days: Age threshold for soft-deleted documents

        Returns:
            Dict with combined cleanup results
        """
        logger.info("Starting all cleanup operations")

        results = {
            "old_documents": await self.cleanup_old_documents(age_days),
            "orphan_embeddings": await self.cleanup_orphan_embeddings(),
            "orphan_graph_nodes": await self.cleanup_orphan_graph_nodes(),
            "soft_deleted_documents": await self.cleanup_soft_deleted_documents(
                soft_delete_age_days
            ),
        }

        total_errors = sum(len(r.get("errors", [])) for r in results.values())
        logger.info(
            f"All cleanup operations completed with {total_errors} total errors"
        )

        return results

    def get_cleanup_statistics(self) -> dict[str, Any]:
        """
        Get statistics about documents that might need cleanup.

        Returns:
            Dict with cleanup statistics
        """
        stats = self.id_registry.get_statistics()

        # Add cleanup-specific statistics
        stats["cleanup_ready"] = {
            "soft_deleted_count": 0,  # Would query document DB
            "old_documents_count": len(self.id_registry.get_documents_older_than(30)),
            "orphan_embeddings_count": 0,  # Would query vector DB
            "orphan_graph_nodes_count": 0,  # Would query knowledge graph
        }

        return stats
