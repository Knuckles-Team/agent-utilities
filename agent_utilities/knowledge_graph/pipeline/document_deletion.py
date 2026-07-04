"""

CONCEPT:AU-KG.query.object-graph-mapper
Document Deletion Pipeline for Knowledge Graph.

Handle document deletion with cascading cleanup across all storage layers.
"""

import logging
from datetime import datetime
from typing import Any

from ..core.engine import IntelligenceGraphEngine
from ..id_management.ontological_identifier import (
    OntologicalIdentifierManager,
    OntologicalIdentifierRegistry,
)

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
        knowledge_graph: IntelligenceGraphEngine,
        id_manager: OntologicalIdentifierManager | None = None,
        id_registry: OntologicalIdentifierRegistry | None = None,
    ):
        """
        Initialize the document deletion pipeline.

        Args:
            knowledge_graph: Knowledge graph engine
            id_manager: Optional unified ID manager
            id_registry: Optional unified ID registry
        """
        self.knowledge_graph = knowledge_graph
        self.id_manager = id_manager or OntologicalIdentifierManager()
        self.id_registry = id_registry or OntologicalIdentifierRegistry()

    async def delete_document(
        self,
        ontological_identifier: str,
        hard_delete: bool = False,
        delete_from_registry: bool = True,
    ) -> dict[str, Any]:
        """
        Delete document from all storage systems.

        This deletes from:
        1. Knowledge graph (nodes + relationships)
        2. Unified ID registry (if hard_delete and delete_from_registry)

        Args:
            ontological_identifier: Unified document ID
            hard_delete: If True, permanently delete. If False, soft delete.
            delete_from_registry: If True, remove from registry

        Returns:
            Dict with deletion results

        Raises:
            ValueError: If document not found
            Exception: If deletion fails
        """
        # Step 1: Get document information
        existing_doc = self.knowledge_graph.graph.nodes.get(ontological_identifier)

        if not existing_doc:
            logger.warning(
                f"Document {ontological_identifier} not found in registry, attempting deletion anyway"
            )

        # Step 2: Get related chunk IDs
        chunk_ids = await self._get_document_chunk_ids(ontological_identifier)
        entity_ids = await self._get_document_entity_ids(ontological_identifier)

        logger.info(
            f"Deleting document {ontological_identifier} with {len(chunk_ids)} chunks and {len(entity_ids)} entities"
        )

        try:
            if hard_delete:
                # Step 3: Delete from knowledge graph completely
                await self._delete_from_knowledge_graph(
                    ontological_identifier, chunk_ids, entity_ids
                )
                logger.info(f"Deleted from knowledge graph: {ontological_identifier}")
            else:
                # Soft delete in knowledge graph
                await self._soft_delete_from_knowledge_graph(ontological_identifier)
                logger.info(
                    f"Soft deleted from knowledge graph: {ontological_identifier}"
                )

            # Step 6: Remove from registry
            if delete_from_registry and hard_delete:
                self.id_registry.unregister_document(ontological_identifier)
                logger.info(f"Removed from registry: {ontological_identifier}")

            return {
                "ontological_identifier": ontological_identifier,
                "chunks_deleted": len(chunk_ids),
                "entities_deleted": len(entity_ids),
                "hard_delete": hard_delete,
                "status": "completed",
                "deleted_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Document deletion failed for {ontological_identifier}: {e}")
            raise Exception(f"Document deletion failed: {e}") from e

    async def _get_document_chunk_ids(self, ontological_identifier: str) -> list[str]:
        """
        Get chunk IDs for a document.

        Args:
            ontological_identifier: Unified document ID

        Returns:
            List[str]: List of chunk IDs
        """
        chunks = []
        if self.knowledge_graph.graph.has_node(ontological_identifier):
            for _, chunk_id, edge_data in self.knowledge_graph.graph.out_edges(
                ontological_identifier, data=True
            ):
                if edge_data.get("relationship_type") == "HAS_CHUNK":
                    chunks.append(chunk_id)
        return chunks

    async def _get_document_entity_ids(self, ontological_identifier: str) -> list[str]:
        """
        Get entity IDs for a document.

        Args:
            ontological_identifier: Unified document ID

        Returns:
            List[str]: List of entity IDs
        """
        # For now, return empty list
        # Entity extraction will be implemented in Phase 3
        return []

    async def _delete_from_knowledge_graph(
        self, ontological_identifier: str, chunk_ids: list[str], entity_ids: list[str]
    ):
        """
        Delete document from knowledge graph.

        Args:
            ontological_identifier: Unified document ID
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
        if self.knowledge_graph.graph.has_node(ontological_identifier):
            self.knowledge_graph.graph.remove_node(ontological_identifier)

    async def _soft_delete_from_knowledge_graph(self, ontological_identifier: str):
        """
        Soft delete document from knowledge graph.

        Sets status to ARCHIVED so QueryMixin filters exclude it
        from all standard search and retrieval operations.

        Args:
            ontological_identifier: Unified document ID
        """
        if self.knowledge_graph.graph.has_node(ontological_identifier):
            node_data = self.knowledge_graph.graph.nodes[ontological_identifier]
            node_data["status"] = "ARCHIVED"
            node_data["deleted_at"] = datetime.now().isoformat()
            self.knowledge_graph.graph.add_node(ontological_identifier, **node_data)

    async def restore_document(self, ontological_identifier: str) -> dict[str, Any]:
        """
        Restore a soft-deleted document.

        Args:
            ontological_identifier: Unified document ID

        Returns:
            Dict with restoration results

        Raises:
            ValueError: If document not found or not soft-deleted
        """
        # Get document
        doc = self.knowledge_graph.graph.nodes.get(ontological_identifier)

        if not doc:
            raise ValueError(f"Document {ontological_identifier} not found")

        if doc.get("status") != "ARCHIVED":
            raise ValueError(f"Document {ontological_identifier} is not soft-deleted")

        # Restore document
        doc["status"] = "ACTIVE"
        doc["deleted_at"] = None
        doc["updated_at"] = datetime.now().isoformat()
        self.knowledge_graph.graph.add_node(ontological_identifier, **doc)

        logger.info(f"Restored document: {ontological_identifier}")

        return {
            "ontological_identifier": ontological_identifier,
            "status": "restored",
            "restored_at": datetime.now().isoformat(),
        }

    async def batch_delete_documents(
        self, ontological_identifiers: list[str], hard_delete: bool = False
    ) -> dict[str, Any]:
        """
        Delete multiple documents.

        Args:
            ontological_identifiers: List of unified document IDs
            hard_delete: Whether to hard delete

        Returns:
            Dict with batch deletion results
        """
        results: dict[str, Any] = {
            "successful": [],
            "failed": [],
            "total": len(ontological_identifiers),
        }

        for ontological_identifier in ontological_identifiers:
            try:
                await self.delete_document(
                    ontological_identifier, hard_delete=hard_delete
                )
                results["successful"].append(ontological_identifier)
            except Exception as e:
                logger.error(f"Failed to delete {ontological_identifier}: {e}")
                results["failed"].append(
                    {"id": ontological_identifier, "error": str(e)}
                )

        logger.info(
            f"Batch delete completed: {len(results['successful'])} successful, {len(results['failed'])} failed"
        )

        return results
