"""
Document Update Pipeline for Knowledge Graph.

Handle document updates with cascading sync across all storage layers,
including embedding regeneration and knowledge graph relationship updates.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..backends.document_storage.base import DocumentDB
from ..engine import IntelligenceGraphEngine
from ..id_management.unified_id import UnifiedIDManager, UnifiedIDRegistry

logger = logging.getLogger(__name__)


class DocumentUpdatePipeline:
    """
    Handle document updates with cascading sync across all systems.

    This pipeline updates:
    1. Document database (content, metadata)
    2. Vector database (regenerate embeddings)
    3. Knowledge graph (update nodes, relationships)
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
        Initialize the document update pipeline.

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

    async def update_document(
        self,
        unified_id: str,
        new_content: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
        regenerate_embeddings: bool = True,
    ) -> dict[str, Any]:
        """
        Update document and regenerate embeddings.

        This updates:
        1. Document database (content, metadata)
        2. Vector database (regenerate embeddings if content changed)
        3. Knowledge graph (update nodes, relationships)

        Args:
            unified_id: Unified document ID
            new_content: New document content (None to keep existing)
            metadata_updates: Dictionary of metadata updates
            regenerate_embeddings: Whether to regenerate embeddings

        Returns:
            Dict with update results

        Raises:
            ValueError: If document not found
            Exception: If update fails
        """
        # Step 1: Verify document exists
        if asyncio.iscoroutinefunction(self.document_db.find_document):
            existing_doc = await self.document_db.find_document(unified_id, "documents")
        else:
            existing_doc = self.document_db.find_document(unified_id, "documents")
        if not existing_doc:
            raise ValueError(f"Document {unified_id} not found in document database")

        # Check if document is soft-deleted
        if existing_doc.get("is_deleted"):
            raise ValueError(
                f"Document {unified_id} is soft-deleted and cannot be updated"
            )

        rollback_actions: list[Callable] = []

        try:
            # Step 2: Update document database
            updated_doc = existing_doc.copy()

            if new_content is not None:
                updated_doc["content"] = new_content
                content_changed = True
            else:
                content_changed = False

            if metadata_updates:
                updated_doc["metadata"].update(metadata_updates)

            updated_doc["updated_at"] = datetime.now().isoformat()

            # Store old content for rollback
            old_content = existing_doc.get("content")
            old_metadata = existing_doc.get("metadata", {}).copy()

            await self._update_document_with_rollback(
                unified_id,
                updated_doc,
                "documents",
                old_content,
                old_metadata,
                rollback_actions,
            )

            logger.info(f"Updated document in database: {unified_id}")

            # Step 3: Re-chunk document (if content changed)
            if new_content is not None and regenerate_embeddings:
                old_chunks = await self._get_document_chunks(unified_id)
                new_chunks = self._chunk_document(new_content)

                # Step 4: Update knowledge graph nodes
                await self._update_graph_nodes(
                    unified_id, old_chunks, new_chunks, rollback_actions
                )

                # Step 5: Regenerate embeddings for changed chunks
                await self._regenerate_embeddings(
                    unified_id, old_chunks, new_chunks, rollback_actions
                )

                # Step 6: Update chunks in document database
                await self._update_document_chunks(
                    unified_id, new_chunks, rollback_actions
                )

                embeddings_regenerated = True
                old_chunk_count = len(old_chunks)
                new_chunk_count = len(new_chunks)
            else:
                embeddings_regenerated = False
                old_chunk_count = 0
                new_chunk_count = 0

            # Step 7: Update registry
            if unified_id in self.id_registry.document_ids:
                self.id_registry.update_document_metadata(
                    unified_id, metadata_updates or {}
                )

            logger.info(f"Successfully updated document: {unified_id}")

            return {
                "unified_id": unified_id,
                "content_changed": content_changed,
                "metadata_updated": bool(metadata_updates),
                "embeddings_regenerated": embeddings_regenerated,
                "old_chunk_count": old_chunk_count,
                "new_chunk_count": new_chunk_count,
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Document update failed for {unified_id}: {e}")
            # Perform rollback
            await self._rollback(rollback_actions, unified_id)
            raise Exception(f"Document update failed and was rolled back: {e}") from e

    async def _update_document_with_rollback(
        self,
        doc_id: str,
        updated_doc: dict[str, Any],
        collection_name: str,
        old_content: str,
        old_metadata: dict[str, Any],
        rollback_actions: list[Callable],
    ):
        """
        Update document with rollback capability.

        Args:
            doc_id: Document ID
            updated_doc: Updated document
            collection_name: Collection name
            old_content: Old content for rollback
            old_metadata: Old metadata for rollback
            rollback_actions: List to append rollback action
        """
        # Handle both sync and async document DB
        if asyncio.iscoroutinefunction(self.document_db.update_document):
            success = await self.document_db.update_document(
                doc_id, updated_doc, collection_name
            )
        else:
            success = self.document_db.update_document(
                doc_id, updated_doc, collection_name
            )

        if not success:
            raise Exception(f"Failed to update document {doc_id}")

        # Add rollback action
        async def rollback():
            try:
                rollback_doc = {
                    "content": old_content,
                    "metadata": old_metadata,
                    "updated_at": datetime.now().isoformat(),
                }
                if asyncio.iscoroutinefunction(self.document_db.update_document):
                    await self.document_db.update_document(
                        doc_id, rollback_doc, collection_name
                    )
                else:
                    self.document_db.update_document(
                        doc_id, rollback_doc, collection_name
                    )
            except Exception as e:
                logger.warning(f"Rollback failed for document {doc_id}: {e}")

        rollback_actions.append(rollback)

    async def _get_document_chunks(self, unified_id: str) -> list[str]:
        """
        Get existing chunks for a document.

        Args:
            unified_id: Unified document ID

        Returns:
            List[str]: List of chunk contents
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

        # Sort by chunk_index
        chunks_sorted = sorted(chunks, key=lambda x: x.get("chunk_index", 0))
        return [chunk.get("content", "") for chunk in chunks_sorted]

    def _chunk_document(self, content: str) -> list[str]:
        """
        Chunk document into manageable pieces.

        Args:
            content: Document content

        Returns:
            List[str]: List of chunks
        """
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            return [content]

        return paragraphs

    async def _update_graph_nodes(
        self,
        unified_id: str,
        old_chunks: list[str],
        new_chunks: list[str],
        rollback_actions: list[Callable],
    ):
        """
        Update knowledge graph nodes for document.

        Args:
            unified_id: Unified document ID
            old_chunks: Old chunks
            new_chunks: New chunks
            rollback_actions: List to append rollback actions
        """
        # Remove old chunk nodes
        old_chunk_ids = [
            self.id_manager.generate_chunk_id(unified_id, i)
            for i in range(len(old_chunks))
        ]

        # Store old node data for rollback
        old_node_data = {}
        for chunk_id in old_chunk_ids:
            if self.knowledge_graph.graph.has_node(chunk_id):
                old_node_data[chunk_id] = self.knowledge_graph.graph.nodes[
                    chunk_id
                ].copy()
                self.knowledge_graph.graph.remove_node(chunk_id)

        # Remove old edges
        edges_to_restore = []
        for chunk_id in old_chunk_ids:
            if self.knowledge_graph.graph.has_edge(unified_id, chunk_id):
                edge_data = self.knowledge_graph.graph.get_edge_data(
                    unified_id, chunk_id
                )
                edges_to_restore.append((chunk_id, edge_data))
                self.knowledge_graph.graph.remove_edge(unified_id, chunk_id)

        # Add rollback action
        def rollback_graph():
            # Restore old nodes and edges
            for chunk_id, node_data in old_node_data.items():
                self.knowledge_graph.graph.add_node(chunk_id, **node_data)

            for chunk_id, edge_data in edges_to_restore:
                self.knowledge_graph.graph.add_edge(unified_id, chunk_id, **edge_data)

        rollback_actions.append(rollback_graph)

        # Create new chunk nodes
        for i, chunk in enumerate(new_chunks):
            chunk_id = self.id_manager.generate_chunk_id(unified_id, i)
            chunk_node_data = {
                "id": chunk_id,
                "parent_doc_id": unified_id,
                "chunk_index": i,
                "content": chunk,
                "metadata": {"unified_id": unified_id},
                "updated_at": datetime.now().isoformat(),
            }

            self.knowledge_graph.graph.add_node(chunk_id, **chunk_node_data)

            # Add edge from document to chunk
            self.knowledge_graph.graph.add_edge(
                unified_id,
                chunk_id,
                relationship_type="HAS_CHUNK",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )

    async def _regenerate_embeddings(
        self,
        unified_id: str,
        old_chunks: list[str],
        new_chunks: list[str],
        rollback_actions: list[Callable],
    ):
        """
        Regenerate embeddings for updated document.

        Args:
            unified_id: Unified document ID
            old_chunks: Old chunks
            new_chunks: New chunks
            rollback_actions: List to append rollback actions
        """
        # Delete old embeddings from vector database
        old_chunk_ids = [
            self.id_manager.generate_chunk_id(unified_id, i)
            for i in range(len(old_chunks))
        ]

        # Store old embeddings for rollback (if possible)
        old_embeddings: dict[str, list[float] | None] = {}
        try:
            for chunk_id in old_chunk_ids:
                # Try to get old embedding from vector DB
                # This depends on vector-mcp API
                old_embeddings[chunk_id] = None  # Placeholder
        except Exception:  # nosec B110
            pass

        for chunk_id in old_chunk_ids:
            self.vector_db.delete_documents(
                [chunk_id], collection_name="knowledge_graph"
            )

        # Generate new embeddings
        new_embeddings = await self._generate_embeddings(new_chunks)

        # Store new embeddings
        new_chunk_ids = [
            self.id_manager.generate_chunk_id(unified_id, i)
            for i in range(len(new_chunks))
        ]

        for chunk_id, embedding in zip(new_chunk_ids, new_embeddings, strict=True):
            vector_doc = {
                "id": chunk_id,
                "content": new_chunks[new_chunk_ids.index(chunk_id)],
                "metadata": {
                    "parent_doc_id": unified_id,
                    "chunk_index": new_chunk_ids.index(chunk_id),
                    "unified_id": unified_id,
                },
                "embedding": embedding,
            }
            self.vector_db.insert_documents(
                [vector_doc], collection_name="knowledge_graph"
            )

        # Add rollback action (delete new embeddings, restore old ones)
        async def rollback_embeddings():
            # Delete new embeddings
            for chunk_id in new_chunk_ids:
                self.vector_db.delete_documents(
                    [chunk_id], collection_name="knowledge_graph"
                )

            # Restore old embeddings (if we stored them)
            # This would require vector-mcp to support restoring embeddings
            logger.warning("Embedding rollback may not restore original embeddings")

        rollback_actions.append(rollback_embeddings)

    async def _update_document_chunks(
        self, unified_id: str, new_chunks: list[str], rollback_actions: list[Callable]
    ):
        """
        Update chunks in document database.

        Args:
            unified_id: Unified document ID
            new_chunks: New chunks
            rollback_actions: List to append rollback actions
        """
        # Delete old chunks
        if asyncio.iscoroutinefunction(self.document_db.find_documents):
            old_chunks = await self.document_db.find_documents(
                {"parent_doc_id": unified_id}, "chunks"
            )
        else:
            old_chunks = self.document_db.find_documents(
                {"parent_doc_id": unified_id}, "chunks"
            )

        old_chunk_ids = [chunk.get("id") for chunk in old_chunks]
        old_chunk_data = {chunk.get("id"): chunk.copy() for chunk in old_chunks}

        for chunk_id in old_chunk_ids:
            if asyncio.iscoroutinefunction(self.document_db.delete_document):
                await self.document_db.delete_document(chunk_id, "chunks")
            else:
                self.document_db.delete_document(chunk_id, "chunks")

        # Insert new chunks
        new_chunk_ids = []
        for i, chunk in enumerate(new_chunks):
            chunk_id = self.id_manager.generate_chunk_id(unified_id, i)
            chunk_record = {
                "id": chunk_id,
                "parent_doc_id": unified_id,
                "chunk_index": i,
                "content": chunk,
                "metadata": {"unified_id": unified_id},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            if asyncio.iscoroutinefunction(self.document_db.insert_document):
                await self.document_db.insert_document(chunk_record, "chunks")
            else:
                self.document_db.insert_document(chunk_record, "chunks")
            new_chunk_ids.append(chunk_id)

        # Add rollback action
        async def rollback_chunks():
            # Delete new chunks
            for chunk_id in new_chunk_ids:
                if asyncio.iscoroutinefunction(self.document_db.delete_document):
                    await self.document_db.delete_document(chunk_id, "chunks")
                else:
                    self.document_db.delete_document(chunk_id, "chunks")

            # Restore old chunks
            for chunk_id, chunk_data in old_chunk_data.items():
                if asyncio.iscoroutinefunction(self.document_db.insert_document):
                    await self.document_db.insert_document(chunk_data, "chunks")
                else:
                    self.document_db.insert_document(chunk_data, "chunks")

        rollback_actions.append(rollback_chunks)

    async def _generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List[List[float]]: List of embeddings
        """
        # Placeholder for embedding generation
        # This should integrate with LM Studio or the existing embedding utilities
        logger.warning("Using dummy embeddings - integrate with LM Studio")

        # Return dummy embeddings (768-dimensional)
        dummy_embedding = [0.0] * 768
        return [dummy_embedding.copy() for _ in chunks]

    async def _rollback(self, rollback_actions: list[Callable], unified_id: str):
        """
        Perform rollback actions in reverse order.

        Args:
            rollback_actions: List of rollback actions
            unified_id: Document ID for logging
        """
        logger.info(f"Starting rollback for {unified_id}")

        # Execute rollbacks in reverse order
        for action in reversed(rollback_actions):
            try:
                await action()
            except Exception as e:
                logger.warning(f"Rollback action failed: {e}")

        logger.info(f"Rollback completed for {unified_id}")
