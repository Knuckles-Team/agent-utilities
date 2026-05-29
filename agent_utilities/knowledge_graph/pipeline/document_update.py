"""

CONCEPT:KG-2.0
Document Update Pipeline for Knowledge Graph.

Handle document updates with cascading sync across all storage layers,
including embedding regeneration and knowledge graph relationship updates.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..core.engine import IntelligenceGraphEngine
from ..id_management.ontological_identifier import (
    OntologicalIdentifierManager,
    OntologicalIdentifierRegistry,
)

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
        knowledge_graph: IntelligenceGraphEngine,
        id_manager: OntologicalIdentifierManager | None = None,
        id_registry: OntologicalIdentifierRegistry | None = None,
    ):
        """
        Initialize the document update pipeline.

        Args:
            knowledge_graph: Knowledge graph engine
            id_manager: Optional unified ID manager
            id_registry: Optional unified ID registry
        """
        self.knowledge_graph = knowledge_graph
        self.id_manager = id_manager or OntologicalIdentifierManager()
        self.id_registry = id_registry or OntologicalIdentifierRegistry()

    async def update_document(
        self,
        ontological_identifier: str,
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
            ontological_identifier: Unified document ID
            new_content: New document content (None to keep existing)
            metadata_updates: Dictionary of metadata updates
            regenerate_embeddings: Whether to regenerate embeddings

        Returns:
            Dict with update results

        Raises:
            ValueError: If document not found
            Exception: If update fails
        """
        # Step 1: Verify document exists in graph
        existing_doc = self.knowledge_graph.graph.nodes.get(ontological_identifier)
        if not existing_doc:
            raise ValueError(
                f"Document {ontological_identifier} not found in knowledge graph"
            )

        # Check if document is soft-deleted (converged lifecycle: status='ARCHIVED')
        if existing_doc.get("status") == "ARCHIVED":
            raise ValueError(
                f"Document {ontological_identifier} is archived and cannot be updated"
            )

        rollback_actions: list[Callable] = []

        try:
            # Step 2: Update document node
            updated_doc = existing_doc.copy()

            if new_content is not None:
                updated_doc["content"] = new_content
                content_changed = True
            else:
                content_changed = False

            if metadata_updates:
                updated_doc["metadata"].update(metadata_updates)

            updated_doc["updated_at"] = datetime.now().isoformat()

            # Store old for rollback
            old_doc = existing_doc.copy()
            self.knowledge_graph.graph.add_node(ontological_identifier, **updated_doc)

            def rollback_doc_update():
                self.knowledge_graph.graph.add_node(ontological_identifier, **old_doc)

            rollback_actions.append(rollback_doc_update)

            logger.info(f"Updated document in graph: {ontological_identifier}")

            # Step 3: Re-chunk document (if content changed)
            if new_content is not None and regenerate_embeddings:
                old_chunks = await self._get_document_chunks(ontological_identifier)
                new_chunks = self._chunk_document(new_content)

                # Step 4: Update knowledge graph nodes
                await self._update_graph_nodes(
                    ontological_identifier, old_chunks, new_chunks, rollback_actions
                )

                embeddings_regenerated = True
                old_chunk_count = len(old_chunks)
                new_chunk_count = len(new_chunks)
            else:
                embeddings_regenerated = False
                old_chunk_count = 0
                new_chunk_count = 0

            # Step 7: Update registry
            if ontological_identifier in self.id_registry.document_ids:
                self.id_registry.update_document_metadata(
                    ontological_identifier, metadata_updates or {}
                )

            logger.info(f"Successfully updated document: {ontological_identifier}")

            return {
                "ontological_identifier": ontological_identifier,
                "content_changed": content_changed,
                "metadata_updated": bool(metadata_updates),
                "embeddings_regenerated": embeddings_regenerated,
                "old_chunk_count": old_chunk_count,
                "new_chunk_count": new_chunk_count,
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Document update failed for {ontological_identifier}: {e}")
            # Perform rollback
            await self._rollback(rollback_actions, ontological_identifier)
            raise Exception(f"Document update failed and was rolled back: {e}") from e

    async def _get_document_chunks(self, ontological_identifier: str) -> list[str]:
        """
        Get existing chunks for a document.

        Args:
            ontological_identifier: Unified document ID

        Returns:
            List[str]: List of chunk contents
        """
        chunks = []
        for _, chunk_id, edge_data in self.knowledge_graph.graph.out_edges(
            ontological_identifier, data=True
        ):
            if edge_data.get("relationship_type") == "HAS_CHUNK":
                node_data = self.knowledge_graph.graph.nodes.get(chunk_id)
                if node_data:
                    chunks.append(node_data)

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
        ontological_identifier: str,
        old_chunks: list[str],
        new_chunks: list[str],
        rollback_actions: list[Callable],
    ):
        """
        Update knowledge graph nodes for document.

        Args:
            ontological_identifier: Unified document ID
            old_chunks: Old chunks
            new_chunks: New chunks
            rollback_actions: List to append rollback actions
        """
        # Remove old chunk nodes
        old_chunk_ids = [
            self.id_manager.generate_chunk_id(ontological_identifier, i)
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
            if self.knowledge_graph.graph.has_edge(ontological_identifier, chunk_id):
                edge_data = self.knowledge_graph.graph.get_edge_data(
                    ontological_identifier, chunk_id
                )
                edges_to_restore.append((chunk_id, edge_data))
                self.knowledge_graph.graph.remove_edge(ontological_identifier, chunk_id)

        # Add rollback action
        def rollback_graph():
            # Restore old nodes and edges
            for chunk_id, node_data in old_node_data.items():
                self.knowledge_graph.graph.add_node(chunk_id, **node_data)

            for chunk_id, edge_data in edges_to_restore:
                self.knowledge_graph.graph.add_edge(
                    ontological_identifier, chunk_id, **edge_data
                )

        rollback_actions.append(rollback_graph)

        # Create new chunk nodes
        new_embeddings = await self._generate_embeddings(new_chunks)
        for i, chunk in enumerate(new_chunks):
            chunk_id = self.id_manager.generate_chunk_id(ontological_identifier, i)
            chunk_node_data = {
                "id": chunk_id,
                "parent_doc_id": ontological_identifier,
                "chunk_index": i,
                "content": chunk,
                "embedding": new_embeddings[i],
                "metadata": {"ontological_identifier": ontological_identifier},
                "updated_at": datetime.now().isoformat(),
            }

            self.knowledge_graph.graph.add_node(chunk_id, **chunk_node_data)

            # Add edge from document to chunk
            self.knowledge_graph.graph.add_edge(
                ontological_identifier,
                chunk_id,
                relationship_type="HAS_CHUNK",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )

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

    async def _rollback(
        self, rollback_actions: list[Callable], ontological_identifier: str
    ):
        """
        Perform rollback actions in reverse order.

        Args:
            rollback_actions: List of rollback actions
            ontological_identifier: Document ID for logging
        """
        logger.info(f"Starting rollback for {ontological_identifier}")

        # Execute rollbacks in reverse order
        for action in reversed(rollback_actions):
            try:
                await action()
            except Exception as e:
                logger.warning(f"Rollback action failed: {e}")

        logger.info(f"Rollback completed for {ontological_identifier}")
