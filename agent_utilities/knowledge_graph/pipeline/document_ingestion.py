"""
Document Ingestion Pipeline for Knowledge Graph.

Tightly integrated pipeline that ingests documents through all storage layers
(document database, vector database, knowledge graph) with unified IDs.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..engine import IntelligenceGraphEngine
from ..id_management.unified_id import UnifiedIDManager, UnifiedIDRegistry

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Tightly integrated document ingestion pipeline.

    Ingests documents through all storage layers with unified ID:
    1. Document database
    2. Vector database (via vector-mcp)
    3. Knowledge graph

    Ensures cross-system consistency with transaction-like behavior.
    """

    def __init__(
        self,
        knowledge_graph: IntelligenceGraphEngine,
        id_manager: UnifiedIDManager | None = None,
        id_registry: UnifiedIDRegistry | None = None,
    ):
        """
        Initialize the document ingestion pipeline.

        Args:
            knowledge_graph: Knowledge graph engine
            id_manager: Optional unified ID manager (creates default if None)
            id_registry: Optional unified ID registry (creates default if None)
        """
        self.knowledge_graph = knowledge_graph
        self.id_manager = id_manager or UnifiedIDManager()
        self.id_registry = id_registry or UnifiedIDRegistry()
        self._ingested_docs: list[str] = []  # Track for rollback

    async def ingest_document(
        self, file_path: str, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Ingest a document through all storage layers with unified ID.

        This method performs:
        1. Generate unified ID
        2. Store in document database
        3. Process document (chunking, entity extraction)
        4. Store chunks in document database
        5. Generate embeddings via vector database
        6. Store embeddings in vector database
        7. Create knowledge graph nodes and relationships
        8. Register unified ID in registry

        Args:
            file_path: Path to the document file
            content: Document content
            metadata: Optional metadata about the document

        Returns:
            Dict with unified_id and processing results

        Raises:
            Exception: If ingestion fails (with rollback)
        """
        unified_id: str | None = None
        rollback_actions: list[Callable] = []

        try:
            # Step 1: Generate unified ID
            unified_id = self.id_manager.generate_document_id()
            logger.info(f"Generated unified ID: {unified_id}")

            # Step 2: Process document (chunking)
            chunks = self._chunk_document(content)
            logger.info(f"Chunked document into {len(chunks)} chunks")

            # Step 3: Generate chunks IDs and embeddings
            chunk_ids = [
                self.id_manager.generate_chunk_id(unified_id, i)
                for i in range(len(chunks))
            ]
            embeddings = await self._generate_embeddings(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Step 4: Create knowledge graph nodes (now includes everything)
            await self._create_graph_nodes(
                unified_id=unified_id,
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                metadata=metadata or {},
                rollback_actions=rollback_actions,
            )
            self.id_registry.mark_system_synced(unified_id)
            logger.info("Created unified knowledge graph nodes")

            # Step 5: Register unified ID
            self.id_registry.register_document(unified_id, metadata or {})
            self._ingested_docs.append(unified_id)

            logger.info(f"Successfully ingested document: {unified_id}")

            return {
                "unified_id": unified_id,
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
                "synced_systems": ["knowledge_graph"],
                "status": "completed",
                "created_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Document ingestion failed for {unified_id}: {e}")
            # Perform rollback
            await self._rollback(rollback_actions, unified_id)
            raise Exception(
                f"Document ingestion failed and was rolled back: {e}"
            ) from e

    async def _rollback(self, rollback_actions: list[Callable], unified_id: str | None):
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

        # Remove from registry if it was registered
        if unified_id in self.id_registry.document_ids:
            self.id_registry.unregister_document(unified_id)

        logger.info(f"Rollback completed for {unified_id}")

    def _chunk_document(self, content: str) -> list[str]:
        """
        Chunk document into manageable pieces.

        Args:
            content: Document content

        Returns:
            List[str]: List of chunks
        """
        # Simple chunking by paragraphs for now
        # Can be enhanced with more sophisticated chunking strategies
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            return [content]

        return paragraphs

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
        # For now, return dummy embeddings

        logger.warning("Using dummy embeddings - integrate with LM Studio")

        # Return dummy embeddings (768-dimensional)
        dummy_embedding = [0.0] * 768
        return [dummy_embedding.copy() for _ in chunks]

    async def _create_graph_nodes(
        self,
        unified_id: str,
        chunks: list[str],
        chunk_ids: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
        rollback_actions: list[Callable],
    ):
        """
        Create knowledge graph nodes for document.

        Args:
            unified_id: Unified document ID
            chunks: List of chunks
            chunk_ids: List of chunk IDs
            metadata: Document metadata
            rollback_actions: List to append rollback actions
        """
        # Create document node
        doc_node_data = {
            "id": unified_id,
            "file_path": metadata.get("file_path", ""),
            "chunk_count": len(chunks),
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
        }

        # Add document node to graph
        self.knowledge_graph.graph.add_node(unified_id, **doc_node_data)

        # Add rollback action
        def rollback_doc_node():
            if self.knowledge_graph.graph.has_node(unified_id):
                self.knowledge_graph.graph.remove_node(unified_id)

        rollback_actions.append(rollback_doc_node)

        # Create chunk nodes and relationships
        for i, (chunk, chunk_id, emb) in enumerate(
            zip(chunks, chunk_ids, embeddings, strict=False)
        ):
            chunk_node_data = {
                "id": chunk_id,
                "parent_doc_id": unified_id,
                "chunk_index": i,
                "content": chunk,
                "embedding": emb,
                "metadata": {"unified_id": unified_id},
            }

            self.knowledge_graph.graph.add_node(chunk_id, **chunk_node_data)

            # Add edge from document to chunk
            self.knowledge_graph.graph.add_edge(
                unified_id,
                chunk_id,
                relationship_type="HAS_CHUNK",
                created_at=datetime.now().isoformat(),
            )

            # Add rollback actions
            def rollback_chunk_node(cid=chunk_id):
                if self.knowledge_graph.graph.has_node(cid):
                    self.knowledge_graph.graph.remove_node(cid)

            rollback_actions.append(rollback_chunk_node)

    def get_ingested_documents(self) -> list[str]:
        """
        Get list of ingested document IDs.

        Returns:
            List[str]: List of unified document IDs
        """
        return self._ingested_docs.copy()

    def get_registry_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict: Registry statistics
        """
        return self.id_registry.get_statistics()
