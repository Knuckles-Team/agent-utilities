"""

CONCEPT:AU-KG.query.object-graph-mapper
Document Ingestion Pipeline for Knowledge Graph.

Tightly integrated pipeline that ingests documents through all storage layers
(document database, vector database, knowledge graph) with unified IDs.
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
        id_manager: OntologicalIdentifierManager | None = None,
        id_registry: OntologicalIdentifierRegistry | None = None,
    ):
        """
        Initialize the document ingestion pipeline.

        Args:
            knowledge_graph: Knowledge graph engine
            id_manager: Optional unified ID manager (creates default if None)
            id_registry: Optional unified ID registry (creates default if None)
        """
        self.knowledge_graph = knowledge_graph
        self.id_manager = id_manager or OntologicalIdentifierManager()
        self.id_registry = id_registry or OntologicalIdentifierRegistry()
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
            Dict with ontological_identifier and processing results

        Raises:
            Exception: If ingestion fails (with rollback)
        """
        ontological_identifier: str | None = None
        rollback_actions: list[Callable] = []

        try:
            # Step 1: Generate unified ID
            ontological_identifier = self.id_manager.generate_document_id()
            logger.info(f"Generated unified ID: {ontological_identifier}")

            # Step 2: Process document (chunking)
            chunks = self._chunk_document(content)
            logger.info(f"Chunked document into {len(chunks)} chunks")

            # Step 3: Generate chunks IDs and embeddings
            chunk_ids = [
                self.id_manager.generate_chunk_id(ontological_identifier, i)
                for i in range(len(chunks))
            ]
            embeddings = await self._generate_embeddings(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Step 4: Create knowledge graph nodes (now includes everything)
            await self._create_graph_nodes(
                ontological_identifier=ontological_identifier,
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                metadata=metadata or {},
                rollback_actions=rollback_actions,
            )
            self.id_registry.mark_system_synced(ontological_identifier)
            logger.info("Created unified knowledge graph nodes")

            # Step 5: Register unified ID
            self.id_registry.register_document(ontological_identifier, metadata or {})
            self._ingested_docs.append(ontological_identifier)

            logger.info(f"Successfully ingested document: {ontological_identifier}")

            return {
                "ontological_identifier": ontological_identifier,
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
                "synced_systems": ["knowledge_graph"],
                "status": "completed",
                "created_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Document ingestion failed for {ontological_identifier}: {e}")
            # Perform rollback
            await self._rollback(rollback_actions, ontological_identifier)
            raise Exception(
                f"Document ingestion failed and was rolled back: {e}"
            ) from e

    async def _rollback(
        self, rollback_actions: list[Callable], ontological_identifier: str | None
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

        # Remove from registry if it was registered
        if ontological_identifier in self.id_registry.document_ids:
            self.id_registry.unregister_document(ontological_identifier)

        logger.info(f"Rollback completed for {ontological_identifier}")

    def _chunk_document(self, content: str) -> list[str]:
        """
        Chunk document into manageable pieces using sentence-boundary-aware splitting.

        Uses the distillation engine's chunking utility for intelligent
        sentence-boundary-aware splitting with 10% overlap.

        Args:
            content: Document content

        Returns:
            List[str]: List of chunks
        """
        from ..distillation.distillation_engine import chunk_text

        chunks = chunk_text(content, chunk_size=2000, overlap=200)
        return chunks if chunks else [content]

    async def _generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """
        Generate embeddings for chunks using the configured embedding model.

        Uses ``create_embedding_model()`` from ``agent_utilities.core.embedding_utilities``.
        Falls back to zero-vector placeholders if no model is available.

        Args:
            chunks: List of text chunks

        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            from agent_utilities.core.embedding_utilities import create_embedding_model

            embed_model = create_embedding_model()
            embeddings = []
            for chunk in chunks:
                emb = embed_model.get_text_embedding(chunk)
                embeddings.append(emb)
            return embeddings
        except Exception as exc:
            raise RuntimeError(
                "Integration not configured: Embedding model is unavailable. Silent fallback to zero-vectors is disabled for zero-stub compliance."
            ) from exc

    async def _create_graph_nodes(
        self,
        ontological_identifier: str,
        chunks: list[str],
        chunk_ids: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
        rollback_actions: list[Callable],
    ):
        """
        Create knowledge graph nodes for document.

        Args:
            ontological_identifier: Unified document ID
            chunks: List of chunks
            chunk_ids: List of chunk IDs
            metadata: Document metadata
            rollback_actions: List to append rollback actions
        """
        # Create document node
        doc_node_data = {
            "id": ontological_identifier,
            "file_path": metadata.get("file_path", ""),
            "chunk_count": len(chunks),
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
        }

        # Add document node to graph
        self.knowledge_graph.graph.add_node(ontological_identifier, **doc_node_data)

        # Add rollback action
        def rollback_doc_node():
            if self.knowledge_graph.graph.has_node(ontological_identifier):
                self.knowledge_graph.graph.remove_node(ontological_identifier)

        rollback_actions.append(rollback_doc_node)

        # Create chunk nodes and relationships
        for i, (chunk, chunk_id, emb) in enumerate(
            zip(chunks, chunk_ids, embeddings, strict=False)
        ):
            chunk_node_data = {
                "id": chunk_id,
                "parent_doc_id": ontological_identifier,
                "chunk_index": i,
                "content": chunk,
                "embedding": emb,
                "metadata": {"ontological_identifier": ontological_identifier},
            }

            self.knowledge_graph.graph.add_node(chunk_id, **chunk_node_data)

            # Add edge from document to chunk
            self.knowledge_graph.graph.add_edge(
                ontological_identifier,
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
