"""
Vector-MCP Backend with Unified ID Support.

Integrates vector-mcp for enhanced vector database operations with unified ID support.
"""

import logging
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)


class VectorMCPBackend(GraphBackend):
    """
    Vector-MCP backend with unified ID support.

    Integrates vector-mcp's factory pattern for enhanced vector database operations
    while maintaining unified ID consistency across all storage layers.

    This backend provides:
    - Multi-database vector support (ChromaDB, PGVector, MongoDB, Qdrant, Couchbase)
    - Unified ID compatibility
    - Hybrid routing with LadybugDB fallback
    - Embedding management
    """

    def __init__(
        self,
        db_type: str = "chroma",
        fallback_graph_db: GraphBackend | None = None,
        **kwargs,
    ):
        """
        Initialize Vector-MCP backend.

        Args:
            db_type: Vector database type (chroma, pgvector, mongodb, qdrant, couchbase)
            fallback_graph_db: Fallback graph database for non-vector operations
            **kwargs: Additional configuration for vector-mcp
        """
        try:
            from vector_mcp.vectordb.base import VectorDBFactory

            self.VectorDBFactory = VectorDBFactory
        except ImportError as e:
            raise ImportError(
                "Vector-MCP backend requires vector-mcp package. "
                "Install with: pip install vector-mcp"
            ) from e

        self.db_type = db_type
        self.vector_db = self.VectorDBFactory.create_vector_database(db_type, **kwargs)
        self.graph_db = fallback_graph_db or self._create_fallback_graph_db()

        from ..id_management.unified_id import UnifiedIDManager

        self.id_manager = UnifiedIDManager()

        logger.info(f"Initialized Vector-MCP backend with {db_type} database")

    def _create_fallback_graph_db(self) -> GraphBackend | None:
        """
        Create fallback LadybugDB for graph operations.

        Returns:
            GraphBackend: LadybugDB backend instance
        """
        try:
            from .ladybug_backend import LadybugBackend

            return LadybugBackend()
        except ImportError:
            logger.warning(
                "Could not create LadybugDB fallback, graph operations may fail"
            )
            return None

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute query with intelligent routing.

        Routes vector queries to vector-mcp and graph queries to fallback graph DB.

        Args:
            query: Query string
            params: Query parameters

        Returns:
            List[Dict]: Query results
        """
        if self._is_vector_query(query):
            return self._execute_vector_query(query, params or {})
        elif self.graph_db:
            return self.graph_db.execute(query, params or {})
        else:
            logger.warning(f"No fallback graph DB available for query: {query}")
            return []

    def _is_vector_query(self, query: str) -> bool:
        """
        Determine if query is a vector operation.

        Args:
            query: Query string

        Returns:
            bool: True if vector query, False otherwise
        """
        vector_keywords = [
            "semantic_search",
            "similarity_search",
            "vector_search",
            "embedding",
            "nearest",
        ]
        return any(keyword in query.lower() for keyword in vector_keywords)

    def _execute_vector_query(
        self, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Execute vector query via vector-mcp.

        Args:
            query: Vector query string
            params: Query parameters

        Returns:
            List[Dict]: Query results
        """
        try:
            # Parse query and execute appropriate vector-mcp operation
            if "semantic_search" in query.lower():
                query_text = params.get("query_text", "")
                n_results = params.get("n_results", 5)
                return self.semantic_search(query_text, n_results=n_results)
            else:
                logger.warning(f"Unknown vector query type: {query}")
                return []
        except Exception as e:
            logger.error(f"Vector query execution failed: {e}")
            return []

    def add_document_embeddings(
        self, unified_id: str, chunks: list[str], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Add document embeddings with unified ID support.

        Args:
            unified_id: Unified document ID
            chunks: List of text chunks
            metadata: Optional metadata

        Returns:
            Dict: Result with chunk_ids
        """
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = self.id_manager.generate_chunk_id(unified_id, i)
            chunk_ids.append(chunk_id)

            # Create vector document for vector-mcp
            vector_doc = {
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    "parent_doc_id": unified_id,
                    "chunk_index": i,
                    "unified_id": unified_id,
                    **(metadata or {}),
                },
            }

            # Insert into vector-mcp
            self.vector_db.insert_documents(
                [vector_doc], collection_name="knowledge_graph"
            )

        logger.info(f"Added {len(chunk_ids)} embeddings for document {unified_id}")

        return {"unified_id": unified_id, "chunk_ids": chunk_ids}

    def delete_document_embeddings(self, unified_id: str) -> None:
        """
        Delete all embeddings for a document.

        Args:
            unified_id: Unified document ID
        """
        # Get all chunk IDs for the document
        chunk_ids = self._get_document_chunk_ids(unified_id)

        if chunk_ids:
            self.vector_db.delete_documents(
                chunk_ids, collection_name="knowledge_graph"
            )
            logger.info(
                f"Deleted {len(chunk_ids)} embeddings for document {unified_id}"
            )

    def _get_document_chunk_ids(self, unified_id: str) -> list[str]:
        """
        Get chunk IDs for a document.

        Args:
            unified_id: Unified document ID

        Returns:
            List[str]: List of chunk IDs
        """
        # This would query the vector-mcp to find all documents with parent_doc_id = unified_id
        # For now, return empty list as this depends on vector-mcp API
        logger.warning("Document chunk ID retrieval not fully implemented")
        return []

    def semantic_search(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search via vector-mcp.

        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_dict: Optional filter criteria

        Returns:
            List[Dict]: Search results
        """
        try:
            results = self.vector_db.semantic_search(
                [query_text], n_results=n_results, filter_dict=filter_dict
            )

            # Post-process results to ensure unified ID compatibility
            for result in results:
                if "metadata" in result:
                    result["metadata"]["unified_id"] = result["metadata"].get(
                        "parent_doc_id"
                    )

            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def create_schema(self) -> None:
        """
        Initialize vector-mcp schema.

        Creates the knowledge_graph collection if it doesn't exist.
        """
        try:
            self.vector_db.create_collection("knowledge_graph", overwrite=False)
            logger.info("Created knowledge_graph collection in vector-mcp")
        except Exception as e:
            logger.error(f"Failed to create vector-mcp schema: {e}")

    def prune(self, criteria: dict[str, Any]) -> None:
        """
        Prune using vector-mcp capabilities.

        Args:
            criteria: Pruning criteria (e.g., node_ids, age_threshold)
        """
        if "node_ids" in criteria:
            self.vector_db.delete_documents(
                criteria["node_ids"], collection_name="knowledge_graph"
            )
            logger.info(f"Pruned {len(criteria['node_ids'])} nodes from vector-mcp")

        # Additional pruning logic can be added here

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """
        Add an embedding vector to a specific node.

        This is a stub method to comply with GraphBackend interface.
        The actual embedding addition is handled through add_document_embeddings.

        Args:
            node_id: Node identifier
            embedding: Embedding vector
        """
        # This method is required by GraphBackend interface but not directly used
        # Embeddings are added through add_document_embeddings with unified IDs
        logger.debug(
            f"add_embedding called for {node_id} (handled by add_document_embeddings)"
        )

    def get_embedding(self, node_id: str) -> list[float] | None:
        """
        Get embedding for a specific node.

        Args:
            node_id: Node ID (chunk ID)

        Returns:
            Optional[List[float]]: Embedding vector
        """
        try:
            results = self.vector_db.get_documents_by_ids(
                [node_id], collection_name="knowledge_graph"
            )
            if results and len(results) > 0:
                return results[0].get("embedding")
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding for {node_id}: {e}")
            return None

    def get_document_embeddings(self, unified_id: str) -> dict[str, list[float]]:
        """
        Get all embeddings for a document.

        Args:
            unified_id: Unified document ID

        Returns:
            Dict: Mapping of chunk_id to embedding
        """
        chunk_ids = self._get_document_chunk_ids(unified_id)
        embeddings = {}

        for chunk_id in chunk_ids:
            embedding = self.get_embedding(chunk_id)
            if embedding:
                embeddings[chunk_id] = embedding

        return embeddings

    def close(self) -> None:
        """Close the vector database and fallback graph database."""
        if hasattr(self, "vector_db") and hasattr(self.vector_db, "close"):
            self.vector_db.close()
        if self.graph_db:
            self.graph_db.close()
