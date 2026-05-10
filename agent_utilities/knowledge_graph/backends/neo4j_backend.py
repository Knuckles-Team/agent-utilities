#!/usr/bin/python
from __future__ import annotations

"""Neo4j Backend Implementation."""


import logging
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class Neo4jBackend(GraphBackend):
    """Neo4j backend for the unified graph."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",  # nosec B107
    ):
        if GraphDatabase is None:
            raise ImportError(
                "Neo4j driver is not installed. Please install with `pip install agent-utilities[neo4j]`"
            )
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Initialized Neo4j backend at {uri}")

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a batch query using Cypher UNWIND."""
        if not self.driver:
            raise RuntimeError("Neo4j backend not connected.")

        # Ensure query contains UNWIND if the user hasn't explicitly structured it
        if "UNWIND" not in query.upper():
            logger.warning(
                "Batch query does not contain UNWIND. Execution might be slow or fail."
            )

        with self.driver.session() as session:
            try:
                result = session.run(query, {"batch": batch})
                return [dict(record) for record in result]
            except Exception as e:
                logger.error(f"Neo4j batch execution error: {e}")
                return []

    def create_schema(self) -> None:
        logger.info("Creating Neo4j vector index for embeddings.")
        query = """
        CREATE VECTOR INDEX idx_embedding IF NOT EXISTS
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 768,
          `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            self.execute(query)
        except Exception as e:
            logger.warning(f"Could not create vector index in Neo4j: {e}")

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        query = "MATCH (n {id: $id}) SET n.embedding = $embedding"
        self.execute(query, {"id": node_id, "embedding": embedding})

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Perform a semantic vector search returning top matching nodes using Neo4j 5.11+."""
        query = """
        CALL db.index.vector.queryNodes('idx_embedding', $n_results, $query_embedding)
        YIELD node, score
        RETURN node
        """
        return self.execute(
            query, {"query_embedding": query_embedding, "n_results": n_results}
        )

    def prune(self, criteria: dict[str, Any]) -> None:
        query = "MATCH (n) WHERE n.last_accessed < $timestamp DETACH DELETE n"
        if "last_accessed" in criteria:
            self.execute(query, {"timestamp": criteria["last_accessed"]})

    def close(self) -> None:
        self.driver.close()
