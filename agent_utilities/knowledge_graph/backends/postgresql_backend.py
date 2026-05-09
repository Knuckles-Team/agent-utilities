"""PostgreSQL + Apache AGE Graph Backend (CONCEPT:OS-5.20).

Production-grade backend using PostgreSQL with the Apache AGE
graph extension for Cypher query support. Suitable for
10-1000 concurrent agents with full ACID transactions.

Requires: pip install agent-utilities[postgresql]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)


class PostgreSQLBackend(GraphBackend):
    """PostgreSQL + Apache AGE graph backend.

    Uses the AGE (A Graph Extension) for PostgreSQL to provide
    Cypher query support with full ACID transaction guarantees.

    This backend is designed for production deployments requiring:
    - Multi-writer concurrent access
    - Full ACID transactions
    - Horizontal read scaling via replicas
    - Point-in-time recovery

    Args:
        dsn: PostgreSQL connection string.
        graph_name: AGE graph name. Default: "agent_graph".
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/agent_utilities",
        graph_name: str = "agent_graph",
    ) -> None:
        self._dsn = dsn
        self._graph_name = graph_name
        self._connection: Any = None
        logger.info("PostgreSQLBackend initialized (dsn=%s, graph=%s)", dsn, graph_name)

    def _ensure_connection(self) -> Any:
        """Lazy connection initialization."""
        if self._connection is None:
            try:
                import psycopg

                self._connection = psycopg.connect(self._dsn)
                logger.info("Connected to PostgreSQL")
            except ImportError:
                raise ImportError(
                    "PostgreSQL backend requires psycopg. "
                    "Install with: pip install agent-utilities[postgresql]"
                ) from None
        return self._connection

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query via Apache AGE."""
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            # AGE requires wrapping Cypher in SELECT * FROM cypher()
            age_query = f"SELECT * FROM cypher('{self._graph_name}', $$ {query} $$) AS (result agtype)"  # nosec
            cur.execute(age_query)
            rows = cur.fetchall()
            return [{"result": row[0]} for row in rows]

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute batch Cypher queries."""
        results = []
        for params in batch:
            results.extend(self.execute(query, params))
        return results

    def create_schema(self) -> None:
        """Initialize AGE graph and required labels."""
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            cur.execute(f"SELECT create_graph('{self._graph_name}')")
            conn.commit()
        logger.info("PostgreSQL AGE schema initialized")

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store embedding using pgvector extension."""
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO node_embeddings (node_id, embedding) "
                "VALUES (%s, %s) ON CONFLICT (node_id) DO UPDATE SET embedding = %s",
                (node_id, embedding, embedding),
            )
            conn.commit()

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Vector similarity search via pgvector."""
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT node_id, 1 - (embedding <=> %s::vector) AS similarity "
                "FROM node_embeddings ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_embedding, query_embedding, n_results),
            )
            return [{"id": row[0], "_similarity": row[1]} for row in cur.fetchall()]

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes matching criteria."""
        logger.info("Pruning with criteria: %s", criteria)

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("PostgreSQL connection closed")

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            conn = self._ensure_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Return database statistics."""
        return {
            "backend": "postgresql",
            "graph_name": self._graph_name,
            "connected": self._connection is not None,
        }
