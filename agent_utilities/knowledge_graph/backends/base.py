#!/usr/bin/python
"""Graph Backend Base Interface.

CONCEPT:KG-2.7 — Vendor-Agnostic Graph Backend Abstraction

Provides the ``GraphBackend`` ABC that all graph storage backends must implement.
Backends may optionally support SPARQL via ``supports_sparql`` / ``execute_sparql()``.
"""

from abc import ABC, abstractmethod
from typing import Any


class GraphBackend(ABC):
    """Abstract interface for Graph Database operations.

    All concrete backends (Memory, LadybugDB, Neo4j, FalkorDB, PostgreSQL,
    Fuseki, Stardog) must implement the core methods below.

    Backends that support SPARQL should override ``supports_sparql`` to return
    ``True`` and implement ``execute_sparql()`` with real SPARQL execution.
    """

    # ------------------------------------------------------------------
    # Core CRUD & Query
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a graph query (e.g., Cypher) and return results."""
        pass

    @abstractmethod
    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a graph query over a batch of parameters for high-throughput ingestion."""
        pass

    @abstractmethod
    def create_schema(self) -> None:
        """Initialize required database schema (DDL for Ladybug, etc)."""
        pass

    # ------------------------------------------------------------------
    # Vector / Embedding Support
    # ------------------------------------------------------------------

    @abstractmethod
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add an embedding vector to a specific node."""
        pass

    @abstractmethod
    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Perform a semantic vector search returning top matching nodes."""
        pass

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @abstractmethod
    def prune(self, criteria: dict[str, Any]) -> None:
        """Run pruning logic based on criteria."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    # ------------------------------------------------------------------
    # Optional SPARQL Capability  (CONCEPT:KG-2.7)
    # ------------------------------------------------------------------

    @property
    def supports_sparql(self) -> bool:
        """Whether this backend supports SPARQL queries.

        Override to ``True`` in backends backed by an RDF store
        (Fuseki, Stardog).
        """
        return False

    def execute_sparql(
        self,
        query: str,
        *,
        default_graph: str | None = None,
        timeout_ms: int = 30_000,
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT/ASK/CONSTRUCT query.

        Backends that advertise ``supports_sparql = True`` must override
        this with a real implementation.

        Args:
            query: W3C SPARQL 1.1 query string.
            default_graph: Optional default graph IRI.
            timeout_ms: Query timeout in milliseconds.

        Returns:
            List of solution dicts (SELECT) or ``[{"result": bool}]`` (ASK).

        Raises:
            NotImplementedError: If the backend does not support SPARQL.
        """
        raise RuntimeError(
            f"{type(self).__name__} does not support SPARQL queries. "
            f"Use a SPARQL-capable backend (jena_fuseki, stardog)."
        )
