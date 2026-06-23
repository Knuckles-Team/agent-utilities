#!/usr/bin/python
"""Graph Backend Base Interface.

CONCEPT:KG-2.7 — Vendor-Agnostic Graph Backend Abstraction

Provides the ``GraphBackend`` ABC that all graph storage backends must implement.
Backends may optionally support SPARQL via ``supports_sparql`` / ``execute_sparql()``.
"""

import json
from abc import ABC, abstractmethod
from typing import Any


def coerce_cypher_property(value: Any) -> Any:
    """Coerce a property value to something a Cypher backend (Neo4j/FalkorDB) accepts.

    Cypher property values must be primitives or *arrays of primitives*. A Map (dict),
    or a list containing non-primitives, raises ``Neo.ClientError.Statement.TypeError``
    ("Property values can only be of primitive types or arrays thereof") — and on a
    fan-out *mirror* that error stalls replication permanently (the outbox entry retries
    forever, dragging the write path). Serialize such values to a JSON string so the
    write persists losslessly (readers can ``json.loads`` it back); primitives and
    primitive arrays (e.g. embedding vectors) pass through untouched. (CONCEPT:KG-2.74)
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        if all(v is None or isinstance(v, str | int | float | bool) for v in value):
            return list(value)
        return json.dumps(value, default=str)
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", "replace")
    # dict (Map), set, or any other non-primitive → lossless JSON string.
    return json.dumps(value, default=str)


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

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        """Atomic compare-and-set on a node's fields (CONCEPT:KG-2.141).

        Optional capability: backends that support an atomic conditional update
        (engine L1, tiered) override this; the default declines so a caller can
        feature-detect rather than silently no-op.
        """
        raise NotImplementedError(  # ABSTRACT-OK — optional CAS capability
            f"{type(self).__name__} does not support compare_and_set_node_fields"
        )

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
    # Cypher portability tier  (CONCEPT:KG-2.63)
    # ------------------------------------------------------------------

    @property
    def cypher_support(self) -> str:
        """How much of openCypher this backend can run unchanged.

        Drives capability-aware multi-connection fan-out (CONCEPT:KG-2.63): when
        the SAME query is run against several connections, a backend that can
        only serve a bounded subset can be surfaced honestly rather than failing
        silently. Values:

        * ``"full"`` — native openCypher (neo4j, falkordb, Apache AGE). Default.
        * ``"subset"`` — only the bounded operational subset the engine emits
          runs (the regex Postgres transpiler, the in-memory epistemic graph).

        Override to ``"subset"`` in backends that do not run native Cypher.
        """
        return "full"

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
