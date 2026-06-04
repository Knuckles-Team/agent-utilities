#!/usr/bin/python
from __future__ import annotations

"""SPARQL Adapter Interface.

CONCEPT:KG-2.7 — Vendor-Agnostic Graph Backend Abstraction

Provides the ``SparqlAdapter`` base class, which is a specialized
``GraphBackend`` designed for W3C SPARQL 1.1 compliant triple stores
(e.g., Fuseki, Stardog, Neptune).
"""

from abc import abstractmethod
from typing import Any

from ..base import GraphBackend


class SparqlAdapter(GraphBackend):
    """Abstract interface for SPARQL 1.1 compliant graph backends.

    This standardizes the HTTP interactions for queries, updates, and
    bulk graph lifecycle operations.
    """

    @property
    def supports_sparql(self) -> bool:
        """All SparqlAdapters support SPARQL natively."""
        return True

    @abstractmethod
    def execute_sparql_query(
        self, query: str, timeout_ms: int = 30_000
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT, ASK, or CONSTRUCT query."""
        pass

    @abstractmethod
    def execute_sparql_update(self, update: str, timeout_ms: int = 30_000) -> None:
        """Execute a SPARQL INSERT or DELETE update."""
        pass

    @abstractmethod
    def upload_graph(self, ttl_content: str, graph_uri: str | None = None) -> None:
        """Upload a full Turtle (.ttl) graph representation."""
        pass

    @abstractmethod
    def download_graph(self, graph_uri: str | None = None) -> str:
        """Download the full graph as a Turtle (.ttl) string."""
        pass

    # Provide default routing from the base GraphBackend execute_sparql
    def execute_sparql(
        self,
        query: str,
        *,
        default_graph: str | None = None,
        timeout_ms: int = 30_000,
    ) -> list[dict[str, Any]]:
        """Unified execution entry point, routes to query or update based on content."""
        _ = default_graph
        stripped = query.strip().upper()
        if stripped.startswith(("INSERT", "DELETE", "LOAD", "CLEAR", "DROP", "CREATE")):
            self.execute_sparql_update(query, timeout_ms=timeout_ms)
            return [{"status": "ok"}]
        else:
            return self.execute_sparql_query(query, timeout_ms=timeout_ms)
