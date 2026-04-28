#!/usr/bin/python
"""Graph Backend Base Interface."""

from abc import ABC, abstractmethod
from typing import Any


class GraphBackend(ABC):
    """Abstract interface for Graph Database operations."""

    @abstractmethod
    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a graph query (e.g., Cypher) and return results."""
        pass

    @abstractmethod
    def create_schema(self) -> None:
        """Initialize required database schema (DDL for Ladybug, etc)."""
        pass

    @abstractmethod
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add an embedding vector to a specific node."""
        pass

    @abstractmethod
    def prune(self, criteria: dict[str, Any]) -> None:
        """Run pruning logic based on criteria."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
