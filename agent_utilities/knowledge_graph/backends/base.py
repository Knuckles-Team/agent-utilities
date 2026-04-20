#!/usr/bin/python
# coding: utf-8
"""Graph Backend Base Interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class GraphBackend(ABC):
    """Abstract interface for Graph Database operations."""

    @abstractmethod
    def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a graph query (e.g., Cypher) and return results."""
        pass

    @abstractmethod
    def create_schema(self) -> None:
        """Initialize required database schema (DDL for Ladybug, etc)."""
        pass

    @abstractmethod
    def add_embedding(self, node_id: str, embedding: List[float]) -> None:
        """Add an embedding vector to a specific node."""
        pass

    @abstractmethod
    def prune(self, _criteria: Dict[str, Any]) -> None:
        """Run pruning logic based on criteria."""
        pass
