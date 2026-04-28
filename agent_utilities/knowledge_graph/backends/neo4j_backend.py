#!/usr/bin/python
"""Neo4j Backend Implementation."""

from __future__ import annotations

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

    def create_schema(self) -> None:
        logger.info(
            "Neo4j does not require strict DDL schema. Creating vector indices if needed."
        )

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        query = """
        MATCH (n {id: $id})
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        """
        self.execute(query, {"id": node_id, "embedding": embedding})

    def prune(self, criteria: dict[str, Any]) -> None:
        query = "MATCH (n) WHERE n.last_accessed < $timestamp DETACH DELETE n"
        if "last_accessed" in criteria:
            self.execute(query, {"timestamp": criteria["last_accessed"]})

    def close(self) -> None:
        self.driver.close()
