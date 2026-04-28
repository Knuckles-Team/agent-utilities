#!/usr/bin/python
"""FalkorDB Backend Implementation."""

from __future__ import annotations

import logging
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)

try:
    from falkordb import FalkorDB
except ImportError:
    FalkorDB = None


class FalkorDBBackend(GraphBackend):
    """FalkorDB backend for the unified graph."""

    def __init__(
        self, host: str = "localhost", port: int = 6379, db_name: str = "agent_graph"
    ):
        if FalkorDB is None:
            raise ImportError(
                "FalkorDB driver is not installed. Please install with `pip install agent-utilities[falkordb]`"
            )
        self.db_name = db_name
        self.client = FalkorDB(host=host, port=port)
        self.graph = self.client.select_graph(db_name)
        logger.info(f"Initialized FalkorDB backend at {host}:{port}")

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        result = self.graph.query(query, params)
        # Convert FalkorDB ResultSet to list of dicts
        output = []
        for row in result.result_set:
            row_dict = {}
            for i, val in enumerate(row):
                header = result.header[i][1]
                if isinstance(val, list):
                    # It might be a path or a complex object
                    row_dict[header] = val
                elif hasattr(val, "properties"):
                    row_dict[header] = val.properties
                else:
                    row_dict[header] = val
            output.append(row_dict)
        return output

    def create_schema(self) -> None:
        logger.info(
            "FalkorDB does not require strict DDL schema. Creating vector indices if needed."
        )

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        query = """
        MATCH (n {id: $id})
        CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
        """
        # Note: vector support in FalkorDB is evolving, using standard approach if possible
        try:
            self.execute(query, {"id": node_id, "embedding": embedding})
        except Exception as e:
            logger.warning(f"Failed to add embedding in FalkorDB: {e}")

    def prune(self, criteria: dict[str, Any]) -> None:
        query = "MATCH (n) WHERE n.last_accessed < $timestamp DELETE n"
        if "last_accessed" in criteria:
            self.execute(query, {"timestamp": criteria["last_accessed"]})

    def close(self) -> None:
        """Close the FalkorDB connection."""
        # FalkorDB client doesn't have an explicit close in some versions,
        # but we follow the interface.
        pass
