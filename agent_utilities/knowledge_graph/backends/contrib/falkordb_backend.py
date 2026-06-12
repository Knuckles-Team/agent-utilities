#!/usr/bin/python
from __future__ import annotations

"""FalkorDB Backend Implementation."""

# CONCEPT:KG-2.0


import logging
from typing import Any

from ..base import GraphBackend

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

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results = []
        for params in batch:
            results.extend(self.execute(query, params))
        return results

    def create_schema(self) -> None:
        # Index a SHARED ``Embeddable`` label (every embedded node is tagged with
        # it in add_embedding). The old index targeted ``Chunk`` — a label no node
        # carries — so vector search silently returned nothing.
        logger.info("Creating FalkorDB vector index for embeddings (Embeddable).")
        query = "CALL db.idx.vector.create('idx_embedding', 'Embeddable', 'embedding', 768, 'cosine')"
        try:
            self.execute(query)
        except Exception as e:
            logger.warning(f"Could not create vector index in FalkorDB: {e}")

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        # Tag ``:Embeddable`` so the node enters the vector index regardless of label.
        query = "MATCH (n {id: $id}) SET n:Embeddable, n.embedding = vecf32($embedding)"
        try:
            self.execute(query, {"id": node_id, "embedding": embedding})
        except Exception as e:
            logger.warning(f"Failed to add embedding in FalkorDB: {e}")

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Perform a semantic vector search returning top matching nodes using FalkorDB."""
        query = """
        CALL db.idx.vector.queryNodes('Embeddable', 'embedding', $n_results, vecf32($query_embedding))
        YIELD node, score
        RETURN node
        """
        try:
            return self.execute(
                query, {"query_embedding": query_embedding, "n_results": n_results}
            )
        except Exception as e:
            logger.error(f"FalkorDB semantic search failed: {e}")
            return []

    def prune(self, criteria: dict[str, Any]) -> None:
        query = "MATCH (n) WHERE n.last_accessed < $timestamp DELETE n"
        if "last_accessed" in criteria:
            self.execute(query, {"timestamp": criteria["last_accessed"]})

    def close(self) -> None:
        """Close the FalkorDB connection."""
        # FalkorDB client doesn't have an explicit close in some versions,
        # but we follow the interface by clearing our references to free resources.
        if hasattr(self, "client"):
            del self.client
        if hasattr(self, "graph"):
            del self.graph
