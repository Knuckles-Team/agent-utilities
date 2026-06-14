#!/usr/bin/python
from __future__ import annotations

"""FalkorDB Backend Implementation."""

# CONCEPT:KG-2.0


import logging
import re
from typing import Any

from ..base import GraphBackend

logger = logging.getLogger(__name__)

# FalkorDB's query-parameter parser rejects strings containing C0/C1 control
# characters (e.g. \x01 from PDF/binary text extraction) with "Failed to parse
# query parameter value", dropping the whole node. Neo4j accepts them, so this is
# FalkorDB-specific. Strip control chars except tab/newline/carriage-return.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _clean_param_value(value: Any) -> Any:
    """Recursively strip control characters from string param values so FalkorDB's
    parser accepts them; non-strings (and dict/list containers) pass through."""
    if isinstance(value, str):
        return _CONTROL_CHARS.sub("", value)
    if isinstance(value, list):
        return [_clean_param_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_param_value(v) for k, v in value.items()}
    return value


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
        params = {k: _clean_param_value(v) for k, v in (params or {}).items()}
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
        # FalkorDB >=4 uses Cypher DDL for vector indexes; the older
        # `db.idx.vector.create` procedure is not registered. Index a shared
        # `:Embeddable` label (add_embedding tags nodes with it); FalkorDB
        # backfills existing matching nodes when the index is created.
        query = (
            "CREATE VECTOR INDEX FOR (n:Embeddable) ON (n.embedding) "
            "OPTIONS {dimension: 768, similarityFunction: 'cosine'}"
        )
        try:
            self.execute(query)
        except Exception as e:
            if "already" not in str(e).lower():
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
