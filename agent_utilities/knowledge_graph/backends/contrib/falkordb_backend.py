#!/usr/bin/python
from __future__ import annotations

"""FalkorDB Backend Implementation."""

# CONCEPT:KG-2.0


import logging
import re
from typing import Any

from ..base import GraphBackend, coerce_cypher_property

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
        # coerce_cypher_property first (Map/nested → JSON string so FalkorDB doesn't
        # reject a Map-valued prop and stall a mirror), then strip control chars.
        params = {
            k: _clean_param_value(coerce_cypher_property(v))
            for k, v in (params or {}).items()
        }
        # Bind params only when there ARE any. FalkorDB's redis-graph client prepends a
        # ``CYPHER `` parameter header whenever a (non-None) ``params`` map is supplied
        # — even an EMPTY ``{}`` — and the server then rejects the headered query with
        # "Missing parameters". Passing ``None`` for an empty map omits the header so a
        # parameter-free mirror query applies cleanly instead of stalling/retrying
        # forever (CONCEPT:KG-2.74).
        result = self.graph.query(query, params or None)
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

    # Native vector search is intentionally NOT served by FalkorDB. In the
    # one-authority + mirrors architecture (CONCEPT:KG-2.74), vector search is
    # served by the epistemic-graph engine authority (and the pgvector/AGE mirror);
    # a graph mirror like FalkorDB only carries the node/edge topology. This is also
    # a hard necessity: ``falkordb:latest``'s vector engine (CREATE VECTOR INDEX +
    # ``vecf32`` writes / ``db.idx.vector.queryNodes``) crashes the server process on
    # our workload, taking the whole container down. So FalkorDB is a graph-only
    # mirror; embeddings are no-ops and ``semantic_search`` returns nothing (a
    # documented parity gap the conformance suite skips, not a hard failure).
    supports_native_vector_search: bool = False

    def create_schema(self) -> None:
        # Graph-only mirror: no vector index (see ``supports_native_vector_search``).
        logger.info(
            "FalkorDB backend ready (graph-only mirror; native vector search is "
            "served by the engine authority, not FalkorDB)."
        )

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        # No-op: FalkorDB is a graph-only mirror. Sending a vecf32 write to
        # falkordb:latest crashes the server, and vectors live on the authority.
        return

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """No native vector search on FalkorDB (graph-only mirror) — see
        ``supports_native_vector_search``. Returns ``[]`` so callers fall back to
        the engine authority's vector path."""
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
