#!/usr/bin/python
from __future__ import annotations

"""FalkorDB Backend Implementation."""

# CONCEPT:AU-KG.query.object-graph-mapper


import logging
import re
from typing import Any

from ..base import GraphBackend, coerce_cypher_property
from ..mirror_target import (
    MirrorTarget,
    cypher_target_has_data,
    resolve_mirror_target,
)

logger = logging.getLogger(__name__)

# FalkorDB's isolation unit is a **graph key** in the Redis keyspace — there is no
# server-side "default graph", so this name plays that role when a connection
# names none (CONCEPT:AU-KG.backend.mirror-target-graph).
DEFAULT_GRAPH_KEY = "agent_graph"

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
        self,
        host: str = "localhost",
        port: int = 6379,
        db_name: str | None = None,
        *,
        mirror_target: MirrorTarget | None = None,
    ):
        if FalkorDB is None:
            raise ImportError(
                "FalkorDB driver is not installed. Please install with `pip install agent-utilities[falkordb]`"
            )
        # CONCEPT:AU-KG.backend.mirror-target-graph — FalkorDB is graph-per-key, so
        # the resolved target IS the key handed to ``select_graph``. A key springs
        # into existence on its first write, so a dedicated target needs no
        # creation step (``ensure_mirror_target`` is a no-op by omission).
        self.mirror_target = mirror_target or resolve_mirror_target(
            None,
            backend_type="falkordb",
            named_selector=db_name,
            default_name=DEFAULT_GRAPH_KEY,
        )
        self.db_name = self.mirror_target.name or DEFAULT_GRAPH_KEY
        self.client = FalkorDB(host=host, port=port)
        self.graph = self.client.select_graph(self.db_name)
        logger.info("Initialized FalkorDB backend")

    # ------------------------------------------------------------------
    # Mirror target (CONCEPT:AU-KG.backend.mirror-target-graph)
    # ------------------------------------------------------------------
    def mirror_target_locator(self) -> str:
        """Name the resolved target the way a FalkorDB operator would."""
        return f"{self.mirror_target.describe()} — FalkorDB graph key {self.db_name!r}"

    def mirror_target_has_data(self) -> bool:
        """Whether the selected FalkorDB graph key already holds nodes."""
        return cypher_target_has_data(self)

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        if include_epistemic:
            # CONCEPT:AU-KB-CURRENCY (Seam 1) — no id-seeded epistemic-envelope
            # primitive on this backend; degrade to ``[]`` per the ABC contract.
            logger.debug(
                "FalkorDBBackend.execute(include_epistemic=True): no epistemic "
                "envelope primitive; returning []"
            )
            return []
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
        # forever (CONCEPT:AU-KG.backend.mirror-health-repair).
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

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute with FalkorDB's read-only query command."""
        if include_epistemic:
            return []
        read_query = getattr(self.graph, "ro_query", None)
        if not callable(read_query):
            raise RuntimeError("FalkorDB read-only query support is unavailable")
        cleaned = {
            key: _clean_param_value(coerce_cypher_property(value))
            for key, value in (params or {}).items()
        }
        result = read_query(query, cleaned or None)
        output: list[dict[str, Any]] = []
        for row in result.result_set:
            row_dict: dict[str, Any] = {}
            for index, value in enumerate(row):
                header = result.header[index][1]
                row_dict[header] = (
                    value.properties if hasattr(value, "properties") else value
                )
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
    # one-authority + mirrors architecture (CONCEPT:AU-KG.backend.mirror-health-repair), vector search is
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
