from __future__ import annotations

"""Epistemic Graph authority adapter.

This backend is deliberately thin. Cypher is parsed, planned, authorized, and
executed by the Rust engine through its explicit ``cypher_read`` and
``cypher_write`` wire modes. Python does not interpret patterns, traverse the
graph, scan labels, simulate aggregation, compile mutations, or translate
``UNWIND`` batches.

Public reads reach this adapter only after a verified
:class:`~agent_utilities.knowledge_graph.core.session.GraphSession` has passed
the guarded query service. Internal query-language writes use the engine's
durable mutation gateway; structured ingestion uses native ``ChangeEnvelope``
instead of raw Cypher batches.
"""

import datetime
import hashlib
import json
import logging
import re
from typing import Any

from .base import GraphBackend, is_write

logger = logging.getLogger(__name__)

_PARAM_TOKEN_RE = re.compile(r"\$(\w+)")
_CURRENT_TIMESTAMP_RE = re.compile(r"\bcurrent_timestamp\(\)", re.I)
_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _query_reference(query: str) -> str:
    """Return a non-reversible query identifier safe for logs and errors."""
    return hashlib.sha256(str(query).encode("utf-8", errors="replace")).hexdigest()[:16]


class CypherEngineError(RuntimeError):
    """The native Cypher authority rejected or failed a request.

    Query text, parameters, endpoints, paths, and backend error details are not
    retained or exposed. Operators can correlate the stable reference with the
    engine's governed audit record.
    """

    def __init__(self, query: str, mode: str, cause: BaseException) -> None:
        self.query_reference = _query_reference(query)
        self.mode = mode
        self.error_type = type(cause).__name__
        super().__init__(
            "native Cypher authority rejected request "
            f"(query_ref={self.query_reference}, mode={mode}, "
            f"error_type={self.error_type})"
        )


def _cypher_literal(value: Any) -> str:
    """Render a bound value in the engine's dependency-free Cypher grammar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError(
            "Cypher parameters cannot encode NULL literals; use IS NULL or IS NOT NULL"
        )
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError("Cypher parameters cannot encode negative number literals")
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_cypher_literal(item) for item in value) + "]"
    raise TypeError(
        f"Cypher parameters do not support {type(value).__name__}; "
        "use scalar values or scalar lists"
    )


class EpistemicGraphBackend(GraphBackend):
    """One-process adapter over the authoritative Rust graph service."""

    @property
    def typed_mutation_support(self) -> str:
        """Declare the explicit lossless mutation seam used by engine writers."""
        return "native"

    @property
    def cypher_support(self) -> str:
        """Report the sole Cypher implementation used by this backend."""
        return "native"

    def __init__(self, graph_name: str | None = None) -> None:
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine.get_or_create(
            graph_name=graph_name,
            backend_type="rust",
        )
        self.graph_name = getattr(self._graph, "graph_name", graph_name)
        logger.info("EpistemicGraphBackend bound to the process graph authority")

    def for_graph(self, graph_name: str) -> EpistemicGraphBackend:
        """Return a graph-scoped view without opening another transport."""
        target = str(graph_name or self.graph_name)
        if target == self.graph_name:
            return self
        view = object.__new__(type(self))
        view._graph = self._graph.for_graph(target)
        view.graph_name = target
        return view

    @property
    def graph(self) -> Any:
        """Return the non-owning process graph view."""
        return self._graph

    @staticmethod
    def _inline_cypher_params(
        query: str,
        params: dict[str, Any],
    ) -> str:
        """Bind parameters for the current literal-only Cypher wire contract.

        The native engine remains the only parser and execution authority. This
        renderer merely serializes values because ``CypherQuery`` currently
        carries query text rather than a separate parameter map.
        """
        if not str(query or "").strip():
            raise ValueError("an explicit Cypher statement is required")
        rendered = _CURRENT_TIMESTAMP_RE.sub(
            lambda _match: _cypher_literal(
                datetime.datetime.now(datetime.UTC).isoformat()
            ),
            query,
        )

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in params:
                raise ValueError(
                    "Cypher query is missing a referenced parameter "
                    f"(query_ref={_query_reference(query)})"
                )
            return _cypher_literal(params[name])

        return _PARAM_TOKEN_RE.sub(replace, rendered)

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Dispatch an internal statement to an explicit native engine mode.

        Public query surfaces call :meth:`execute_read` directly. This
        compatibility-free internal method never guesses authorization: a
        lexical write indication can only select the stricter write mode, and
        the engine reparses the complete statement and rejects every mode
        mismatch before execution.
        """
        if is_write(query):
            if include_epistemic:
                raise ValueError("epistemic row projection is available only for reads")
            return self.execute_write(query, params)
        return self.execute_read(
            query,
            params,
            include_epistemic=include_epistemic,
        )

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute through the engine's parser-enforced read-only mode."""
        from ..core.session import resolve_session

        resolve_session(required_scope="kg:read")
        rendered = self._inline_cypher_params(query, params or {})
        try:
            rows = list(self._graph.query_cypher(rendered) or [])
        except Exception as exc:  # noqa: BLE001 - replace unsafe driver details
            raise CypherEngineError(rendered, "read", exc) from None
        if not include_epistemic:
            return rows
        from ..core.epistemic_row import attach_epistemic_rows

        return attach_epistemic_rows(rows, self._graph.explain_provenance_by_ids)  # type: ignore[return-value]

    def execute_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute an internal query-language mutation through MutationBatch."""
        from ..core.session import resolve_session

        resolve_session(required_scope="kg:write")
        rendered = self._inline_cypher_params(query, params or {})
        try:
            return list(self._graph.query_cypher_write(rendered) or [])
        except Exception as exc:  # noqa: BLE001 - replace unsafe driver details
            raise CypherEngineError(rendered, "write", exc) from None

    def execute_batch(
        self,
        query: str,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reject raw query-language batches.

        Translating ``UNWIND`` into per-row statements created a second Python
        mutation compiler and split durability. Connector and materialization
        batches must use native ``ChangeEnvelope`` ingestion.
        """
        del query, batch
        raise RuntimeError(
            "raw Cypher batches are not supported by the graph authority; "
            "commit a native ChangeEnvelope graph slice"
        )

    def create_schema(self) -> None:
        """Validate that the native authority is reachable."""
        if not self.health_check():
            raise RuntimeError("native graph authority health check failed")

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Write an embedding to the engine-maintained ANN index."""
        self._graph.add_embedding(node_id, embedding)

    def semantic_search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Run the engine-maintained ANN query without an O(N) Python fallback."""
        hits = self._graph.semantic_search(query_embedding, n_results) or []
        results: list[dict[str, Any]] = []
        for item in hits:
            if isinstance(item, list | tuple) and len(item) >= 2:
                node_id, score = str(item[0]), float(item[1])
            elif isinstance(item, dict):
                node_id = str(item.get("id", ""))
                score = float(item.get("_similarity", item.get("score", 0.0)) or 0.0)
            else:
                continue
            if not node_id:
                continue
            data = self._graph._get_node_properties(node_id) or {}
            results.append({**data, "id": node_id, "_similarity": score})
        return results

    def hydrate_engine_embeddings(self, batch_log_every: int = 5000) -> int:
        """Run the one-time persisted-state embedding-index migration."""
        count = 0
        for node_id, props in self._graph._get_all_nodes_with_properties():
            embedding = (props or {}).get("embedding")
            if not embedding:
                continue
            self._graph.add_embedding(node_id, list(embedding))
            count += 1
            if count % batch_log_every == 0:
                logger.info("embedding-index migration processed %d rows", count)
        logger.info("embedding-index migration completed (%d rows)", count)
        return count

    def prune(self, criteria: dict[str, Any]) -> None:
        """Remove nodes selected by a native read followed by typed removals."""
        if not criteria:
            raise ValueError("prune criteria must not be empty")
        if any(not _CYPHER_IDENTIFIER_RE.fullmatch(str(key)) for key in criteria):
            raise ValueError("prune criteria contains an invalid property identifier")
        predicates = " AND ".join(f"n.{key} = ${key}" for key in criteria)
        rows = self.execute_read(
            f"MATCH (n) WHERE {predicates} RETURN n AS node_id",
            criteria,
        )
        for row in rows:
            node_id = row.get("node_id")
            if isinstance(node_id, str) and node_id:
                self._graph.remove_node(node_id)
        logger.info("graph prune completed (%d nodes)", len(rows))

    def close(self) -> None:
        """Keep the shared process transport alive; graph views own no resource."""

    def health_check(self) -> bool:
        """Return native service health."""
        try:
            return bool(self._graph.client.health())
        except Exception:  # noqa: BLE001 - health is a boolean probe
            return False

    def get_stats(self) -> dict[str, Any]:
        """Return native graph cardinalities."""
        return {
            "backend": "epistemic-graph",
            "nodes": self._graph.node_count(),
            "edges": self._graph.edge_count(),
        }

    def add_node(self, node_id: str, label: str = "", **properties: Any) -> None:
        """Atomically create or field-merge one node through ``BatchUpdate``.

        ``upsert_node`` is the native counterpart of ``MERGE`` plus ``SET``: a
        missing node is created and an existing node keeps fields omitted by this
        call.  Keeping the operation typed preserves nested and list-valued
        properties without compiling a second mutation language in Python.
        """
        if "type" in properties:
            raise ValueError("node property 'type' is retired; use node_type")
        node_type = str(properties.get("node_type") or label).strip()
        if not node_type:
            raise ValueError("node_type is required")
        self._graph.batch_update(
            [
                {
                    "op": "upsert_node",
                    "id": node_id,
                    "properties": {
                        **properties,
                        "id": node_id,
                        "node_type": node_type,
                    },
                }
            ]
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "",
        /,
        **properties: Any,
    ) -> None:
        """Add or replace one edge through the typed native operation."""
        aliases = {"type", "rel_type", "relationship_type", "relation"}.intersection(
            properties
        )
        if aliases:
            raise ValueError("edge relationship aliases are retired; use relationship")
        relationship = str(properties.get("relationship") or rel_type).strip()
        if not relationship:
            raise ValueError("relationship is required")
        self._graph.batch_update(
            [
                {
                    "op": "upsert_edge",
                    "source": source_id,
                    "target": target_id,
                    "properties": {
                        **properties,
                        "relationship": relationship,
                    },
                }
            ]
        )

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        """Return one node through the typed native point read."""
        if not self._graph.has_node(node_id):
            return None
        props = self._graph._get_node_properties(node_id)
        return dict(props) if isinstance(props, dict) else {}

    def nodes_by_label(
        self,
        label: str,
        limit: int = 0,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read a label through the engine-maintained label index."""
        return self._graph.get_nodes_by_label(label, limit) or []

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        """Apply an atomic native conditional field update."""
        return self._graph.compare_and_set_node_fields(node_id, conditions, updates)

    def save_to_json(self, path: str) -> None:
        """Export an operator-requested snapshot without logging its location."""
        data = json.loads(self._graph.to_json())
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
        logger.info("graph export completed")

    def load_from_json(self, path: str) -> None:
        """Import an operator-requested snapshot without logging its location."""
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        self._graph.from_json(json.dumps(data))
        logger.info("graph import completed")
