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

from agent_utilities.models.knowledge_graph import (
    RETIRED_EDGE_RELATIONSHIP_PROPERTIES,
    retired_edge_relationship_property_error,
    retired_node_type_property_error,
)
from agent_utilities.security.identifiers import (
    InvalidIdentifierError,
    validate_identifier,
)

from .base import GraphBackend, is_write

logger = logging.getLogger(__name__)

_PARAM_TOKEN_RE = re.compile(r"\$(\w+)")
_CURRENT_TIMESTAMP_RE = re.compile(r"\bcurrent_timestamp\(\)", re.I)


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


# ``$name`` parameter placeholders and the ``current_timestamp()`` pseudo-literal
# get inlined into the query text before it reaches the native engine — its wire
# method (``CypherQuery``) carries only the literal query string, no params map
# (CONCEPT:AU-KG.query.object-graph-mapper).


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (the ``current_timestamp()`` value)."""
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()


def _cypher_literal(value: Any) -> str:
    """Render ``value`` as a literal the native Cypher engine's grammar accepts.

    The engine's hand-written parser (``eg-query/src/cypher/parser.rs``) only
    implements ``literal := string | number | true | false`` — there is no
    ``null`` literal (only ``IS [NOT] NULL`` predicates) and no negative-number
    literal (``-`` never fuses with a following digit in its tokenizer). A value
    this grammar cannot express raises ``NotImplementedError`` naming the exact
    gap rather than silently mis-inlining it or dropping the predicate.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise NotImplementedError(  # ABSTRACT-OK: intentional unsupported-native-Cypher-shape raise, contractually unimplemented (AU-P0-2 contract)
            "the native Cypher engine has no NULL literal (only IS [NOT] NULL "
            "predicates); cannot inline a None value as an equality/SET literal "
            "— rewrite the query to use IS NULL / IS NOT NULL instead"
        )
    if isinstance(value, int | float):
        if value < 0:
            raise NotImplementedError(  # ABSTRACT-OK: intentional unsupported-native-Cypher-shape raise, contractually unimplemented (AU-P0-2 contract)
                "the native Cypher engine's literal grammar has no negative "
                f"number literal (value={value!r}); rewrite the query so the "
                "comparison/assignment does not need a negative literal"
            )
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_cypher_literal(v) for v in value) + "]"
    raise NotImplementedError(  # ABSTRACT-OK: intentional unsupported-native-Cypher-shape raise, contractually unimplemented (AU-P0-2 contract)
        f"cannot inline a {type(value).__name__} value as a native Cypher "
        f"literal (value={value!r}); the engine's grammar only accepts "
        "string/number/bool literals and lists of them"
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
        from ..core.shard_topology import resolve_routing_graph

        # The one process transport may have been initialized first by the
        # encrypted secret resolver and therefore be rooted at ``__secrets__``.
        # ``None`` here means the operational tenant/default graph, never
        # "inherit whichever graph happened to create the transport first".
        # get_or_create() still reuses that one transport and returns only a
        # lightweight graph-scoped view.
        target_graph = resolve_routing_graph(graph_name)
        self._graph = GraphComputeEngine.get_or_create(
            graph_name=target_graph,
            backend_type="rust",
        )
        self.graph_name = getattr(self._graph, "graph_name", target_graph)
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
        parsed_hits: list[tuple[str, float]] = []
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
            parsed_hits.append((node_id, score))

        node_ids = [node_id for node_id, _ in parsed_hits]
        batch_get = getattr(self._graph, "_get_node_properties_batch", None)
        if callable(batch_get):
            properties = batch_get(node_ids)
        else:
            # Compatibility for injected graph adapters that predate the native
            # batch property surface. Production GraphComputeEngine takes the
            # single-RPC path above.
            properties = {
                node_id: self._graph._get_node_properties(node_id) or {}
                for node_id in node_ids
            }

        from ..enrichment.semantic import EMBEDDING_INDEX_READY_FIELD

        results: list[dict[str, Any]] = []
        for node_id, score in parsed_hits:
            data = properties.get(node_id, {})
            # Defense in depth for direct/fake graph surfaces that bypass
            # GraphComputeEngine's bounded stale-ANN fence.  The durable node
            # property is the source of truth while older engines lack an atomic
            # vector-only removal operation.
            if data.get(EMBEDDING_INDEX_READY_FIELD) is False:
                continue
            embedding = data.get("embedding")
            if not isinstance(embedding, list | tuple) or not embedding:
                continue
            results.append({**data, "id": node_id, "_similarity": score})
        return results

    def hydrate_engine_embeddings(self, batch_log_every: int = 5000) -> int:
        """Run the persisted-state embedding-index migration and fence repair."""
        from ..enrichment.semantic import EMBEDDING_INDEX_READY_FIELD

        count = 0
        for node_id, props in self._graph._get_all_nodes_with_properties():
            embedding = (props or {}).get("embedding")
            if not embedding:
                continue
            if (props or {}).get(EMBEDDING_INDEX_READY_FIELD) is False:
                # A process can stop after the durable cross-modal transaction
                # committed but before the served-read readiness CAS. Re-run the
                # idempotent transaction so startup hydration repairs that safe,
                # intentionally hidden state instead of hiding it forever.
                if self._graph.compare_and_set_node_embedding(
                    node_id,
                    {
                        "embedding": list(embedding),
                        EMBEDDING_INDEX_READY_FIELD: False,
                    },
                    {"embedding": list(embedding)},
                    list(embedding),
                ):
                    count += 1
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
        try:
            for key in criteria:
                validate_identifier(key, kind="property")
        except InvalidIdentifierError as exc:
            raise ValueError(
                "prune criteria contains an invalid property identifier"
            ) from exc
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
        return None

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
            raise retired_node_type_property_error()
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
        aliases = RETIRED_EDGE_RELATIONSHIP_PROPERTIES.intersection(properties)
        if aliases:
            raise retired_edge_relationship_property_error(aliases)
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

    def compare_and_set_node_embedding(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
        embedding: list[float],
    ) -> bool:
        """Condition fields and replace the ANN vector in one native transaction."""
        return self._graph.compare_and_set_node_embedding(
            node_id, conditions, updates, embedding
        )

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
