#!/usr/bin/python
"""Tiered (write-through) graph backend.

CONCEPT:KG-2.7 — Vendor-Agnostic Graph Backend Abstraction

Composes two :class:`GraphBackend` instances into a two-tier store:

* **L1 — working store** (default ``EpistemicGraphBackend``): serves all reads
  and graph compute. Fast, in-process, low latency.
* **L3 — durable persistence** (default ``PostgreSQLBackend`` / pggraph): every
  mutation is mirrored here so state survives process restarts.

Writes are applied to L1 first (authoritative for reads), then mirrored to L3.
**L3 mirror failures are logged and non-fatal** — a transient durability hiccup
must not abort an ingestion run; the gap is closed by
:meth:`reconcile_to_durable`. Reads never touch L3.

Selected via ``GRAPH_BACKEND=tiered`` (see ``create_backend``); the L1 type is
``GRAPH_BACKEND_L1`` (default ``epistemic_graph``) and L3 is a PostgreSQL DSN
(``GRAPH_DB_URI`` / ``PGGRAPH_DSN``).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)

# Cypher clauses that mutate the graph. A query containing any of these (even
# alongside a leading MATCH) is treated as a write and mirrored to L3.
_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def _is_write(query: str) -> bool:
    """True if the Cypher query mutates the graph."""
    return bool(_WRITE_RE.search(query or ""))


# A relationship pattern ``-[...]-`` (single-hop traversal). The L1 epistemic
# interpreter can't traverse edges (it returns every node), so READ queries with
# a relationship pattern are served from L3 (durable, relational kg_edges), which
# transpiles them to JOINs. Node-only reads stay on the fast L1 store.
_TRAVERSAL_RE = re.compile(r"-\s*\[[^\]]*\]\s*->?|<-\s*\[[^\]]*\]\s*-")


def _is_traversal(query: str) -> bool:
    """True if the (read) query traverses a relationship pattern."""
    return bool(_TRAVERSAL_RE.search(query or ""))


def _sanitize_label(label: str) -> str:
    """Reduce a label/relationship to a transpiler-safe identifier (``\\w+``)."""
    s = re.sub(r"\W+", "_", str(label or "Node")).strip("_")
    return s or "Node"


class TieredGraphBackend(GraphBackend):
    """Write-through wrapper: L1 working store + L3 durable persistence."""

    def __init__(self, l1: GraphBackend, l3: GraphBackend) -> None:
        self.l1 = l1
        self.l3 = l3
        self._l3_failures = 0
        self._l3_writes = 0
        logger.info(
            "TieredGraphBackend initialized (L1=%s, L3=%s)",
            type(l1).__name__,
            type(l3).__name__,
        )

    # ------------------------------------------------------------------
    # L3 mirroring helper — never raises
    # ------------------------------------------------------------------
    def _mirror(self, op: str, fn) -> None:
        try:
            fn()
            self._l3_writes += 1
        except Exception as exc:  # noqa: BLE001 - durability is best-effort
            self._l3_failures += 1
            logger.warning(
                "TieredGraphBackend: L3 mirror of %s failed (#%d): %s",
                op,
                self._l3_failures,
                exc,
            )

    # ------------------------------------------------------------------
    # Core CRUD & Query
    # ------------------------------------------------------------------
    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Reads → L1; writes → L1 (authoritative) then mirrored to L3.

        Exception: relationship-traversal READS go to L3, because the L1
        epistemic interpreter cannot traverse edges (it returns all nodes). L3
        transpiles ``(a)-[:R]->(b)`` to a JOIN over ``kg_edges``. Falls back to
        L1 if L3 is unavailable. (CONCEPT:KG-2.7 traversal correctness)
        """
        if _is_write(query):
            result = self.l1.execute(query, params)
            self._mirror("execute", lambda: self.l3.execute(query, params))
            return result
        if _is_traversal(query):
            try:
                return self.l3.execute(query, params)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash a read
                logger.warning(
                    "TieredGraphBackend: L3 traversal read failed (%s); "
                    "falling back to L1.",
                    exc,
                )
                return self.l1.execute(query, params)
        return self.l1.execute(query, params)

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """High-throughput ingestion — always a write; mirror to L3."""
        result = self.l1.execute_batch(query, batch)
        self._mirror("execute_batch", lambda: self.l3.execute_batch(query, batch))
        return result

    def create_schema(self) -> None:
        self.l1.create_schema()
        self._mirror("create_schema", self.l3.create_schema)

    # ------------------------------------------------------------------
    # Vector / Embedding Support
    # ------------------------------------------------------------------
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        self.l1.add_embedding(node_id, embedding)
        self._mirror("add_embedding", lambda: self.l3.add_embedding(node_id, embedding))

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Vector search served from L3 (durable pgvector), falling back to L1.

        Embeddings are persisted to L3 (pgvector over the per-label tables); the
        L1 working store's vector index is typically empty, so serving vector
        search from L1 silently returned nothing (breaking concept→code linking
        and semantic retrieval). Prefer L3; fall back to L1 only if L3 yields
        nothing or errors. (CONCEPT:KG-2.7 retrieval correctness)
        """
        try:
            res = self.l3.semantic_search(query_embedding, n_results)
            if res:
                return res
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TieredGraphBackend: L3 semantic_search failed (%s); "
                "falling back to L1.",
                exc,
            )
        return self.l1.semantic_search(query_embedding, n_results)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def prune(self, criteria: dict[str, Any]) -> None:
        self.l1.prune(criteria)
        self._mirror("prune", lambda: self.l3.prune(criteria))

    def close(self) -> None:
        try:
            self.l1.close()
        finally:
            try:
                self.l3.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TieredGraphBackend: L3 close failed: %s", exc)

    # ------------------------------------------------------------------
    # SPARQL — delegate to whichever tier supports it (L3 first)
    # ------------------------------------------------------------------
    @property
    def supports_sparql(self) -> bool:
        return self.l3.supports_sparql or self.l1.supports_sparql

    def execute_sparql(
        self,
        query: str,
        *,
        default_graph: str | None = None,
        timeout_ms: int = 30_000,
    ) -> list[dict[str, Any]]:
        target = self.l3 if self.l3.supports_sparql else self.l1
        return target.execute_sparql(
            query, default_graph=default_graph, timeout_ms=timeout_ms
        )

    # ------------------------------------------------------------------
    # Facade compatibility — expose the L1 compute graph + delegate the rest
    # ------------------------------------------------------------------
    @property
    def graph(self) -> Any:
        """The L1 compute engine (used by the KnowledgeGraph facade L1 path)."""
        return getattr(self.l1, "graph", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate backend-specific attributes/methods to the L1 store.

        The facade/engine sometimes call EpistemicGraph-specific helpers
        (e.g. ``save_to_json``, ``_get_node_properties``). Anything not defined
        on the tier wrapper resolves against L1. (``__getattr__`` only fires for
        names not found normally, so the explicit methods above take priority.)
        """
        # self.l1 may not be set yet during unpickling/partial init
        l1 = self.__dict__.get("l1")
        if l1 is not None and hasattr(l1, name):
            return getattr(l1, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ------------------------------------------------------------------
    # Durability reconciliation
    # ------------------------------------------------------------------
    def reconcile_to_durable(self) -> dict[str, int]:
        """Bulk-copy the L1 graph into L3 to close any mirror gaps.

        Returns a summary dict with counts of nodes/edges reconciled and the
        number of L3 errors encountered. Best-effort: individual failures are
        counted, not raised.
        """
        summary = {
            "nodes": 0,
            "edges": 0,
            "errors": 0,
            "prior_l3_failures": self._l3_failures,
        }
        graph = getattr(self.l1, "graph", None)
        if graph is None:
            logger.warning(
                "reconcile_to_durable: L1 exposes no compute graph; skipping"
            )
            return summary

        # Nodes
        try:
            node_ids = list(graph._get_all_nodes())
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconcile_to_durable: cannot enumerate L1 nodes: %s", exc)
            node_ids = []

        for nid in node_ids:
            try:
                props = dict(graph._get_node_properties(nid) or {})
                label = _sanitize_label(
                    props.get("type") or props.get("label") or "Node"
                )
                name = props.get("name") or nid
                # Backstop only: write the universally-present columns (id/name/type)
                # so an L1-only node at least EXISTS durably. Full property fidelity
                # is the engine's backend-first write path's responsibility.
                self.l3.execute(
                    f"CREATE (n:{label} {{id: $id, name: $name, type: $type}})",
                    {"id": nid, "name": str(name), "type": label},
                )
                summary["nodes"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("reconcile node %s failed: %s", nid, exc)

        # Edges — prefer the underlying client list() which carries edge data
        # (``_get_all_edges`` drops it). Fall back to the 2-tuple enumeration.
        try:
            edges = list(graph._client.edges.list())
        except Exception:
            try:
                edges = list(graph._get_all_edges())
            except Exception:
                edges = []
        for edge in edges:
            try:
                src, dst = edge[0], edge[1]
                edata = edge[2] if len(edge) > 2 else {}
                rel = _sanitize_label((edata or {}).get("type", "RELATED_TO"))
                # Transpiler-recognized edge-upsert form.
                self.l3.execute(
                    f"MERGE (a)-[r:{rel}]->(b)",
                    {"source_id": src, "target_id": dst},
                )
                summary["edges"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("reconcile edge %s->%s failed: %s", edge[0], edge[1], exc)

        logger.info(
            "TieredGraphBackend reconcile: %d nodes, %d edges, %d errors",
            summary["nodes"],
            summary["edges"],
            summary["errors"],
        )
        return summary

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def durability_stats(self) -> dict[str, int]:
        """Mirror counters for monitoring the durable tier."""
        return {"l3_writes": self._l3_writes, "l3_failures": self._l3_failures}
