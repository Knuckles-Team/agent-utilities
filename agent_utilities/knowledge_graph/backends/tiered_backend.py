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
:meth:`reconcile_to_durable`. Reads are served from L1, including id-anchored
relationship traversals (resolved natively on the engine); only traversals L1
can't anchor fall through to L3.

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


# A relationship pattern ``-[...]-`` (single-hop or variable-length traversal).
_TRAVERSAL_RE = re.compile(r"-\s*\[[^\]]*\]\s*->?|<-\s*\[[^\]]*\]\s*-")

# An ``{id: ...}`` anchor — the entry point the L1 engine needs to walk a
# traversal natively (single-hop ``->``/``<-`` or bounded ``[*lo..hi]``) over its
# neighbour/BFS ops. (CONCEPT:KG-2.7 P1 — L1 native traversal.)
_ID_ANCHOR_RE = re.compile(
    r"\{\s*id\s*:\s*(\$\w+|'[^']*'|\"[^\"]*\")\s*\}", re.IGNORECASE
)


def _is_traversal(query: str) -> bool:
    """True if the (read) query traverses a relationship pattern."""
    return bool(_TRAVERSAL_RE.search(query or ""))


def _l1_can_traverse(query: str) -> bool:
    """True if the L1 epistemic engine can resolve this traversal natively.

    L1 handles **id-anchored** traversals — single-hop (``->``/``<-``) and bounded
    variable-length (``[*lo..hi]``) — by walking the engine's neighbour/BFS ops
    from the anchor. Without an ``{id: ...}`` anchor there is no entry point, so
    those defer to L3's relational ``kg_edges`` JOINs.
    """
    q = query or ""
    return bool(_ID_ANCHOR_RE.search(q)) and _is_traversal(q)


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
        self._l1_reads = 0
        self._l3_reads = 0
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

        Traversal reads: an **id-anchored** relationship traversal (single-hop
        ``->``/``<-`` or bounded ``[*lo..hi]``) is now resolved natively on the
        fast L1 engine — its reason for existing. Only traversals L1 can't anchor
        (no ``{id:...}`` entry point) fall through to L3, which transpiles
        ``(a)-[:R]->(b)`` to a ``kg_edges`` JOIN; that path falls back to L1 if L3
        is unavailable. (CONCEPT:KG-2.7 P1 — L1 native traversal.)
        """
        if _is_write(query):
            result = self.l1.execute(query, params)
            self._mirror("execute", lambda: self.l3.execute(query, params))
            return result
        if _is_traversal(query):
            if _l1_can_traverse(query):
                self._l1_reads += 1
                return self.l1.execute(query, params)
            self._l3_reads += 1
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
    def _l1_edges(self, graph: Any) -> list[tuple[str, str, dict]]:
        """Enumerate L1 edges as ``(src, dst, data)`` via the bulk edge view."""
        view = getattr(graph, "edges", None)
        if view is not None:
            try:
                seq = view(data=True) if callable(view) else view
                out = []
                for e in seq:
                    if isinstance(e, tuple | list) and len(e) >= 2:
                        out.append((e[0], e[1], e[2] if len(e) > 2 else {}))
                if out:
                    return out
            except Exception:  # noqa: BLE001
                pass
        for meth in ("_get_all_edges",):  # NX-style enumeration (carries data)
            fn = getattr(graph, meth, None)
            if callable(fn):
                try:
                    return [(e[0], e[1], e[2] if len(e) > 2 else {}) for e in fn()]
                except Exception:  # noqa: BLE001
                    pass
        try:  # underlying client list()
            return [
                (e[0], e[1], e[2] if len(e) > 2 else {})
                for e in graph._client.edges.list()
            ]
        except Exception:  # noqa: BLE001
            return []

    def _l3_label_count(self, label: str) -> int | None:
        """Durable node count for ``label`` (``None`` if it can't be measured)."""
        try:
            rows = self.l3.execute(f"MATCH (n:{label}) RETURN count(n) AS c")
            if rows and isinstance(rows[0], dict):
                v = rows[0].get("c")
                return int(v) if v is not None else None
        except Exception:  # noqa: BLE001
            return None
        return None

    def reconcile_to_durable(self) -> dict[str, int]:
        """Mirror the L1 graph into L3 and report **exact** remaining drift.

        Writes every L1 node/edge to the durable tier (auto-DDL self-heals any
        missing table/column), then measures the *actual* divergence by comparing
        per-type L1 vs L3 counts — so the metric reflects what truly landed, not
        the count of best-effort writes (``execute`` swallows failures and returns
        ``[]``, so per-write counting over-reports). ``nodes_missing`` /
        ``edges_missing`` are the honest drift after the pass; non-zero means a
        write was dropped despite auto-DDL.
        """
        summary: dict[str, int] = {
            "nodes": 0,
            "edges": 0,
            "errors": 0,
            "nodes_missing": 0,
            "edges_missing": 0,
            "prior_l3_failures": self._l3_failures,
        }
        graph = getattr(self.l1, "graph", None)
        if graph is None:
            logger.warning(
                "reconcile_to_durable: L1 exposes no compute graph; skipping"
            )
            return summary

        # --- Nodes: write each (auto-DDL on the L3 write path heals new types). ---
        try:
            node_ids = list(graph._get_all_nodes())
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconcile_to_durable: cannot enumerate L1 nodes: %s", exc)
            node_ids = []
        l1_by_label: dict[str, int] = {}
        for nid in node_ids:
            try:
                props = dict(graph._get_node_properties(nid) or {})
                label = _sanitize_label(
                    props.get("type") or props.get("label") or "Node"
                )
                l1_by_label[label] = l1_by_label.get(label, 0) + 1
                self.l3.execute(
                    f"CREATE (n:{label} {{id: $id, name: $name, type: $type}})",
                    {"id": nid, "name": str(props.get("name") or nid), "type": label},
                )
                summary["nodes"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("reconcile node %s failed: %s", nid, exc)

        # --- Edges: mirror into the unified kg_edges table. ---
        l1_edges = self._l1_edges(graph)
        for src, dst, edata in l1_edges:
            try:
                rel = _sanitize_label((edata or {}).get("type", "RELATED_TO"))
                # The transpiler recognizes UPSERT_EDGE only with the node-MATCH
                # prefix (→ INSERT INTO kg_edges); the bare ``MERGE (a)-[r]->(b)``
                # is UNKNOWN and silently no-ops.
                self.l3.execute(
                    f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                    f"MERGE (a)-[r:{rel}]->(b)",
                    {"source_id": src, "target_id": dst},
                )
                summary["edges"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("reconcile edge %s->%s failed: %s", src, dst, exc)

        # --- EXACT drift: post-condition L1 vs L3 counts (not per-write trust). ---
        for label, l1n in l1_by_label.items():
            l3n = self._l3_label_count(label)
            if l3n is not None:
                summary["nodes_missing"] += max(0, l1n - l3n)
            else:
                summary["nodes_missing"] += l1n  # couldn't verify → assume unmirrored
        l3e = getattr(self.l3, "edge_count", lambda: None)()
        if l3e is not None:
            summary["edges_missing"] = max(0, len(l1_edges) - l3e)
        else:
            summary["edges_missing"] = 0  # not measurable on this backend

        logger.info(
            "TieredGraphBackend reconcile: %d nodes, %d edges written; "
            "drift after: %d nodes / %d edges missing; %d write errors",
            summary["nodes"],
            summary["edges"],
            summary["nodes_missing"],
            summary["edges_missing"],
            summary["errors"],
        )
        return summary

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def durability_stats(self) -> dict[str, int]:
        """Mirror counters for monitoring the durable tier + read routing."""
        return {
            "l3_writes": self._l3_writes,
            "l3_failures": self._l3_failures,
            "l1_reads": self._l1_reads,
            "l3_reads": self._l3_reads,
        }
