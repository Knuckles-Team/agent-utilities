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


# A WHERE/label-anchored traversal — the graph_code_nav shapes
# (``MATCH (a:La)-[:REL]->(b:Lb) WHERE b.prop = $x``). L1 resolves these by
# scanning the labeled anchor for the property match, then walking — so they too
# belong on L1 (where the resolved code graph is authoritative), NOT L3.
# (CONCEPT:KG-2.9g)
_LABELED_NODE_RE = re.compile(r"\(\s*\w+\s*:\s*`?\w+", re.IGNORECASE)
_WHERE_PROP_ANCHOR_RE = re.compile(r"\bWHERE\b[^;]*?\b\w+\.\w+\s*=", re.IGNORECASE)


# An aggregate (``count(...)``) over a relationship pattern whose WHERE carries a
# node-property filter that NEITHER tier can faithfully honor — e.g.
# ``MATCH (d)-[r]->(c) WHERE d.doc_type='news_article' RETURN count(c)``. L1's
# unanchored aggregate path would drop the filter and return the *global* edge
# count (silently wrong); L3's regex transpiler can't anchor an unlabeled pattern
# and returns an empty (also wrong) result. Sending it to L1 makes the engine
# FAIL LOUD (raise) rather than fabricate a number. (CONCEPT:KG-2.9h)
_AGG_COUNT_RE = re.compile(r"\bRETURN\b[^;]*\bcount\s*\(", re.IGNORECASE)
_WHERE_RE = re.compile(r"\bWHERE\b(.+?)(?:\bRETURN\b|\bSET\b|$)", re.IGNORECASE | re.S)


def _is_unhonorable_aggregate(query: str) -> bool:
    """True if this is a relationship-aggregate whose WHERE filter no tier can honor.

    Such a query has no faithful answer on either tier, so it must fail loudly
    rather than return a fabricated count/empty. (CONCEPT:KG-2.9h)
    """
    q = query or ""
    if not (_is_traversal(q) and _AGG_COUNT_RE.search(q)):
        return False
    if _l1_can_traverse(q):
        return False  # id- or label+WHERE-anchored → a tier can honor it
    wm = _WHERE_RE.search(q)
    if not wm:
        return False
    # A WHERE predicate beyond bare ``.id`` equality (which would have anchored the
    # query above) cannot be applied by the unanchored aggregate path.
    residual = re.sub(
        r"\w+\.id\s*=\s*(?:\$\w+|'[^']*'|\"[^\"]*\")",
        "",
        wm.group(1),
        flags=re.IGNORECASE,
    )
    return bool(re.search(r"\w+\.\w+", residual))


def _l1_can_traverse(query: str) -> bool:
    """True if the L1 epistemic engine can resolve this traversal natively.

    L1 handles two anchored shapes by walking the engine's neighbour/BFS ops:
      * **id-anchored** — single-hop (``->``/``<-``) or bounded ``[*lo..hi]`` from
        an ``{id: ...}`` entry point; and
      * **label+WHERE-anchored** (CONCEPT:KG-2.9g) — a labeled pattern with a
        ``WHERE <var>.<prop> = …`` anchor (the code-nav find_references /
        trace_call_graph / impact_of_change shapes), resolved by scanning the
        labeled anchor then walking.
    Both are served from L1 (authoritative for the resolved graph). Only
    genuinely unanchored traversals defer to L3's relational ``kg_edges`` JOINs.
    """
    q = query or ""
    if not _is_traversal(q):
        return False
    if _ID_ANCHOR_RE.search(q):
        return True
    return bool(_LABELED_NODE_RE.search(q)) and bool(_WHERE_PROP_ANCHOR_RE.search(q))


def _sanitize_label(label: str) -> str:
    """Reduce a label/relationship to a transpiler-safe identifier (``\\w+``)."""
    s = re.sub(r"\W+", "_", str(label or "Node")).strip("_")
    return s or "Node"


class TieredGraphBackend(GraphBackend):
    """Write-through wrapper: L1 working store + L3 durable persistence."""

    def __init__(
        self,
        l1: GraphBackend,
        l3: GraphBackend,
        *,
        write_behind: bool | None = None,
        wb_queue_size: int = 10000,
    ) -> None:
        self.l1 = l1
        self.l3 = l3
        self._l3_failures = 0
        self._l3_writes = 0
        self._l1_reads = 0
        self._l3_reads = 0
        # Write-behind (CONCEPT:KG-2.7, B4): when enabled, node/edge mirrors to the
        # durable L3 are queued and drained on a background thread so L1 (the
        # authoritative working store) acks immediately instead of paying L3 latency
        # inline. This is the DEFAULT, not a knob (CONCEPT:KG-2.7): the durable tier's
        # latency is a property of the system, not a per-deployment setting, and a
        # slow/erroring L3 must never stall the hot write path — so write-behind is a
        # free, always-on capability. It is safe because L1 is authoritative for reads
        # (read-after-write holds) and L1's snapshot+WAL let L3 reconcile from L1 on
        # restart; embeddings still mirror SYNCHRONOUSLY (semantic_search reads L3, so
        # the vector must be there before the next query). The queue is bounded; on
        # saturation a mirror runs inline (backpressure, never dropped). Callers may
        # pass write_behind=False for a strict-synchronous case (e.g. tests).
        if write_behind is None:
            write_behind = True
        self._write_behind = bool(write_behind)
        self._wb_inline = 0
        self._wb_queue: Any = None
        self._wb_thread: Any = None
        if self._write_behind:
            import queue as _queue
            import threading

            self._wb_queue = _queue.Queue(maxsize=max(1, wb_queue_size))
            self._wb_thread = threading.Thread(
                target=self._wb_drain, name="kg-l3-backfeed", daemon=True
            )
            self._wb_thread.start()
        logger.info(
            "TieredGraphBackend initialized (L1=%s, L3=%s, write_behind=%s)",
            type(l1).__name__,
            type(l3).__name__,
            self._write_behind,
        )

    # ------------------------------------------------------------------
    # L3 mirroring helper — never raises
    # ------------------------------------------------------------------
    def _mirror(self, op: str, fn, *, force_sync: bool = False) -> None:
        # In write-behind mode, queue the mirror for the background drainer so L1
        # acks without waiting on L3. `force_sync` (embeddings) bypasses the queue.
        if self._write_behind and not force_sync and self._wb_queue is not None:
            try:
                self._wb_queue.put_nowait((op, fn))
                return
            except Exception:  # noqa: BLE001 — queue.Full → backpressure, run inline
                self._wb_inline += 1
                logger.warning(
                    "TieredGraphBackend: L3 backfeed queue full; mirroring %s inline",
                    op,
                )
        self._mirror_sync(op, fn)

    def _mirror_sync(self, op: str, fn) -> None:
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

    def _wb_drain(self) -> None:
        """Background drainer: applies queued L3 mirrors in order. A ``None`` item
        is the shutdown sentinel."""
        while True:
            item = self._wb_queue.get()
            try:
                if item is None:
                    return
                op, fn = item
                self._mirror_sync(op, fn)
            finally:
                self._wb_queue.task_done()

    def flush_backfeed(self) -> None:
        """Block until all queued L3 mirrors have drained (called at checkpoint, so
        the durable tier is caught up to L1 before the snapshot)."""
        if self._write_behind and self._wb_queue is not None:
            self._wb_queue.join()

    # ------------------------------------------------------------------
    # Core CRUD & Query
    # ------------------------------------------------------------------
    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        is_write: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Reads → L1; writes → L1 (authoritative) then mirrored to L3.

        Traversal reads: an **id-anchored** relationship traversal (single-hop
        ``->``/``<-`` or bounded ``[*lo..hi]``) is now resolved natively on the
        fast L1 engine — its reason for existing. Only traversals L1 can't anchor
        (no ``{id:...}`` entry point) fall through to L3, which transpiles
        ``(a)-[:R]->(b)`` to a ``kg_edges`` JOIN; that path falls back to L1 if L3
        is unavailable. (CONCEPT:KG-2.7 P1 — L1 native traversal.)
        """
        # Prefer the caller's explicit intent over the keyword regex. The regex
        # false-positives on READS that merely mention a mutation keyword (e.g.
        # `WHERE n.x = 'SET'`), needlessly mirroring them to L3; an explicit
        # `is_write` is authoritative. The regex remains only the fallback for
        # genuinely ad-hoc cypher whose intent the caller didn't declare.
        write = _is_write(query) if is_write is None else is_write
        if write:
            result = self.l1.execute(query, params)
            self._mirror("execute", lambda: self.l3.execute(query, params))
            return result
        if _is_traversal(query):
            if _l1_can_traverse(query):
                self._l1_reads += 1
                return self.l1.execute(query, params)
            # A relationship aggregate with a WHERE filter no tier can honor: route
            # to L1 so it FAILS LOUD instead of L3 fabricating an empty/global count
            # (silent-wrong is the cardinal sin). (CONCEPT:KG-2.9h)
            if _is_unhonorable_aggregate(query):
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

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        """Atomic compare-and-set on a node's fields (CONCEPT:KG-2.141).

        Runs on L1 (the authoritative engine) and returns its bool. Only when
        the CAS won (``True``) is the resulting node state mirrored to L3 —
        best-effort, never raising — so a loser leaves L3 untouched. The mirror
        re-applies the won updates via a guarded Cypher ``SET`` keyed on the
        node id so the durable tier converges to L1.
        """
        won = self.l1.compare_and_set_node_fields(node_id, conditions, updates)
        if won:

            def _mirror_cas() -> None:
                set_clause = ", ".join(f"n.{k} = ${k}" for k in updates)
                params: dict[str, Any] = {"_casid": node_id, **updates}
                self.l3.execute(
                    f"MATCH (n {{id: $_casid}}) SET {set_clause}",
                    params,
                )

            self._mirror("compare_and_set_node_fields", _mirror_cas)
        return won

    # ------------------------------------------------------------------
    # Vector / Embedding Support
    # ------------------------------------------------------------------
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        self.l1.add_embedding(node_id, embedding)
        # Synchronous even in write-behind mode: semantic_search reads L3, so the
        # vector must be durable before the next query (read-after-write).
        self._mirror(
            "add_embedding",
            lambda: self.l3.add_embedding(node_id, embedding),
            force_sync=True,
        )

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
        # Drain pending L3 mirrors and stop the drainer BEFORE closing the tiers,
        # so a write-behind backlog isn't lost on shutdown.
        if self._write_behind and self._wb_thread is not None:
            try:
                self._wb_queue.join()
                self._wb_queue.put(None)  # shutdown sentinel
                self._wb_thread.join(timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "TieredGraphBackend: backfeed drain on close failed: %s", exc
                )
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
        """Mirror the L1 graph into the durable L3 and report **exact** drift.

        Delegates to the native cross-backend migration
        (:func:`agent_utilities.knowledge_graph.migration.copy_graph`), which writes
        every L1 node/edge (+ embeddings) through the engine's dialect-aware MERGE
        upserts — so a native-cypher L3 (Neo4j/FalkorDB/AGE) or strict-schema Kuzu
        gets a correct write, not the reconstructed ``CREATE (n:Label {`k`: $k})``
        cypher that double-escaped reserved keys and dropped edges. Idempotent
        (MERGE); ``nodes_missing`` / ``edges_missing`` are the honest post-pass drift.
        """
        from ..migration import copy_graph

        if getattr(self.l1, "graph", None) is None:
            logger.warning(
                "reconcile_to_durable: L1 exposes no compute graph; skipping"
            )
            return {
                "nodes": 0,
                "edges": 0,
                "errors": 0,
                "nodes_missing": 0,
                "edges_missing": 0,
                "prior_l3_failures": self._l3_failures,
            }
        summary = copy_graph(self.l1, self.l3)
        summary["prior_l3_failures"] = self._l3_failures
        return summary

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def durability_stats(self) -> dict[str, int]:
        """Mirror counters for monitoring the durable tier + read routing.

        In write-behind mode a growing ``backfeed_queued`` (or any
        ``backfeed_inline`` from queue saturation) is the LOUD signal that L3 is
        falling behind L1 — alarm on it rather than letting durability drift
        silently."""
        depth = (
            self._wb_queue.qsize()
            if (self._write_behind and self._wb_queue is not None)
            else 0
        )
        return {
            "l3_writes": self._l3_writes,
            "l3_failures": self._l3_failures,
            "l1_reads": self._l1_reads,
            "l3_reads": self._l3_reads,
            "backfeed_queued": depth,
            "backfeed_inline": self._wb_inline,
        }
