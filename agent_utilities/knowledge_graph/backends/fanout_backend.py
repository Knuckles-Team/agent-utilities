#!/usr/bin/python
# CONCEPT:KG-2.74 - Concurrent N-way mirrored graph backend: one configurable authority store serves reads and acks writes synchronously while every mutation fans out, lossless, to any set of durable mirrors (Postgres/AGE, Neo4j, FalkorDB, LadybugDB) via a durable per-mirror outbox with replay-on-reconnect and a reconcile drift-repair backstop.
"""Fan-out (N-way mirror) graph backend.

CONCEPT:KG-2.74 — Concurrent Multi-Store Mirroring.

Generalises the two-tier :class:`TieredGraphBackend` (L1 + one durable L3) into a
**one-authority, N-mirror** store: a single configurable **authority** backend
serves every read and acks every write synchronously, and each write is then
mirrored — losslessly and asynchronously — to any number of durable backends.

The shape, and why each piece is here:

* **One authority, reads + write-ack.** The authority (e.g. the epistemic-graph
  L1 working store, or Postgres) is the source of truth: reads and
  ``semantic_search`` go there, and a write returns as soon as the authority
  commits **and** the mutation is durably enqueued for the mirrors. Reads are
  therefore always consistent; mirrors are eventually-consistent (seconds of lag).
* **Durable per-mirror outbox (no loss).** Each write is appended to
  :class:`GraphOutbox` (sqlite/WAL) once per mirror *before* the write is acked. A
  per-mirror drainer thread applies entries in append order and advances a
  persisted cursor; a mirror that is offline or slow keeps its unapplied tail and
  **replays from its cursor on reconnect / restart** — a transient outage never
  drops a write. (Contrast the tiered write-behind queue, which is in-memory and
  lost on crash.)
* **One drainer per mirror = natural single-writer.** Because each mirror is
  applied by exactly one thread, a single-writer store (LadybugDB / Kuzu, file-
  locked) is serialised for free — no special-casing, it is simply the slowest
  mirror.
* **Reconcile drift-repair backstop.** :meth:`reconcile` does a full authority→
  mirror re-sync and reports exact remaining drift — the backstop for the only
  un-mirrorable window (a crash between the authority commit and the outbox
  append) and for a mirror whose outbox tail was lost. It reuses the proven
  node/edge enumeration + drift counting of :class:`TieredGraphBackend`.

Selected via ``GRAPH_BACKEND=fanout`` with ``graph_authority`` +
``graph_mirror_targets`` naming connections declared in ``kg_connections``
(CONCEPT:KG-2.63). The zero-infra default is unchanged: this backend is only
built when an operator configures a mirror set.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .base import GraphBackend
from .outbox import GraphOutbox, OutboxEntry
from .tiered_backend import _is_write

logger = logging.getLogger(__name__)

# Drainer cadence (config discipline: bounded constants, not per-deploy knobs).
_BATCH = 256  # entries applied per drain pass
_IDLE_POLL_S = 0.25  # how long a drainer waits for new work before re-checking
_BASE_BACKOFF_S = 0.5  # first retry delay after an apply failure
_MAX_BACKOFF_S = 30.0  # cap on exponential backoff for an unreachable mirror
_STALL_THRESHOLD = 5  # consecutive failures before a mirror is flagged "stalled"


@dataclass
class _MirrorState:
    """Live per-mirror counters surfaced by :meth:`durability_stats`."""

    writes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    stalled: bool = False
    wake: threading.Event = field(default_factory=threading.Event)
    thread: Any = None


class FanOutBackend(GraphBackend):
    """One authority backend mirrored, losslessly, to N durable backends."""

    def __init__(
        self,
        authority: GraphBackend,
        mirrors: dict[str, GraphBackend],
        *,
        outbox_path: str,
    ) -> None:
        self._authority = authority
        self._mirrors: dict[str, GraphBackend] = dict(mirrors)
        self._state: dict[str, _MirrorState] = {
            name: _MirrorState() for name in self._mirrors
        }
        self._authority_writes = 0
        self._stop = threading.Event()
        self._outbox: GraphOutbox | None = None
        if self._mirrors:
            self._outbox = GraphOutbox(outbox_path, list(self._mirrors))
            for name in self._mirrors:
                t = threading.Thread(
                    target=self._drain,
                    args=(name,),
                    name=f"kg-mirror-{name}",
                    daemon=True,
                )
                self._state[name].thread = t
                t.start()
        logger.info(
            "FanOutBackend initialized (authority=%s, mirrors=[%s])",
            type(authority).__name__,
            ", ".join(self._mirrors) or "none",
        )

    # ------------------------------------------------------------------
    # Producer side — authority sync + durable enqueue
    # ------------------------------------------------------------------
    def _enqueue(self, op: str, payload: dict[str, Any]) -> None:
        """Durably append a mutation for every mirror and wake the drainers."""
        if not self._mirrors or self._outbox is None:
            return
        self._outbox.append(op, payload)
        for st in self._state.values():
            st.wake.set()

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        is_write: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Reads → authority; writes → authority (sync) then mirrored durably."""
        write = _is_write(query) if is_write is None else is_write
        if not write:
            return self._authority.execute(query, params)
        result = self._authority.execute(query, params)
        self._authority_writes += 1
        self._enqueue("execute", {"query": query, "params": params})
        return result

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """High-throughput ingestion — always a write; mirror durably."""
        result = self._authority.execute_batch(query, batch)
        self._authority_writes += 1
        self._enqueue("execute_batch", {"query": query, "batch": batch})
        return result

    def create_schema(self) -> None:
        self._authority.create_schema()
        self._enqueue("create_schema", {})

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        self._authority.add_embedding(node_id, embedding)
        self._enqueue(
            "add_embedding", {"node_id": node_id, "embedding": list(embedding)}
        )

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Vector search served from the authority (the read source of truth)."""
        return self._authority.semantic_search(query_embedding, n_results)

    def prune(self, criteria: dict[str, Any]) -> None:
        self._authority.prune(criteria)
        self._enqueue("prune", {"criteria": criteria})

    # ------------------------------------------------------------------
    # Consumer side — per-mirror drainer
    # ------------------------------------------------------------------
    def _apply(self, backend: GraphBackend, entry: OutboxEntry) -> None:
        """Apply one outbox entry to a single mirror backend."""
        p = entry.payload
        op = entry.op
        if op == "execute":
            backend.execute(p["query"], p.get("params"))
        elif op == "execute_batch":
            backend.execute_batch(p["query"], p.get("batch") or [])
        elif op == "add_embedding":
            backend.add_embedding(p["node_id"], p["embedding"])
        elif op == "create_schema":
            backend.create_schema()
        elif op == "prune":
            backend.prune(p.get("criteria") or {})
        else:  # pragma: no cover — forward-compat guard
            logger.warning("FanOutBackend: unknown outbox op %r; skipping", op)

    def _drain(self, mirror: str) -> None:
        """Background loop: apply this mirror's outbox tail, in order, with
        replay. Never advances the cursor past an entry that failed, so an
        offline/slow mirror catches up on its own when it returns."""
        assert self._outbox is not None
        st = self._state[mirror]
        backend = self._mirrors[mirror]
        backoff = _BASE_BACKOFF_S
        while not self._stop.is_set():
            st.wake.clear()  # cleared BEFORE draining so a concurrent append re-sets it
            try:
                pending = self._outbox.pending(mirror, limit=_BATCH)
            except Exception as exc:  # noqa: BLE001 — outbox read hiccup
                logger.warning(
                    "FanOutBackend: outbox read for %s failed: %s", mirror, exc
                )
                self._stop.wait(backoff)
                continue
            if not pending:
                st.wake.wait(_IDLE_POLL_S)
                continue
            progressed = False
            for entry in pending:
                if self._stop.is_set():
                    break
                try:
                    self._apply(backend, entry)
                    self._outbox.ack(mirror, entry.seq)
                    st.writes += 1
                    st.consecutive_failures = 0
                    st.last_error = None
                    st.stalled = False
                    progressed = True
                except Exception as exc:  # noqa: BLE001 — transient mirror outage
                    st.failures += 1
                    st.consecutive_failures += 1
                    st.last_error = str(exc)
                    st.stalled = st.consecutive_failures >= _STALL_THRESHOLD
                    # Do NOT advance the cursor: the entry replays after backoff.
                    if st.stalled:
                        logger.warning(
                            "FanOutBackend: mirror %s stalled at seq %d after %d "
                            "consecutive failures: %s",
                            mirror,
                            entry.seq,
                            st.consecutive_failures,
                            exc,
                        )
                    break
            if progressed:
                backoff = _BASE_BACKOFF_S
            else:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)

    def flush_mirrors(self, timeout: float = 30.0) -> bool:
        """Block until every mirror has applied the full outbox (or timeout).

        Returns ``True`` if all mirrors fully caught up. Used at checkpoint and in
        tests to assert convergence.
        """
        if not self._mirrors or self._outbox is None:
            return True
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(self._outbox.lag(m) == 0 for m in self._mirrors):
                return True
            self._stop.wait(0.05)
        return all(self._outbox.lag(m) == 0 for m in self._mirrors)

    # ------------------------------------------------------------------
    # Reconcile — full authority→mirror drift repair (backstop)
    # ------------------------------------------------------------------
    def reconcile(self, mirror: str | None = None) -> dict[str, Any]:
        """Full re-sync of the authority graph into one (or every) mirror.

        Reuses :meth:`TieredGraphBackend.reconcile_to_durable` — the proven
        node/edge enumeration + exact drift counting — by composing the authority
        as L1 and the chosen mirror as L3. The backstop for the crash-gap window
        and for a mirror whose outbox tail was lost.
        """
        from .tiered_backend import TieredGraphBackend

        targets = [mirror] if mirror else list(self._mirrors)
        out: dict[str, Any] = {}
        for name in targets:
            backend = self._mirrors.get(name)
            if backend is None:
                out[name] = {"error": "unknown mirror"}
                continue
            t = TieredGraphBackend(l1=self._authority, l3=backend, write_behind=False)
            try:
                out[name] = t.reconcile_to_durable()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": str(exc)}
        return out

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def durability_stats(self) -> dict[str, Any]:
        """Per-mirror replication health — alarm on ``lag`` / ``stalled``."""
        mirrors: dict[str, Any] = {}
        for name, st in self._state.items():
            lag = self._outbox.lag(name) if self._outbox is not None else 0
            mirrors[name] = {
                "writes": st.writes,
                "failures": st.failures,
                "consecutive_failures": st.consecutive_failures,
                "lag": lag,
                "stalled": st.stalled,
                "last_error": st.last_error,
            }
        return {
            "authority": type(self._authority).__name__,
            "authority_writes": self._authority_writes,
            "outbox_depth": self._outbox.depth() if self._outbox is not None else 0,
            "mirrors": mirrors,
        }

    # ------------------------------------------------------------------
    # SPARQL / Cypher capability — delegate to the authority
    # ------------------------------------------------------------------
    @property
    def cypher_support(self) -> str:
        return getattr(self._authority, "cypher_support", "full")

    @property
    def supports_sparql(self) -> bool:
        return self._authority.supports_sparql

    def execute_sparql(
        self,
        query: str,
        *,
        default_graph: str | None = None,
        timeout_ms: int = 30_000,
    ) -> list[dict[str, Any]]:
        return self._authority.execute_sparql(
            query, default_graph=default_graph, timeout_ms=timeout_ms
        )

    # ------------------------------------------------------------------
    # Facade compatibility — expose the authority compute graph + delegate
    # ------------------------------------------------------------------
    @property
    def graph(self) -> Any:
        return getattr(self._authority, "graph", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate backend-specific helpers to the authority store.

        Mirrors :class:`TieredGraphBackend`: the facade/engine sometimes call
        EpistemicGraph-specific helpers (``save_to_json`` …); anything not defined
        here resolves against the authority. ``__getattr__`` only fires for names
        not found normally, so the explicit methods above take priority.
        """
        authority = self.__dict__.get("_authority")
        if authority is not None and hasattr(authority, name):
            return getattr(authority, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        # Stop drainers first so they don't touch a closing outbox/backend.
        self._stop.set()
        for st in self._state.values():
            st.wake.set()
            if st.thread is not None:
                try:
                    st.thread.join(timeout=10.0)
                except Exception:  # noqa: BLE001
                    logger.debug("FanOutBackend: drainer join failed", exc_info=True)
        if self._outbox is not None:
            self._outbox.close()
        try:
            self._authority.close()
        finally:
            for name, backend in self._mirrors.items():
                try:
                    backend.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "FanOutBackend: mirror %s close failed: %s", name, exc
                    )
