#!/usr/bin/python
# CONCEPT:KG-2.74 - Concurrent N-way mirrored graph backend: one configurable authority store serves reads and acks writes synchronously while every mutation fans out, lossless, to any set of durable mirrors (Postgres/AGE, Neo4j, FalkorDB, LadybugDB) via a durable per-mirror outbox with replay-on-reconnect and a reconcile drift-repair backstop.
"""Fan-out (N-way mirror) graph backend.

CONCEPT:KG-2.74 — Concurrent Multi-Store Mirroring.

The **one-authority, N-mirror** store: a single configurable **authority** backend
serves every read and acks every write synchronously, and each write is then
mirrored — losslessly and asynchronously — to any number of durable backends.

The shape, and why each piece is here:

* **One authority, reads + write-ack.** The authority (normally the epistemic-graph
  engine — the one database) is the source of truth: reads and
  ``semantic_search`` go there, and a write returns as soon as the authority
  commits **and** the mutation is durably enqueued for the mirrors. Reads are
  therefore always consistent; mirrors are eventually-consistent (seconds of lag).
* **Async hand-off — the ack never waits on the mirror enqueue (CONCEPT:KG-2.273).**
  The authority commit is the source of truth; a write returns the instant the
  authority acks. The mirror fan-out is a **non-blocking hand-off**: the write puts
  the mutation onto a bounded in-memory ring and returns — it does **not** wait on
  the sqlite outbox ``append``. A single **persister** thread drains the ring into
  the durable outbox (batched) and wakes the per-mirror drainers. So a slow/locked
  outbox throttles only the persister, never the authority ack (operator's law:
  blocked time is wasted compute; the durable write must not wait on the mirror).
  The ring is bounded + auto-sized; on overflow the producer falls back to a
  **synchronous** durable-outbox append (loud + reconcilable — a mirror write is
  never silently dropped, memory never grows unbounded).
* **Durable per-mirror outbox (no loss).** The persister appends each mutation to
  :class:`GraphOutbox` (sqlite/WAL) once per mirror. A per-mirror drainer thread
  applies entries in append order and advances a persisted cursor; a mirror that is
  offline or slow keeps its unapplied tail and **replays from its cursor on
  reconnect / restart** — once a write is in the outbox a transient outage never
  drops it (vs. an in-memory write-behind queue, which is lost on crash). The only
  un-mirrorable window is the brief gap between the authority ack and the persister
  landing the entry durably (was microseconds, now a ring-drain latency); the
  periodic ``reconcile`` drift-repair pass is the backstop for it, unchanged.
* **One drainer per mirror = natural single-writer.** Because each mirror is
  applied by exactly one thread, a single-writer store (LadybugDB / Kuzu, file-
  locked) is serialised for free — no special-casing, it is simply the slowest
  mirror.
* **Reconcile drift-repair backstop.** :meth:`reconcile` does a full authority→
  mirror re-sync and reports exact remaining drift — the backstop for the only
  un-mirrorable window (a crash between the authority commit and the outbox
  append) and for a mirror whose outbox tail was lost.

Selected via ``GRAPH_BACKEND=fanout`` with ``graph_authority`` +
``graph_mirror_targets`` naming connections declared in ``kg_connections``
(CONCEPT:KG-2.63). The zero-infra default is unchanged: this backend is only
built when an operator configures a mirror set.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from .base import GraphBackend
from .base import is_write as _is_write
from .outbox import GraphOutbox, OutboxEntry

logger = logging.getLogger(__name__)


def _auto_handoff_capacity() -> int:
    """Auto-size the in-memory mirror hand-off ring from the host's core count.

    The ring absorbs a write burst while the single persister thread drains it to
    the durable outbox (CONCEPT:KG-2.273). Bigger boxes get a deeper ring (more
    in-flight writes tolerated before backpressure); a Pi stays capped so memory is
    bounded. No knob — config discipline says auto-size, don't expose a flag.
    """
    cpu = os.cpu_count() or 2
    return max(2048, min(cpu * 1024, 16384))


# The engine's edge-write shape (IntelligenceGraphEngine._upsert_edge):
# ``MATCH (s.. {id: $sid}) MATCH (t.. {id: $tid}) MERGE (s)-[r:REL]->(t) [SET ...]``.
# Matching it lets the fan-out replay edges STRUCTURALLY (src/dst/rel/props) so
# each mirror gets a dialect-correct write — e.g. LadybugDB's strict REL schema
# needs the props folded into its ``properties`` JSON column, which the raw
# per-prop ``SET r.`k` = $k`` cypher (correct for Neo4j/FalkorDB/AGE) cannot do.
_EDGE_MERGE_RE = re.compile(r"MERGE\s*\(s\)-\[r:(\w+)\]->\(t\)")


def _edge_upsert_payload(
    query: str, params: dict[str, Any] | None
) -> dict[str, Any] | None:
    """If ``query`` is the engine's edge MERGE, return a structured upsert payload
    (``source_id`` / ``target_id`` / ``rel_type`` / ``props``); else ``None``."""
    params = params or {}
    if "sid" not in params or "tid" not in params:
        return None
    m = _EDGE_MERGE_RE.search(query or "")
    if not m:
        return None
    props = {k: v for k, v in params.items() if k not in ("sid", "tid")}
    return {
        "source_id": params["sid"],
        "target_id": params["tid"],
        "rel_type": m.group(1),
        "props": props,
    }


# Drainer cadence (config discipline: bounded constants, not per-deploy knobs).
_BATCH = 256  # entries applied per drain pass
_HANDOFF_BATCH = 256  # ring items the persister coalesces into one outbox flush
_IDLE_POLL_S = 0.25  # how long a drainer waits for new work before re-checking
_BASE_BACKOFF_S = 0.5  # first retry delay after an apply failure
_MAX_BACKOFF_S = 30.0  # cap on exponential backoff for an unreachable mirror
_STALL_THRESHOLD = 5  # consecutive failures before a mirror is flagged "stalled"
_POISON_DROP_AFTER = 3  # permanent-error confirmations on ONE entry before skipping it

# Substrings (lowercased) that mark an apply failure as PERMANENT — a malformed or
# dialect-incompatible query that REPLAY can never fix (vs. a transient mirror outage
# such as a refused connection or timeout, which must keep retrying). A permanent
# poison entry is dropped after a few confirmations instead of blocking the mirror and
# spamming the log forever; reconcile() is the backstop for the dropped mutation.
_PERMANENT_APPLY_ERROR_MARKERS = (
    "unknown function",  # neo4j: a singular ``label(n)`` against a full-cypher store
    "missing parameters",  # falkordb: a ``$param`` query reached the mirror unbound
    "syntaxerror",
    "syntax error",
    "invalid input",
    "invalid syntax",
    "type mismatch",
    "neo.clienterror",  # neo4j non-retryable client-side error class
)


def _is_permanent_apply_error(exc: BaseException) -> bool:
    """True if an outbox-entry apply failed for a reason replay can NEVER fix.

    A malformed / dialect-incompatible query (unknown function, missing parameters,
    syntax error) fails deterministically on every replay, so retrying it forever
    only blocks the mirror and floods the log. A transient outage (connection
    refused, timeout, mirror restarting) is NOT permanent and keeps retrying. Used by
    the drainer to drop a poison entry instead of stalling (CONCEPT:KG-2.74)."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _PERMANENT_APPLY_ERROR_MARKERS)


@dataclass
class _MirrorState:
    """Live per-mirror counters surfaced by :meth:`durability_stats`."""

    writes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    stalled: bool = False
    # Poison-entry tracking: the seq currently failing with a PERMANENT error and how
    # many times it has been confirmed, so a malformed entry is skipped (not retried
    # forever). Reset on any successful apply or when the failing seq changes.
    poison_seq: int | None = None
    poison_count: int = 0
    dropped: int = 0  # poison entries skipped (surfaced by durability_stats)
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
        # Cached per-mirror portable writers (lazy) for structural edge replay.
        self._writers: dict[int, Any] = {}
        self._state: dict[str, _MirrorState] = {
            name: _MirrorState() for name in self._mirrors
        }
        self._authority_writes = 0
        self._stop = threading.Event()
        self._outbox: GraphOutbox | None = None
        # Async mirror hand-off (CONCEPT:KG-2.273): the write path PUTs onto this
        # bounded ring and returns; one persister thread drains it into the durable
        # outbox. ``_inflight`` counts ring items not yet durably persisted, so a
        # convergence wait (flush_mirrors) can tell when the durable hand-off caught
        # up. The ring is auto-sized; overflow falls back to a synchronous append.
        self._handoff: queue.Queue[tuple[str, dict[str, Any]]] | None = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._persister: Any = None
        if self._mirrors:
            self._outbox = GraphOutbox(outbox_path, list(self._mirrors))
            self._handoff = queue.Queue(maxsize=_auto_handoff_capacity())
            self._persister = threading.Thread(
                target=self._persist_handoff,
                name="kg-mirror-persist",
                daemon=True,
            )
            self._persister.start()
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
            "FanOutBackend initialized (authority=%s, mirrors=[%s], handoff_ring=%d)",
            type(authority).__name__,
            ", ".join(self._mirrors) or "none",
            self._handoff.maxsize if self._handoff is not None else 0,
        )

    # ------------------------------------------------------------------
    # Producer side — authority sync + durable enqueue
    # ------------------------------------------------------------------
    def _enqueue(self, op: str, payload: dict[str, Any]) -> None:
        """Hand a mutation off to the mirror fan-out **without blocking the ack**.

        CONCEPT:KG-2.273. The authority has already acked; this must not wait on the
        durable outbox ``append``. So it PUTs the mutation onto the bounded in-memory
        ring (O(1), never the sqlite write lock) and returns — the persister thread
        lands it durably. The only time this touches sqlite synchronously is the
        **overflow** backstop: if the ring is full (the persister can't keep up), the
        producer appends straight to the durable outbox rather than block the ack or
        drop the write — loud + reconcilable, and it bounds memory.
        """
        if not self._mirrors or self._outbox is None or self._handoff is None:
            return
        try:
            self._handoff.put_nowait((op, payload))
        except queue.Full:
            logger.warning(
                "FanOutBackend: mirror hand-off ring full (cap=%d) — persister is "
                "behind; appending this mirror write to the durable outbox "
                "synchronously (backpressure, not loss). seq op=%s",
                self._handoff.maxsize,
                op,
            )
            self._outbox.append(op, payload)
            for st in self._state.values():
                st.wake.set()
            return
        with self._inflight_lock:
            self._inflight += 1

    def _persist_handoff(self) -> None:
        """Drain the in-memory ring into the durable outbox (CONCEPT:KG-2.273).

        One thread, so the global outbox ``seq`` order matches the hand-off order. It
        coalesces a burst into one batch (amortizing the sqlite commit), appends each
        mutation durably, then wakes the per-mirror drainers. If the outbox append
        fails (a transient sqlite hiccup), it KEEPS the un-persisted batch and retries
        after a backoff — a handed-off write is never dropped before it is durable.
        """
        assert self._outbox is not None and self._handoff is not None
        pending: list[tuple[str, dict[str, Any]]] = []
        while not self._stop.is_set():
            if not pending:
                try:
                    pending.append(self._handoff.get(timeout=_IDLE_POLL_S))
                except queue.Empty:
                    continue
                while len(pending) < _HANDOFF_BATCH:
                    try:
                        pending.append(self._handoff.get_nowait())
                    except queue.Empty:
                        break
            try:
                for op, payload in pending:
                    self._outbox.append(op, payload)
            except Exception as exc:  # noqa: BLE001 — transient outbox write hiccup
                logger.warning(
                    "FanOutBackend: durable outbox append failed (%d pending), "
                    "retrying: %s",
                    len(pending),
                    exc,
                )
                self._stop.wait(_BASE_BACKOFF_S)
                continue  # keep `pending` — retry, never drop
            with self._inflight_lock:
                self._inflight -= len(pending)
            pending = []
            for st in self._state.values():
                st.wake.set()

    def _drain_handoff_remaining(self) -> None:
        """Flush any ring items still in memory straight to the durable outbox.

        Called on ``close`` after the persister has stopped, so a graceful shutdown
        loses nothing: every handed-off-but-not-yet-persisted mutation lands durably
        before the outbox is closed (CONCEPT:KG-2.273)."""
        if self._handoff is None or self._outbox is None:
            return
        drained = 0
        while True:
            try:
                op, payload = self._handoff.get_nowait()
            except queue.Empty:
                break
            try:
                self._outbox.append(op, payload)
                drained += 1
            except Exception as exc:  # noqa: BLE001 — best-effort shutdown flush
                logger.warning(
                    "FanOutBackend: shutdown outbox flush failed for op=%s: %s",
                    op,
                    exc,
                )
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - drained)
        if drained:
            logger.info("FanOutBackend: flushed %d ring items at close", drained)

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
        # Edge writes mirror STRUCTURALLY so each backend gets a dialect-correct
        # write (Ladybug folds props into its `properties` JSON column); everything
        # else forwards the raw cypher (portable for node MERGE + ad-hoc writes).
        edge = _edge_upsert_payload(query, params)
        if edge is not None:
            self._enqueue("upsert_edge", edge)
        else:
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
        elif op == "upsert_edge":
            # Dialect-correct edge write per backend (reuses the engine's
            # backend-aware _upsert_edge: native props for Neo4j/FalkorDB/AGE,
            # `properties` JSON column for strict-schema LadybugDB).
            self._edge_writer(backend)._upsert_edge(
                p["source_id"], p["target_id"], p["rel_type"], p.get("props") or {}
            )
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

    def _edge_writer(self, backend: GraphBackend) -> Any:
        """A cached portable writer for ``backend`` (reuses the engine's
        dialect-aware ``_upsert_edge`` without the engine's heavy machinery)."""
        key = id(backend)
        w = self._writers.get(key)
        if w is None:
            from ..migration import _portable_writer

            w = _portable_writer(backend)
            self._writers[key] = w
        return w

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
                    st.poison_seq = None
                    st.poison_count = 0
                    progressed = True
                except Exception as exc:  # noqa: BLE001 — transient mirror outage
                    st.failures += 1
                    st.consecutive_failures += 1
                    st.last_error = str(exc)
                    st.stalled = st.consecutive_failures >= _STALL_THRESHOLD
                    # PERMANENT errors (malformed/dialect-incompatible cypher) never
                    # apply on replay — after a few confirmations on the SAME entry,
                    # SKIP it (advance the cursor) so the mirror keeps draining instead
                    # of blocking + spamming the log forever. Transient outages fall
                    # through and replay after backoff as before. reconcile() repairs
                    # any dropped mutation (CONCEPT:KG-2.74).
                    if _is_permanent_apply_error(exc):
                        if st.poison_seq == entry.seq:
                            st.poison_count += 1
                        else:
                            st.poison_seq = entry.seq
                            st.poison_count = 1
                        if st.poison_count >= _POISON_DROP_AFTER:
                            logger.error(
                                "FanOutBackend: mirror %s DROPPING poison entry "
                                "seq=%d op=%s after %d permanent failures "
                                "(reconcile is the backstop): %s",
                                mirror,
                                entry.seq,
                                entry.op,
                                st.poison_count,
                                exc,
                            )
                            self._outbox.ack(mirror, entry.seq)
                            st.dropped += 1
                            st.poison_seq = None
                            st.poison_count = 0
                            st.consecutive_failures = 0
                            st.stalled = False
                            progressed = True
                            continue  # move on to the next entry this pass
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
        # Convergence has TWO stages now (CONCEPT:KG-2.273): the in-memory ring must
        # be drained into the durable outbox (``_inflight == 0``), then every mirror
        # cursor must catch up to the outbox tail (``lag == 0``). The first stage
        # matters because the async hand-off means a just-acked write may not be in
        # the outbox yet — ``lag`` alone would read 0 prematurely.
        while time.monotonic() < deadline:
            if self._durable_caught_up() and all(
                self._outbox.lag(m) == 0 for m in self._mirrors
            ):
                return True
            self._stop.wait(0.05)
        return self._durable_caught_up() and all(
            self._outbox.lag(m) == 0 for m in self._mirrors
        )

    def _durable_caught_up(self) -> bool:
        """True when the persister has landed every handed-off write in the outbox."""
        with self._inflight_lock:
            return self._inflight == 0

    # ------------------------------------------------------------------
    # Reconcile — full authority→mirror drift repair (backstop)
    # ------------------------------------------------------------------
    def reconcile(self, mirror: str | None = None) -> dict[str, Any]:
        """Full re-sync of the authority graph into one (or every) mirror.

        Uses the native cross-backend migration (:func:`copy_graph`), which writes
        through the engine's dialect-aware MERGE upserts — so each mirror gets a
        correct native write (not one backend's raw cypher forwarded to all). The
        backstop for the crash-gap window and for a mirror whose outbox tail was
        lost; also the path that backfills a freshly-added mirror.
        """
        from ..migration import copy_graph

        targets = [mirror] if mirror else list(self._mirrors)
        out: dict[str, Any] = {}
        for name in targets:
            backend = self._mirrors.get(name)
            if backend is None:
                out[name] = {"error": "unknown mirror"}
                continue
            try:
                out[name] = copy_graph(self._authority, backend)
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
                "dropped": st.dropped,
                "last_error": st.last_error,
            }
        with self._inflight_lock:
            inflight = self._inflight
        return {
            "authority": type(self._authority).__name__,
            "authority_writes": self._authority_writes,
            "outbox_depth": self._outbox.depth() if self._outbox is not None else 0,
            # Async hand-off backlog (CONCEPT:KG-2.273): ring items the write path
            # handed off but the persister has not yet landed in the durable outbox.
            "handoff_inflight": inflight,
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
        # Stop the persister + drainers first so they don't touch a closing
        # outbox/backend. Then flush any ring items still in memory to the durable
        # outbox so a graceful shutdown loses nothing (CONCEPT:KG-2.273).
        self._stop.set()
        if self._persister is not None:
            try:
                self._persister.join(timeout=10.0)
            except Exception:  # noqa: BLE001
                logger.debug("FanOutBackend: persister join failed", exc_info=True)
        self._drain_handoff_remaining()
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
