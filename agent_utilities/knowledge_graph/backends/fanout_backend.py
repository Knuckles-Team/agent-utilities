#!/usr/bin/python
# CONCEPT:AU-KG.backend.mirror-health-repair - Concurrent N-way mirrored graph backend: the fixed epistemic-graph authority serves reads and acks writes synchronously while every mutation fans out, lossless, to durable external projections via an outbox with replay and reconciliation.
"""Fan-out (N-way mirror) graph backend.

CONCEPT:AU-KG.backend.mirror-health-repair — Concurrent Multi-Store Mirroring.

The **one-authority, N-mirror** store: the fixed ``EpistemicGraphBackend`` serves
every read and acks every write synchronously, and each write is then mirrored —
losslessly and asynchronously — to any number of durable external projections.

The shape, and why each piece is here:

* **One authority, reads + write-ack.** The epistemic-graph engine is the source
  of truth: reads and
  ``semantic_search`` go there, and a write returns as soon as the authority
  commits **and** the mutation is durably enqueued for the mirrors. Reads are
  therefore always consistent; mirrors are eventually-consistent (seconds of lag).
* **Async hand-off — the ack never waits on the mirror enqueue (CONCEPT:AU-KG.backend.authority-has-already-acked).**
  The authority commit is the source of truth; a write returns the instant the
  authority acks. The mirror fan-out is a **non-blocking hand-off**: the write puts
  the mutation onto a bounded in-memory ring and returns — it does **not** wait on
  the sqlite outbox ``append``. A single **persister** thread drains the ring into
  the durable outbox (batched) and wakes the per-mirror drainers. So a slow/locked
  outbox throttles only the persister, never the authority ack (operator's law:
  blocked time is wasted compute; the durable write must not wait on the mirror).
  The ring is bounded + auto-sized; on overflow the producer synchronously helps
  persist the **oldest** queued mutation before admitting its own (loud +
  reconcilable — writes never overtake, silently drop, or grow memory unbounded).
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

Built automatically when ``graph_mirror_targets`` names connections declared in
``kg_connections`` (CONCEPT:AU-KG.backend.multi-connection-registry), or when a
connection declares ``role=mirror``. The zero-infra default remains the bare
authority when no projection is configured.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any

from .base import GraphBackend
from .base import is_write as _is_write
from .outbox import GraphOutbox, OutboxEntry

logger = logging.getLogger(__name__)


class AuthorityCommittedMirrorHandoffError(RuntimeError):
    """A native authority batch committed before its mirror handoff failed.

    The authoritative graph remains the durable acknowledgement source.  Callers
    must not retry the same batch as though it were atomic rejection, but the
    handoff failure stays explicit so the mirror reconciler/operator can repair
    the eventual projection.
    """

    authority_committed = True


def _auto_handoff_capacity() -> int:
    """Auto-size the in-memory mirror hand-off ring from the host's core count.

    The ring absorbs a write burst while the single persister thread drains it to
    the durable outbox (CONCEPT:AU-KG.backend.authority-has-already-acked). Bigger boxes get a deeper ring (more
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
_MUTATION_LOCK_STRIPES = 256  # same-entity ordering without global write serialization
_NON_DROPPABLE_REPLAY_OPS = frozenset(
    {"add_embedding", "compare_and_set_node_embedding"}
)

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
    the drainer to drop a poison entry instead of stalling (CONCEPT:AU-KG.backend.mirror-health-repair)."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _PERMANENT_APPLY_ERROR_MARKERS)


def _overrides_backend_method(backend: GraphBackend, method_name: str) -> bool:
    """Return whether ``backend`` implements an optional base capability.

    Optional GraphBackend methods deliberately raise ``NotImplementedError``.
    A callable check therefore reports false capabilities for ordinary mirrors
    such as PostgreSQL and Neo4j that simply inherit the default method.
    """
    return getattr(type(backend), method_name, None) is not getattr(
        GraphBackend, method_name, None
    )


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


def _new_epistemic_authority() -> GraphBackend:
    """Construct the sole allowed fan-out authority.

    Kept as a small private seam so unit tests can substitute an inert recording
    implementation without exposing authority selection in the production API.
    """
    from .epistemic_graph_backend import EpistemicGraphBackend

    return EpistemicGraphBackend()


class FanOutBackend(GraphBackend):
    """The epistemic-graph authority mirrored to N external projections."""

    def __init__(
        self,
        mirrors: dict[str, GraphBackend],
        *,
        outbox_path: str,
    ) -> None:
        self._authority = _new_epistemic_authority()
        self._mirrors: dict[str, GraphBackend] = dict(mirrors)
        # Cached per-mirror portable writers (lazy) for structural edge replay.
        self._writers: dict[int, Any] = {}
        self._state: dict[str, _MirrorState] = {
            name: _MirrorState() for name in self._mirrors
        }
        self._authority_writes = 0
        # Lifecycle fencing (D-LRR-1). Every producer registers BEFORE its
        # authority mutation and remains active through mirror admission. close()
        # flips open -> closing under the same condition, rejects later writers,
        # and waits the registered set to reach zero before stopping persistence.
        self._lifecycle_condition = threading.Condition()
        self._lifecycle_state = "open"
        self._active_producers = 0
        self._close_lock = threading.Lock()
        # Same-entity writes must keep authority-commit -> mirror-enqueue order,
        # especially CAS followed by an ANN update. A fixed striped lock table
        # provides that ordering without holding one global lock across every
        # authority RPC: independent entities still write concurrently and the
        # memory footprint cannot grow with node cardinality.
        self._mutation_locks = tuple(
            threading.RLock() for _ in range(_MUTATION_LOCK_STRIPES)
        )
        self._stop = threading.Event()
        self._outbox: GraphOutbox | None = None
        # Async mirror hand-off (CONCEPT:AU-KG.backend.authority-has-already-acked): the write path PUTs onto this
        # bounded ring and returns; one persister thread drains it into the durable
        # outbox. ``_inflight`` counts ring items not yet durably persisted, so a
        # convergence wait (flush_mirrors) can tell when the durable hand-off caught
        # up. The ring is auto-sized; overflow helps persist its oldest item.
        self._handoff: queue.Queue[tuple[str, dict[str, Any]]] | None = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        # Exactly one thread appends the current hand-off head. Producers may
        # still enqueue behind that head while sqlite is slow. On overflow a
        # producer helps persist OLDER entries first, so the overflow mutation
        # can never overtake items that were already in the ring.
        self._handoff_condition = threading.Condition()
        self._handoff_active: tuple[str, dict[str, Any]] | None = None
        self._handoff_appending = False
        # FIFO admission tickets cover the overflow slow path. Without them, a
        # helper pops the oldest ring item (freeing one slot), then a later
        # producer can steal that slot before the helper admits its own mutation.
        self._admission_next_ticket = 0
        self._admission_serving_ticket = 0
        self._admission_waiters = 0
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
            type(self._authority).__name__,
            ", ".join(self._mirrors) or "none",
            self._handoff.maxsize if self._handoff is not None else 0,
        )

    # ------------------------------------------------------------------
    # Producer side — authority sync + durable enqueue
    # ------------------------------------------------------------------
    @contextmanager
    def _producer(self) -> Iterator[None]:
        """Fence one authority mutation plus its mirror hand-off from close."""
        with self._lifecycle_condition:
            if self._lifecycle_state != "open":
                raise RuntimeError(
                    f"FanOutBackend is {self._lifecycle_state}; writes are fenced"
                )
            self._active_producers += 1
        try:
            yield
        finally:
            with self._lifecycle_condition:
                self._active_producers -= 1
                self._lifecycle_condition.notify_all()

    def _mutation_lock(self, node_id: str) -> threading.RLock:
        """Return the fixed ordering stripe for one entity identity."""
        return self._mutation_locks[hash(str(node_id)) % len(self._mutation_locks)]

    def _enqueue(self, op: str, payload: dict[str, Any]) -> None:
        """Hand a mutation off to the mirror fan-out **without blocking the ack**.

        CONCEPT:AU-KG.backend.authority-has-already-acked. The authority has already acked; this must not wait on the
        durable outbox ``append``. So it PUTs the mutation onto the bounded in-memory
        ring (O(1), never the sqlite write lock) and returns — the persister thread
        lands it durably. The only time this touches sqlite synchronously is the
        **overflow** backstop: if the ring is full (the persister can't keep up), the
        producer helps persist older queued items before admitting its own — loud +
        reconcilable, ordered, and memory-bounded.
        """
        if not self._mirrors or self._outbox is None or self._handoff is None:
            return
        warned = False
        ticket: int | None = None
        while True:
            should_help = False
            with self._handoff_condition:
                if ticket is None:
                    if self._admission_waiters == 0:
                        try:
                            self._handoff.put_nowait((op, payload))
                        except queue.Full:  # noqa: BLE001 — expected saturation; fall through to ordered overflow admission
                            pass
                        else:
                            # Increment while the hand-off condition excludes the
                            # persister from claiming this item. That closes the old
                            # put-then-increment race that could make inflight negative.
                            with self._inflight_lock:
                                self._inflight += 1
                            self._handoff_condition.notify_all()
                            return
                    ticket = self._admission_next_ticket
                    self._admission_next_ticket += 1
                    self._admission_waiters += 1

                if ticket == self._admission_serving_ticket:
                    try:
                        self._handoff.put_nowait((op, payload))
                    except queue.Full:
                        should_help = not self._handoff_appending
                    else:
                        with self._inflight_lock:
                            self._inflight += 1
                        self._admission_waiters -= 1
                        self._admission_serving_ticket += 1
                        self._handoff_condition.notify_all()
                        return

                if not should_help:
                    self._handoff_condition.wait(timeout=_IDLE_POLL_S)
                    continue

            if not warned:
                logger.warning(
                    "FanOutBackend: mirror hand-off ring full (cap=%d) — persister "
                    "is behind; synchronously helping persist older writes before "
                    "admitting op=%s (ordered backpressure, not loss)",
                    self._handoff.maxsize,
                    op,
                )
                warned = True
            try:
                progressed = self._persist_one_handoff()
            except Exception as exc:  # noqa: BLE001 — overflow backpressure retries; authority already committed
                logger.warning(
                    "FanOutBackend: ordered overflow persistence failed; retrying: %s",
                    exc,
                )
                progressed = False
            if not progressed:
                # Another thread owns the older head (or the outbox is
                # recovering). Wait briefly for space; never append this newer
                # mutation ahead of it and never grow the bounded ring.
                with self._handoff_condition:
                    self._handoff_condition.wait(timeout=_IDLE_POLL_S)

    def _claim_handoff(self) -> tuple[str, dict[str, Any]] | None:
        """Claim the oldest ring item without blocking concurrent producers."""
        assert self._handoff is not None
        with self._handoff_condition:
            if self._handoff_appending:
                return None
            if self._handoff_active is None:
                try:
                    self._handoff_active = self._handoff.get_nowait()
                except queue.Empty:
                    return None
            self._handoff_appending = True
            self._handoff_condition.notify_all()
            return self._handoff_active

    def _finish_handoff(
        self,
        item: tuple[str, dict[str, Any]],
        *,
        committed: bool,
    ) -> None:
        """Release an append claim, retaining a failed head for ordered retry."""
        with self._handoff_condition:
            if self._handoff_active != item:  # pragma: no cover - invariant guard
                raise RuntimeError("fan-out hand-off completion lost its active item")
            self._handoff_appending = False
            if committed:
                self._handoff_active = None
                with self._inflight_lock:
                    self._inflight -= 1
            self._handoff_condition.notify_all()

    def _persist_one_handoff(self) -> bool:
        """Persist exactly one ordered hand-off, returning whether one existed."""
        assert self._outbox is not None
        item = self._claim_handoff()
        if item is None:
            return False
        try:
            self._outbox.append(*item)
        except Exception:
            self._finish_handoff(item, committed=False)
            raise
        self._finish_handoff(item, committed=True)
        return True

    def _persist_handoff(self) -> None:
        """Drain the in-memory ring into the durable outbox (CONCEPT:AU-KG.backend.authority-has-already-acked).

        One thread, so the global outbox ``seq`` order matches the hand-off order. It
        coalesces a burst into one batch (amortizing the sqlite commit), appends each
        mutation durably, then wakes the per-mirror drainers. If the outbox append
        fails (a transient sqlite hiccup), it KEEPS the un-persisted batch and retries
        after a backoff — a handed-off write is never dropped before it is durable.
        """
        assert self._outbox is not None and self._handoff is not None
        while not self._stop.is_set():
            persisted = 0
            try:
                while persisted < _HANDOFF_BATCH and self._persist_one_handoff():
                    persisted += 1
            except Exception as exc:  # noqa: BLE001 — transient outbox write hiccup
                logger.warning(
                    "FanOutBackend: durable outbox append failed; retrying ordered "
                    "head: %s",
                    exc,
                )
                self._stop.wait(_BASE_BACKOFF_S)
                continue
            if persisted:
                for st in self._state.values():
                    st.wake.set()
                continue
            with self._handoff_condition:
                if self._stop.is_set():
                    break
                self._handoff_condition.wait(timeout=_IDLE_POLL_S)

    def _drain_handoff_remaining(self) -> None:
        """Flush any ring items still in memory straight to the durable outbox.

        Called on ``close`` after the persister has stopped, so a graceful shutdown
        loses nothing: every handed-off-but-not-yet-persisted mutation lands durably
        before the outbox is closed (CONCEPT:AU-KG.backend.authority-has-already-acked)."""
        if self._handoff is None or self._outbox is None:
            return
        drained = 0
        while True:
            try:
                if not self._persist_one_handoff():
                    break
                drained += 1
            except Exception as exc:  # noqa: BLE001 — best-effort shutdown flush
                logger.warning(
                    "FanOutBackend: shutdown outbox flush failed: %s",
                    exc,
                )
                raise
        if drained:
            logger.info("FanOutBackend: flushed %d ring items at close", drained)

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        is_write: bool | None = None,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Reads → authority; writes → authority (sync) then mirrored durably.

        ``include_epistemic`` (CONCEPT:AU-KB-CURRENCY, Seam 1) forwards straight to
        the authority backend's own ``execute`` on the read path — it is only
        honored when the authority is an ``EpistemicGraphBackend`` (or another
        backend that itself supports it); otherwise it degrades to ``[]`` exactly
        as the authority's own ``execute`` documents. Ignored on the write path
        (a write's return value is the mutation result, not a query result to
        currency-upgrade).
        """
        write = _is_write(query) if is_write is None else is_write
        if not write:
            return self._authority.execute(
                query, params, include_epistemic=include_epistemic
            )
        with self._producer():
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

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Route public reads only through the authority's read-only contract."""
        return self._authority.execute_read(
            query,
            params,
            include_epistemic=include_epistemic,
        )

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """High-throughput ingestion — always a write; mirror durably."""
        with self._producer():
            result = self._authority.execute_batch(query, batch)
            self._authority_writes += 1
            self._enqueue("execute_batch", {"query": query, "batch": batch})
            return result

    @staticmethod
    def _prepare_typed_mutations(
        operations: list[dict[str, Any]],
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]], list[str]]:
        """Validate + render typed upsert operations into mirror-outbox payloads.

        Shared by :meth:`apply_typed_batch` (which also commits the authority)
        and :meth:`replay_committed_change` (whose authority commit already
        happened elsewhere — e.g. a native ``ApplyChangeEnvelope``) so the two
        callers cannot drift into two different mutation dialects.
        """
        prepared: list[tuple[str, dict[str, Any]]] = []
        copied_operations: list[dict[str, Any]] = []
        mutation_keys: list[str] = []
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("typed batch operations must be mappings")
            copied = dict(operation)
            props = copied.get("properties")
            if not isinstance(props, dict):
                raise ValueError("typed batch operation properties must be a mapping")
            copied["properties"] = dict(props)
            op = str(copied.get("op") or "")
            if op == "upsert_node":
                node_id = str(copied.get("id") or "").strip()
                node_type = str(copied["properties"].get("node_type") or "").strip()
                if not node_id or not node_type:
                    raise ValueError("typed node batch requires id and node_type")
                payload = {"id": node_id, **copied["properties"]}
                prepared.append(
                    (
                        "upsert_node",
                        {
                            "node_id": node_id,
                            "label": node_type,
                            "properties": payload,
                        },
                    )
                )
                mutation_keys.append(node_id)
            elif op == "upsert_edge":
                source_id = str(copied.get("source") or "").strip()
                target_id = str(copied.get("target") or "").strip()
                relationship = str(
                    copied["properties"].get("relationship") or ""
                ).strip()
                if not source_id or not target_id or not relationship:
                    raise ValueError(
                        "typed edge batch requires source, target, and relationship"
                    )
                prepared.append(
                    (
                        "upsert_edge",
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "rel_type": relationship,
                            "props": copied["properties"],
                        },
                    )
                )
                mutation_keys.append(
                    f"edge\x00{source_id}\x00{target_id}\x00{relationship}"
                )
            else:
                raise ValueError(f"unsupported typed batch operation: {op!r}")
            copied_operations.append(copied)
        return prepared, copied_operations, mutation_keys

    def apply_typed_batch(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Commit native typed upserts once, then enqueue ordered mirror replay.

        The authority's ``BatchUpdate`` validates the complete operation list and
        commits it atomically.  Prepare every mirror payload *before* that commit
        so an unsupported operation cannot leave an authority-only mutation.
        Mirrors keep their established per-mutation outbox ordering; their durable
        replay remains eventual, while the authority remains the read/write ack
        source of truth.
        """
        prepared, copied_operations, mutation_keys = self._prepare_typed_mutations(
            operations
        )

        apply = getattr(self._authority, "apply_typed_batch", None)
        if not callable(apply):
            raise RuntimeError(
                "fan-out authority does not support typed batch mutations"
            )
        # Match the lifecycle and same-mutation ordering contract of the scalar
        # typed methods.  Acquire every touched stripe in index order so
        # overlapping batches cannot deadlock, and retain the producer fence
        # through the last mirror handoff so close() cannot seal the outbox in
        # the authority-committed window.
        stripe_indexes = sorted(
            {hash(key) % len(self._mutation_locks) for key in mutation_keys}
        )
        with ExitStack() as stack:
            stack.enter_context(self._producer())
            for stripe_index in stripe_indexes:
                stack.enter_context(self._mutation_locks[stripe_index])
            result = apply(copied_operations)
            self._authority_writes += 1
            try:
                for op, payload in prepared:
                    self._enqueue(op, payload)
            except Exception as exc:  # noqa: BLE001 — authority has already committed; reconciliation is now required for mirrors
                logger.critical(
                    "FanOutBackend: authority committed typed batch but mirror handoff "
                    "failed; reconciliation required: %s",
                    exc,
                )
                raise AuthorityCommittedMirrorHandoffError(
                    "authority committed typed batch but mirror handoff failed"
                ) from exc
            return result

    def replay_committed_change(self, operations: list[dict[str, Any]]) -> None:
        """Durably enqueue mirror replay for a mutation ALREADY committed at authority.

        A native ``ApplyChangeEnvelope``/``ApplyChangeEnvelopes`` commit bypasses
        this backend's own ``execute``/``apply_typed_batch`` entirely — the caller
        holds a direct client onto the unwrapped native authority (see
        ``envelope_ingest._resolve_native_authority``) so the graph slice never
        compiles a second Python mutation authority. This method is the seam that
        lets that already-committed node/edge delta still reach the SAME durable
        mirror outbox ordinary typed writes use (D-BFR-12): it validates and
        renders the operations exactly like :meth:`apply_typed_batch`, but never
        calls the authority — only enqueues. Idempotent replay of an already-
        applied envelope must not call this a second time (the caller skips it
        when the native commit reports ``replayed``); a mirror always converges
        on the SAME final node/edge state either way, but re-enqueueing a no-op
        delta would still cost an unnecessary outbox write per replay.
        """
        if not self._mirrors or not operations:
            return
        prepared, _copied_operations, mutation_keys = self._prepare_typed_mutations(
            operations
        )
        stripe_indexes = sorted(
            {hash(key) % len(self._mutation_locks) for key in mutation_keys}
        )
        with ExitStack() as stack:
            stack.enter_context(self._producer())
            for stripe_index in stripe_indexes:
                stack.enter_context(self._mutation_locks[stripe_index])
            try:
                for op, payload in prepared:
                    self._enqueue(op, payload)
            except Exception as exc:  # noqa: BLE001 — authority already committed via native ChangeEnvelope; reconciliation is now required for mirrors
                logger.critical(
                    "FanOutBackend: native ChangeEnvelope committed but mirror "
                    "handoff failed; reconciliation required: %s",
                    exc,
                )
                raise AuthorityCommittedMirrorHandoffError(
                    "native ChangeEnvelope committed but mirror handoff failed"
                ) from exc

    def add_node(self, node_id: str, label: str = "", **properties: Any) -> None:
        """Commit one typed node mutation and mirror its structured payload.

        The native authority accepts the full MessagePack property domain.  Keeping
        this operation structured avoids lossy Cypher literal rendering while the
        mirror drainer still uses its backend-aware portable writer.
        """
        node_type = str(properties.get("node_type") or label).strip()
        if not node_type:
            raise ValueError("node_type is required")
        payload = {
            **properties,
            "id": node_id,
            "node_type": node_type,
        }
        typed_add = getattr(self._authority, "add_node", None)
        if not callable(typed_add):
            raise RuntimeError(
                "fan-out authority does not support typed node mutations"
            )
        with self._producer(), self._mutation_lock(node_id):
            typed_add(node_id, **payload)
            self._authority_writes += 1
            self._enqueue(
                "upsert_node",
                {
                    "node_id": node_id,
                    "label": node_type,
                    "properties": payload,
                },
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "",
        /,
        **properties: Any,
    ) -> None:
        """Commit one typed edge mutation and mirror its structured payload."""
        relationship = str(properties.get("relationship") or rel_type).strip()
        if not relationship:
            raise ValueError("relationship is required")
        payload = {**properties, "relationship": relationship}
        typed_add = getattr(self._authority, "add_edge", None)
        if not callable(typed_add):
            raise RuntimeError(
                "fan-out authority does not support typed edge mutations"
            )
        edge_key = f"edge\x00{source_id}\x00{target_id}\x00{relationship}"
        with self._producer(), self._mutation_lock(edge_key):
            typed_add(source_id, target_id, **payload)
            self._authority_writes += 1
            self._enqueue(
                "upsert_edge",
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": relationship,
                    "props": payload,
                },
            )

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        """Apply authority CAS and replay its winning full node to mirrors.

        The base backend's optional default raises ``NotImplementedError`` and
        therefore cannot fall through ``__getattr__`` to the authority. This
        explicit implementation preserves the native winner/loser result. A
        loser emits no mirror mutation. A winner snapshots the complete updated
        authoritative node and reuses the existing structured-node outbox path,
        preserving ACL/classification and converging mirrors without a second
        backend-specific partial-update dialect.
        """
        compare_and_set = getattr(self._authority, "compare_and_set_node_fields", None)
        get_properties = getattr(self._authority, "get_node_properties", None)
        if not callable(compare_and_set) or not callable(get_properties):
            raise NotImplementedError(  # ABSTRACT-OK — optional CAS capability, mirrors base.py's own feature-detection guard
                "fan-out authority does not support node compare-and-set snapshots"
            )

        with self._producer(), self._mutation_lock(node_id):
            before = get_properties(node_id)
            if before is None:
                return False
            label = str(before.get("node_type") or before.get("label") or "").strip()
            if self._mirrors and not label:
                raise RuntimeError(
                    "fan-out compare-and-set requires a typed authority node"
                )
            if not bool(compare_and_set(node_id, conditions, updates)):
                return False
            self._authority_writes += 1
            if self._mirrors:
                after = get_properties(node_id)
                if after is None:
                    raise RuntimeError(
                        "fan-out authority node disappeared after compare-and-set"
                    )
                self._enqueue(
                    "upsert_node",
                    {
                        "node_id": node_id,
                        "label": label,
                        "properties": dict(after),
                    },
                )
            return True

    def compare_and_set_node_embedding(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
        embedding: list[float],
    ) -> bool:
        """Atomically update authority fields + ANN, then replay one mirror op."""
        return self._compare_and_set_node_embedding_with_authority(
            self._authority,
            node_id,
            conditions,
            updates,
            embedding,
        )

    def compare_and_set_node_embedding_for_graph(
        self,
        graph_name: str,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
        embedding: list[float],
    ) -> bool:
        """Publish a graph-scoped native embedding update through fan-out."""
        scoped = getattr(self._authority, "for_graph", None)
        if not callable(scoped):
            raise RuntimeError("fan-out authority does not expose graph-scoped views")
        return self._compare_and_set_node_embedding_with_authority(
            scoped(graph_name),
            node_id,
            conditions,
            updates,
            embedding,
        )

    def _compare_and_set_node_embedding_with_authority(
        self,
        authority: GraphBackend,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
        embedding: list[float],
    ) -> bool:
        """Apply one scoped authority update and enqueue its mirror snapshot."""
        atomic_update = getattr(authority, "compare_and_set_node_embedding", None)
        if not callable(atomic_update):
            raise RuntimeError(
                "fan-out authority does not support atomic embedding updates"
            )
        get_properties = getattr(authority, "get_node_properties", None)
        snapshotter: Callable[[str], Any] | None = (
            get_properties if callable(get_properties) else None
        )
        if self._mirrors and snapshotter is None:
            raise RuntimeError(
                "fan-out authority cannot snapshot an atomic embedding update"
            )
        vector = list(embedding)
        with self._producer(), self._mutation_lock(node_id):
            applied = bool(atomic_update(node_id, conditions, updates, vector))
            if not applied:
                return False
            self._authority_writes += 1
            properties: dict[str, Any] = {}
            label = ""
            if self._mirrors:
                assert snapshotter is not None
                snapshot = snapshotter(node_id)
                if not isinstance(snapshot, dict):
                    raise RuntimeError(
                        "fan-out authority node disappeared after atomic embedding update"
                    )
                properties = dict(snapshot)
                label = str(
                    properties.get("node_type")
                    or properties.get("label")
                    or properties.get("type")
                    or ""
                ).strip()
                if not label:
                    raise RuntimeError(
                        "fan-out atomic embedding update requires a typed authority node"
                    )
            self._enqueue(
                "compare_and_set_node_embedding",
                {
                    "node_id": node_id,
                    "conditions": dict(conditions),
                    "updates": dict(updates),
                    "embedding": vector,
                    "label": label,
                    "properties": properties,
                },
            )
            return True

    def create_schema(self) -> None:
        with self._producer():
            self._authority.create_schema()
            self._enqueue("create_schema", {})

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        with self._producer(), self._mutation_lock(node_id):
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
        with self._producer():
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
        elif op == "upsert_node":
            self._node_writer(backend)._upsert_node(
                p["label"],
                p["node_id"],
                p.get("properties") or {},
            )
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
            if getattr(backend, "supports_native_vector_search", True) is False:
                return
            if not _overrides_backend_method(backend, "verify_node_embedding"):
                raise RuntimeError(
                    "mirror embedding replay lacks read-after-write verification"
                )
            backend.add_embedding(p["node_id"], p["embedding"])
            if not backend.verify_node_embedding(p["node_id"], p["embedding"]):
                raise RuntimeError("mirror embedding replay was not verified")
        elif op == "compare_and_set_node_embedding":
            if _overrides_backend_method(backend, "compare_and_set_node_embedding"):
                atomic_update = backend.compare_and_set_node_embedding
                try:
                    applied = bool(
                        atomic_update(
                            p["node_id"],
                            p.get("conditions") or {},
                            p.get("updates") or {},
                            p["embedding"],
                        )
                    )
                except NotImplementedError:
                    applied = False
                else:
                    if applied:
                        return

            # Compatibility mirrors may not own a cross-modal transaction. The
            # authority is the only read source, so replay is allowed to converge
            # the mirror in two idempotent steps. If a prior replay crashed after
            # its field CAS, matching updates prove it is safe to retry ANN add.
            if _overrides_backend_method(backend, "compare_and_set_node_fields"):
                try:
                    applied = backend.compare_and_set_node_fields(
                        p["node_id"],
                        p.get("conditions") or {},
                        p.get("updates") or {},
                    )
                except NotImplementedError:
                    applied = False
                if not applied:
                    get_properties = getattr(backend, "get_node_properties", None)
                    current = (
                        get_properties(p["node_id"])
                        if callable(get_properties)
                        else None
                    )
                    applied = isinstance(current, dict) and all(
                        current.get(field) == expected
                        for field, expected in (p.get("updates") or {}).items()
                    )
                if applied:
                    if getattr(backend, "supports_native_vector_search", True) is False:
                        return
                    if not backend.embedding_is_node_property:
                        backend.add_embedding(p["node_id"], p["embedding"])
                    if not _overrides_backend_method(
                        backend, "verify_node_embedding"
                    ) or not backend.verify_node_embedding(
                        p["node_id"], p["embedding"]
                    ):
                        raise RuntimeError("mirror embedding replay was not verified")
                    return

            # Standard mirrors do not implement either optional CAS contract.
            # Replay the authority's full post-commit node snapshot through the
            # existing dialect-aware writer, then project the vector.  This is
            # idempotent and lets the cursor advance instead of poison-retrying
            # an inherited NotImplementedError forever.
            properties = p.get("properties")
            label = str(p.get("label") or "").strip()
            if not isinstance(properties, dict) or not label:
                raise RuntimeError(
                    "mirror atomic embedding replay lacks an authority snapshot"
                )
            # PostgreSQL/Ladybug/Neo4j store the vector as the node property, so
            # the full snapshot upsert is their ONE vector write. Side-index
            # backends omit it structurally and perform ONE add_embedding call.
            # Either storage mode must verify before cursor advance.
            structural_properties = {
                field: value
                for field, value in properties.items()
                if field != "embedding"
            }
            replay_properties = (
                properties
                if backend.embedding_is_node_property
                else structural_properties
            )
            self._node_writer(backend)._upsert_node(
                label,
                p["node_id"],
                replay_properties,
            )
            if getattr(backend, "supports_native_vector_search", True) is False:
                return
            if not backend.embedding_is_node_property:
                backend.add_embedding(p["node_id"], p["embedding"])
            if not _overrides_backend_method(backend, "verify_node_embedding"):
                raise RuntimeError(
                    "mirror embedding replay lacks read-after-write verification"
                )
            if not backend.verify_node_embedding(p["node_id"], p["embedding"]):
                raise RuntimeError("mirror embedding replay was not verified")
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

    def _node_writer(self, backend: GraphBackend) -> Any:
        """Return the cached portable writer used for structured node replay."""
        return self._edge_writer(backend)

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
                    # PERMANENT malformed/dialect-incompatible graph mutations are
                    # skipped after repeated confirmation so one poison query cannot
                    # block the mirror forever; reconcile() repairs that graph state.
                    # Vector publication is deliberately exempt: an exception whose
                    # text happens to contain a permanent marker may have come from
                    # durable read-back verification. Only successful verification or
                    # an explicit graph-only capability may advance a vector cursor.
                    if (
                        entry.op not in _NON_DROPPABLE_REPLAY_OPS
                        and _is_permanent_apply_error(exc)
                    ):
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
        # Convergence has TWO stages now (CONCEPT:AU-KG.backend.authority-has-already-acked): the in-memory ring must
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
            "lifecycle_state": self._lifecycle_state,
            "active_producers": self._active_producers,
            "outbox_depth": self._outbox.depth() if self._outbox is not None else 0,
            # Async hand-off backlog (CONCEPT:AU-KG.backend.authority-has-already-acked): ring items the write path
            # handed off but the persister has not yet landed in the durable outbox.
            "handoff_inflight": inflight,
            "mirrors": mirrors,
        }

    # ------------------------------------------------------------------
    # SPARQL / Cypher capability — delegate to the authority
    # ------------------------------------------------------------------
    @property
    def typed_mutation_support(self) -> str:
        """Expose typed writes only when the authority explicitly declares them."""
        return str(getattr(self._authority, "typed_mutation_support", ""))

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
    # Facade surface — expose the authority compute graph + delegate
    # ------------------------------------------------------------------
    @property
    def authority(self) -> GraphBackend:
        """Return the fixed operational authority for health introspection."""
        return self._authority

    @property
    def graph(self) -> Any:
        return getattr(self._authority, "graph", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate backend-specific helpers to the authority store.

        The facade/engine sometimes call authority-specific helpers
        (``save_to_json`` …); anything not defined
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
        # One close owner establishes CLOSING before any resource teardown. New
        # producers fail before authority mutation; already-registered producers
        # finish authority + ring admission before the persister is stopped.
        with self._close_lock:
            with self._lifecycle_condition:
                if self._lifecycle_state == "closed":
                    return
                self._lifecycle_state = "closing"
                while self._active_producers:
                    self._lifecycle_condition.wait(timeout=_IDLE_POLL_S)

            self._stop.set()
            with self._handoff_condition:
                self._handoff_condition.notify_all()
            if self._persister is not None:
                self._persister.join(timeout=10.0)
                if self._persister.is_alive():
                    raise RuntimeError(
                        "FanOutBackend persister did not stop; refusing to close outbox"
                    )

            # No producer or persister can append now. Strictly persist every
            # admitted mutation before the SQLite handle is closed; an append
            # failure aborts close and leaves the handle open for a retry.
            self._drain_handoff_remaining()
            with self._handoff_condition:
                if self._handoff_appending or self._handoff_active is not None:
                    raise RuntimeError(
                        "FanOutBackend append claim remained active during close"
                    )
            with self._inflight_lock:
                if self._inflight:
                    raise RuntimeError(
                        f"FanOutBackend close left {self._inflight} handoff(s) volatile"
                    )

            for st in self._state.values():
                st.wake.set()
                if st.thread is not None:
                    st.thread.join(timeout=10.0)
                    if st.thread.is_alive():
                        raise RuntimeError(
                            "FanOutBackend mirror drainer did not stop; refusing close"
                        )
            if self._outbox is not None:
                self._outbox.close()

            close_error: BaseException | None = None
            try:
                self._authority.close()
            except BaseException as exc:  # noqa: BLE001 - close all mirrors, then relay
                close_error = exc
            for name, backend in self._mirrors.items():
                try:
                    backend.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "FanOutBackend: mirror %s close failed: %s", name, exc
                    )
            with self._lifecycle_condition:
                self._lifecycle_state = "closed"
                self._lifecycle_condition.notify_all()
            if close_error is not None:
                raise close_error
