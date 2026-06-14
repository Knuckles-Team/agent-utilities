#!/usr/bin/python
# CONCEPT:KG-2.74 - Durable per-mirror outbox: a crash-safe append log that lets the FanOutBackend mirror every write to N durable graph stores without loss — a store that is down or slow simply replays its unapplied tail from a persisted cursor when it returns.
"""Durable per-mirror write outbox for lossless multi-store mirroring.

CONCEPT:KG-2.74 — Lossless N-way mirror outbox.

The :class:`~agent_utilities.knowledge_graph.backends.fanout_backend.FanOutBackend`
commits each write to a single **authority** store synchronously, then hands the
mutation to this outbox once per **mirror**. A per-mirror background drainer
applies entries in order and advances a persisted cursor; a mirror that is
offline or slow keeps its unapplied tail in the log and **replays from its cursor
on reconnect / process restart** — so a transient outage never drops a write.

Why a dedicated durable log (and not the in-memory write-behind queue the
``TieredGraphBackend`` uses): that queue lives in process memory, so a crash or a
mirror outage longer than the process lifetime loses the backlog. This log is
``sqlite`` in WAL mode (durable across restarts, zero external infra) — the same
zero-infra-default shape the state-store seam uses. The append is committed
*before* the write is acked, so the only un-mirrorable window is a crash in the
microseconds between the authority commit and the outbox append; that residual
gap is the job of the periodic ``reconcile`` drift-repair pass, not this log.

Schema (one sqlite file, WAL):

* ``outbox(mirror, seq, op, payload, created_at)`` — one row per (mirror,
  mutation). ``seq`` is a global monotonic append order shared across mirrors so
  every mirror applies mutations in the same order. ``payload`` is the JSON-
  encoded operation arguments.
* ``cursor(mirror, applied_seq)`` — the highest ``seq`` durably applied to each
  mirror. Replay resumes from here.

Entries are pruned once every registered mirror has applied them (past the
slowest cursor), so the log self-trims and only a permanently-stalled mirror
grows it (which the lag metric alarms on).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutboxEntry:
    """A single durable mutation awaiting application to one mirror."""

    seq: int
    op: str
    payload: dict[str, Any]


class GraphOutbox:
    """Crash-safe, thread-safe append log of pending mirror mutations.

    One sqlite file (WAL mode) shared by all mirrors of a FanOutBackend. A single
    connection guarded by an :class:`~threading.RLock` — writes serialize in
    sqlite anyway and every op here is short, so one lock keeps it simple and
    correct across the append path and the per-mirror drainer threads.
    """

    def __init__(self, path: str | Path, mirrors: list[str]) -> None:
        self._path = str(path)
        self._mirrors = list(mirrors)
        self._lock = threading.RLock()
        # check_same_thread=False: the append path and the drainer threads share
        # this one connection under ``self._lock`` (sqlite serializes writes).
        self._conn = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None
        )
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema / lifecycle
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn
            # WAL = durable across restarts AND lets a reader run during a write.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS outbox ("
                "  mirror TEXT NOT NULL,"
                "  seq INTEGER NOT NULL,"
                "  op TEXT NOT NULL,"
                "  payload TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  PRIMARY KEY (mirror, seq)"
                ")"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_mirror_seq "
                "ON outbox (mirror, seq)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS cursor ("
                "  mirror TEXT PRIMARY KEY,"
                "  applied_seq INTEGER NOT NULL DEFAULT 0"
                ")"
            )
            for m in self._mirrors:
                cur.execute(
                    "INSERT OR IGNORE INTO cursor (mirror, applied_seq) "
                    "VALUES (?, 0)",
                    (m,),
                )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.debug("GraphOutbox close failed", exc_info=True)

    # ------------------------------------------------------------------
    # Append (producer side)
    # ------------------------------------------------------------------
    def append(self, op: str, payload: dict[str, Any]) -> int:
        """Durably append one mutation for **every** mirror; return its ``seq``.

        Allocates the next global sequence and writes one row per mirror inside a
        single transaction, so the mutation is atomically enqueued for all mirrors
        or none. Returns once committed (the write may then be acked).
        """
        blob = json.dumps(payload, default=str)
        now = time.time()
        with self._lock:
            cur = self._conn
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute("SELECT COALESCE(MAX(seq), 0) FROM outbox").fetchone()
                seq = int(row[0]) + 1
                cur.executemany(
                    "INSERT INTO outbox (mirror, seq, op, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(m, seq, op, blob, now) for m in self._mirrors],
                )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        return seq

    # ------------------------------------------------------------------
    # Drain (consumer side, per mirror)
    # ------------------------------------------------------------------
    def applied_seq(self, mirror: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT applied_seq FROM cursor WHERE mirror = ?", (mirror,)
            ).fetchone()
            return int(row[0]) if row else 0

    def pending(self, mirror: str, limit: int = 256) -> list[OutboxEntry]:
        """The next batch of unapplied entries for ``mirror``, in ``seq`` order."""
        with self._lock:
            applied = self.applied_seq(mirror)
            rows = self._conn.execute(
                "SELECT seq, op, payload FROM outbox "
                "WHERE mirror = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                (mirror, applied, limit),
            ).fetchall()
        return [
            OutboxEntry(seq=int(r[0]), op=str(r[1]), payload=json.loads(r[2]))
            for r in rows
        ]

    def ack(self, mirror: str, seq: int) -> None:
        """Advance ``mirror``'s cursor to ``seq`` (it has been durably applied)."""
        with self._lock:
            self._conn.execute(
                "UPDATE cursor SET applied_seq = ? WHERE mirror = ? "
                "AND applied_seq < ?",
                (seq, mirror, seq),
            )
            self._prune_locked()

    def lag(self, mirror: str) -> int:
        """How many appended entries this mirror has not yet applied."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE mirror = ? AND seq > "
                "(SELECT applied_seq FROM cursor WHERE mirror = ?)",
                (mirror, mirror),
            ).fetchone()
            return int(row[0]) if row else 0

    def depth(self) -> int:
        """Total un-pruned rows across all mirrors (overall backlog size)."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM outbox").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Self-trim
    # ------------------------------------------------------------------
    def _prune_locked(self) -> None:
        """Drop rows every mirror has applied (≤ the slowest cursor).

        Caller must hold ``self._lock``. Only the registered mirrors gate
        pruning, so a row is removed strictly after all of them have it durably.
        """
        if not self._mirrors:
            return
        row = self._conn.execute(
            "SELECT MIN(applied_seq) FROM cursor WHERE mirror IN "
            f"({','.join('?' for _ in self._mirrors)})",
            tuple(self._mirrors),
        ).fetchone()
        floor = int(row[0]) if row and row[0] is not None else 0
        if floor > 0:
            self._conn.execute("DELETE FROM outbox WHERE seq <= ?", (floor,))
