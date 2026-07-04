"""Cross-host daemon leadership election.

CONCEPT:AU-OS.state.cross-host-daemon-leadership — Cross-host daemon leadership election via Postgres advisory locks so singleton background ticks run on exactly one host fleet-wide

When durable state is externalized (``state_db_uri`` set, CONCEPT:AU-OS.state.unified-durable-state-externalization),
N hosts may each hold their *per-host* flock and would otherwise all run the
singleton background ticks (golden loop, maintenance scheduler, SDD watcher,
…) — duplicating LLM work and racing writes. :class:`DaemonLeadership` makes
those ticks run on exactly ONE host fleet-wide using a Postgres session
advisory lock (``pg_try_advisory_lock``) with a stable per-role key:

* the lock is tied to a dedicated connection, so a crashed leader's lock is
  released by Postgres when the connection dies — no stale-lockfile problem;
* followers re-try acquisition on every :meth:`is_leader` poll, so leadership
  fails over within one scheduler tick.

Under the SQLite default (no ``state_db_uri``) :meth:`is_leader` is always
``True``: single-host deployments keep the existing flock-only behavior
unchanged.

Tick classification (documented here, enforced in
``knowledge_graph/core/engine_tasks.py``):

* **Leader-only** — everything driven by the maintenance scheduler (analysis,
  golden loop, failure ingest, anomaly consumer, fuseki publish, compaction,
  evolution, durable reconcile, enrichment, file/SDD watch, hygiene, task
  reaper) plus the embedding-backfill drain. These are whole-graph/singleton
  passes where N copies means duplicated LLM spend or double writes.
* **Per-host** — ingestion capacity: the task workers, the submission-queue
  drain, and the graph-writer drain. These scale out across hosts safely
  because queue claims are atomic (SKIP LOCKED / advisory claim guard,
  CONCEPT:AU-KG.ingest.cross-host-safe-kg).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent_utilities.core.state_store import (
    advisory_key,
    postgres_state_enabled,
    state_db_uri,
)

logger = logging.getLogger(__name__)


class DaemonLeadership:
    """Holds (or keeps trying to win) the fleet-wide leader lock for a role."""

    def __init__(self, role: str, dsn: str | None = None):
        self.role = role
        self._dsn = dsn
        self._key = advisory_key(f"leader:{role}")
        self._conn: Any = None
        self._held = False
        self._lock = threading.Lock()
        self._warned = False

    def _resolve_dsn(self) -> str | None:
        return self._dsn or state_db_uri()

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self._resolve_dsn(), autocommit=True)

    def is_leader(self) -> bool:
        """True iff this process currently holds the role's leader lock.

        SQLite default → always True (per-host flock already guarantees a
        single daemon). Postgres → non-blocking try-acquire on a dedicated
        connection; connection loss drops leadership (Postgres releases the
        advisory lock server-side) and the next poll re-elects.
        """
        if self._dsn is None and not postgres_state_enabled():
            return True
        with self._lock:
            if self._held and self._conn is not None:
                try:
                    self._conn.execute("SELECT 1")
                    return True
                except Exception:  # noqa: BLE001 — connection died → leadership lost
                    logger.warning(
                        "leader connection lost for role %r — re-electing", self.role
                    )
                    self._teardown()
            try:
                if self._conn is None:
                    self._conn = self._connect()
                cur = self._conn.execute(
                    "SELECT pg_try_advisory_lock(%s)", (self._key,)
                )
                row = cur.fetchone()
                self._held = bool(row and row[0])
                if self._held:
                    logger.info(
                        "elected fleet leader for role %r (advisory key %d)",
                        self.role,
                        self._key,
                    )
                    self._warned = False
                return self._held
            except Exception as e:  # noqa: BLE001 — PG unreachable → nobody leads
                self._teardown()
                if not self._warned:
                    logger.warning(
                        "leadership probe failed for role %r (%s) — leader-only "
                        "ticks paused until the state store is reachable",
                        self.role,
                        e,
                    )
                    self._warned = True
                return False

    def _teardown(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._conn = None
        self._held = False

    def release(self) -> None:
        """Voluntarily drop leadership (closes the lock connection)."""
        with self._lock:
            if self._held and self._conn is not None:
                try:
                    self._conn.execute("SELECT pg_advisory_unlock(%s)", (self._key,))
                except Exception:  # noqa: BLE001
                    pass
            self._teardown()


_instances: dict[str, DaemonLeadership] = {}
_instances_lock = threading.Lock()


def get_leadership(role: str) -> DaemonLeadership:
    """Process-wide shared :class:`DaemonLeadership` per role."""
    with _instances_lock:
        inst = _instances.get(role)
        if inst is None:
            inst = DaemonLeadership(role)
            _instances[role] = inst
        return inst
