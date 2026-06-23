"""Durable delta-ingestion manifest (CONCEPT:KG-2.8).

A single, durable record of "what content has already been ingested" keyed by
``(graph_name, category, source_uri) -> content_hash``. The ``IngestionEngine``
consults it before dispatching an adaptor so unchanged sources are skipped
across process restarts; adaptors never re-parse or duplicate work.

Storage: in practice the durable tier (pggraph/PostgreSQL) is SCHEMA-CONSTRAINED
and has no ``:IngestManifest`` table, and the pure in-memory L1
(``EpistemicGraphBackend``) isn't durable across restart — so for all real
backends the manifest uses a small **SQLite store under ``data_dir()``**
(``kg_ingest_manifest.db``, WAL, mirroring the proven ``SQLiteTaskQueue``
pattern), which is durable + robust + backend-agnostic. A graph-native
``:IngestManifest`` path (via the ``GraphBackend.execute()`` MERGE contract) is
retained for any future backend that can durably store arbitrary labels; see
``_NON_DURABLE_BACKENDS`` for the SQLite-fallback set.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Backends for which the manifest must use the SQLite fallback instead of
# graph-native :IngestManifest nodes:
#   - PostgreSQLBackend: schema-constrained tables have no :IngestManifest table,
#     so a graph-native MERGE errors ("relation does not exist").
# SQLite under data_dir() is durable + robust, so it's the manifest store for
# these. (The epistemic-graph engine authority holds arbitrary nodes, so the
# fanout/engine path keeps graph-native manifests.)
_NON_DURABLE_BACKENDS = {
    "EpistemicGraphBackend",
    "PostgreSQLBackend",
}

_LABEL = "IngestManifest"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_durable_backend(backend: Any) -> bool:
    """True if writes through ``backend.execute()`` survive a restart.

    A pure in-memory backend cannot persist the manifest graph-side, so it is
    treated as non-durable (→ SQLite fallback). A schema-backed durable store
    (pggraph) is durable.
    """
    if backend is None:
        return False
    if not hasattr(backend, "execute"):
        return False
    return type(backend).__name__ not in _NON_DURABLE_BACKENDS


def _key(graph_name: str, category: str, source_uri: str) -> str:
    return f"{graph_name}|{category}|{source_uri}"


class DeltaManifest:
    """Durable per-(graph, category, source) content-hash manifest.

    CONCEPT:KG-2.8

    Args:
        backend: A ``GraphBackend``. If durable, the manifest is stored as
            ``:IngestManifest`` nodes via ``execute()``. If ``None`` or a pure
            in-memory backend, the SQLite fallback is used.
        db_path: Override the SQLite fallback path (defaults to
            ``data_dir()/"kg_ingest_manifest.db"``).
    """

    def __init__(self, backend: Any = None, db_path: str | None = None) -> None:
        self._backend: Any = backend if _is_durable_backend(backend) else None
        self.mode = "graph" if self._backend is not None else "sqlite"
        self._lock = threading.Lock()
        if self.mode == "sqlite":
            self._db_path = db_path or self._default_db_path()
            self._init_sqlite()
            logger.debug("DeltaManifest: SQLite mode at %s", self._db_path)
        else:
            logger.debug(
                "DeltaManifest: graph mode via %s", type(self._backend).__name__
            )

    # ── SQLite fallback ──────────────────────────────────────────────────
    @staticmethod
    def _default_db_path() -> str:
        from agent_utilities.core.paths import data_dir

        return str(data_dir() / "kg_ingest_manifest.db")

    def _init_sqlite(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            try:
                with conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS ingest_manifest ("
                        "graph_name TEXT NOT NULL, category TEXT NOT NULL, "
                        "source_uri TEXT NOT NULL, content_hash TEXT NOT NULL, "
                        "updated_at TEXT NOT NULL, "
                        "PRIMARY KEY (graph_name, category, source_uri))"
                    )
            finally:
                conn.close()

    # ── Public API ───────────────────────────────────────────────────────
    def get(self, graph_name: str, category: str, source_uri: str) -> str | None:
        """Return the stored content_hash for a source, or ``None``."""
        if self.mode == "sqlite":
            with self._lock:
                conn = sqlite3.connect(self._db_path, timeout=30.0)
                try:
                    cur = conn.execute(
                        "SELECT content_hash FROM ingest_manifest "
                        "WHERE graph_name=? AND category=? AND source_uri=?",
                        (graph_name, category, source_uri),
                    )
                    row = cur.fetchone()
                    return row[0] if row else None
                finally:
                    conn.close()
        rows = self._backend.execute(
            f"MATCH (m:{_LABEL} {{id: $id}}) RETURN m",
            {"id": _key(graph_name, category, source_uri)},
        )
        node = self._first_node(rows)
        return node.get("content_hash") if node else None

    def seen(
        self, graph_name: str, category: str, source_uri: str, content_hash: str
    ) -> bool:
        """True if this exact content_hash was already ingested for the source."""
        return self.get(graph_name, category, source_uri) == content_hash

    def record(
        self, graph_name: str, category: str, source_uri: str, content_hash: str
    ) -> None:
        """Upsert the content_hash for a source (idempotent)."""
        ts = _now()
        if self.mode == "sqlite":
            with self._lock:
                conn = sqlite3.connect(self._db_path, timeout=30.0)
                try:
                    with conn:
                        conn.execute(
                            "INSERT INTO ingest_manifest "
                            "(graph_name, category, source_uri, content_hash, updated_at) "
                            "VALUES (?,?,?,?,?) "
                            "ON CONFLICT(graph_name, category, source_uri) "
                            "DO UPDATE SET content_hash=excluded.content_hash, "
                            "updated_at=excluded.updated_at",
                            (graph_name, category, source_uri, content_hash, ts),
                        )
                finally:
                    conn.close()
            return
        self._backend.execute(
            f"MERGE (m:{_LABEL} {{id: $id}}) SET "
            "m.graph_name = $graph_name, m.category = $category, "
            "m.source_uri = $source_uri, m.content_hash = $content_hash, "
            "m.updated_at = $updated_at",
            {
                "id": _key(graph_name, category, source_uri),
                "graph_name": graph_name,
                "category": category,
                "source_uri": source_uri,
                "content_hash": content_hash,
                "updated_at": ts,
            },
        )

    def load_for_graph(self, graph_name: str, category: str) -> dict[str, str]:
        """Bulk-load ``{source_uri: content_hash}`` for one (graph, category).

        One query → in-memory dict; used to seed finer-grained per-file skip
        (e.g. ``EnrichmentPipeline.hash_seen``) without per-file round-trips.
        """
        if self.mode == "sqlite":
            with self._lock:
                conn = sqlite3.connect(self._db_path, timeout=30.0)
                try:
                    cur = conn.execute(
                        "SELECT source_uri, content_hash FROM ingest_manifest "
                        "WHERE graph_name=? AND category=?",
                        (graph_name, category),
                    )
                    return {r[0]: r[1] for r in cur.fetchall()}
                finally:
                    conn.close()
        rows = self._backend.execute(
            f"MATCH (m:{_LABEL}) WHERE m.graph_name = $graph_name "
            "AND m.category = $category RETURN m",
            {"graph_name": graph_name, "category": category},
        )
        out: dict[str, str] = {}
        for row in rows if isinstance(rows, list) else []:
            node = self._row_node(row)
            if node and node.get("source_uri"):
                out[node["source_uri"]] = node.get("content_hash", "")
        return out

    def freshness(self, graph_name: str, category: str) -> dict[str, str]:
        """Bulk-load ``{source_uri: updated_at}`` for one (graph, category).

        The last-sync watermark per ingested source, used by the ingestion-coverage
        doctor check to enforce a freshness SLA (CONCEPT:OS-5.47) — flagging repos
        whose last delta sync is older than the threshold.
        """
        if self.mode == "sqlite":
            with self._lock:
                conn = sqlite3.connect(self._db_path, timeout=30.0)
                try:
                    cur = conn.execute(
                        "SELECT source_uri, updated_at FROM ingest_manifest "
                        "WHERE graph_name=? AND category=?",
                        (graph_name, category),
                    )
                    return {r[0]: r[1] for r in cur.fetchall()}
                finally:
                    conn.close()
        rows = self._backend.execute(
            f"MATCH (m:{_LABEL}) WHERE m.graph_name = $graph_name "
            "AND m.category = $category RETURN m",
            {"graph_name": graph_name, "category": category},
        )
        out: dict[str, str] = {}
        for row in rows if isinstance(rows, list) else []:
            node = self._row_node(row)
            if node and node.get("source_uri"):
                out[node["source_uri"]] = node.get("updated_at", "")
        return out

    def clear(self, graph_name: str | None = None, category: str | None = None) -> None:
        """Remove manifest rows (all, or scoped by graph/category).

        Used by graph-wipe paths and tests. Graph mode relies on the KG wipe
        clearing ``:IngestManifest`` nodes; this provides an explicit scoped
        clear for the SQLite fallback and targeted resets.
        """
        if self.mode == "sqlite":
            clauses, args = [], []
            if graph_name is not None:
                clauses.append("graph_name=?")
                args.append(graph_name)
            if category is not None:
                clauses.append("category=?")
                args.append(category)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            with self._lock:
                conn = sqlite3.connect(self._db_path, timeout=30.0)
                try:
                    with conn:
                        conn.execute(f"DELETE FROM ingest_manifest{where}", tuple(args))
                finally:
                    conn.close()
            return
        # Graph mode: DETACH DELETE matching manifest nodes.
        if graph_name is None and category is None:
            self._backend.execute(f"MATCH (m:{_LABEL}) DETACH DELETE m", {})
            return
        conds, params = [], {}
        if graph_name is not None:
            conds.append("m.graph_name = $graph_name")
            params["graph_name"] = graph_name
        if category is not None:
            conds.append("m.category = $category")
            params["category"] = category
        self._backend.execute(
            f"MATCH (m:{_LABEL}) WHERE {' AND '.join(conds)} DETACH DELETE m",
            params,
        )

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _row_node(row: Any) -> dict[str, Any] | None:
        """Extract a node dict from a result row (``{'m': {...}}`` or ``{...}``)."""
        if not isinstance(row, dict):
            return None
        inner = row.get("m")
        if isinstance(inner, dict):
            return inner
        return row

    def _first_node(self, rows: Any) -> dict[str, Any] | None:
        if not isinstance(rows, list) or not rows:
            return None
        return self._row_node(rows[0])
