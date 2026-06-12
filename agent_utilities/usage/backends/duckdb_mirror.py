"""DuckDB usage backend — optional columnar analytics mirror.

CONCEPT:ECO-4.39. Portable, fast for heavy aggregation. DuckDB speaks most
SQLite SQL but has no FTS5, so search falls back to a substring scan. Imported
lazily: a missing ``duckdb`` package raises a clear, actionable error rather
than failing at import time.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from agent_utilities.core.config import setting

from ..models import SearchHit
from ..schema import sqlite_ddl
from .sql_base import SqlUsageBackend


def _default_path() -> Path:
    override = setting("USAGE_DUCKDB_PATH")
    if override:
        return Path(override)
    base = Path.home() / ".local" / "share" / "agent-utilities"
    base.mkdir(parents=True, exist_ok=True)
    return base / "usage.duckdb"


class DuckDBUsageBackend(SqlUsageBackend):
    name = "duckdb"
    placeholder = "?"

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _default_path()
        self._schema_ready = False

    @contextlib.contextmanager
    def _connect(self):
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "USAGE_DB_BACKEND=duckdb requires the 'duckdb' package "
                "(pip install duckdb)."
            ) from exc
        conn = duckdb.connect(str(self._path))
        try:
            yield conn
            conn.commit()
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        # DuckDB autoincrement differs; use sequences via a simplified DDL.
        ddl = sqlite_ddl().replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGINT")
        # DuckDB has no FTS5 virtual table — drop that statement.
        ddl = ddl.split("CREATE VIRTUAL TABLE", 1)[0]
        with self._connect() as conn:
            for stmt in ddl.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
        self._schema_ready = True

    def _ensure_search(self, conn) -> None:
        return None

    def _clear_search(self, conn, session_id: str) -> None:
        return None

    def _index_messages(self, conn, session_id, msgs) -> None:
        return None  # substring search reads messages.content directly

    def search(self, query, *, limit=50, **filters):
        if not query or not query.strip():
            return []
        with self._connect() as conn:
            cur = conn.execute(
                """SELECT m.session_id, m.ordinal, m.role,
                      substr(m.content, 1, 160), s.project, s.agent
                    FROM messages m JOIN sessions s ON s.id = m.session_id
                    WHERE m.content ILIKE ? LIMIT ?""",
                (f"%{query}%", limit),
            )
            rows = cur.fetchall()
        return [
            SearchHit(
                session_id=r[0],
                ordinal=int(r[1]),
                role=r[2],
                snippet=r[3] or "",
                project=r[4] or "",
                agent=r[5] or "claude",
            )
            for r in rows
        ]
