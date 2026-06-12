"""SQLite + FTS5 usage backend — the zero-dependency native default.

CONCEPT:ECO-4.39. Connects directly via :mod:`sqlite3` so FTS5 and connection
lifecycle are fully controlled. Path resolves to a per-host XDG data file, the
same convention the sessions/durable-exec stores use.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path

from ..models import SearchHit
from ..schema import sqlite_ddl
from .sql_base import SqlUsageBackend


def default_db_path() -> Path:
    override = os.environ.get("USAGE_DB_PATH")
    if override:
        return Path(override)
    base = Path.home() / ".local" / "share" / "agent-utilities"
    base.mkdir(parents=True, exist_ok=True)
    return base / "usage.db"


class SqliteUsageBackend(SqlUsageBackend):
    """Default backend: SQLite WAL + FTS5 full-text search."""

    name = "sqlite"
    placeholder = "?"

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else default_db_path()
        self._schema_ready = False

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._connect() as conn:
            conn.executescript(sqlite_ddl())
        self._schema_ready = True

    # FTS5 search structure already created by ensure_schema; nothing per-write.
    def _ensure_search(self, conn) -> None:
        return None

    def _clear_search(self, conn, session_id: str) -> None:
        conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))

    def _index_messages(self, conn, session_id, msgs) -> None:
        for m in msgs:
            if not m.content:
                continue
            conn.execute(
                "INSERT INTO messages_fts (content, session_id, ordinal, role) "
                "VALUES (?, ?, ?, ?)",
                (m.content, m.session_id, m.ordinal, m.role),
            )

    def search(self, query, *, limit=50, **filters):
        if not query or not query.strip():
            return []
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    """SELECT f.session_id, f.ordinal, f.role,
                          snippet(messages_fts, 0, '[', ']', '…', 12),
                          s.project, s.agent
                        FROM messages_fts f
                        JOIN sessions s ON s.id = f.session_id
                        WHERE messages_fts MATCH ?
                        LIMIT ?""",
                    (query, limit),
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                # Malformed FTS query → fall back to a LIKE scan.
                cur = conn.execute(
                    """SELECT f.session_id, f.ordinal, f.role,
                          substr(f.content, 1, 160), s.project, s.agent
                        FROM messages_fts f
                        JOIN sessions s ON s.id = f.session_id
                        WHERE f.content LIKE ? LIMIT ?""",
                    (f"%{query}%", limit),
                )
                rows = cur.fetchall()
        return [
            SearchHit(
                session_id=r[0], ordinal=int(r[1]), role=r[2], snippet=r[3] or "",
                project=r[4] or "", agent=r[5] or "claude",
            )
            for r in rows
        ]
