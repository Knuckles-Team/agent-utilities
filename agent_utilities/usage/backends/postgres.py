"""Postgres usage backend — enterprise-scale shared store.

CONCEPT:ECO-4.39. Opens through the ``state_store`` seam (the same pool the KG
Postgres backend uses) so a single ``STATE_DB_URI`` promotes the whole platform
— including usage analytics — onto shared Postgres. Search uses ``tsvector``/
GIN; everything else shares ``SqlUsageBackend`` for query-shape parity.
"""

from __future__ import annotations

from agent_utilities.core.state_store import open_state_connection

from ..models import SearchHit
from ..schema import postgres_ddl
from .sql_base import SqlUsageBackend


class PostgresUsageBackend(SqlUsageBackend):
    name = "postgres"
    placeholder = "?"  # open_state_connection rewrites ? -> %s for psycopg

    def __init__(self) -> None:
        self._schema_ready = False

    def _connect(self):
        # store name "usage" so ensure_state_schema runs the DDL once per process.
        return open_state_connection("usage", lambda: "", postgres_ddl())

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._connect() as conn:  # ensure_state_schema runs inside open
            conn.commit()
        self._schema_ready = True

    def _ensure_search(self, conn) -> None:
        return None

    def _clear_search(self, conn, session_id: str) -> None:
        return None  # tsvector lives on messages rows, removed with the DELETE

    def _index_messages(self, conn, session_id, msgs) -> None:
        # Populate the tsvector column for the just-inserted rows.
        conn.execute(
            "UPDATE messages SET content_tsv = to_tsvector('english', content) "
            "WHERE session_id = ? AND content_tsv IS NULL",
            (session_id,),
        )

    def search(self, query, *, limit=50, **filters):
        if not query or not query.strip():
            return []
        with self._connect() as conn:
            cur = conn.execute(
                """SELECT m.session_id, m.ordinal, m.role,
                      substr(m.content, 1, 160), s.project, s.agent
                    FROM messages m JOIN sessions s ON s.id = m.session_id
                    WHERE m.content_tsv @@ plainto_tsquery('english', ?)
                    LIMIT ?""",
                (query, limit),
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
