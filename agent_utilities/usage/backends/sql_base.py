"""Shared SQL implementation for usage backends (CONCEPT:ECO-4.39).

Holds the analytics/aggregation queries that are identical across SQLite,
Postgres, and DuckDB (standard SQL). Subclasses supply connection, parameter
placeholder, and the search implementation (FTS5 vs tsvector vs substring),
keeping observable query-shape parity.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from datetime import UTC, datetime

from agent_utilities.pricing import ModelPricing

from ..backend import UsageBackend
from ..models import (
    ActivityCell,
    BreakdownEntry,
    ParsedSessionBundle,
    SessionDetail,
    SessionRow,
    TokenTotals,
    ToolStat,
    UsageEvent,
    UsageMessage,
    UsageSummary,
    UsageToolCall,
)

_SESSION_FILTER_COLS = {
    "project": "project",
    "agent": "agent",
    "origin": "origin",
    "tenant_id": "tenant_id",
    "health_grade": "health_grade",
    "outcome": "outcome",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqlUsageBackend(UsageBackend):
    """Base for SQL-dialect usage backends. Subclasses implement the hooks."""

    placeholder = "?"

    # ── hooks subclasses must provide ───────────────────────────────────
    @abstractmethod
    def _connect(self):  # returns a DB-API-ish connection (context manager)
        """Open a DB-API-style connection usable as a context manager."""

    @abstractmethod
    def _ensure_search(self, conn) -> None:
        """Create the search structure (FTS5 table / tsvector index)."""

    @abstractmethod
    def _index_messages(self, conn, session_id: str, msgs: list[UsageMessage]) -> None:
        """Index a session's messages for search after they are written."""

    @abstractmethod
    def _clear_search(self, conn, session_id: str) -> None:
        """Remove a session's search rows before re-indexing."""

    @abstractmethod
    def search(self, query, *, limit=50, **filters):
        """Full-text search session messages (dialect-specific implementation)."""

    # ── helpers ─────────────────────────────────────────────────────────
    def _ph(self) -> str:
        return self.placeholder

    def _where(self, filters: dict, *, alias: str = "") -> tuple[str, list]:
        """Build a WHERE clause from session-scoped filters + date range."""
        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list = []
        ph = self._ph()
        for key, col in _SESSION_FILTER_COLS.items():
            val = filters.get(key)
            if val:
                clauses.append(f"{prefix}{col} = {ph}")
                params.append(val)
        if filters.get("from_date"):
            clauses.append(f"{prefix}started_at >= {ph}")
            params.append(filters["from_date"])
        if filters.get("to_date"):
            clauses.append(f"{prefix}started_at <= {ph}")
            params.append(filters["to_date"])
        sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return sql, params

    # ── lifecycle ───────────────────────────────────────────────────────
    def close(self) -> None:
        return None

    # ── writes ──────────────────────────────────────────────────────────
    def write_bundle(self, bundle: ParsedSessionBundle) -> None:
        s = bundle.session
        ph = self._ph()
        with self._connect() as conn:
            self._ensure_search(conn)
            # Replace existing rows for idempotent re-ingest.
            for tbl in ("messages", "tool_calls", "usage_events"):
                conn.execute(f"DELETE FROM {tbl} WHERE session_id = {ph}", (s.id,))
            self._clear_search(conn, s.id)
            conn.execute(f"DELETE FROM sessions WHERE id = {ph}", (s.id,))
            conn.execute(
                f"""INSERT INTO sessions
                  (id, project, machine, agent, first_message, started_at, ended_at,
                   message_count, user_message_count, total_output_tokens,
                   peak_context_tokens, is_automated, outcome, health_grade,
                   termination_status, parent_session_id, relationship_type, origin,
                   tenant_id, correlation_id, file_path, file_hash, file_mtime,
                   file_inode, created_at)
                  VALUES ({", ".join([ph] * 25)})""",
                (
                    s.id,
                    s.project,
                    s.machine,
                    s.agent,
                    s.first_message,
                    s.started_at,
                    s.ended_at,
                    s.message_count,
                    s.user_message_count,
                    s.total_output_tokens,
                    s.peak_context_tokens,
                    1 if s.is_automated else 0,
                    s.outcome,
                    s.health_grade,
                    s.termination_status,
                    s.parent_session_id,
                    s.relationship_type,
                    s.origin,
                    s.tenant_id,
                    s.correlation_id,
                    s.file_path,
                    s.file_hash,
                    s.file_mtime,
                    s.file_inode,
                    _now(),
                ),
            )
            for m in bundle.messages:
                conn.execute(
                    f"""INSERT INTO messages
                      (session_id, ordinal, role, content, thinking_text, timestamp,
                       model, context_tokens, output_tokens, has_tool_use,
                       content_length)
                      VALUES ({", ".join([ph] * 11)})""",
                    (
                        m.session_id,
                        m.ordinal,
                        m.role,
                        m.content,
                        m.thinking_text,
                        m.timestamp,
                        m.model,
                        m.context_tokens,
                        m.output_tokens,
                        1 if m.has_tool_use else 0,
                        m.content_length or len(m.content),
                    ),
                )
            for t in bundle.tool_calls:
                self._insert_tool_call(conn, t)
            for e in bundle.usage_events:
                self._insert_usage_event(conn, e)
            self._index_messages(conn, s.id, bundle.messages)

    def _insert_tool_call(self, conn, t: UsageToolCall) -> None:
        ph = self._ph()
        conn.execute(
            f"""INSERT INTO tool_calls
              (session_id, message_ordinal, tool_name, category, tool_use_id,
               input_json, skill_name, result_content_length, subagent_session_id,
               status, occurred_at, origin, tenant_id, correlation_id)
              VALUES ({", ".join([ph] * 14)})""",
            (
                t.session_id,
                t.message_ordinal,
                t.tool_name,
                t.category,
                t.tool_use_id,
                t.input_json,
                t.skill_name,
                t.result_content_length,
                t.subagent_session_id,
                t.status,
                t.occurred_at or _now(),
                t.origin,
                t.tenant_id,
                t.correlation_id,
            ),
        )

    def _insert_usage_event(self, conn, e: UsageEvent) -> None:
        ph = self._ph()
        # Dedup-aware: skip when (session, source, dedup_key) already present.
        if e.dedup_key:
            cur = conn.execute(
                f"""SELECT 1 FROM usage_events
                  WHERE session_id={ph} AND source={ph} AND dedup_key={ph}""",
                (e.session_id, e.source, e.dedup_key),
            )
            if cur.fetchone() is not None:
                return
        conn.execute(
            f"""INSERT INTO usage_events
              (session_id, message_ordinal, source, model, input_tokens,
               output_tokens, cache_creation_input_tokens, cache_read_input_tokens,
               reasoning_tokens, cost_usd, cost_status, cost_source, occurred_at,
               dedup_key, origin, tenant_id, correlation_id)
              VALUES ({", ".join([ph] * 17)})""",
            (
                e.session_id,
                e.message_ordinal,
                e.source,
                e.model,
                e.input_tokens,
                e.output_tokens,
                e.cache_creation_input_tokens,
                e.cache_read_input_tokens,
                e.reasoning_tokens,
                e.cost_usd,
                e.cost_status,
                e.cost_source,
                e.occurred_at or _now(),
                e.dedup_key,
                e.origin,
                e.tenant_id,
                e.correlation_id,
            ),
        )

    def record_usage_event(self, event: UsageEvent) -> None:
        with self._connect() as conn:
            self._insert_usage_event(conn, event)

    def record_tool_call(self, call: UsageToolCall) -> None:
        with self._connect() as conn:
            self._insert_tool_call(conn, call)

    def upsert_pricing(self, entries: Iterable[ModelPricing]) -> None:
        ph = self._ph()
        with self._connect() as conn:
            for p in entries:
                conn.execute(
                    f"DELETE FROM model_pricing WHERE model_pattern = {ph}",
                    (p.model_pattern,),
                )
                conn.execute(
                    f"""INSERT INTO model_pricing
                      (model_pattern, input_per_mtok, output_per_mtok,
                       cache_creation_per_mtok, cache_read_per_mtok, updated_at)
                      VALUES ({", ".join([ph] * 6)})""",
                    (
                        p.model_pattern,
                        p.input_per_mtok,
                        p.output_per_mtok,
                        p.cache_creation_per_mtok,
                        p.cache_read_per_mtok,
                        _now(),
                    ),
                )

    # ── sync skip cache ─────────────────────────────────────────────────
    def should_sync(self, path: str, mtime: int, size: int) -> bool:
        ph = self._ph()
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT mtime, size FROM skipped_files WHERE path = {ph}", (path,)
            )
            row = cur.fetchone()
        if row is None:
            return True
        return int(row[0]) != int(mtime) or int(row[1]) != int(size)

    def mark_synced(self, path: str, mtime: int, size: int) -> None:
        ph = self._ph()
        with self._connect() as conn:
            conn.execute(f"DELETE FROM skipped_files WHERE path = {ph}", (path,))
            conn.execute(
                f"INSERT INTO skipped_files (path, mtime, size) "
                f"VALUES ({ph}, {ph}, {ph})",
                (path, mtime, size),
            )

    # ── queries ─────────────────────────────────────────────────────────
    def summary(self, **filters) -> UsageSummary:
        where, params = self._where(filters, alias="s")
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT
                      COALESCE(SUM(e.input_tokens),0),
                      COALESCE(SUM(e.output_tokens),0),
                      COALESCE(SUM(e.cache_creation_input_tokens),0),
                      COALESCE(SUM(e.cache_read_input_tokens),0),
                      COALESCE(SUM(e.reasoning_tokens),0),
                      COALESCE(SUM(e.cost_usd),0),
                      COUNT(DISTINCT s.id)
                    FROM sessions s LEFT JOIN usage_events e ON e.session_id = s.id
                    {where}""",
                params,
            )
            r = cur.fetchone()
        inp, out, cc, cr, reason, cost, sessions = (
            int(r[0]),
            int(r[1]),
            int(r[2]),
            int(r[3]),
            int(r[4]),
            float(r[5]),
            int(r[6]),
        )
        cache_total = cc + cr
        hit = (cr / cache_total) if cache_total else 0.0
        return UsageSummary(
            from_date=filters.get("from_date"),
            to_date=filters.get("to_date"),
            session_count=sessions,
            totals=TokenTotals(
                input_tokens=inp,
                output_tokens=out,
                cache_creation_tokens=cc,
                cache_read_tokens=cr,
                reasoning_tokens=reason,
                cost_usd=cost,
            ),
            cache_hit_rate=round(hit, 4),
        )

    def breakdown(self, dimension: str, **filters) -> list[BreakdownEntry]:
        col = {"model": "e.model", "project": "s.project", "agent": "s.agent"}.get(
            dimension
        )
        if col is None:
            raise ValueError(f"unknown dimension: {dimension}")
        where, params = self._where(filters, alias="s")
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT {col} AS k,
                      COUNT(DISTINCT s.id),
                      COALESCE(SUM(e.input_tokens),0),
                      COALESCE(SUM(e.output_tokens),0),
                      COALESCE(SUM(e.cost_usd),0)
                    FROM sessions s LEFT JOIN usage_events e ON e.session_id = s.id
                    {where}
                    GROUP BY {col}
                    ORDER BY COALESCE(SUM(e.cost_usd),0) DESC""",
                params,
            )
            rows = cur.fetchall()
        return [
            BreakdownEntry(
                key=(row[0] or "(unknown)"),
                session_count=int(row[1]),
                input_tokens=int(row[2]),
                output_tokens=int(row[3]),
                cost_usd=float(row[4]),
            )
            for row in rows
        ]

    def tool_stats(self, **filters) -> list[ToolStat]:
        where, params = self._where(filters, alias="s")
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT t.tool_name, t.category, COUNT(*),
                      SUM(CASE WHEN t.status IN ('error','failed','rejected')
                               THEN 0 ELSE 1 END)
                    FROM tool_calls t JOIN sessions s ON s.id = t.session_id
                    {where}
                    GROUP BY t.tool_name, t.category
                    ORDER BY COUNT(*) DESC""",
                params,
            )
            rows = cur.fetchall()
        out = []
        for row in rows:
            calls, success = int(row[2]), int(row[3])
            out.append(
                ToolStat(
                    name=row[0] or "(tool)",
                    category=row[1] or "other",
                    calls=calls,
                    success=success,
                    success_rate=round(success / calls, 4) if calls else 0.0,
                )
            )
        return out

    def activity(self, **filters) -> list[ActivityCell]:
        where, params = self._where(filters, alias="s")
        # Pull started_at + cost per session; bucket in Python for dialect-safety.
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT s.started_at, COALESCE(SUM(e.cost_usd),0)
                    FROM sessions s LEFT JOIN usage_events e ON e.session_id = s.id
                    {where}
                    GROUP BY s.id, s.started_at""",
                params,
            )
            rows = cur.fetchall()
        cells: dict[tuple[int, int], ActivityCell] = {}
        for started_at, cost in rows:
            if not started_at:
                continue
            try:
                dt = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            key = (dt.weekday(), dt.hour)
            cell = cells.get(key)
            if cell is None:
                cell = ActivityCell(day_of_week=key[0], hour=key[1])
                cells[key] = cell
            cell.sessions += 1
            cell.cost_usd += float(cost or 0.0)
        return sorted(cells.values(), key=lambda c: (c.day_of_week, c.hour))

    def _session_rows(self, where: str, params: list, order: str, limit: int):
        ph = self._ph()
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT s.id, s.project, s.agent, s.started_at, s.ended_at,
                      s.message_count, s.total_output_tokens,
                      COALESCE(SUM(e.cost_usd),0), s.health_grade, s.outcome, s.origin
                    FROM sessions s LEFT JOIN usage_events e ON e.session_id = s.id
                    {where}
                    GROUP BY s.id
                    ORDER BY {order}
                    LIMIT {ph}""",
                [*params, limit],
            )
            rows = cur.fetchall()
        return [
            SessionRow(
                id=row[0],
                project=row[1] or "",
                agent=row[2] or "claude",
                started_at=row[3],
                ended_at=row[4],
                message_count=int(row[5] or 0),
                total_output_tokens=int(row[6] or 0),
                cost_usd=float(row[7] or 0.0),
                health_grade=row[8],
                outcome=row[9] or "unknown",
                origin=row[10] or "ingested",
            )
            for row in rows
        ]

    def top_sessions(self, *, limit: int = 20, **filters) -> list[SessionRow]:
        where, params = self._where(filters, alias="s")
        return self._session_rows(
            where, params, "COALESCE(SUM(e.cost_usd),0) DESC", limit
        )

    def list_sessions(self, *, limit: int = 100, **filters) -> list[SessionRow]:
        where, params = self._where(filters, alias="s")
        return self._session_rows(where, params, "s.started_at DESC", limit)

    def session_detail(self, session_id: str) -> SessionDetail | None:
        rows = self._session_rows(
            f"WHERE s.id = {self._ph()}", [session_id], "s.started_at DESC", 1
        )
        if not rows:
            return None
        ph = self._ph()
        with self._connect() as conn:
            mcur = conn.execute(
                f"""SELECT session_id, ordinal, role, content, thinking_text,
                      timestamp, model, context_tokens, output_tokens, has_tool_use,
                      content_length
                    FROM messages WHERE session_id = {ph} ORDER BY ordinal""",
                (session_id,),
            )
            messages = [
                UsageMessage(
                    session_id=m[0],
                    ordinal=m[1],
                    role=m[2],
                    content=m[3],
                    thinking_text=m[4],
                    timestamp=m[5],
                    model=m[6],
                    context_tokens=int(m[7] or 0),
                    output_tokens=int(m[8] or 0),
                    has_tool_use=bool(m[9]),
                    content_length=int(m[10] or 0),
                )
                for m in mcur.fetchall()
            ]
            tcur = conn.execute(
                f"""SELECT session_id, message_ordinal, tool_name, category,
                      tool_use_id, input_json, skill_name, result_content_length,
                      subagent_session_id, status, occurred_at, origin, tenant_id,
                      correlation_id
                    FROM tool_calls WHERE session_id = {ph}
                    ORDER BY message_ordinal""",
                (session_id,),
            )
            tool_calls = [
                UsageToolCall(
                    session_id=t[0],
                    message_ordinal=t[1],
                    tool_name=t[2],
                    category=t[3],
                    tool_use_id=t[4],
                    input_json=t[5],
                    skill_name=t[6],
                    result_content_length=t[7],
                    subagent_session_id=t[8],
                    status=t[9] or "",
                    occurred_at=t[10],
                    origin=t[11] or "ingested",
                    tenant_id=t[12] or "",
                    correlation_id=t[13] or "",
                )
                for t in tcur.fetchall()
            ]
            ecur = conn.execute(
                f"""SELECT session_id, message_ordinal, source, model, input_tokens,
                      output_tokens, cache_creation_input_tokens,
                      cache_read_input_tokens, reasoning_tokens, cost_usd,
                      cost_status, cost_source, occurred_at, dedup_key, origin,
                      tenant_id, correlation_id
                    FROM usage_events WHERE session_id = {ph}""",
                (session_id,),
            )
            usage_events = [
                UsageEvent(
                    session_id=e[0],
                    message_ordinal=e[1],
                    source=e[2],
                    model=e[3],
                    input_tokens=int(e[4] or 0),
                    output_tokens=int(e[5] or 0),
                    cache_creation_input_tokens=int(e[6] or 0),
                    cache_read_input_tokens=int(e[7] or 0),
                    reasoning_tokens=int(e[8] or 0),
                    cost_usd=(float(e[9]) if e[9] is not None else None),
                    cost_status=e[10] or "",
                    cost_source=e[11] or "",
                    occurred_at=e[12],
                    dedup_key=e[13] or "",
                    origin=e[14] or "ingested",
                    tenant_id=e[15] or "",
                    correlation_id=e[16] or "",
                )
                for e in ecur.fetchall()
            ]
        return SessionDetail(
            session=rows[0],
            messages=messages,
            tool_calls=tool_calls,
            usage_events=usage_events,
        )
