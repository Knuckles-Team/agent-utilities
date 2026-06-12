#!/usr/bin/python
"""Apache AGE graph backend — real openCypher on PostgreSQL.

CONCEPT:KG-2.7 — the durable PostgreSQL graph tier executed via a bounded regex
Cypher→SQL transpiler that silently returned ``[]`` for any unsupported shape
(``count(r)``, ``RETURN ... AS alias``, multi-hop, variable-length). This backend
runs Cypher natively through Apache AGE's ``cypher()`` function, so the full
openCypher surface the engine emits works, with **fail-loud** behaviour on a
genuine error instead of a silent empty result.

Storage split:
  * graph (nodes/edges + properties) → AGE graph ``agent_graph`` (agtype).
  * embeddings → a ``kg_embeddings`` side table (pgvector), keyed by node id.

Selected via ``GRAPH_PG_AGE=1`` (or ``backend_type="age"``) — see ``create_backend``.
Requires the AGE + pgvector image (``docker/pggraph-age.compose.yml``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .postgresql_backend import PostgreSQLBackend

logger = logging.getLogger(__name__)

# `name`/`prop` backtick-quoted identifiers the engine emits — AGE uses plain
# identifiers, so strip the backticks (our identifiers are always ``\w+``).
_BACKTICK_IDENT = re.compile(r"`(\w+)`")
# A ``$param`` placeholder.
_PARAM = re.compile(r"\$([A-Za-z_]\w*)")
# Tail clauses that follow the RETURN projection.
_RETURN_TAIL = re.compile(r"\s+(ORDER\s+BY|LIMIT|SKIP)\b", re.IGNORECASE)


def _cypher_literal(value: Any) -> str:
    """Render a Python value as an openCypher literal for inlining into AGE."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_cypher_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        # Store maps as a JSON string property (parity with the other backends,
        # which JSON-encode nested values).
        value = json.dumps(value, default=str)
    s = str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"'{s}'"


def _parse_agtype(raw: Any) -> Any:
    """Convert an AGE ``agtype`` cell into a plain Python value.

    Vertices/edges (``{...}::vertex`` / ``::edge``) collapse to their ``properties``
    dict so callers see node/edge properties directly (id, name, …) — matching the
    other backends' read shape.
    """
    if raw is None or not isinstance(raw, str):
        return raw
    s = raw
    for suffix in ("::vertex", "::edge", "::path"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return raw
    if (
        isinstance(obj, dict)
        and "properties" in obj
        and ("label" in obj or "id" in obj)
    ):
        return dict(obj.get("properties") or {})
    return obj


class AGEBackend(PostgreSQLBackend):
    """openCypher-on-Postgres via Apache AGE (graph) + pgvector (embeddings)."""

    @property
    def cypher_support(self) -> str:
        """Apache AGE runs native openCypher (count/alias/multi-hop/variable-length/
        edge-props), so the full query surface is portable here (CONCEPT:KG-2.63)."""
        return "full"

    # ── Cypher → AGE translation ─────────────────────────────────────────────

    def _inline_params(self, cypher: str, params: dict[str, Any]) -> str:
        """Inline ``$param`` placeholders as Cypher literals.

        AGE's native parameter passing is finicky outside prepared statements, so
        we substitute escaped literals (queries are engine-generated and internal).
        """

        def _sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in params:
                return m.group(0)  # leave unknown placeholders (e.g. _clearance_level)
            return _cypher_literal(params[name])

        return _PARAM.sub(_sub, cypher)

    @staticmethod
    def _return_columns(cypher: str) -> list[str]:
        """Parse the RETURN projection into AGE result column names.

        Returns ``[]`` for a query with no RETURN (a write) — the caller then uses
        a single throwaway ``AS (result agtype)`` column.
        """
        idx = cypher.upper().rfind("RETURN ")
        if idx == -1:
            return []
        tail = cypher[idx + len("RETURN ") :]
        cut = _RETURN_TAIL.search(tail)
        if cut:
            tail = tail[: cut.start()]
        tail = re.sub(r"^\s*DISTINCT\s+", "", tail, flags=re.IGNORECASE)

        cols: list[str] = []
        depth = 0
        cur = ""
        for ch in tail:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                cols.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            cols.append(cur)

        names: list[str] = []
        for i, raw in enumerate(cols):
            am = re.search(r"\bAS\s+(\w+)\s*$", raw.strip(), re.IGNORECASE)
            if am:
                names.append(am.group(1))
                continue
            # No explicit alias: a bare variable (``RETURN n``) keeps its name;
            # an expression gets a synthetic column name.
            expr = raw.strip()
            names.append(expr if re.fullmatch(r"\w+", expr) else f"col{i}")
        return names

    def _age_session(self, cur: Any) -> None:
        """Prepare a connection to run AGE Cypher (idempotent, per checkout)."""
        cur.execute("LOAD 'age'")
        cur.execute('SET search_path = ag_catalog, "$user", public')

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        cypher = (query or "").strip()
        if not cypher:
            return []
        cypher = _BACKTICK_IDENT.sub(r"\1", cypher)
        inlined = self._inline_params(cypher, params)

        cols = self._return_columns(inlined)
        col_def = ", ".join(f"{c} agtype" for c in cols) if cols else "result agtype"
        # Dollar-quote with a tag unlikely to collide with the (escaped) Cypher.
        sql = (
            f"SELECT * FROM cypher('{self._graph_name}', $ag${inlined}$ag$) "
            f"AS ({col_def})"
        )
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._age_session(cur)
                cur.execute(sql)
                rows = cur.fetchall()  # no-RETURN writes yield 0 rows, not an error
                colnames = [d.name for d in cur.description] if cur.description else []
            conn.commit()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({colnames[i]: _parse_agtype(v) for i, v in enumerate(row)})
        return out

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in batch:
            results.extend(self.execute(query, item))
        return results

    # ── Schema / embeddings ──────────────────────────────────────────────────

    def create_schema(self) -> None:
        """Ensure the AGE graph + pgvector embeddings table exist (idempotent)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS age CASCADE")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self._age_session(cur)
                cur.execute(
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
                    (self._graph_name,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        "SELECT ag_catalog.create_graph(%s)", (self._graph_name,)
                    )
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS kg_embeddings "
                    "(node_id TEXT PRIMARY KEY, embedding vector(768))"
                )
            conn.commit()

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        vec = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO kg_embeddings (node_id, embedding) VALUES (%s, %s::vector) "
                    "ON CONFLICT (node_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                    (node_id, vec),
                )
            conn.commit()

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        vec = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT node_id, 1 - (embedding <=> %s::vector) AS score "
                    "FROM kg_embeddings ORDER BY embedding <=> %s::vector LIMIT %s",
                    (vec, vec, n_results),
                )
                rows = cur.fetchall()
            conn.commit()
        return [{"id": r[0], "score": float(r[1])} for r in rows]
