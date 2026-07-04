from __future__ import annotations

"""Native database traversal tools for agents (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

Gives an agent (including RLM-driven recursive agents, CONCEPT:AU-ORCH.planning.recursion-nesting-depth) the
ability to *natively traverse* a relational/NoSQL database — list tables, inspect
schema, and run live read queries — over the existing :class:`UniversalConnector`
(CONCEPT:AU-KG.ingest.enterprise-source-extractor), which speaks **PostgreSQL, MySQL/MariaDB, MS SQL Server, Oracle,
SQLite, and MongoDB**. This is a capability Onyx does not have (Onyx ships zero
database connectors); here it is both an *ingestion* source (the ``database``
document-source connector, ECO-4.25) and an *interactive* agent tool.

Safety: queries are **read-only by default** — a deny-list blocks DDL/DML
(``INSERT``/``UPDATE``/``DELETE``/``DROP``/``ALTER``/…) unless the operator opts
into writes via ``DB_TOOLS_ALLOW_WRITE=1``. Connection strings come from the
caller or ``{ALIAS}_DSN`` env vars so secrets are never embedded in agent text.
Registered under the ``DB_TOOLS`` env gate in ``tool_registry``.
"""

import json
import logging
import os
import re

from pydantic_ai import RunContext

from agent_utilities.core.config import setting

from ..models import AgentDeps

logger = logging.getLogger(__name__)

# Statements that mutate data/schema — blocked unless writes are explicitly allowed.
_WRITE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|replace|merge|"
    r"call|exec|execute|attach|vacuum|reindex|pragma)\b",
    re.IGNORECASE,
)


def _allow_write() -> bool:
    return setting("DB_TOOLS_ALLOW_WRITE", "0").lower() in ("1", "true", "yes")


def _resolve_dsn(dsn: str) -> str:
    """Resolve a DSN, or an ``{ALIAS}`` that maps to a ``{ALIAS}_DSN`` env var.

    So an agent can reference ``warehouse`` and the operator sets
    ``WAREHOUSE_DSN=postgresql://…`` without the secret ever entering a prompt.
    """
    if "://" in dsn or dsn.startswith("sqlite") or os.path.exists(dsn):
        return dsn
    env_key = f"{dsn.upper().replace('-', '_')}_DSN"
    return setting(env_key, dsn)


def _connect(dsn: str, kind: str | None):
    from ..protocols.universal_connector import UniversalConnector

    return UniversalConnector(_resolve_dsn(dsn), kind=kind or None)


async def db_tables(ctx: RunContext[AgentDeps], dsn: str, kind: str = "") -> str:
    """List the tables/collections of a database (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

    Args:
        ctx: The agent run context.
        dsn: Connection string (``postgresql://…``, ``mysql://…`` (MariaDB),
            ``mssql://…``, ``oracle://…``, ``sqlite:///path.db``, ``mongodb://…``)
            or an alias resolving to ``{ALIAS}_DSN``.
        kind: Optional explicit backend kind (inferred from the DSN otherwise).

    Returns:
        JSON ``{tables: [...]}`` (the introspected entity names), or ``{error}``.
    """
    try:
        batch = _connect(dsn, kind).introspect()
        tables = sorted(
            n.props.get("name", "")
            for n in batch.nodes
            if n.type in ("Table", "Collection") and n.props.get("name")
        )
        return json.dumps({"tables": tables})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


async def db_schema(ctx: RunContext[AgentDeps], dsn: str, kind: str = "") -> str:
    """Return the schema (tables → columns) of a database (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

    Args:
        ctx: The agent run context.
        dsn: Connection string or ``{ALIAS}`` (see :func:`db_tables`).
        kind: Optional explicit backend kind.

    Returns:
        JSON ``{schema: {table: [columns...]}}``, or ``{error}``.
    """
    try:
        batch = _connect(dsn, kind).introspect()
        by_id = {n.id: n for n in batch.nodes}
        tables = {
            n.id: n.props.get("name", n.id)
            for n in batch.nodes
            if n.type in ("Table", "Collection")
        }
        schema: dict[str, list[str]] = {name: [] for name in tables.values()}
        for e in batch.edges:
            tgt = by_id.get(e.target)
            if (
                tgt is not None
                and tgt.type in ("Column", "Field")
                and e.source in tables
            ):
                col = tgt.props.get("name")
                if col and col not in schema[tables[e.source]]:
                    schema[tables[e.source]].append(col)
        return json.dumps({"schema": schema})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


async def db_query(
    ctx: RunContext[AgentDeps],
    dsn: str,
    query: str,
    kind: str = "",
    limit: int = 200,
) -> str:
    """Run a read-only query and return rows (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

    Lets an agent natively traverse a database. By default only read queries are
    permitted; mutating statements are rejected unless ``DB_TOOLS_ALLOW_WRITE=1``.

    Args:
        ctx: The agent run context.
        dsn: Connection string or ``{ALIAS}`` (see :func:`db_tables`).
        query: The SQL (or Mongo find JSON) to execute.
        kind: Optional explicit backend kind.
        limit: Max rows to return (truncates large results for the context window).

    Returns:
        JSON ``{rows: [...], row_count, truncated}``, or ``{error}``.
    """
    if not query.strip():
        return json.dumps({"error": "empty query"})
    is_write = bool(_WRITE_RE.search(query))
    if is_write and not _allow_write():
        return json.dumps(
            {
                "error": "write/DDL statements are blocked; set DB_TOOLS_ALLOW_WRITE=1 to permit"
            }
        )
    try:
        conn = _connect(dsn, kind)
        if is_write:
            # Route mutations through write() so they are committed (read() does not).
            affected = conn.write(query)
            return json.dumps({"rows": [], "row_count": 0, "rows_affected": affected})
        rows = [r for r in conn.read(query) if isinstance(r, dict)]
        truncated = len(rows) > limit
        out = rows[:limit]
        return json.dumps(
            {"rows": out, "row_count": len(out), "truncated": truncated}, default=str
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


db_tools = [db_tables, db_schema, db_query]
