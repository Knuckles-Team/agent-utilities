from __future__ import annotations

"""Native database traversal tools for agents (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

Gives an agent (including RLM-driven recursive agents, CONCEPT:AU-ORCH.planning.recursion-nesting-depth) the
ability to *natively traverse* a relational/NoSQL database — list tables, inspect
schema, and run live read queries — over the existing :class:`UniversalConnector`
(CONCEPT:AU-KG.ingest.enterprise-source-extractor), which speaks **PostgreSQL, MySQL/MariaDB, MS SQL Server, Oracle,
SQLite, and MongoDB**. This is a capability Onyx does not have (Onyx ships zero
database connectors); here it is both an *ingestion* source (the ``database``
document-source connector, ECO-4.25) and an *interactive* agent tool.

Safety: queries are permanently read-only and independently bounded by both the
tool and connector. Connection values are accepted only as runtime secret refs
(``vault://``, ``secret://``, or ``env://``); literal endpoints,
credentials, and host paths never enter agent text or durable configuration.
Registered under the ``DB_TOOLS`` env gate in ``tool_registry``.
"""

import json
import re

from pydantic_ai import RunContext

from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

from ..models import AgentDeps

# Interactive database tools are permanently read-only. Mutations flow through
# governed connector ChangeEnvelopes and native MutationBatch, never ad-hoc SQL.
_WRITE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|replace|merge|"
    r"call|exec|execute|attach|vacuum|reindex|pragma)\b",
    re.IGNORECASE,
)
_REF_RE = re.compile(r"^(?:vault|secret|env)://[A-Za-z0-9_./#-]+$")
_KINDS = frozenset({"postgresql", "mysql", "mssql", "oracle", "sqlite", "mongodb"})


def _resolve_dsn(dsn: str) -> str:
    """Resolve a runtime secret reference; literal DSNs/paths are forbidden."""
    reference = str(dsn or "").strip()
    if not _REF_RE.fullmatch(reference):
        raise ValueError("database connection must be a runtime secret reference")
    from agent_utilities.security.secrets_client import create_secrets_client

    value = create_secrets_client().resolve_ref(reference)
    rendered = value.decode("utf-8") if isinstance(value, bytes) else str(value or "")
    if not rendered or len(rendered.encode("utf-8")) > 8_192 or "\x00" in rendered:
        raise ValueError("database connection reference is unavailable")
    return rendered


def _connect(dsn: str, kind: str | None):
    from ..protocols.universal_connector import UniversalConnector

    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind and normalized_kind not in _KINDS:
        raise ValueError("database kind is not supported")
    return UniversalConnector(_resolve_dsn(dsn), kind=normalized_kind or None)


def _safe_payload(value: object) -> object:
    clean, _ = PersistencePrivacyGuard().sanitize(value)
    return clean


async def db_tables(ctx: RunContext[AgentDeps], dsn: str, kind: str = "") -> str:
    """List the tables/collections of a database (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

    Args:
        ctx: The agent run context.
        dsn: Runtime secret reference resolving to a connection profile/DSN.
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
        return json.dumps({"tables": _safe_payload(tables)})
    except Exception:  # noqa: BLE001
        return json.dumps({"error": "database introspection failed"})


async def db_schema(ctx: RunContext[AgentDeps], dsn: str, kind: str = "") -> str:
    """Return the schema (tables → columns) of a database (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

    Args:
        ctx: The agent run context.
        dsn: Runtime secret reference (see :func:`db_tables`).
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
        return json.dumps({"schema": _safe_payload(schema)})
    except Exception:  # noqa: BLE001
        return json.dumps({"error": "database schema introspection failed"})


async def db_query(
    ctx: RunContext[AgentDeps],
    dsn: str,
    query: str,
    kind: str = "",
    limit: int = 200,
) -> str:
    """Run a read-only query and return rows (CONCEPT:AU-ECO.toolkit.database-traversal-tools).

    Lets an agent natively traverse a database. Only one bounded read query is
    permitted; mutations are always rejected.

    Args:
        ctx: The agent run context.
        dsn: Runtime secret reference (see :func:`db_tables`).
        query: The SQL (or Mongo find JSON) to execute.
        kind: Optional explicit backend kind.
        limit: Max rows to return (truncates large results for the context window).

    Returns:
        JSON ``{rows: [...], row_count, truncated}``, or ``{error}``.
    """
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query.encode("utf-8")) > 65_536
    ):
        return json.dumps({"error": "empty query"})
    if not isinstance(limit, int) or not 1 <= limit <= 1_000:
        return json.dumps({"error": "query limit is invalid"})
    is_write = bool(_WRITE_RE.search(query))
    normalized_kind = str(kind or "").strip().lower()
    if is_write or (
        normalized_kind != "mongodb" and not re.match(r"^\s*select\b", query, re.I)
    ):
        return json.dumps({"error": "interactive database tools are read-only"})
    if normalized_kind != "mongodb" and any(
        token in query for token in (";", "--", "/*", "*/")
    ):
        return json.dumps(
            {"error": "multi-statement or commented queries are not supported"}
        )
    try:
        conn = _connect(dsn, kind)
        rows = [
            row for row in conn.read(query, max_rows=limit + 1) if isinstance(row, dict)
        ]
        truncated = len(rows) > limit
        page = rows[:limit]
        out = _safe_payload(page)
        return json.dumps(
            {"rows": out, "row_count": len(page), "truncated": truncated}, default=str
        )
    except Exception:  # noqa: BLE001
        return json.dumps({"error": "database query failed"})


db_tools = [db_tables, db_schema, db_query]
