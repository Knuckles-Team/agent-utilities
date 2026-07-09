#!/usr/bin/python
from __future__ import annotations

"""Connector / ETL → native engine SQL tables (CONCEPT:AU-KG.ingest.mirror-inbound).

The engine gained arbitrary user tables (DataFusion + pg-wire). This module makes
that reachable from the platform: mirror any registered source connector's documents
— or arbitrary ETL output rows — into a native engine SQL table via ``CREATE TABLE``
+ bulk ``INSERT``, so "ingest tables from any connector / mirror data into our DB"
works end-to-end.

It composes the parts that already exist:

* **Connector registry** (``protocols.source_connectors``) — ``build_connector`` +
  the ``LoadConnector.load()`` mixin yield ``SourceDocument`` rows for any registered
  connector (rest / database / web / rss / filesystem / reader / mcp_tool / …).
* **Engine SQL write surface** — ``GraphComputeEngine.sql_exec`` (KG-2.266) runs the
  ``CREATE TABLE`` / ``INSERT`` DDL/DML against the engine's user-table store; reads
  go back through the engine's read-only ``sql`` surface.

The write/read engine handle is reached the same way the ontology lifecycle reaches
it — ``engine.graph_compute`` (the ``GraphComputeEngine``) — so this stays mockable
and engine-optional.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# The flattened columns every mirrored connector document gets. ``metadata`` is kept
# as a JSON-encoded text column so heterogeneous provenance survives without an
# unbounded schema.
_DOC_COLUMNS: list[str] = [
    "id",
    "source_uri",
    "title",
    "text",
    "doc_type",
    "updated_at",
    "metadata",
]

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _graph_compute(engine: Any) -> Any:
    """The engine's ``GraphComputeEngine`` (the SQL/RDF wire handle), or None."""
    return getattr(engine, "graph_compute", None) if engine is not None else None


def _safe_ident(name: str) -> str:
    """Validate a SQL identifier (table / column) — reject anything injectable."""
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal (single-quote-escaped)."""
    import json as _json

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict | list):
        value = _json.dumps(value, default=str)
    return "'" + str(value).replace("'", "''") + "'"


def ensure_table(
    engine: Any, table: str, columns: list[str], *, col_type: str = "VARCHAR"
) -> dict[str, Any]:
    """``CREATE TABLE IF NOT EXISTS`` for ``table`` with the given text columns."""
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return {"created": False, "reason": "no engine SQL surface"}
    tbl = _safe_ident(table)
    cols = ", ".join(f"{_safe_ident(c)} {col_type}" for c in columns)
    stmt = f"CREATE TABLE IF NOT EXISTS {tbl} ({cols})"
    gc.sql_exec(stmt)
    return {"created": True, "table": tbl, "columns": list(columns), "ddl": stmt}


def insert_rows(
    engine: Any,
    table: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    batch_size: int = 500,
) -> int:
    """Bulk-``INSERT`` ``rows`` into ``table`` (multi-row VALUES, batched)."""
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec") or not rows:
        return 0
    tbl = _safe_ident(table)
    cols = [_safe_ident(c) for c in columns]
    col_clause = ", ".join(cols)
    written = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        values = ", ".join(
            "(" + ", ".join(_sql_literal(row.get(c)) for c in columns) + ")"
            for row in batch
        )
        gc.sql_exec(f"INSERT INTO {tbl} ({col_clause}) VALUES {values}")
        written += len(batch)
    return written


def list_tables(engine: Any) -> list[str]:
    """List the engine's user SQL tables (best-effort over information_schema)."""
    gc = _graph_compute(engine)
    sql_fn = getattr(engine, "sql", None)
    if gc is None or not callable(sql_fn):
        return []
    try:
        rows = sql_fn(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('information_schema')"
        )
        return [
            str(name)
            for r in rows
            if isinstance(r, dict) and (name := r.get("table_name")) is not None
        ]
    except Exception as exc:  # noqa: BLE001 — engine may not expose info_schema
        logger.debug("list_tables failed: %s", exc)
        return []


def drop_table(engine: Any, table: str) -> dict[str, Any]:
    """``DROP TABLE IF EXISTS`` for ``table``."""
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return {"dropped": False, "reason": "no engine SQL surface"}
    tbl = _safe_ident(table)
    gc.sql_exec(f"DROP TABLE IF EXISTS {tbl}")
    return {"dropped": True, "table": tbl}


def _flatten_doc(doc: Any) -> dict[str, Any]:
    """A ``SourceDocument`` → flat row dict over :data:`_DOC_COLUMNS`."""
    import json as _json

    get = (lambda k: getattr(doc, k, None)) if not isinstance(doc, dict) else doc.get
    meta = get("metadata") or {}
    return {
        "id": get("id"),
        "source_uri": get("source_uri") or "",
        "title": get("title") or "",
        "text": get("text") or "",
        "doc_type": get("doc_type") or "document",
        "updated_at": get("updated_at"),
        "metadata": _json.dumps(meta, default=str) if meta else "{}",
    }


def ingest_connector_to_table(
    engine: Any,
    source: str,
    *,
    table: str | None = None,
    config: dict[str, Any] | None = None,
    limit: int = 1000,
    replace: bool = False,
) -> dict[str, Any]:
    """Mirror a registered connector's documents into a native engine SQL table.

    Builds ``source`` from the connector registry, drains its ``load()`` documents
    (capped at ``limit``), flattens each into a row, then ``CREATE TABLE`` +
    bulk ``INSERT`` into ``table`` (defaults to ``conn_<source>``). Returns a
    manifest coerced through :class:`~..etl.result.EtlResult`
    (CONCEPT:AU-KG.etl.result-contract) — a validated ``status``/``counts``
    contract instead of a hand-assembled dict, still a plain ``dict`` at the
    return boundary so existing callers keep indexing it unchanged.
    """
    from agent_utilities.knowledge_graph.etl.result import EtlResult
    from agent_utilities.protocols.source_connectors.registry import build_connector

    def _result(payload: dict[str, Any]) -> dict[str, Any]:
        return EtlResult.coerce(payload, source=source).model_dump()

    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return _result({"status": "skipped", "reason": "no engine SQL surface"})

    table = table or f"conn_{re.sub(r'[^A-Za-z0-9_]', '_', source)}"
    try:
        connector = build_connector(source, config or {})
    except Exception as exc:  # noqa: BLE001 — bad source / config
        return _result({"status": "error", "error": str(exc)})

    loader = getattr(connector, "load", None)
    if not callable(loader):
        return _result(
            {
                "status": "error",
                "error": f"connector {source!r} has no load() surface (not a LoadConnector)",
            }
        )

    rows: list[dict[str, Any]] = []
    for doc in loader():
        rows.append(_flatten_doc(doc))
        if len(rows) >= limit:
            break

    if replace:
        drop_table(engine, table)
    ensure_table(engine, table, _DOC_COLUMNS)
    written = insert_rows(engine, table, rows, _DOC_COLUMNS)
    return _result(
        {
            "status": "ok",
            "table": _safe_ident(table),
            "columns": list(_DOC_COLUMNS),
            "rows_seen": len(rows),
            "rows_written": written,
        }
    )


def ingest_rows_to_table(
    engine: Any,
    table: str,
    rows: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Mirror arbitrary ETL output rows into a native engine SQL table.

    Generic sink for ETL pipelines: infers columns from the row keys (or uses the
    supplied ``columns``), then ``CREATE TABLE`` + bulk ``INSERT``.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return {"status": "skipped", "reason": "no engine SQL surface", "table": table}
    if not rows:
        return {"status": "ok", "table": table, "rows_written": 0, "columns": []}

    if columns is None:
        seen: dict[str, None] = {}
        for row in rows:
            for k in row:
                seen.setdefault(k, None)
        columns = list(seen)

    if replace:
        drop_table(engine, table)
    ensure_table(engine, table, columns)
    written = insert_rows(engine, table, rows, columns)
    return {
        "status": "ok",
        "table": _safe_ident(table),
        "columns": list(columns),
        "rows_written": written,
    }
