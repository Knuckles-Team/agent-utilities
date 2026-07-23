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
import math
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
_MAX_IDENTIFIER_BYTES = 63
_MAX_COLUMNS = 512
_MAX_ROWS = 100_000
_MAX_CELL_BYTES = 1024 * 1024
_MAX_STATEMENT_BYTES = 4 * 1024 * 1024
_MAX_BATCH_SIZE = 1_000
_ALLOWED_COLUMN_TYPES = frozenset(
    {
        "BIGINT",
        "BLOB",
        "BOOLEAN",
        "BYTES",
        "DOUBLE",
        "INTEGER",
        "REAL",
        "TEXT",
        "VARCHAR",
    }
)


def _graph_compute(engine: Any) -> Any:
    """The engine's ``GraphComputeEngine`` (the SQL/RDF wire handle), or None."""
    return getattr(engine, "graph_compute", None) if engine is not None else None


def _safe_ident(name: str) -> str:
    """Validate a SQL identifier (table / column) — reject anything injectable."""
    if (
        not isinstance(name, str)
        or len(name.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
        or not _IDENT_RE.fullmatch(name)
    ):
        raise ValueError("invalid SQL identifier")
    return name


def _safe_column_type(value: str) -> str:
    """Return one engine-supported primitive type, never caller-supplied SQL."""
    normalized = str(value or "").strip().upper()
    if normalized not in _ALLOWED_COLUMN_TYPES:
        raise ValueError("invalid SQL column type")
    return normalized


def _bounded_columns(columns: list[str]) -> list[str]:
    if not isinstance(columns, list) or not 1 <= len(columns) <= _MAX_COLUMNS:
        raise ValueError("SQL column count is outside the supported limit")
    validated = [_safe_ident(column) for column in columns]
    if len(set(validated)) != len(validated):
        raise ValueError("SQL columns must be unique")
    return validated


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal (single-quote-escaped)."""
    import json as _json

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite SQL numeric values are not supported")
    if isinstance(value, int | float):
        rendered_numeric = str(value)
        if len(rendered_numeric.encode("ascii")) > _MAX_CELL_BYTES:
            raise ValueError("SQL cell exceeds the supported limit")
        return rendered_numeric
    if isinstance(value, dict | list):
        try:
            value = _json.dumps(
                value,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("SQL cell is not serializable") from exc
    rendered = str(value)
    if "\x00" in rendered or len(rendered.encode("utf-8")) > _MAX_CELL_BYTES:
        raise ValueError("SQL cell exceeds the supported limit")
    return "'" + rendered.replace("'", "''") + "'"


def ensure_table(
    engine: Any, table: str, columns: list[str], *, col_type: str = "VARCHAR"
) -> dict[str, Any]:
    """``CREATE TABLE IF NOT EXISTS`` for ``table`` with the given text columns."""
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return {"created": False, "reason": "no engine SQL surface"}
    tbl = _safe_ident(table)
    validated_columns = _bounded_columns(columns)
    validated_type = _safe_column_type(col_type)
    cols = ", ".join(f"{column} {validated_type}" for column in validated_columns)
    stmt = f"CREATE TABLE IF NOT EXISTS {tbl} ({cols})"
    gc.sql_exec(stmt)
    return {
        "created": True,
        "table": tbl,
        "columns": validated_columns,
        "ddl": stmt,
    }


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
    if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
        raise ValueError("SQL row count exceeds the supported limit")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError("SQL batch size is outside the supported limit")
    tbl = _safe_ident(table)
    cols = _bounded_columns(columns)
    col_clause = ", ".join(cols)
    written = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        pending: list[str] = []
        prefix = f"INSERT INTO {tbl} ({col_clause}) VALUES "
        prefix_bytes = len(prefix.encode("utf-8"))
        pending_bytes = 0
        for row in batch:
            if not isinstance(row, dict):
                raise ValueError("SQL rows must be mappings")
            rendered = "(" + ", ".join(_sql_literal(row.get(c)) for c in cols) + ")"
            rendered_bytes = len(rendered.encode("utf-8"))
            if prefix_bytes + rendered_bytes > _MAX_STATEMENT_BYTES:
                raise ValueError("SQL row exceeds the statement size limit")
            projected = (
                prefix_bytes + pending_bytes + rendered_bytes + (2 if pending else 0)
            )
            if projected > _MAX_STATEMENT_BYTES:
                gc.sql_exec(prefix + ", ".join(pending))
                written += len(pending)
                pending = []
                pending_bytes = 0
            pending.append(rendered)
            pending_bytes += rendered_bytes + (2 if len(pending) > 1 else 0)
        if pending:
            gc.sql_exec(prefix + ", ".join(pending))
            written += len(pending)
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
    except Exception:  # noqa: BLE001 — engine may not expose info_schema
        logger.debug("list_tables failed")
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
    strict :class:`~..etl.result.EtlResult` manifest
    (CONCEPT:AU-KG.etl.result-contract). Table diagnostics are namespaced under
    ``details`` and the row count is explicit.
    """
    from agent_utilities.knowledge_graph.etl.result import EtlResult
    from agent_utilities.protocols.source_connectors.registry import build_connector

    def _result(payload: dict[str, Any]) -> dict[str, Any]:
        canonical = {
            key: payload[key]
            for key in ("status", "counts", "watermark", "error", "reason")
            if key in payload
        }
        canonical["source"] = source
        canonical["details"] = {
            key: value
            for key, value in payload.items()
            if key not in canonical and key != "source"
        }
        return EtlResult.model_validate(canonical).model_dump()

    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return _result({"status": "skipped", "reason": "no engine SQL surface"})

    table = _safe_ident(table or f"conn_{re.sub(r'[^A-Za-z0-9_]', '_', source)}")
    if not isinstance(limit, int) or not 1 <= limit <= _MAX_ROWS:
        return _result({"status": "error", "error": "ingest limit is invalid"})
    try:
        connector = build_connector(source, config or {})
    except Exception:  # noqa: BLE001 — bad source / config
        return _result({"status": "error", "error": "connector is unavailable"})

    loader = getattr(connector, "load", None)
    if not callable(loader):
        return _result(
            {
                "status": "error",
                "error": "connector does not provide a load surface",
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
            "counts": {"rows": written},
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
    tbl = _safe_ident(table)
    if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
        raise ValueError("SQL row count exceeds the supported limit")
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return {"status": "skipped", "reason": "no engine SQL surface", "table": tbl}
    if not rows:
        return {"status": "ok", "table": tbl, "rows_written": 0, "columns": []}

    if columns is None:
        seen: dict[str, None] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("SQL rows must be mappings")
            for k in row:
                seen.setdefault(k, None)
                if len(seen) > _MAX_COLUMNS:
                    raise ValueError("SQL column count exceeds the supported limit")
        columns = list(seen)
    columns = _bounded_columns(columns)

    if replace:
        drop_table(engine, tbl)
    ensure_table(engine, tbl, columns)
    written = insert_rows(engine, tbl, rows, columns)
    return {
        "status": "ok",
        "table": tbl,
        "columns": list(columns),
        "rows_written": written,
    }
