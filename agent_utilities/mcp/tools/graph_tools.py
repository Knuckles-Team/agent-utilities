"""Read-only SQL **catalog introspection** for the engine's SQL surface.

CONCEPT:AU-KG.query.raw-python — SQL over the KG (read path)
CONCEPT:AU-ECO.mcp.full-api-mcp-surface — one action core, two surfaces

The webui catalog browser (``agent-webui`` ``TableExplorerView``) needs the
``catalogs → tables/views → columns`` projection that ``psql``'s ``\\d`` shows.
Until now the browser could reach *no* route that touches ``information_schema``
at all, so it rendered its "capability not yet activated" state forever
(``plans/semantic-indexing/DESIGN-embedding-bindings.md`` §4).

**This module carries no user-supplied SQL, by construction.** The obvious fix —
a raw ``POST /graph/sql-query`` passthrough — would hand a browser arbitrary SQL
against the tenant catalog: a far larger surface than a schema tree needs, with
an injection surface and a path to row data. Instead:

* every statement issued here is a module-level **constant** (:data:`CATALOG_STATEMENTS`).
  There is no f-string, no ``%``, no ``.format()``, and therefore no identifier
  or literal interpolation anywhere on this path;
* the one caller-supplied control — an optional schema-name filter — is
  validated as a SQL identifier (:func:`validate_schema_filter`) and then
  applied **in Python**, over rows already returned. It never reaches the
  engine, so even a validator bug cannot become an injection;
* only the engine's synthesized read-only ``information_schema`` relations are
  read; no user table is touched.

Dispatch goes through the ONE shared action core — ``kg_server._execute_tool``
against the already-registered ``graph_table`` tool's read-only ``query``
action — so the REST route added in :mod:`agent_utilities.gateway.graph_api` and
the MCP surface execute the same code, under the same session/RLS enforcement,
with no second implementation to drift.

Engine fidelity (verified against ``epistemic-graph``
``crates/eg-query/src/sql/catalog.rs``, read 2026-08-27) — reported honestly in
the response's ``capabilities`` block rather than fabricated:

* ``information_schema.key_column_usage`` / ``.table_constraints`` are shaped
  but **empty**: PK/UNIQUE metadata lives in the redb table store and is not
  threaded into the per-query catalog build. So ``primary_key`` is ``None``
  ("unknown"), never a confident ``false``.
* ``information_schema.columns.is_nullable`` is hardcoded ``'YES'`` for every
  column — nullability is not tracked structurally by the schema-on-read
  projection. ``nullable`` is passed through, and ``capabilities.nullability``
  says it is not authoritative.

Both flags flip to ``True`` automatically, with no change here, once the engine
populates those relations.
"""

from __future__ import annotations

import json
import logging
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

#: The fixed, server-authored catalog statements. Constants on purpose: the only
#: SQL this module can ever issue is one of these three strings, verbatim.
CATALOG_STATEMENTS: Any = MappingProxyType(
    {
        "tables": (
            "SELECT table_catalog, table_schema, table_name, table_type "
            "FROM information_schema.tables"
        ),
        "columns": (
            "SELECT table_catalog, table_schema, table_name, column_name, "
            "ordinal_position, is_nullable, data_type, udt_name "
            "FROM information_schema.columns"
        ),
        "primary_keys": (
            "SELECT table_catalog, table_schema, table_name, column_name "
            "FROM information_schema.key_column_usage"
        ),
    }
)

_TABLE_KINDS = MappingProxyType({"BASE TABLE": "table", "VIEW": "view"})


class SqlSchemaUnavailable(RuntimeError):
    """Typed, fail-closed catalog-introspection error.

    Never raised to report "healthy, nothing there": an empty projection is
    indistinguishable at the call site from a broken engine read, which is the
    exact failure mode AGENTS.md's "Fail closed" section exists to stop.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "catalog_unavailable",
        status_code: int = 503,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    def as_payload(self) -> dict[str, Any]:
        """The wire error envelope (no engine internals, no caller echo)."""
        return {"status": "error", "code": self.code, "message": str(self)}


def validate_schema_filter(value: Any) -> str | None:
    """Return a validated schema name, or ``None`` for "every schema".

    The result is used as a **Python** comparison key, never as SQL text; the
    validation is defence in depth so a hostile value is rejected at the edge
    rather than travelling further into the projection.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise SqlSchemaUnavailable(
            "schema filter must be a string",
            code="invalid_schema_filter",
            status_code=422,
        )
    text = value.strip()
    if not text:
        return None
    from agent_utilities.security.identifiers import (
        InvalidIdentifierError,
        validate_sql_identifier,
    )

    try:
        return validate_sql_identifier(text, kind="schema")
    except InvalidIdentifierError as exc:
        raise SqlSchemaUnavailable(
            "schema filter is not a valid SQL identifier",
            code="invalid_schema_filter",
            status_code=422,
        ) from exc


def _decode_rows(raw: Any) -> list[dict[str, Any]] | None:
    """Rows from a ``graph_table`` result, or ``None`` when the read failed.

    ``graph_table`` returns a JSON string: a list of row objects on success, or
    a single ``{"error": ...}`` object on failure. Distinguishing those two is
    what makes the caller able to fail closed instead of reporting "no tables".
    """
    payload = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except ValueError:
            logger.warning("sql-schema: engine returned non-JSON catalog payload")
            return None
    if not isinstance(payload, list):
        return None
    return [row for row in payload if isinstance(row, dict)]


async def _read_catalog(key: str) -> list[dict[str, Any]] | None:
    """Run ONE constant catalog statement through the shared action core."""
    from agent_utilities.mcp import kg_server

    try:
        raw = await kg_server._execute_tool(
            "graph_table", action="query", sql=CATALOG_STATEMENTS[key]
        )
    except Exception as exc:  # noqa: BLE001 — mapped to a typed, fail-closed error
        raise SqlSchemaUnavailable(
            f"engine SQL catalog read failed ({type(exc).__name__})"
        ) from exc
    return _decode_rows(raw)


def _relation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("table_catalog") or ""),
        str(row.get("table_schema") or ""),
        str(row.get("table_name") or ""),
    )


def _group_by_relation(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_relation_key(row), []).append(row)
    return grouped


def _primary_key_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], set[str]]:
    index: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        name = str(row.get("column_name") or "")
        if name:
            index.setdefault(_relation_key(row), set()).add(name)
    return index


def _column_entry(row: dict[str, Any], pk_names: set[str] | None) -> dict[str, Any]:
    name = str(row.get("column_name") or "")
    try:
        position = int(row.get("ordinal_position") or 0)
    except (TypeError, ValueError):
        position = 0
    return {
        "name": name,
        "position": position,
        "data_type": str(row.get("data_type") or ""),
        "udt_name": str(row.get("udt_name") or "") or None,
        "nullable": str(row.get("is_nullable") or "").strip().upper() == "YES",
        "primary_key": (name in pk_names) if pk_names is not None else None,
    }


def _relation_entry(
    row: dict[str, Any],
    columns_by_relation: dict[tuple[str, str, str], list[dict[str, Any]]],
    pk_index: dict[tuple[str, str, str], set[str]],
    *,
    primary_keys_supported: bool,
) -> dict[str, Any]:
    key = _relation_key(row)
    pk_names = pk_index.get(key, set()) if primary_keys_supported else None
    columns = [
        _column_entry(column_row, pk_names)
        for column_row in columns_by_relation.get(key, [])
    ]
    columns.sort(key=lambda entry: (entry["position"], entry["name"]))
    raw_type = str(row.get("table_type") or "").strip().upper()
    return {
        "catalog": key[0],
        "schema": key[1],
        "name": key[2],
        "kind": _TABLE_KINDS.get(raw_type, "table"),
        "table_type": raw_type or "BASE TABLE",
        "columns": columns,
    }


def _nest_by_catalog(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``[relation]`` → ``[{catalog, schemas: [{schema, tables: [...]}]}]``."""
    catalogs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for relation in relations:
        schemas = catalogs.setdefault(relation["catalog"], {})
        schemas.setdefault(relation["schema"], []).append(relation)
    return [
        {
            "catalog": catalog,
            "schemas": [
                {"schema": schema, "tables": sorted(tables, key=_relation_sort_key)}
                for schema, tables in sorted(schemas.items())
            ],
        }
        for catalog, schemas in sorted(catalogs.items())
    ]


def _relation_sort_key(relation: dict[str, Any]) -> tuple[str, str]:
    return (relation["kind"], relation["name"])


def _counts(catalogs: list[dict[str, Any]]) -> dict[str, int]:
    schemas = [schema for catalog in catalogs for schema in catalog["schemas"]]
    tables = [table for schema in schemas for table in schema["tables"]]
    return {
        "catalogs": len(catalogs),
        "schemas": len(schemas),
        "tables": len(tables),
        "columns": sum(len(table["columns"]) for table in tables),
    }


def build_projection(
    table_rows: list[dict[str, Any]],
    column_rows: list[dict[str, Any]],
    pk_rows: list[dict[str, Any]],
    *,
    schema_filter: str | None = None,
) -> dict[str, Any]:
    """Shape three ``information_schema`` reads into the browser projection."""
    columns_by_relation = _group_by_relation(column_rows)
    pk_index = _primary_key_index(pk_rows)
    primary_keys_supported = bool(pk_index)
    relations = [
        _relation_entry(
            row,
            columns_by_relation,
            pk_index,
            primary_keys_supported=primary_keys_supported,
        )
        for row in table_rows
        if schema_filter is None or str(row.get("table_schema") or "") == schema_filter
    ]
    catalogs = _nest_by_catalog(relations)
    return {
        "status": "success",
        "catalogs": catalogs,
        "capabilities": {
            "primary_keys": primary_keys_supported,
            "nullability": False,
        },
        "counts": _counts(catalogs),
    }


async def sql_schema(*, schema: str | None = None) -> dict[str, Any]:
    """The ``catalogs → tables/views → columns`` projection. No caller SQL.

    Args:
        schema: Optional schema-name filter, validated as a SQL identifier and
            matched in Python. ``None`` returns every readable schema.

    Raises:
        SqlSchemaUnavailable: the filter was malformed, the engine catalog was
            unreadable, or it produced nothing — the engine always synthesizes
            at least ``nodes``/``edges``, so an empty relation list means a
            failed read, not an empty database.
    """
    schema_filter = validate_schema_filter(schema)

    table_rows = await _read_catalog("tables")
    if not table_rows:
        raise SqlSchemaUnavailable(
            "engine SQL catalog returned no relations; the synthesized "
            "information_schema is unreadable"
        )
    column_rows = await _read_catalog("columns")
    if column_rows is None:
        raise SqlSchemaUnavailable("engine column catalog is unreadable")
    pk_rows = await _read_catalog("primary_keys")

    projection = build_projection(
        table_rows, column_rows, pk_rows or [], schema_filter=schema_filter
    )
    if schema_filter is not None and not projection["catalogs"]:
        raise SqlSchemaUnavailable(
            "no readable schema matches that name",
            code="schema_not_found",
            status_code=404,
        )
    return projection


__all__ = [
    "CATALOG_STATEMENTS",
    "SqlSchemaUnavailable",
    "build_projection",
    "sql_schema",
    "validate_schema_filter",
]
