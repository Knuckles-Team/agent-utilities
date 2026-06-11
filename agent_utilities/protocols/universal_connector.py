from __future__ import annotations

"""Universal multi-DB / GraphQL DataConnector with KG schema introspection.

CONCEPT:KG-2.9 — Universal DataConnector

A single, driver-based connector that speaks SQL (PostgreSQL, MySQL, MS SQL,
Oracle, SQLite), MongoDB, and GraphQL through one uniform surface:
``read`` / ``write`` / ``update`` / ``health_check`` plus schema
``introspect`` that emits the data source's schema as Knowledge-Graph nodes and
edges (``DataSource`` → ``Table``/``Collection``/``GraphQLType`` → ``Column``/
``Field`` with ``HAS_TABLE`` / ``HAS_COLUMN`` / ``FOREIGN_KEY`` relationships).

This module is **importable with no database drivers installed**: every driver
import is performed lazily inside :meth:`UniversalConnector._driver` and a
missing driver raises a clear :class:`RuntimeError` (never an ``ImportError``
leak). It emits the same backend-agnostic ``ExtractionBatch`` shape that the KG
enrichment pipeline (``knowledge_graph.enrichment.registry.write_batch``)
already persists.

Usage::

    from agent_utilities.protocols.universal_connector import UniversalConnector

    conn = UniversalConnector("sqlite:///./app.db")     # kind inferred
    rows = conn.read("SELECT * FROM users WHERE id = ?", (1,))
    conn.write("INSERT INTO users(name) VALUES (?)", ("ada",))
    batch = conn.introspect()                            # -> ExtractionBatch
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from ..knowledge_graph.enrichment.models import (
    EnrichmentEdge,
    ExtractionBatch,
    GraphNode,
)

logger = logging.getLogger(__name__)

# Supported backend kinds.
SQL_KINDS = frozenset({"postgresql", "mysql", "mssql", "oracle", "sqlite"})
SUPPORTED_KINDS = SQL_KINDS | frozenset({"mongodb", "graphql"})

# Map common DSN URL schemes to a canonical ``kind``.
_SCHEME_TO_KIND: dict[str, str] = {
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "psql": "postgresql",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "mssql",
    "sqlserver": "mssql",
    "oracle": "oracle",
    "oracledb": "oracle",
    "sqlite": "sqlite",
    "file": "sqlite",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "graphql": "graphql",
    "http": "graphql",
    "https": "graphql",
}


def slug(value: str) -> str:
    """Collapse a DSN/identifier to a stable, id-safe slug.

    Credentials embedded in a DSN are stripped so the resulting node id never
    leaks secrets into the graph.
    """
    parsed = urlparse(value)
    if parsed.scheme:
        host = parsed.hostname or ""
        path = (parsed.path or "").strip("/")
        base = f"{parsed.scheme}_{host}_{path}"
    else:
        base = value
    base = base.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return base or "datasource"


def infer_kind(dsn: str) -> str:
    """Infer the backend ``kind`` from a DSN scheme.

    Falls back to ``sqlite`` for bare filesystem paths (no scheme).
    """
    parsed = urlparse(dsn)
    scheme = (parsed.scheme or "").lower()
    # A bare Windows drive path (``c:\...``) parses a 1-char scheme; treat as path.
    if scheme and len(scheme) > 1:
        kind = _SCHEME_TO_KIND.get(scheme)
        if kind is not None:
            return kind
        raise ValueError(f"Cannot infer connector kind from DSN scheme {scheme!r}")
    # No (usable) scheme → assume a local sqlite file path.
    return "sqlite"


class UniversalConnector:
    """Driver-based connector for SQL + MongoDB + GraphQL with KG introspection.

    CONCEPT:KG-2.9

    Args:
        dsn: A data-source connection string. SQL/Mongo use a URL DSN
            (``postgresql://...``, ``mongodb://...``); sqlite accepts
            ``sqlite:///path`` or a bare filesystem path; graphql uses the
            endpoint URL.
        kind: Explicit backend kind; inferred from the DSN scheme when ``None``.
    """

    def __init__(self, dsn: str, kind: str | None = None) -> None:
        self.dsn = dsn
        self.kind = (kind or infer_kind(dsn)).lower()
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported connector kind {self.kind!r}; "
                f"expected one of {sorted(SUPPORTED_KINDS)}"
            )
        self.name = f"universal:{self.kind}:{slug(dsn)}"

    # ------------------------------------------------------------------ #
    # Driver resolution (lazy; no hard dependencies).
    # ------------------------------------------------------------------ #
    def _driver(self) -> Any:
        """Lazily import and return the client module for this ``kind``.

        Each import is guarded so importing this module never requires any
        database driver. A missing driver yields a clear ``RuntimeError``.
        """
        try:
            if self.kind == "sqlite":
                import sqlite3

                return sqlite3
            if self.kind == "postgresql":
                import psycopg  # type: ignore[import-not-found]

                return psycopg
            if self.kind == "mysql":
                import pymysql  # type: ignore[import-not-found,import-untyped]

                return pymysql
            if self.kind == "mssql":
                import pyodbc  # type: ignore[import-not-found]

                return pyodbc
            if self.kind == "oracle":
                import oracledb  # type: ignore[import-not-found]

                return oracledb
            if self.kind == "mongodb":
                import pymongo  # type: ignore[import-not-found]

                return pymongo
            if self.kind == "graphql":
                import httpx  # type: ignore[import-not-found]

                return httpx
        except ImportError as exc:
            raise RuntimeError(
                f"Driver for kind {self.kind!r} is not installed "
                f"({exc}). Install the relevant client to use this connector."
            ) from exc
        raise RuntimeError(f"No driver mapping for kind {self.kind!r}")

    def _sqlite_path(self) -> str:
        """Resolve the on-disk sqlite path from the DSN."""
        parsed = urlparse(self.dsn)
        if parsed.scheme in ("sqlite", "file"):
            # sqlite:///abs/path  ->  netloc="" path="/abs/path"
            # sqlite:///:memory:  ->  in-memory
            path = parsed.path
            if parsed.netloc and not path:
                path = parsed.netloc
            path = path.lstrip("/") if path.startswith("///") else path
            # Common forms: sqlite:///foo.db, sqlite:////abs/foo.db
            if path.startswith("/") and not parsed.netloc:
                # sqlite:////abs -> keep absolute; sqlite:///rel -> strip one
                candidate = parsed.path
                # urlparse gives path="/rel.db" for sqlite:///rel.db
                return candidate[1:] if candidate.startswith("/") else candidate
            return path or ":memory:"
        return self.dsn

    def _connect(self) -> Any:
        """Open a new DBAPI/Mongo/HTTP connection appropriate for ``kind``."""
        driver = self._driver()
        if self.kind == "sqlite":
            return driver.connect(self._sqlite_path())
        if self.kind == "postgresql":
            return driver.connect(self.dsn)
        if self.kind == "mysql":
            parsed = urlparse(self.dsn)
            return driver.connect(
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                database=(parsed.path or "").lstrip("/") or None,
            )
        if self.kind == "mssql":
            return driver.connect(self.dsn)
        if self.kind == "oracle":
            return driver.connect(self.dsn)
        if self.kind == "mongodb":
            return driver.MongoClient(self.dsn)
        if self.kind == "graphql":
            return driver.Client()
        raise RuntimeError(f"No connection path for kind {self.kind!r}")

    # ------------------------------------------------------------------ #
    # Read / write / update.
    # ------------------------------------------------------------------ #
    def read(self, query: str, params: Any = None) -> list[dict[str, Any]]:
        """Execute a read query and return rows as a list of dicts.

        For SQL backends ``query`` is SQL; for MongoDB ``query`` is a
        ``"collection"`` name (params is an optional filter dict); for GraphQL
        ``query`` is a GraphQL document string.
        """
        if self.kind in SQL_KINDS:
            return self._sql_read(query, params)
        if self.kind == "mongodb":
            return self._mongo_read(query, params)
        if self.kind == "graphql":
            return self._graphql_read(query, params)
        raise RuntimeError(f"read() unsupported for kind {self.kind!r}")

    def write(self, query: str, params: Any = None) -> int:
        """Execute a mutating statement; return the affected row count."""
        return self._execute(query, params)

    def update(self, query: str, params: Any = None) -> int:
        """Alias for :meth:`write` — same execute path, kept for clarity."""
        return self._execute(query, params)

    def _execute(self, query: str, params: Any) -> int:
        if self.kind in SQL_KINDS:
            return self._sql_execute(query, params)
        if self.kind == "mongodb":
            # `query` = collection; params = {"op": ..., "doc": ...} style dict.
            raise RuntimeError(
                "Use read() with a MongoDB collection; write() requires a "
                "command dict (not yet wired for MongoDB)."
            )
        if self.kind == "graphql":
            # A mutation is just a read with a mutation document.
            self._graphql_read(query, params)
            return 1
        raise RuntimeError(f"write()/update() unsupported for kind {self.kind!r}")

    # --- SQL ---------------------------------------------------------- #
    def _sql_read(self, query: str, params: Any) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(query, params or ())
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchall()
            return [dict(zip(cols, row, strict=False)) for row in rows]
        finally:
            conn.close()

    def _sql_execute(self, query: str, params: Any) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(query, params or ())
            affected = cur.rowcount
            conn.commit()
            return int(affected if affected is not None and affected >= 0 else 0)
        finally:
            conn.close()

    # --- MongoDB ------------------------------------------------------ #
    def _mongo_read(self, collection: str, params: Any) -> list[dict[str, Any]]:
        client = self._connect()
        parsed = urlparse(self.dsn)
        db_name = (parsed.path or "").lstrip("/") or "test"
        db = client[db_name]
        return list(db[collection].find(params or {}))

    # --- GraphQL ------------------------------------------------------ #
    def _graphql_read(self, query: str, params: Any) -> list[dict[str, Any]]:
        httpx = self._driver()
        resp = httpx.post(
            self.dsn,
            json={"query": query, "variables": params or {}},
            timeout=30.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data")
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []

    def health_check(self) -> bool:
        """Return True if a trivial round-trip to the backend succeeds."""
        try:
            if self.kind == "sqlite":
                self._sql_read("SELECT 1", None)
                return True
            if self.kind in SQL_KINDS:
                self._sql_read("SELECT 1", None)
                return True
            if self.kind == "mongodb":
                client = self._connect()
                client.admin.command("ping")
                return True
            if self.kind == "graphql":
                self._graphql_read("{ __typename }", None)
                return True
        except Exception as exc:
            logger.debug("health_check failed for %s: %s", self.name, exc)
            return False
        return False

    # ------------------------------------------------------------------ #
    # Schema introspection → KG ExtractionBatch.
    # ------------------------------------------------------------------ #
    def introspect(self) -> ExtractionBatch:
        """Return the data source's schema as a KG ``ExtractionBatch``.

        Best-effort and tolerant: the ``DataSource`` node is always emitted; any
        per-table/collection/type failure is logged and skipped rather than
        aborting the whole introspection.
        """
        ds_id = f"datasource:{slug(self.dsn)}"
        nodes: list[GraphNode] = [
            GraphNode(
                id=ds_id,
                type="DataSource",
                props={"kind": self.kind, "name": self.name},
            )
        ]
        edges: list[EnrichmentEdge] = []
        try:
            if self.kind == "sqlite":
                self._introspect_sqlite(ds_id, nodes, edges)
            elif self.kind in SQL_KINDS:
                self._introspect_sql(ds_id, nodes, edges)
            elif self.kind == "mongodb":
                self._introspect_mongo(ds_id, nodes, edges)
            elif self.kind == "graphql":
                self._introspect_graphql(ds_id, nodes, edges)
        except Exception as exc:  # pragma: no cover - defensive best-effort
            logger.warning("introspect() partial failure for %s: %s", self.name, exc)
        return ExtractionBatch(category="datasource", nodes=nodes, edges=edges)

    # --- sqlite introspection (PRAGMA) -------------------------------- #
    def _introspect_sqlite(
        self,
        ds_id: str,
        nodes: list[GraphNode],
        edges: list[EnrichmentEdge],
    ) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [r[0] for r in cur.fetchall()]
            for table in tables:
                table_id = f"table:{slug(self.dsn)}:{table}"
                nodes.append(
                    GraphNode(id=table_id, type="Table", props={"name": table})
                )
                edges.append(
                    EnrichmentEdge(source=ds_id, target=table_id, rel_type="HAS_TABLE")
                )
                cur.execute(f'PRAGMA table_info("{table}")')
                for col in cur.fetchall():
                    # (cid, name, type, notnull, dflt_value, pk)
                    col_name, col_type, pk = col[1], col[2], col[5]
                    col_id = f"column:{slug(self.dsn)}:{table}:{col_name}"
                    nodes.append(
                        GraphNode(
                            id=col_id,
                            type="Column",
                            props={
                                "name": col_name,
                                "data_type": col_type,
                                "primary_key": bool(pk),
                            },
                        )
                    )
                    edges.append(
                        EnrichmentEdge(
                            source=table_id, target=col_id, rel_type="HAS_COLUMN"
                        )
                    )
                # Foreign keys.
                cur.execute(f'PRAGMA foreign_key_list("{table}")')
                for fk in cur.fetchall():
                    # (id, seq, table, from, to, on_update, on_delete, match)
                    from_col, ref_table, to_col = fk[3], fk[2], fk[4]
                    src_col_id = f"column:{slug(self.dsn)}:{table}:{from_col}"
                    tgt_col_id = (
                        f"column:{slug(self.dsn)}:{ref_table}:{to_col}"
                        if to_col
                        else f"table:{slug(self.dsn)}:{ref_table}"
                    )
                    edges.append(
                        EnrichmentEdge(
                            source=src_col_id,
                            target=tgt_col_id,
                            rel_type="FOREIGN_KEY",
                        )
                    )
        finally:
            conn.close()

    # --- generic SQL introspection (information_schema) --------------- #
    def _introspect_sql(
        self,
        ds_id: str,
        nodes: list[GraphNode],
        edges: list[EnrichmentEdge],
    ) -> None:
        col_rows = self._sql_read(
            "SELECT table_name, column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema NOT IN "
            "('pg_catalog','information_schema','sys','mysql','performance_schema') "
            "ORDER BY table_name, ordinal_position",
            None,
        )
        seen_tables: set[str] = set()
        for row in col_rows:
            table = row.get("table_name") or row.get("TABLE_NAME")
            col_name = row.get("column_name") or row.get("COLUMN_NAME")
            data_type = row.get("data_type") or row.get("DATA_TYPE")
            if not table or not col_name:
                continue
            table_id = f"table:{slug(self.dsn)}:{table}"
            if table not in seen_tables:
                seen_tables.add(table)
                nodes.append(
                    GraphNode(id=table_id, type="Table", props={"name": table})
                )
                edges.append(
                    EnrichmentEdge(source=ds_id, target=table_id, rel_type="HAS_TABLE")
                )
            col_id = f"column:{slug(self.dsn)}:{table}:{col_name}"
            nodes.append(
                GraphNode(
                    id=col_id,
                    type="Column",
                    props={"name": col_name, "data_type": data_type},
                )
            )
            edges.append(
                EnrichmentEdge(source=table_id, target=col_id, rel_type="HAS_COLUMN")
            )
        # Foreign keys (best-effort; standard information_schema join).
        try:
            fk_rows = self._sql_read(
                "SELECT kcu.table_name AS src_table, kcu.column_name AS src_col, "
                "ccu.table_name AS ref_table, ccu.column_name AS ref_col "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                "JOIN information_schema.constraint_column_usage ccu "
                "  ON tc.constraint_name = ccu.constraint_name "
                "WHERE tc.constraint_type = 'FOREIGN KEY'",
                None,
            )
            for fk in fk_rows:
                src = f"column:{slug(self.dsn)}:{fk['src_table']}:{fk['src_col']}"
                tgt = f"column:{slug(self.dsn)}:{fk['ref_table']}:{fk['ref_col']}"
                edges.append(
                    EnrichmentEdge(source=src, target=tgt, rel_type="FOREIGN_KEY")
                )
        except Exception as exc:  # pragma: no cover - dialect variance
            logger.debug("FK introspection skipped for %s: %s", self.name, exc)

    # --- MongoDB introspection (sampled) ------------------------------ #
    def _introspect_mongo(
        self,
        ds_id: str,
        nodes: list[GraphNode],
        edges: list[EnrichmentEdge],
    ) -> None:
        client = self._connect()
        parsed = urlparse(self.dsn)
        db_name = (parsed.path or "").lstrip("/") or "test"
        db = client[db_name]
        for coll_name in db.list_collection_names():
            coll_id = f"collection:{slug(self.dsn)}:{coll_name}"
            nodes.append(
                GraphNode(id=coll_id, type="Collection", props={"name": coll_name})
            )
            edges.append(
                EnrichmentEdge(source=ds_id, target=coll_id, rel_type="HAS_TABLE")
            )
            sample = db[coll_name].find_one() or {}
            for field, value in sample.items():
                field_id = f"field:{slug(self.dsn)}:{coll_name}:{field}"
                nodes.append(
                    GraphNode(
                        id=field_id,
                        type="Field",
                        props={"name": str(field), "data_type": type(value).__name__},
                    )
                )
                edges.append(
                    EnrichmentEdge(
                        source=coll_id, target=field_id, rel_type="HAS_COLUMN"
                    )
                )

    # --- GraphQL introspection --------------------------------------- #
    def _introspect_graphql(
        self,
        ds_id: str,
        nodes: list[GraphNode],
        edges: list[EnrichmentEdge],
    ) -> None:
        introspection_query = (
            "query { __schema { types { name kind "
            "fields { name type { name kind } } } } }"
        )
        results = self._graphql_read(introspection_query, None)
        if not results:
            return
        schema = results[0].get("__schema", {}) if results else {}
        for gql_type in schema.get("types", []):
            type_name = gql_type.get("name") or ""
            if not type_name or type_name.startswith("__"):
                continue
            if gql_type.get("kind") not in ("OBJECT", "INTERFACE", "INPUT_OBJECT"):
                continue
            type_id = f"gqltype:{slug(self.dsn)}:{type_name}"
            nodes.append(
                GraphNode(
                    id=type_id,
                    type="GraphQLType",
                    props={"name": type_name, "kind": gql_type.get("kind")},
                )
            )
            edges.append(
                EnrichmentEdge(source=ds_id, target=type_id, rel_type="HAS_TABLE")
            )
            for field in gql_type.get("fields") or []:
                field_name = field.get("name") or ""
                if not field_name:
                    continue
                field_id = f"gqlfield:{slug(self.dsn)}:{type_name}:{field_name}"
                ftype = (field.get("type") or {}).get("name")
                nodes.append(
                    GraphNode(
                        id=field_id,
                        type="Field",
                        props={"name": field_name, "data_type": ftype},
                    )
                )
                edges.append(
                    EnrichmentEdge(
                        source=type_id, target=field_id, rel_type="HAS_COLUMN"
                    )
                )
