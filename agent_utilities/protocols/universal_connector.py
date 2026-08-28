from __future__ import annotations

"""Bounded, read-only multi-database connector with KG schema introspection.

CONCEPT:AU-KG.ingest.universal-data-connector — Universal DataConnector

A single, driver-based connector speaks SQL (PostgreSQL, MySQL, MS SQL,
Oracle, SQLite) and MongoDB through a bounded ``read`` / ``health_check`` /
``introspect`` surface. Mutations use governed ChangeEnvelope/MutationBatch
APIs instead.

GraphQL is handled by the schema-neutral ``GraphQLSourceAdapter``. Keeping HTTP
out of this class ensures all GraphQL traffic receives the central TLS, SSRF,
response-size, schema-approval, and drift controls rather than a second weaker
implementation.

This module is **importable with no database drivers installed**: every driver
import is performed lazily inside :meth:`UniversalConnector._driver` and a
missing driver raises a clear :class:`RuntimeError` (never an ``ImportError``
leak). It emits the same backend-agnostic ``ExtractionBatch`` shape that the KG
enrichment pipeline (``knowledge_graph.enrichment.registry.write_batch``)
already persists.

Connection values are runtime-only. Durable node identifiers are irreversible
persistence references and never contain a host, account, endpoint, database
path, or credential.
"""

import datetime as dt
import decimal
import logging
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..knowledge_graph.enrichment.models import (
    EnrichmentEdge,
    ExtractionBatch,
    GraphNode,
)
from ..security.identifiers import quote_sql_identifier
from ..security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

logger = logging.getLogger(__name__)

# Supported backend kinds.
SQL_KINDS = frozenset({"postgresql", "mysql", "mssql", "oracle", "sqlite"})
SUPPORTED_KINDS = SQL_KINDS | frozenset({"mongodb"})

_MAX_QUERY_BYTES = 65_536
_MAX_PARAM_BYTES = 1_048_576
_MAX_RESULT_BYTES = 16 * 1_048_576
_MAX_CELL_BYTES = 1_048_576
_MAX_ROWS = 10_000
_MAX_COLUMNS = 512
_MAX_SCHEMA_TABLES = 10_000
_MAX_SCHEMA_FIELDS = 100_000
_MAX_IDENTIFIER_BYTES = 255
_FETCH_BATCH = 128
_SOURCE_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MONGO_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_MONGO_OPERATORS = frozenset(
    {
        "$and",
        "$eq",
        "$exists",
        "$gt",
        "$gte",
        "$in",
        "$lt",
        "$lte",
        "$ne",
        "$nin",
        "$nor",
        "$not",
        "$or",
    }
)

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
}


def infer_kind(dsn: str) -> str:
    """Infer the backend ``kind`` from a DSN scheme.

    A scheme is required; bare paths must be paired with ``kind="sqlite"``.
    """
    parsed = urlparse(dsn)
    scheme = (parsed.scheme or "").lower()
    if scheme and len(scheme) > 1:
        kind = _SCHEME_TO_KIND.get(scheme)
        if kind is not None:
            return kind
        raise ValueError(f"Cannot infer connector kind from DSN scheme {scheme!r}")
    raise ValueError("database connector kind requires an explicit DSN scheme")


@dataclass
class _SchemaSink:
    """Bundles the growing node/edge lists during schema introspection.

    Exists so introspection helpers stay within the <=7 parameter cap instead
    of threading ``nodes`` and ``edges`` as two separate positional args.
    """

    nodes: list[GraphNode]
    edges: list[EnrichmentEdge]


class UniversalConnector:
    """Driver-based, bounded read connector for SQL and MongoDB.

    CONCEPT:AU-KG.ingest.universal-data-connector

    Args:
        dsn: A runtime-only data-source connection string. SQL/Mongo use a URL
            DSN; a bare sqlite path requires an explicit ``kind="sqlite"``.
        kind: Explicit backend kind; inferred from the DSN scheme when ``None``.
        source_alias: Optional non-sensitive logical alias. It participates in
            the irreversible source reference but is never persisted verbatim.
    """

    def __init__(
        self,
        dsn: str,
        kind: str | None = None,
        *,
        source_alias: str | None = None,
        tls_service: str | None = None,
        tls_profile: str | None = None,
        tls_profile_ref: str | None = None,
        tls_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        if not isinstance(dsn, str) or not dsn or len(dsn.encode("utf-8")) > 8_192:
            raise ValueError("database connection value is invalid")
        if "\x00" in dsn:
            raise ValueError("database connection value is invalid")
        if source_alias is not None and not _SOURCE_ALIAS_RE.fullmatch(source_alias):
            raise ValueError("source_alias must be a short logical identifier")
        self.dsn = dsn
        self.kind = (kind or infer_kind(dsn)).lower()
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError("database connector kind is not supported")
        identity = source_alias or dsn
        self.source_ref = persistence_reference("datasource", identity)
        self.name = f"universal:{self.kind}:{self.source_ref}"
        self._tls_service = tls_service
        self._tls_profile = tls_profile
        self._tls_profile_ref = tls_profile_ref
        self._tls_resolver = tls_resolver

    def _object_id(self, object_kind: str, *parts: object) -> str:
        framed = "\x00".join(str(part) for part in parts)
        reference = persistence_reference(
            "schema_object",
            framed,
            namespace=self.source_ref,
        )
        return f"{object_kind}:{reference}"

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
        except ImportError as exc:
            raise RuntimeError("database driver is unavailable") from exc
        raise RuntimeError("database driver is unavailable")

    def _sqlite_path(self) -> str:
        """Resolve the on-disk sqlite path from the DSN."""
        if self.dsn in {":memory:", "sqlite:///:memory:"}:
            return ":memory:"
        if self.dsn.startswith("sqlite:///"):
            path = unquote(self.dsn.removeprefix("sqlite:///"))
            if not path:
                raise ValueError("sqlite connection value is invalid")
            return path
        parsed = urlparse(self.dsn)
        if parsed.scheme == "file":
            path = unquote(parsed.path or parsed.netloc)
            if not path:
                raise ValueError("sqlite connection value is invalid")
            return path
        return self.dsn

    def _connect(self) -> Any:
        """Open a bounded connection appropriate for ``kind``."""
        if self.kind == "sqlite":
            return self._connect_sqlite()
        return self._connect_networked()

    def _connect_sqlite(self) -> Any:
        """Open a bounded sqlite connection (read-only, symlink-refusing)."""
        driver = self._driver()
        path = self._sqlite_path()
        if path == ":memory:":
            connection = driver.connect(path, timeout=10.0)
        else:
            connection = driver.connect(
                f"{self._sqlite_resolved_uri(path)}?mode=ro",
                uri=True,
                timeout=10.0,
            )
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @staticmethod
    def _sqlite_resolved_uri(path: str) -> str:
        """Resolve ``path`` to a file URI, refusing symlinks and non-files."""
        candidate = Path(path)
        if candidate.is_symlink():
            raise RuntimeError("sqlite source is unavailable")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("sqlite source is unavailable") from exc
        if not resolved.is_file():
            raise RuntimeError("sqlite source is unavailable")
        return resolved.as_uri()

    def _connect_networked(self) -> Any:
        """Open a TLS-governed connection for every non-sqlite ``kind``."""
        driver = self._driver()
        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        service = self._tls_service or {
            "postgresql": "postgres",
            "mongodb": "mongodb",
        }.get(self.kind, f"database-{self.kind}")
        trust = resolve_configured_tls_profile(
            service,
            profile_name=self._tls_profile,
            profile_ref=self._tls_profile_ref,
            resolver=self._tls_resolver,
        )
        try:
            connectors: dict[str, Callable[[Any, Any], Any]] = {
                "postgresql": self._connect_postgresql,
                "mysql": self._connect_mysql,
                "mssql": self._connect_mssql,
                "oracle": self._connect_oracle,
                "mongodb": self._connect_mongodb,
            }
            connector = connectors.get(self.kind)
            if connector is None:
                raise RuntimeError("database connection kind is unsupported")
            return connector(driver, trust)
        finally:
            trust.cleanup()

    def _connect_postgresql(self, driver: Any, trust: Any) -> Any:
        return driver.connect(self.dsn, **trust.psycopg_kwargs())

    def _connect_mysql(self, driver: Any, trust: Any) -> Any:
        from agent_utilities.core.transport_security import TransportSecurityError

        if trust.proxy_url:
            raise TransportSecurityError("mysql_tls_proxy_unsupported")
        parsed = urlparse(self.dsn)
        if not parsed.hostname:
            raise ValueError("MySQL connection host is required")
        return driver.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=(parsed.path or "").lstrip("/") or None,
            ssl=trust.ssl_context,
        )

    def _connect_mssql(self, driver: Any, trust: Any) -> Any:
        from agent_utilities.core.transport_security import TransportSecurityError

        if (
            trust.proxy_url
            or trust.ca_bundle_path is not None
            or trust.ca_directory is not None
            or trust.client_cert_path is not None
        ):
            raise TransportSecurityError("mssql_tls_profile_unsupported")
        connection = self.dsn.rstrip(";")
        return driver.connect(f"{connection};Encrypt=yes;TrustServerCertificate=no")

    def _connect_oracle(self, driver: Any, trust: Any) -> Any:
        from agent_utilities.core.transport_security import TransportSecurityError

        if trust.proxy_url:
            raise TransportSecurityError("oracle_tls_proxy_unsupported")
        return driver.connect(self.dsn, ssl_context=trust.ssl_context)

    def _connect_mongodb(self, driver: Any, trust: Any) -> Any:
        return driver.MongoClient(
            self.dsn,
            serverSelectionTimeoutMS=10_000,
            connectTimeoutMS=10_000,
            socketTimeoutMS=30_000,
            **trust.pymongo_kwargs(),
        )

    # ------------------------------------------------------------------ #
    # Bounded reads. Mutations use governed MutationBatch APIs.
    # ------------------------------------------------------------------ #
    def read(
        self,
        query: str,
        params: Any = None,
        *,
        max_rows: int = _MAX_ROWS,
    ) -> list[dict[str, Any]]:
        """Execute a read query and return rows as a list of dicts.

        For SQL backends ``query`` is SQL; for MongoDB ``query`` is a
        ``"collection"`` name and params is an optional bounded filter dict.
        """
        if not isinstance(max_rows, int) or isinstance(max_rows, bool):
            raise ValueError("max_rows is invalid")
        if not 1 <= max_rows <= _MAX_ROWS:
            raise ValueError("max_rows is outside the supported range")
        if self.kind in SQL_KINDS:
            self._validate_sql_read(query)
            self._validate_params(params)
            return self._sql_read(query, params, max_rows=max_rows)
        if self.kind == "mongodb":
            return self._mongo_read(query, params, max_rows=max_rows)
        raise RuntimeError("database read kind is unsupported")

    @staticmethod
    def _validate_sql_read(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("database read query is invalid")
        if len(query.encode("utf-8")) > _MAX_QUERY_BYTES or "\x00" in query:
            raise ValueError("database read query is invalid")
        value = query.strip()
        if not UniversalConnector._is_single_readonly_select(value):
            raise PermissionError("database connector permits one read-only SELECT")

    _SQL_READ_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?is)\binto\b"),
        re.compile(r"(?is)\bfor\s+(update|share)\b"),
        re.compile(
            r"(?is)\b(pg_read_file|pg_ls_dir|lo_import|load_file|sleep|benchmark)\s*\("
        ),
    )

    @staticmethod
    def _is_single_readonly_select(value: str) -> bool:
        if not re.match(r"(?is)^select\b", value):
            return False
        if any(token in value for token in (";", "--", "/*", "*/")):
            return False
        return not any(
            pattern.search(value)
            for pattern in UniversalConnector._SQL_READ_FORBIDDEN_PATTERNS
        )

    @classmethod
    def _validate_params(cls, params: Any) -> None:
        if params is not None:
            cls._validate_param_value(params, 0, [0])

    @classmethod
    def _validate_param_value(cls, value: Any, depth: int, budget: list[int]) -> None:
        if depth > 12:
            raise ValueError("database query parameters are too deeply nested")
        if cls._accumulate_scalar_budget(value, budget):
            pass
        elif isinstance(value, Mapping):
            cls._validate_param_mapping(value, depth, budget)
        elif isinstance(value, Sequence):
            cls._validate_param_sequence(value, depth, budget)
        else:
            raise ValueError("database query parameter type is unsupported")
        if budget[0] > _MAX_PARAM_BYTES:
            raise ValueError("database query parameters are too large")

    @staticmethod
    def _accumulate_scalar_budget(value: Any, budget: list[int]) -> bool:
        """Add ``value``'s byte cost to ``budget`` if scalar; return whether it was."""
        if value is None or isinstance(value, bool | int | float):
            budget[0] += 16
            return True
        if isinstance(value, str | bytes | bytearray | memoryview):
            raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
            budget[0] += len(raw)
            return True
        if isinstance(value, dt.date | dt.time | decimal.Decimal | uuid.UUID):
            budget[0] += 64
            return True
        return False

    @classmethod
    def _validate_param_mapping(
        cls, value: Mapping, depth: int, budget: list[int]
    ) -> None:
        if len(value) > 1_000:
            raise ValueError("database query has too many parameters")
        for key, item in value.items():
            if not isinstance(key, str) or len(key.encode("utf-8")) > 255:
                raise ValueError("database parameter name is invalid")
            budget[0] += len(key.encode("utf-8"))
            cls._validate_param_value(item, depth + 1, budget)

    @classmethod
    def _validate_param_sequence(
        cls, value: Sequence, depth: int, budget: list[int]
    ) -> None:
        if len(value) > 1_000:
            raise ValueError("database query has too many parameters")
        for item in value:
            cls._validate_param_value(item, depth + 1, budget)

    @classmethod
    def _bounded_cell_size(cls, value: Any, *, depth: int = 0) -> int:
        if depth > 12:
            raise RuntimeError("database result is too deeply nested")
        scalar_size = cls._scalar_cell_size(value)
        if scalar_size is not None:
            size = scalar_size
        elif isinstance(value, Mapping):
            size = cls._mapping_cell_size(value, depth)
        elif isinstance(value, Sequence):
            size = cls._sequence_cell_size(value, depth)
        else:
            raise TypeError("database result contains an unsupported cell type")
        if size > _MAX_CELL_BYTES:
            raise RuntimeError("database result cell exceeds the configured bound")
        return size

    @staticmethod
    def _scalar_cell_size(value: Any) -> int | None:
        if value is None or isinstance(value, bool | int | float):
            return 16
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, bytes | bytearray | memoryview):
            return len(value)
        if isinstance(value, dt.date | dt.time | decimal.Decimal | uuid.UUID):
            return 64
        return None

    @classmethod
    def _mapping_cell_size(cls, value: Mapping, depth: int) -> int:
        if len(value) > 10_000:
            raise RuntimeError("database result object has too many fields")
        return sum(
            len(str(key).encode("utf-8"))
            + cls._bounded_cell_size(item, depth=depth + 1)
            for key, item in value.items()
        )

    @classmethod
    def _sequence_cell_size(cls, value: Sequence, depth: int) -> int:
        if len(value) > 10_000:
            raise RuntimeError("database result array has too many values")
        return sum(cls._bounded_cell_size(item, depth=depth + 1) for item in value)

    @staticmethod
    def _safe_identifier(value: Any) -> str:
        rendered = str(value or "")
        if (
            not rendered
            or "\x00" in rendered
            or any(ord(char) < 32 for char in rendered)
            or len(rendered.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
        ):
            raise ValueError("database schema identifier is invalid")
        return rendered

    @classmethod
    def _validate_mongo_filter(cls, value: Any, *, depth: int = 0) -> None:
        if depth > 12:
            raise ValueError("MongoDB filter is too deeply nested")
        if isinstance(value, Mapping):
            cls._validate_mongo_filter_mapping(value, depth)
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray | memoryview
        ):
            cls._validate_mongo_filter_sequence(value, depth)
        elif not cls._is_mongo_filter_scalar(value):
            raise ValueError("MongoDB filter value is unsupported")

    @classmethod
    def _validate_mongo_filter_mapping(cls, value: Mapping, depth: int) -> None:
        if len(value) > 1_000:
            raise ValueError("MongoDB filter has too many fields")
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("MongoDB filter key is invalid")
            if raw_key.startswith("$") and raw_key not in _MONGO_OPERATORS:
                raise PermissionError("MongoDB filter operator is not permitted")
            cls._safe_identifier(raw_key)
            cls._validate_mongo_filter(item, depth=depth + 1)

    @classmethod
    def _validate_mongo_filter_sequence(cls, value: Sequence, depth: int) -> None:
        if len(value) > 1_000:
            raise ValueError("MongoDB filter has too many values")
        for item in value:
            cls._validate_mongo_filter(item, depth=depth + 1)

    @staticmethod
    def _is_mongo_filter_scalar(value: Any) -> bool:
        return isinstance(
            value,
            type(None)
            | bool
            | int
            | float
            | str
            | bytes
            | bytearray
            | dt.date
            | dt.time
            | decimal.Decimal
            | uuid.UUID,
        )

    # --- SQL ---------------------------------------------------------- #
    def _sql_read(
        self,
        query: str,
        params: Any,
        *,
        max_rows: int = _MAX_ROWS,
    ) -> list[dict[str, Any]]:
        self._validate_sql_read(query)
        self._validate_params(params)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(query, params or ())
            description = cur.description or []
            if len(description) > _MAX_COLUMNS:
                raise RuntimeError("database result has too many columns")
            cols = [self._safe_identifier(item[0]) for item in description]
            output: list[dict[str, Any]] = []
            total_bytes = 0
            while len(output) < max_rows:
                chunk = cur.fetchmany(min(_FETCH_BATCH, max_rows - len(output)))
                if not chunk:
                    break
                for row in chunk:
                    total_bytes += sum(self._bounded_cell_size(cell) for cell in row)
                    if total_bytes > _MAX_RESULT_BYTES:
                        raise RuntimeError(
                            "database result exceeds the configured bound"
                        )
                    output.append(dict(zip(cols, row, strict=False)))
            return output
        finally:
            conn.close()

    # --- MongoDB ------------------------------------------------------ #
    def _mongo_read(
        self,
        collection: str,
        params: Any,
        *,
        max_rows: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(collection, str) or not _MONGO_COLLECTION_RE.fullmatch(
            collection
        ):
            raise ValueError("MongoDB collection is invalid")
        query_filter = params or {}
        if not isinstance(query_filter, Mapping):
            raise ValueError("MongoDB filter must be an object")
        self._validate_params(query_filter)
        self._validate_mongo_filter(query_filter)
        client = self._connect()
        try:
            return self._mongo_read_rows(client, collection, query_filter, max_rows)
        finally:
            client.close()

    def _mongo_read_rows(
        self,
        client: Any,
        collection: str,
        query_filter: Mapping[str, Any],
        max_rows: int,
    ) -> list[dict[str, Any]]:
        parsed = urlparse(self.dsn)
        db_name = (parsed.path or "").lstrip("/") or "test"
        db_name = self._safe_identifier(db_name)
        db = client[db_name]
        output: list[dict[str, Any]] = []
        total_bytes = 0
        for document in db[collection].find(dict(query_filter)).limit(max_rows):
            if not isinstance(document, Mapping):
                continue
            item = dict(document)
            total_bytes += sum(
                len(str(key).encode("utf-8")) + self._bounded_cell_size(value)
                for key, value in item.items()
            )
            if total_bytes > _MAX_RESULT_BYTES:
                raise RuntimeError("database result exceeds the configured bound")
            output.append(item)
        return output

    def health_check(self) -> bool:
        """Return True if a trivial round-trip to the backend succeeds."""
        try:
            if self.kind == "sqlite":
                self._sql_read("SELECT 1", None, max_rows=1)
                return True
            if self.kind in SQL_KINDS:
                self._sql_read("SELECT 1", None, max_rows=1)
                return True
            if self.kind == "mongodb":
                client = self._connect()
                try:
                    client.admin.command("ping")
                    return True
                finally:
                    client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "database health check failed kind=%s error_type=%s",
                self.kind,
                type(exc).__name__,
            )
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
        ds_id = f"datasource:{self.source_ref}"
        nodes: list[GraphNode] = [
            GraphNode(
                id=ds_id,
                type="DataSource",
                props={"kind": self.kind, "source_ref": self.source_ref},
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
        except Exception as exc:  # pragma: no cover - defensive best-effort
            logger.warning(
                "database introspection partially failed kind=%s error_type=%s",
                self.kind,
                type(exc).__name__,
            )
        privacy_guard = PersistencePrivacyGuard()
        for node in nodes:
            clean_props, _ = privacy_guard.sanitize(node.props)
            node.props = clean_props if isinstance(clean_props, dict) else {}
        return ExtractionBatch(category="datasource", nodes=nodes, edges=edges)

    # --- sqlite introspection (PRAGMA) -------------------------------- #
    def _introspect_sqlite(
        self,
        ds_id: str,
        nodes: list[GraphNode],
        edges: list[EnrichmentEdge],
    ) -> None:
        sink = _SchemaSink(nodes, edges)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [
                self._safe_identifier(row[0])
                for row in cur.fetchmany(_MAX_SCHEMA_TABLES + 1)
            ]
            if len(tables) > _MAX_SCHEMA_TABLES:
                raise RuntimeError("database schema has too many tables")
            field_count = 0
            for table in tables:
                field_count = self._introspect_sqlite_table(
                    cur, ds_id, table, sink, field_count
                )
        finally:
            conn.close()

    def _introspect_sqlite_table(
        self,
        cur: Any,
        ds_id: str,
        table: str,
        sink: _SchemaSink,
        field_count: int,
    ) -> int:
        table_id = self._object_id("table", table)
        sink.nodes.append(GraphNode(id=table_id, type="Table", props={"name": table}))
        sink.edges.append(
            EnrichmentEdge(source=ds_id, target=table_id, rel_type="HAS_TABLE")
        )
        quoted_table = quote_sql_identifier(table, kind="table")
        field_count = self._introspect_sqlite_columns(
            cur, quoted_table, table, table_id, sink, field_count
        )
        self._introspect_sqlite_foreign_keys(cur, quoted_table, table, sink)
        return field_count

    def _introspect_sqlite_columns(
        self,
        cur: Any,
        quoted_table: str,
        table: str,
        table_id: str,
        sink: _SchemaSink,
        field_count: int,
    ) -> int:
        cur.execute(f"PRAGMA table_info({quoted_table})")
        for col in cur.fetchmany(_MAX_SCHEMA_FIELDS - field_count + 1):
            # (cid, name, type, notnull, dflt_value, pk)
            field_count += 1
            if field_count > _MAX_SCHEMA_FIELDS:
                raise RuntimeError("database schema has too many fields")
            col_name = self._safe_identifier(col[1])
            col_type = str(col[2] or "")[:_MAX_IDENTIFIER_BYTES]
            pk = col[5]
            col_id = self._object_id("column", table, col_name)
            sink.nodes.append(
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
            sink.edges.append(
                EnrichmentEdge(source=table_id, target=col_id, rel_type="HAS_COLUMN")
            )
        return field_count

    def _introspect_sqlite_foreign_keys(
        self,
        cur: Any,
        quoted_table: str,
        table: str,
        sink: _SchemaSink,
    ) -> None:
        cur.execute(f"PRAGMA foreign_key_list({quoted_table})")
        for fk in cur.fetchmany(_MAX_SCHEMA_FIELDS + 1):
            # (id, seq, table, from, to, on_update, on_delete, match)
            from_col = self._safe_identifier(fk[3])
            ref_table = self._safe_identifier(fk[2])
            to_col = self._safe_identifier(fk[4]) if fk[4] else ""
            src_col_id = self._object_id("column", table, from_col)
            tgt_col_id = (
                self._object_id("column", ref_table, to_col)
                if to_col
                else self._object_id("table", ref_table)
            )
            sink.edges.append(
                EnrichmentEdge(
                    source=src_col_id,
                    target=tgt_col_id,
                    rel_type="FOREIGN_KEY",
                )
            )

    # --- generic SQL introspection (information_schema) --------------- #
    def _introspect_sql(
        self,
        ds_id: str,
        nodes: list[GraphNode],
        edges: list[EnrichmentEdge],
    ) -> None:
        sink = _SchemaSink(nodes, edges)
        col_rows = self._sql_read(
            "SELECT table_name, column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema NOT IN "
            "('pg_catalog','information_schema','sys','mysql','performance_schema') "
            "ORDER BY table_name, ordinal_position",
            None,
            max_rows=_MAX_SCHEMA_FIELDS,
        )
        seen_tables: set[str] = set()
        for row in col_rows:
            self._introspect_sql_column_row(ds_id, row, sink, seen_tables)
        self._introspect_sql_foreign_keys(sink)

    def _introspect_sql_column_row(
        self,
        ds_id: str,
        row: Mapping[str, Any],
        sink: _SchemaSink,
        seen_tables: set[str],
    ) -> None:
        raw_table = row.get("table_name") or row.get("TABLE_NAME")
        raw_column = row.get("column_name") or row.get("COLUMN_NAME")
        if not raw_table or not raw_column:
            return
        table = self._safe_identifier(raw_table)
        col_name = self._safe_identifier(raw_column)
        data_type = str(row.get("data_type") or row.get("DATA_TYPE") or "")[
            :_MAX_IDENTIFIER_BYTES
        ]
        table_id = self._object_id("table", table)
        if table not in seen_tables:
            if len(seen_tables) >= _MAX_SCHEMA_TABLES:
                raise RuntimeError("database schema has too many tables")
            seen_tables.add(table)
            sink.nodes.append(
                GraphNode(id=table_id, type="Table", props={"name": table})
            )
            sink.edges.append(
                EnrichmentEdge(source=ds_id, target=table_id, rel_type="HAS_TABLE")
            )
        col_id = self._object_id("column", table, col_name)
        sink.nodes.append(
            GraphNode(
                id=col_id,
                type="Column",
                props={"name": col_name, "data_type": data_type},
            )
        )
        sink.edges.append(
            EnrichmentEdge(source=table_id, target=col_id, rel_type="HAS_COLUMN")
        )

    def _introspect_sql_foreign_keys(self, sink: _SchemaSink) -> None:
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
                max_rows=_MAX_SCHEMA_FIELDS,
            )
            for fk in fk_rows:
                src_table = self._safe_identifier(fk.get("src_table"))
                src_col = self._safe_identifier(fk.get("src_col"))
                ref_table = self._safe_identifier(fk.get("ref_table"))
                ref_col = self._safe_identifier(fk.get("ref_col"))
                src = self._object_id("column", src_table, src_col)
                tgt = self._object_id("column", ref_table, ref_col)
                sink.edges.append(
                    EnrichmentEdge(source=src, target=tgt, rel_type="FOREIGN_KEY")
                )
        except Exception as exc:  # pragma: no cover - dialect variance
            logger.debug(
                "database foreign-key introspection skipped kind=%s error_type=%s",
                self.kind,
                type(exc).__name__,
            )

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
        db = client[self._safe_identifier(db_name)]
        try:
            collections = db.list_collection_names()
            if len(collections) > _MAX_SCHEMA_TABLES:
                raise RuntimeError("database schema has too many collections")
            field_count = 0
            for raw_collection in collections:
                coll_name = self._safe_identifier(raw_collection)
                if not _MONGO_COLLECTION_RE.fullmatch(coll_name):
                    continue
                coll_id = self._object_id("collection", coll_name)
                nodes.append(
                    GraphNode(
                        id=coll_id,
                        type="Collection",
                        props={"name": coll_name},
                    )
                )
                edges.append(
                    EnrichmentEdge(
                        source=ds_id,
                        target=coll_id,
                        rel_type="HAS_TABLE",
                    )
                )
                sample = db[coll_name].find_one() or {}
                if not isinstance(sample, Mapping):
                    continue
                for raw_field, value in sample.items():
                    field_count += 1
                    if field_count > _MAX_SCHEMA_FIELDS:
                        raise RuntimeError("database schema has too many fields")
                    field = self._safe_identifier(raw_field)
                    field_id = self._object_id("field", coll_name, field)
                    nodes.append(
                        GraphNode(
                            id=field_id,
                            type="Field",
                            props={
                                "name": field,
                                "data_type": type(value).__name__[:64],
                            },
                        )
                    )
                    edges.append(
                        EnrichmentEdge(
                            source=coll_id,
                            target=field_id,
                            rel_type="HAS_COLUMN",
                        )
                    )
        finally:
            client.close()
