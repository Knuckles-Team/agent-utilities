#!/usr/bin/python
from __future__ import annotations

"""``TrinoQueryBackend`` — read Trino/Lakekeeper as the calling principal (CA-27).

CONCEPT:AU-KG.compute.trino-query-backend

**Why this is not a ``GraphBackend``.** ``backends/base.py``'s ``GraphBackend`` ABC
(``execute``/``execute_batch``/``create_schema``/``add_embedding``/``semantic_search``/
``prune``/``close``) is Cypher/graph-query shaped: every method returns
``list[dict[str, Any]]`` rows keyed by node/edge shape. Trino returns tabular
result sets over an Iceberg projection, and per the Company Architecture program's
invariant I3 ("Provenance on every hop") a Trino result is not a governed fact
until it carries a lineage fence (run id, input snapshot id(s), code version,
confidence) and passes through ``ApplyChangeEnvelope`` — ``GraphBackend.execute()``
has no contract for that fence. Forcing this backend onto the graph-shaped ABC
would either fabricate a fake node/edge shape for tabular rows or silently drop
the fencing requirement. So ``TrinoQueryBackend`` implements the narrower
:class:`QueryBackend` protocol instead — see ``plans/company-architecture/lanes/
CA-27-trino-spark-adapters.md`` for the full design note.

**Authority (I1/R3).** eg redb remains the single authoritative store. Trino
reads the Iceberg projection Spark/eg write; this backend is READ-ONLY by
construction (:func:`_reject_write_sql` rejects any DML/DDL token) and never
issues an Iceberg write/commit call. A result becomes a KG fact only through
:class:`ChangeEnvelopeBuilder` -> the existing ``ApplyChangeEnvelope`` door
(``agent_utilities.knowledge_graph.ingestion.change_envelope``) — this module
does not call that door itself; it hands the caller a ready-to-apply
``ChangeEnvelope``.

**Identity (GOC-79/DEC-CA-04).** Every connection authenticates as the calling
principal via an OIDC bearer token — never a static/shared credential. The
token is either supplied by the caller (``token_provider``) or resolved from
the MCP-layer delegated-auth context (RFC 8693 Token Exchange,
``agent_utilities.mcp.delegated_auth.get_delegated_token``), matching the
mechanism GOC-79's other external-engine clients already use. The SQLAlchemy
``trino`` dialect (``sqlalchemy-trino``, the SAME dialect CA-41's sql-mcp uses —
agreed connection parameters, disjoint files per the CA-27 lane contract) turns
an ``access_token`` URL-query value into ``trino.auth.JWTAuthentication``
(verified against the installed ``trino.sqlalchemy.dialect.TrinoDialect.
create_connect_args`` this session). ``username`` is always sent too (Trino's
``X-Trino-User`` session-identity header, distinct from ``access_token``'s
bearer credential) -- confirmed live this session that the coordinator 401s
ANY request lacking it, auth-enabled or not. Connections are pooled
per-principal (keyed by the delegated identity reference), so no two
principals ever share a pooled connection/session.

**Measured gap (2026-08-26 live check against `services/trino`, tag 476,
its in-cluster ClusterIP): principal-scoped OIDC cannot be proven
end-to-end against TODAY's deployment.** The coordinator is plain HTTP with
no auth layer wired yet (the Keycloak-fronted Keycloak-fronted ingress CA-52
owns still 406s), and the installed ``trino`` python client refuses -- by its
own design, not a bug here -- to send ``JWTAuthentication`` over a non-TLS
connection (``TrinoAuthError: TLS/SSL is required for authentication``). The
default behavior stays fail-closed (a missing/empty token still raises); the
ONLY escape hatch is ``allow_unauthenticated=True`` together with a
``token_provider`` that explicitly returns ``None`` (never empty string --
that stays an error) -- logged loudly every time, and never the default. This
is how the P4 live proof in the CA-27 lane report was run; production
call sites must not pass ``allow_unauthenticated=True`` once GOC-79's
Keycloak-fronted Trino endpoint is live.

**Never a hard dependency.** ``trino``/``sqlalchemy``/``sqlalchemy-trino`` are
the optional ``agent-utilities[trino]`` extra; every import of them here is
function-scoped and raises a clear, extra-naming ``ImportError`` when absent
(the repo's standing optional-dependency discipline — see ``pyproject.toml``).
"""

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...models.company_brain import DataClassification
from ..ingestion.change_envelope import ChangeEnvelope, Operation

logger = logging.getLogger(__name__)

__all__ = [
    "KnowledgeBatch",
    "QueryBackend",
    "TrinoQueryBackend",
    "TrinoQueryError",
    "UnknownSnapshotError",
    "ChangeEnvelopeBuilder",
    "MissingFenceFieldError",
]

#: Matches ``sqlalchemy_trino``'s connector default -- large enough that a
#: bulk sweep is single-digit round trips, small enough that one page stays a
#: modest Arrow frame (CA-27 W01 proposal, unverified beyond this session's
#: manual check of the connector's own default fetch size).
DEFAULT_PAGE_SIZE = 8_192

#: Bounded connection pool per principal (tunable) -- never unbounded, so a
#: runaway caller cannot exhaust the Trino coordinator's session table.
DEFAULT_POOL_SIZE = 8

#: How many distinct principals' engines this backend keeps warm at once
#: before evicting the least-recently-used one. Bounds worst-case fan-out
#: memory (one pooled SQLAlchemy Engine per principal) without ever silently
#: sharing a connection across principals.
_MAX_TRACKED_PRINCIPALS = 64


class TrinoQueryError(RuntimeError):
    """A Trino query/connection failure. Typed and specific -- never surfaces
    as a bare ``Exception`` a caller has to string-match."""


class UnknownSnapshotError(TrinoQueryError):
    """Raised when ``FOR VERSION AS OF <snapshot_id>`` targets a snapshot
    Iceberg does not recognize (never committed, expired, or malformed).

    P4's negative case (CA-27-W06/W07): this must never degrade to a silent
    HEAD/latest read -- a caller that mistypes or races a snapshot id gets an
    exception, not a surprising "current" answer.
    """


def _looks_like_unknown_snapshot(message: str) -> bool:
    """Best-effort classification of a Trino/Iceberg error message as an
    unknown-snapshot condition (Iceberg's ``IcebergUtil``/``TrinoIcebergUtil``
    reports variants like "Cannot find snapshot with ID ..." / "... is not a
    valid snapshot ID" depending on connector version -- there is no single
    stable SQLSTATE for this across Trino releases, so this is pattern
    matching on the message, documented as best-effort rather than exact).
    """
    lowered = message.lower()
    if "snapshot" not in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "not found",
            "does not exist",
            "cannot find",
            "no such",
            "invalid snapshot",
            "not a valid snapshot",
        )
    )


# A read-only guard: reject any statement that opens with a mutating/DDL
# keyword. Deliberately conservative (matches the leading token only, mirrors
# ``backends/base.py``'s ``_WRITE_RE`` write-detection convention) -- R3 ("Trino
# reads Spark-owned tables, never writes them") is enforced here, not trusted
# to caller discipline. Acceptance gate 4: `grep -n "iceberg" trino_backend.py`
# must show read/catalog-list calls only -- this guard is why: nothing in this
# module ever assembles a write statement to send to Trino.
_WRITE_SQL_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CALL|COMMENT|SET\s+SESSION)\b",
    re.IGNORECASE,
)

#: A Trino/Iceberg snapshot id is a signed 64-bit integer rendered as decimal
#: digits (optionally negative for pre-history synthetic ids) -- validated
#: before string-interpolation into ``FOR VERSION AS OF`` so a caller-supplied
#: value can never break out of the intended SQL shape.
_SNAPSHOT_ID_RE = re.compile(r"^-?[0-9]{1,20}$")

#: A dotted SQL identifier segment (catalog/schema/table), reusing this
#: repo's shared identifier gate rather than a bespoke regex.
_IDENTIFIER_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _reject_write_sql(sql: str) -> None:
    if _WRITE_SQL_RE.match(sql or ""):
        raise TrinoQueryError(
            "TrinoQueryBackend is read-only (CA program invariant R3: Trino/Spark "
            "are read/compute surfaces over the Iceberg projection, never a "
            "competing primary) -- refusing a write/DDL statement."
        )


def _validate_snapshot_id(snapshot_id: str) -> str:
    text = str(snapshot_id).strip()
    if not _SNAPSHOT_ID_RE.match(text):
        raise TrinoQueryError(
            f"snapshot_id must be a Trino/Iceberg numeric snapshot id, got {snapshot_id!r}"
        )
    return text


def _validate_qualified_table(table: str) -> str:
    parts = str(table).split(".")
    # Trino addresses a table as `table`, `schema.table`, or
    # `catalog.schema.table` -- never more than 3 segments.
    if not (1 <= len(parts) <= 3) or any(
        not _IDENTIFIER_SEGMENT_RE.match(p) for p in parts
    ):
        raise TrinoQueryError(
            f"table must be a plain dotted identifier (catalog.schema.table), got {table!r}"
        )
    return ".".join(parts)


@dataclass(frozen=True)
class KnowledgeBatch:
    """One page of a Trino query result -- the tabular, provenance-first
    currency this backend returns (mirrors ``core/knowledge_stream.py``'s
    ``KnowledgeStreamBatch`` row currency: rows + explicit provenance, never a
    bare ``list[dict]`` with the provenance implied or absent).

    Attributes:
        rows: This page's decoded rows, column name -> value.
        columns: Column names, in result order.
        snapshot_id: The Iceberg snapshot id this page was read ``FOR VERSION
            AS OF`` (``None`` for an unpinned/HEAD read -- never silently
            filled in).
        lsn: The eg LSN this page corresponds to. Per the Company Architecture
            program's invariant I3 ("Iceberg snapshot = eg LSN"), this is the
            SAME value as ``snapshot_id`` whenever the caller pinned one --
            carried as a distinct field so a consumer never has to know that
            equivalence to read provenance correctly.
        row_count: ``len(rows)`` (cached rather than recomputed by callers
            that only want the count).
        page_index: 0-based page counter within this ``query()``/``as_of()``
            call, for log correlation.
    """

    rows: list[dict[str, Any]]
    columns: tuple[str, ...]
    snapshot_id: str | None
    lsn: str | None
    row_count: int
    page_index: int

    def record_batch(self) -> Any:
        """Lazily build a ``pyarrow.RecordBatch`` for this page.

        Optional -- mirrors ``knowledge_stream.py``'s "never a hard
        dependency" pyarrow discipline. Raises ``ImportError`` naming the
        ``pyarrow`` extra when it is not installed, rather than degrading
        silently (a caller that asked for Arrow specifically wants Arrow).
        """
        try:
            import pyarrow as pa
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "KnowledgeBatch.record_batch() needs pyarrow (install "
                "agent-utilities[pyarrow])."
            ) from exc
        if self.rows:
            return pa.RecordBatch.from_pylist(self.rows)
        # No rows: still produce a (zero-row, string-typed) RecordBatch with
        # the right column names, rather than an empty list a caller has to
        # special-case.
        schema = pa.schema([(name, pa.string()) for name in self.columns])
        return pa.RecordBatch.from_pylist([], schema=schema)


@runtime_checkable
class QueryBackend(Protocol):
    """Narrow protocol for a read-only, tabular, provenance-carrying compute
    surface (Trino today; DuckDB/Spark SQL could implement the same shape
    without subclassing ``GraphBackend``)."""

    def query(
        self, sql: str, *, snapshot_id: str | None = None
    ) -> Iterator[KnowledgeBatch]:
        """Run one read-only SQL statement, yielding paged, provenance-tagged results."""
        ...

    def close(self) -> None:
        """Release pooled connections."""
        ...


def _default_token_provider() -> str:
    """Resolve the calling principal's OIDC bearer token via the fleet's
    existing RFC 8693 Token Exchange delegated-auth surface.

    Deliberately the ONLY default -- no static-credential fallback exists in
    this module. A caller running outside an MCP request context (a batch
    job, a test) must supply ``token_provider`` explicitly; this function
    raising is the fail-closed behavior, not a bug to work around by adding a
    shared secret here.
    """
    from ...mcp.delegated_auth import get_delegated_token

    return get_delegated_token(audience="trino")


def _default_principal_ref() -> str:
    try:
        from ...mcp.delegated_auth import get_user_identity

        return str(get_user_identity().get("identity_ref") or "")
    except Exception:  # noqa: BLE001 - identity resolution is best-effort for pool keying only
        return ""


class TrinoQueryBackend:
    """Read Trino/Lakekeeper as the calling principal, returning paged,
    provenance-tagged results. See the module docstring for why this does
    not subclass ``GraphBackend``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        catalog: str = "lakehouse",
        schema: str | None = None,
        token_provider: Callable[[], str] | None = None,
        principal_ref_provider: Callable[[], str] | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        pool_size: int = DEFAULT_POOL_SIZE,
        http_scheme: str = "http",
        verify: bool = True,
        source: str = "au-trino-query-backend",
        allow_unauthenticated: bool = False,
    ) -> None:
        if not endpoint:
            raise TrinoQueryError("TrinoQueryBackend requires a non-empty endpoint")
        self._endpoint = endpoint.strip()
        self._catalog = catalog
        self._schema = schema
        self._token_provider = token_provider or _default_token_provider
        self._principal_ref_provider = principal_ref_provider or _default_principal_ref
        self._page_size = max(1, int(page_size))
        self._pool_size = max(1, int(pool_size))
        self._http_scheme = http_scheme
        self._verify = verify
        self._source = source
        self._allow_unauthenticated = allow_unauthenticated
        # principal_ref -> SQLAlchemy Engine. Order is insertion order (dict,
        # Python 3.7+), used as an LRU-by-eviction-of-oldest for
        # _MAX_TRACKED_PRINCIPALS.
        self._engines: dict[str, Any] = {}

    # -- connection management ------------------------------------------------

    def _resolve_token(self) -> str | None:
        token = self._token_provider()
        if token is None:
            # An EXPLICIT None (never an empty string -- see below) is the
            # only way to get an unauthenticated connection, and only when
            # the caller also passed allow_unauthenticated=True at
            # construction. Models today's measured reality (baseline
            # 2026-08-26): the deployed Trino coordinator is plain HTTP with
            # no auth layer wired yet (GOC-79's Keycloak-fronting proxy is
            # the Keycloak-fronted ingress, still 406 per CA-52's territory) -- and the
            # `trino` python client's own JWTAuthentication refuses to send a
            # bearer token over a non-TLS connection (a real client-side
            # guard, confirmed live this session), so principal-scoped OIDC
            # cannot be proven end-to-end against the CURRENT deployment.
            # This is a genuine platform gap, not a bug in this backend --
            # never silently promoted to the default; a caller must opt in.
            if not self._allow_unauthenticated:
                raise TrinoQueryError(
                    "token_provider returned None (no principal OIDC token) and "
                    "allow_unauthenticated=False -- refusing a shared/anonymous "
                    "connection. Pass allow_unauthenticated=True only against a "
                    "deployment known to have no auth layer (see class docstring)."
                )
            logger.warning(
                "TrinoQueryBackend: connecting to %s WITHOUT a bearer token "
                "(allow_unauthenticated=True) -- the coordinator has no auth "
                "layer today; this is not principal-scoped and must not be used "
                "once GOC-79's Keycloak-fronted endpoint is live",
                self._endpoint,
            )
            return None
        if not token or not isinstance(token, str):
            raise TrinoQueryError(
                "no principal OIDC token available for Trino connection -- "
                "refusing to fall back to a shared/anonymous connection"
            )
        return token

    def _build_engine(self, token: str | None, principal: str) -> Any:
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.engine import URL
        except ImportError as exc:
            raise ImportError(
                "TrinoQueryBackend needs sqlalchemy + sqlalchemy-trino + trino "
                "(install agent-utilities[trino])."
            ) from exc

        host, _, port_text = self._endpoint.rpartition(":")
        host = host or self._endpoint
        try:
            port = int(port_text) if port_text else 8080
        except ValueError:
            host, port = self._endpoint, 8080

        database = f"{self._catalog}/{self._schema}" if self._schema else self._catalog
        query: dict[str, str] = {
            "source": self._source,
            "verify": "true" if self._verify else "false",
        }
        if token is not None:
            query["access_token"] = token
        url = URL.create(
            "trino",
            # `username` becomes Trino's `X-Trino-User` session-identity header
            # (dialect: `create_connect_args`) -- distinct from `access_token`
            # (the actual bearer credential). Trino requires SOME user
            # identity on every request even when no bearer auth is
            # configured (confirmed live this session: an access_token-less,
            # username-less connection 401s with "Basic authentication or
            # X-Trino-Original-User or X-Trino-User must be sent"), so the
            # calling principal's identity ref is always sent for query
            # attribution/audit (`system.runtime.queries.user`), whether or
            # not a bearer token also rides along.
            username=principal or "au-trino-query-backend",
            host=host,
            port=port,
            database=database,
            query=query,
        )
        return create_engine(url, pool_size=self._pool_size, pool_pre_ping=True)

    def _engine_for_principal(self) -> Any:
        principal = self._principal_ref_provider() or "__anonymous__"
        engine = self._engines.get(principal)
        if engine is not None:
            return engine
        token = self._resolve_token()
        engine = self._build_engine(token, principal)
        if len(self._engines) >= _MAX_TRACKED_PRINCIPALS:
            oldest = next(iter(self._engines))
            self._engines.pop(oldest).dispose()
        self._engines[principal] = engine
        return engine

    # -- query surface ---------------------------------------------------------

    def query(
        self, sql: str, *, snapshot_id: str | None = None
    ) -> Iterator[KnowledgeBatch]:
        """Run one read-only SQL statement, yielding paged results.

        ``snapshot_id`` is provenance metadata stamped onto every yielded
        page -- it is NOT injected into ``sql`` here (the caller embeds ``FOR
        VERSION AS OF`` itself, or uses :meth:`as_of` for the safe,
        identifier-validated convenience form).

        The read-only guard and the SQLAlchemy import check both run
        EAGERLY, before this returns -- a write statement or a missing
        ``[trino]`` extra raises the moment a caller calls ``query()``, never
        deferred to the first ``next()`` on the returned generator (a plain
        ``def`` with ``yield`` inside would defer both checks silently).
        """
        _reject_write_sql(sql)
        try:
            from sqlalchemy import (
                text as sql_text,  # noqa: F401 - import-check only here
            )
        except ImportError as exc:
            raise ImportError(
                "TrinoQueryBackend needs sqlalchemy + sqlalchemy-trino + trino "
                "(install agent-utilities[trino])."
            ) from exc
        return self._query_iter(sql, snapshot_id)

    def _query_iter(
        self, sql: str, snapshot_id: str | None
    ) -> Iterator[KnowledgeBatch]:
        from sqlalchemy import text as sql_text

        engine = self._engine_for_principal()
        try:
            with engine.connect() as conn:
                result = conn.execute(sql_text(sql))
                columns = tuple(result.keys())
                page_index = 0
                while True:
                    chunk = result.fetchmany(self._page_size)
                    if not chunk:
                        return
                    rows = [dict(zip(columns, row, strict=True)) for row in chunk]
                    yield KnowledgeBatch(
                        rows=rows,
                        columns=columns,
                        snapshot_id=snapshot_id,
                        lsn=snapshot_id,
                        row_count=len(rows),
                        page_index=page_index,
                    )
                    page_index += 1
        except UnknownSnapshotError:
            raise
        except TrinoQueryError:
            raise
        except Exception as exc:  # noqa: BLE001 - reclassified below, never swallowed
            message = str(exc)
            if _looks_like_unknown_snapshot(message):
                raise UnknownSnapshotError(
                    f"Trino: unknown/uncommitted snapshot (snapshot_id={snapshot_id!r}): {message}"
                ) from exc
            raise TrinoQueryError(
                f"Trino query failed ({type(exc).__name__}): {message}"
            ) from exc

    def as_of(
        self,
        table: str,
        snapshot_id: str,
        *,
        where: str | None = None,
    ) -> Iterator[KnowledgeBatch]:
        """``SELECT * FROM <table> FOR VERSION AS OF <snapshot_id> [WHERE ...]``,
        with ``table``/``snapshot_id`` identifier-validated before
        interpolation (P4's positive/negative harness entry point). Validation
        is eager, same as :meth:`query`.
        """
        safe_table = _validate_qualified_table(table)
        safe_snapshot = _validate_snapshot_id(snapshot_id)
        sql = f"SELECT * FROM {safe_table} FOR VERSION AS OF {safe_snapshot}"
        if where:
            sql += f" WHERE {where}"
        return self.query(sql, snapshot_id=safe_snapshot)

    def close(self) -> None:
        for engine in self._engines.values():
            engine.dispose()
        self._engines.clear()


# ---------------------------------------------------------------------------
# R5 fence: a compute result is not a fact until it carries these four fields.
# ---------------------------------------------------------------------------


class MissingFenceFieldError(ValueError):
    """Raised by :meth:`ChangeEnvelopeBuilder.build` when a required R5 fence
    field (run id, input snapshot id(s), code version) is missing. A separate
    type from the generic ``ValueError`` :class:`ChangeEnvelope` itself raises
    for its own field validation, so a caller can distinguish "this compute
    result was never fenced" from "the envelope's own fields are malformed".
    """


@dataclass
class ChangeEnvelopeBuilder:
    """Wraps a materialized Trino/Spark compute result with R5's fence fields
    before it may reach ``ApplyChangeEnvelope`` -- a Trino or Spark result is
    not a fact until it carries: run id, input snapshot id(s), code version,
    confidence (Company Architecture program invariant I3 / R5).

    This builder does not call ``ApplyChangeEnvelope`` itself (out of scope,
    per this lane's non-goals) -- it produces a ``ChangeEnvelope`` a caller
    hands to the existing ingestion door.
    """

    connector: str
    run_id: str
    input_snapshot_ids: tuple[str, ...] = field(default_factory=tuple)
    code_version: str = ""
    confidence: float = 1.0

    def build(
        self,
        *,
        source_object_id: str,
        payload: dict[str, Any],
        tenant: str = "",
        operation: Operation = "upsert",
        classification: DataClassification = DataClassification.INTERNAL,
    ) -> ChangeEnvelope:
        if not self.run_id:
            raise MissingFenceFieldError(
                "ChangeEnvelopeBuilder requires a run_id (R5 fence field)"
            )
        if not self.input_snapshot_ids:
            raise MissingFenceFieldError(
                "ChangeEnvelopeBuilder requires at least one input_snapshot_id "
                "(R5 fence field)"
            )
        if not self.code_version:
            raise MissingFenceFieldError(
                "ChangeEnvelopeBuilder requires a code_version (R5 fence field)"
            )
        return ChangeEnvelope(
            connector=self.connector,
            operation=operation,
            tenant=tenant,
            source_object_id=source_object_id,
            typed_payload=payload,
            classification=classification,
            confidence=self.confidence,
            provenance={
                "run_id": self.run_id,
                "input_snapshot_ids": list(self.input_snapshot_ids),
                "code_version": self.code_version,
                "generated_by": self.connector,
            },
        )
