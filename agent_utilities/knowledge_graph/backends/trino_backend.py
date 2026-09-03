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
principal via an OIDC bearer token — never a static/shared credential. Both
the token and opaque principal reference are required injected ports; this
adapter has no knowledge of MCP, configuration, or a concrete identity
provider. The SQLAlchemy
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

**Transport.** Authenticated Trino connections require TLS. An explicit HTTP
endpoint is rejected before engine construction; a bare host/port is treated
as HTTPS. Missing/empty credentials and principal references fail closed.
Tokens are resolved on every dispatch and are part of the bounded pool key, so
rotation or revocation cannot silently reuse a stale principal session.

**Never a hard dependency.** ``trino``/``sqlalchemy``/``sqlalchemy-trino`` are
the optional ``agent-utilities[trino]`` extra; every import of them here is
function-scoped and raises a clear, extra-naming ``ImportError`` when absent
(the repo's standing optional-dependency discipline — see ``pyproject.toml``).
"""

import hashlib
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ...models.company_brain import DataClassification
from ..core.tabular_query_service import KnowledgeBatch
from ..ingestion.change_envelope import ChangeEnvelope, Operation

__all__ = [
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


def _endpoint_text(endpoint: Any) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise TrinoQueryError("TrinoQueryBackend requires a non-empty endpoint")
    return endpoint.strip()


def _validated_endpoint_parts(rendered: str) -> Any:
    parsed = urlsplit(rendered if "://" in rendered else f"//{rendered}")
    if parsed.scheme and parsed.scheme.lower() != "https":
        raise TrinoQueryError("authenticated Trino endpoints require HTTPS")
    if parsed.username or parsed.password:
        raise TrinoQueryError("Trino endpoint must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise TrinoQueryError("Trino endpoint must contain only host and port")
    if not parsed.hostname:
        raise TrinoQueryError("Trino endpoint has no valid host")
    return parsed


def _validated_endpoint(endpoint: Any) -> tuple[str, int]:
    """Return a TLS-only host/port pair without retaining endpoint credentials."""

    parsed = _validated_endpoint_parts(_endpoint_text(endpoint))
    try:
        port = parsed.port
    except ValueError as exc:
        raise TrinoQueryError("Trino endpoint has an invalid port") from exc
    port = {None: 443}.get(port, port)
    if port < 1:
        raise TrinoQueryError("Trino endpoint has an invalid port")
    return parsed.hostname, port


def _required_provider(provider: Any, *, name: str) -> Callable[[], str]:
    if not callable(provider):
        raise TrinoQueryError(f"TrinoQueryBackend requires an injected {name}")
    return provider


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
        token_provider: Callable[[], str],
        principal_ref_provider: Callable[[], str],
        page_size: int = DEFAULT_PAGE_SIZE,
        pool_size: int = DEFAULT_POOL_SIZE,
        verify: bool = True,
        source: str = "au-trino-query-backend",
    ) -> None:
        self._host, self._port = _validated_endpoint(endpoint)
        self._catalog = catalog
        self._schema = schema
        self._token_provider = _required_provider(token_provider, name="token_provider")
        self._principal_ref_provider = _required_provider(
            principal_ref_provider, name="principal_ref_provider"
        )
        self._page_size = max(1, int(page_size))
        self._pool_size = max(1, int(pool_size))
        self._verify = verify
        self._source = source
        self._engines: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._engines_lock = threading.RLock()

    # -- connection management ------------------------------------------------

    def _resolve_token(self) -> str:
        token = self._token_provider()
        if not isinstance(token, str) or not token.strip():
            raise TrinoQueryError(
                "no principal OIDC token available for Trino connection -- "
                "refusing an anonymous connection"
            )
        return token

    def _resolve_principal(self) -> str:
        principal = self._principal_ref_provider()
        if not isinstance(principal, str) or not principal.strip():
            raise TrinoQueryError("verified Trino principal identity is required")
        return principal.strip()

    @staticmethod
    def _token_fingerprint(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _pop_principal_engines(self, principal: str) -> list[Any]:
        return [
            self._engines.pop(key)
            for key in tuple(self._engines)
            if key[0] == principal
        ]

    def _discard_principal_engines(self, principal: str) -> None:
        with self._engines_lock:
            stale = self._pop_principal_engines(principal)
        for engine in stale:
            engine.dispose()

    def _resolve_token_for_principal(self, principal: str) -> str:
        try:
            return self._resolve_token()
        except Exception:
            self._discard_principal_engines(principal)
            raise

    def _build_engine(self, token: str, principal: str) -> Any:
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.engine import URL
        except ImportError as exc:
            raise ImportError(
                "TrinoQueryBackend needs sqlalchemy + sqlalchemy-trino + trino "
                "(install agent-utilities[trino])."
            ) from exc

        database = f"{self._catalog}/{self._schema}" if self._schema else self._catalog
        query: dict[str, str] = {
            "source": self._source,
            "verify": "true" if self._verify else "false",
            "http_scheme": "https",
            "access_token": token,
        }
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
            username=principal,
            host=self._host,
            port=self._port,
            database=database,
            query=query,
        )
        return create_engine(
            url,
            connect_args={"http_scheme": "https"},
            pool_size=self._pool_size,
            pool_pre_ping=True,
        )

    def _engine_for_cache_key(
        self, cache_key: tuple[str, str], token: str, principal: str
    ) -> Any:
        with self._engines_lock:
            engine = self._engines.get(cache_key)
            if engine is not None:
                self._engines.move_to_end(cache_key)
                return engine
            stale = self._pop_principal_engines(principal)
            for old_engine in stale:
                old_engine.dispose()
            engine = self._build_engine(token, principal)
            self._engines[cache_key] = engine
            if len(self._engines) > _MAX_TRACKED_PRINCIPALS:
                _, evicted = self._engines.popitem(last=False)
                evicted.dispose()
            return engine

    def _engine_for_principal(self) -> Any:
        principal = self._resolve_principal()
        token = self._resolve_token_for_principal(principal)
        cache_key = (principal, self._token_fingerprint(token))
        return self._engine_for_cache_key(cache_key, token, principal)

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
        with self._engines_lock:
            engines = tuple(self._engines.values())
            self._engines.clear()
        for engine in engines:
            engine.dispose()


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
