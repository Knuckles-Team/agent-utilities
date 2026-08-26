"""Tenant/principal-scoped read-only registry over the native fleet catalog.

The catalog writer is :mod:`agent_utilities.knowledge_graph.core.fleet_catalog_tables`;
this module is deliberately a reader.  It never starts MCP children, probes a
live server, writes a row, or falls back to a second store.  Authorization is
resolved from the ambient verified :class:`GraphSession` plus the
process-owned broker's current exact OAuth-grant fingerprints, then embedded in
the catalog SQL predicate before filtering, sorting, pagination, or counts.

CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables
CONCEPT:AU-OS.state.unified-durable-state-externalization
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Generic, Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_utilities.knowledge_graph.core.fleet_catalog_tables import (
    DISCOVERY_AUTHORITY_OAUTH_GRANT,
    DISCOVERY_AUTHORITY_TENANT_LOCAL,
)
from agent_utilities.knowledge_graph.core.table_ingest import (
    _safe_ident,
    _sql_literal,
)

logger = logging.getLogger(__name__)

_MAX_LIMIT = 100
_DEFAULT_LIMIT = 50
_MAX_QUERY_BYTES = 128
_MAX_CATALOG_ROWS = 10_000
_MAX_CURSOR_BYTES = 4096
_CURSOR_TTL_SECONDS = 900.0
# Bound on one blocking catalog SQL call (unix-socket engine RPC), offloaded
# to a worker thread via ``_offload_catalog_call``. 30s comfortably covers the
# measured queueing-driven latency of a single call under load (observed up
# to ~11.7s) while still failing a genuinely hung engine call closed instead
# of hanging the request indefinitely.
_CATALOG_READ_TIMEOUT_S = 30.0

registry_router = APIRouter(tags=["registry"])


class _RegistryModel(BaseModel):
    """Strict public projection model; catalog shape errors fail closed."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RegistryServer(_RegistryModel):
    """Public desired registration metadata (never credentials)."""

    id: str
    name: str
    transport: str = ""
    url: str = ""
    enabled: bool = False


class RegistryDiscovery(_RegistryModel):
    """Privacy-safe observed discovery state for one server."""

    id: str
    server_id: str
    server_name: str
    reachable: bool = False
    last_error: str = ""
    tool_count: int = 0
    skill_count: int = 0
    prompt_count: int = 0
    resource_count: int = 0
    observed_at: str = ""


class RegistryTool(_RegistryModel):
    """Public tool contract metadata; raw schema is intentionally omitted."""

    id: str
    server_id: str
    server_name: str
    name: str
    description: str = ""
    schema_digest: str = ""
    tool_mode: str = ""
    enabled: bool = False


class RegistryPrompt(_RegistryModel):
    id: str
    server_id: str
    server_name: str
    name: str
    description: str = ""
    uri: str = ""


class RegistryResource(_RegistryModel):
    id: str
    server_id: str
    server_name: str
    uri: str = ""
    name: str = ""
    description: str = ""
    mime_type: str = ""
    resource_kind: str = ""


class RegistrySkill(_RegistryModel):
    id: str
    name: str
    description: str = ""
    uri: str = ""
    skill_type: str = ""
    classification: str = ""
    provider: str = ""
    mcp_server: str = ""
    enabled: bool = False


RegistryItem = TypeVar("RegistryItem", bound=BaseModel)


class RegistryPage(BaseModel, Generic[RegistryItem]):
    """Typed, bounded page envelope shared by every registry collection."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    kind: str
    items: list[RegistryItem]
    count: int = Field(ge=0)
    next_cursor: str | None = None


class RegistryItemEnvelope(BaseModel, Generic[RegistryItem]):
    status: Literal["ok"] = "ok"
    item: RegistryItem


class RegistryKindResult(BaseModel):
    """One kind's slice of a multi-kind page — degrades independently.

    ``status="unavailable"`` (with ``reason`` set) means THIS kind's catalog
    read failed; it is never conflated with "zero items", which is a
    genuine, successful empty page (``status="ok"``, ``items=[]``). Mirrors
    ``agent_webui.api_extensions._read_fleet_catalog``'s own per-kind
    ``None``-on-failure contract, translated into this route's typed
    envelope shape instead of a raw ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unavailable"] = "ok"
    items: list[dict[str, Any]] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    next_cursor: str | None = None
    reason: str | None = None


class RegistryMultiKindPage(BaseModel):
    """``GET /api/registry?kinds=...`` envelope: one request, N kinds.

    ``kinds`` is keyed by the requested kind name. Pagination is PER KIND —
    each ``RegistryKindResult.next_cursor`` is fed back independently as
    that same kind's ``cursor_<kind>`` query parameter on the next request;
    there is no single global cursor advancing every kind in lockstep (a
    caller may be on page 3 of ``skills`` while still on page 1 of
    ``servers``). See ``_parse_multi_kind_request`` for the exact query
    parameter contract.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    kinds: dict[str, RegistryKindResult]


class _KindSpec:
    __slots__ = (
        "table",
        "columns",
        "model",
        "name_column",
        "authority_column",
        "principal_column",
        "grant_column",
    )

    def __init__(
        self,
        table: str,
        columns: tuple[str, ...],
        model: type[BaseModel],
        *,
        name_column: str = "name",
        authority_column: str | None = None,
        principal_column: str | None = None,
        grant_column: str | None = None,
    ) -> None:
        self.table = _safe_ident(table)
        self.columns = tuple(_safe_ident(column) for column in columns)
        self.model = model
        self.name_column = _safe_ident(name_column)
        self.authority_column = (
            _safe_ident(authority_column) if authority_column else None
        )
        self.principal_column = (
            _safe_ident(principal_column) if principal_column else None
        )
        self.grant_column = _safe_ident(grant_column) if grant_column else None


#
# PHASE A investigation — three sections `/api/enhanced/tools` still serves
# from the filesystem/KG (`builtin_tools`, `skill_graphs`, `skill_workflows`,
# see `agent_utilities.mcp.kg_server._build_tools_payload_sync`'s own
# docstring, "FIX LANE (collapse-tool-endpoints)") were evaluated as
# candidate additional `_KindSpec` entries here. None were added. Evidence:
#
# * ``builtin_tools`` — no SQL table exists at all. These are native,
#   in-process Python callables under ``agent_utilities/tools/*.py``, never
#   MCP-discovered and never written to any fleet-catalog table. Adding a
#   `_KindSpec` for this would require fabricating a table, which is exactly
#   what this task was told not to do.
# * ``skill_graphs`` — the ``skills`` table's schema *can* represent a
#   skill-graph row (``skill_type='graph'``), but
#   ``knowledge_graph/ingestion/skill_workflow_ingest.py`` explicitly SKIPS
#   ``skill_type: graph`` files during ingestion ("left for its own
#   ingester"), and nothing schedules that ingester automatically (only a
#   manual, explicit-``root`` on-demand action reaches it). In a typical
#   deployment the catalog rows for this kind are simply absent even though
#   the on-disk ``skill-graphs`` corpus is real and populated — serving this
#   kind from the catalog would silently show an empty page for a genuinely
#   non-empty corpus, the "losing freshness" failure mode this task called
#   out by name. Left out.
# * ``skill_workflows`` — the ``skills`` table CAN represent
#   id/name/description/enabled for a ``skill_type='workflow'`` row (these
#   DO get ingested on the automatic ``package_install`` tick, unlike
#   ``skill_graphs``), but the table itself
#   (``knowledge_graph/core/fleet_catalog_tables.py``'s ``TABLE_SKILLS`` DDL)
#   has no ``domain``/``tags`` columns at all — real fields on the existing
#   filesystem-sourced payload, parsed from each ``SKILL.md``'s frontmatter.
#   Adding this kind would silently blank those two fields for every caller
#   that switches to it. A prior lane evaluated this EXACT trade-off for
#   this EXACT data (`_build_tools_payload_sync`'s own docstring, "Moving
#   these two sections would silently blank domain/tags ... exactly the
#   'fabricate or silently drop' failure mode this fix lane was told to
#   avoid") and chose to keep it filesystem-sourced; no new evidence here
#   overturns that call, so it stays out too.
#
# All three remain reachable only through the existing filesystem/KG path
# (`GET /tools` / `/api/enhanced/tools`) until a real ingester exists for
# skill-graphs and the `skills` table grows `domain`/`tags` columns.
#
_KIND_SPECS: dict[str, _KindSpec] = {
    "servers": _KindSpec(
        "mcp_servers",
        ("id", "tenant_id", "name", "transport", "url", "enabled"),
        RegistryServer,
    ),
    "discoveries": _KindSpec(
        "mcp_server_discovery",
        (
            "id",
            "tenant_id",
            "server_id",
            "server_name",
            "reachable",
            "last_error",
            "tool_count",
            "skill_count",
            "prompt_count",
            "resource_count",
            "observed_at",
            "discovery_authority_kind",
            "discovery_principal",
            "discovery_grant_digest",
        ),
        RegistryDiscovery,
        name_column="server_name",
        authority_column="discovery_authority_kind",
        principal_column="discovery_principal",
        grant_column="discovery_grant_digest",
    ),
    "tools": _KindSpec(
        "mcp_tools",
        (
            "id",
            "tenant_id",
            "server_id",
            "server_name",
            "name",
            "description",
            "schema_digest",
            "tool_mode",
            "enabled",
            "discovery_authority_kind",
            "discovery_principal",
            "discovery_grant_digest",
        ),
        RegistryTool,
        authority_column="discovery_authority_kind",
        principal_column="discovery_principal",
        grant_column="discovery_grant_digest",
    ),
    "prompts": _KindSpec(
        "mcp_prompts",
        (
            "id",
            "tenant_id",
            "server_id",
            "server_name",
            "name",
            "description",
            "uri",
            "discovery_authority_kind",
            "discovery_principal",
            "discovery_grant_digest",
        ),
        RegistryPrompt,
        authority_column="discovery_authority_kind",
        principal_column="discovery_principal",
        grant_column="discovery_grant_digest",
    ),
    "resources": _KindSpec(
        "mcp_resources",
        (
            "id",
            "tenant_id",
            "server_id",
            "server_name",
            "uri",
            "name",
            "description",
            "mime_type",
            "resource_kind",
            "discovery_authority_kind",
            "discovery_principal",
            "discovery_grant_digest",
        ),
        RegistryResource,
        authority_column="discovery_authority_kind",
        principal_column="discovery_principal",
        grant_column="discovery_grant_digest",
    ),
    "skills": _KindSpec(
        "skills",
        (
            "id",
            "tenant_id",
            "name",
            "description",
            "uri",
            "skill_type",
            "classification",
            "provider",
            "mcp_server",
            "enabled",
            "discovery_authority_kind",
            "discovery_principal",
            "discovery_grant_digest",
        ),
        RegistrySkill,
        authority_column="discovery_authority_kind",
        principal_column="discovery_principal",
        grant_column="discovery_grant_digest",
    ),
}

_RESPONSE_MODELS: dict[str, tuple[Any, Any]] = {
    "servers": (RegistryPage[RegistryServer], RegistryItemEnvelope[RegistryServer]),
    "discoveries": (
        RegistryPage[RegistryDiscovery],
        RegistryItemEnvelope[RegistryDiscovery],
    ),
    "tools": (RegistryPage[RegistryTool], RegistryItemEnvelope[RegistryTool]),
    "prompts": (RegistryPage[RegistryPrompt], RegistryItemEnvelope[RegistryPrompt]),
    "resources": (
        RegistryPage[RegistryResource],
        RegistryItemEnvelope[RegistryResource],
    ),
    "skills": (RegistryPage[RegistrySkill], RegistryItemEnvelope[RegistrySkill]),
}


class CatalogUnavailable(RuntimeError):
    """Raised when the authoritative engine catalog cannot be read."""


def _get_catalog_engine() -> Any:
    """Resolve the existing engine singleton; no reader-owned store exists."""

    from agent_utilities.mcp.kg_server import _get_engine

    return _get_engine()


async def _offload_catalog_call(
    fn: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> Any:
    """Run one blocking catalog SQL call off the ASGI event loop, bounded by
    ``_CATALOG_READ_TIMEOUT_S``.

    ``_authorized_count``/``_authorized_page``/``_authorized_item`` each make
    a synchronous unix-socket engine RPC. Calling one of them directly from
    an ``async def`` route handler blocks the single gateway event loop for
    the RPC's full duration: one slow registry read then stalls *every*
    other in-flight request on the process, not just this one. This mirrors
    the dispatch-isolation pattern already used for engine RPCs elsewhere in
    this system (``asyncio.wait_for(asyncio.to_thread(...))`` around
    ``agent_utilities.mcp.kg_server``'s tool dispatch,
    CONCEPT:AU-ECO.mcp.gateway-dispatch-isolation): run the blocking call on
    a worker thread so the loop stays schedulable, and bound it with a
    deadline so a hung engine call fails this one request cleanly instead of
    hanging indefinitely. A deadline breach surfaces as a plain
    ``TimeoutError``, which the existing catch-all ``except Exception`` in
    ``_list_kind``/``_get_kind`` already maps to the same explicit
    ``catalog_unavailable`` 503 response used for any other catalog failure
    -- no separate handling is required at the call sites.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(fn, *args, **kwargs), timeout=_CATALOG_READ_TIMEOUT_S
    )


def _resolve_current_discovery_grants(actor: Any) -> tuple[str, ...]:
    """Resolve current grant fingerprints from the process-owned broker set."""

    from agent_utilities.mcp.multiplexer import current_remote_oauth_grant_bindings
    from agent_utilities.mcp.remote_oauth_broker import OAuthGrantBinding

    return tuple(
        sorted(
            {
                binding.fingerprint
                for binding in current_remote_oauth_grant_bindings(actor)
                if isinstance(binding, OAuthGrantBinding)
                and isinstance(binding.fingerprint, str)
                and binding.fingerprint
            }
        )
    )


def _require_catalog_authority(
    *, require_discovery_binding: bool
) -> tuple[str, str, tuple[str, ...]]:
    """Require verified session/scope and return tenant, principal, grants."""

    from agent_utilities.knowledge_graph.core.session import resolve_session

    session = resolve_session(required_scope="kg:read")
    actor = session.actor
    tenant = str(session.tenant or actor.tenant_id or "").strip()
    principal = str(actor.actor_id or "").strip()
    if not tenant or not principal or not actor.authenticated:
        raise PermissionError("registry authority is unavailable")
    grant_digests: tuple[str, ...] = ()
    if require_discovery_binding:
        grant_digests = _resolve_current_discovery_grants(actor)
        # A verified tenant may read process-owned local/stdio discovery even
        # when no provider OAuth grant is present.  The SQL predicate below
        # keeps that tenant-local scope disjoint from OAuth rows; an empty
        # grant set therefore removes only the provider-grant branch.
    return tenant, principal, grant_digests


def _parse_request(request: Request) -> tuple[int, str, str | None]:
    """Parse bounded caller controls without putting them into SQL."""

    params = request.query_params
    raw_limit = params.get("limit", str(_DEFAULT_LIMIT))
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid registry limit") from exc
    if not 1 <= limit <= _MAX_LIMIT:
        raise HTTPException(status_code=422, detail="registry limit is out of bounds")
    query = str(params.get("q", "") or "").strip()
    if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        raise HTTPException(status_code=422, detail="registry filter is too long")
    cursor = params.get("cursor") or None
    if cursor is not None and len(cursor.encode("utf-8")) > _MAX_CURSOR_BYTES:
        raise HTTPException(status_code=400, detail="invalid registry cursor")
    return limit, query, cursor


def _parse_multi_kind_request(
    request: Request,
) -> tuple[list[str], int, str, dict[str, str], bool]:
    """Parse the bounded multi-kind controls: ``kinds`` (required,
    comma-separated, validated against ``_KIND_SPECS``), a ``limit``/``q``
    shared across every requested kind (same bounds as the single-kind
    route), one cursor PER kind via ``cursor_<kind>`` query parameters (e.g.
    ``?kinds=tools,skills&cursor_tools=...&cursor_skills=...`` — never a
    single combined cursor, since each kind's keyset position is
    independent), and ``include=toggle``.
    """

    params = request.query_params
    raw_kinds = str(params.get("kinds", "") or "").strip()
    kinds: list[str] = []
    for token in raw_kinds.split(","):
        kind = token.strip()
        if not kind:
            continue
        if kind not in _KIND_SPECS:
            raise HTTPException(
                status_code=422, detail=f"unknown registry kind: {kind}"
            )
        if kind not in kinds:
            kinds.append(kind)
    if not kinds:
        raise HTTPException(status_code=422, detail="registry kinds is required")

    raw_limit = params.get("limit", str(_DEFAULT_LIMIT))
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid registry limit") from exc
    if not 1 <= limit <= _MAX_LIMIT:
        raise HTTPException(status_code=422, detail="registry limit is out of bounds")

    query = str(params.get("q", "") or "").strip()
    if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        raise HTTPException(status_code=422, detail="registry filter is too long")

    cursors: dict[str, str] = {}
    for kind in kinds:
        cursor = params.get(f"cursor_{kind}") or None
        if cursor is None:
            continue
        if len(cursor.encode("utf-8")) > _MAX_CURSOR_BYTES:
            raise HTTPException(status_code=400, detail="invalid registry cursor")
        cursors[kind] = cursor

    include_raw = str(params.get("include", "") or "")
    include_toggle = "toggle" in {
        part.strip() for part in include_raw.split(",") if part.strip()
    }
    return kinds, limit, query, cursors, include_toggle


def _cursor_token(
    *,
    kind: str,
    query: str,
    after: tuple[str, str],
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...] = (),
    grant_digest: str | None = None,
) -> str:
    """Mint an existing HMAC run token bound to the registry read scope."""

    from agent_utilities.security.run_token import mint_token

    if grant_digest is not None:
        grant_digests = (grant_digest,)
    grant_digests = tuple(sorted(set(grant_digests)))
    payload = json.dumps(
        {
            "kind": kind,
            "query": query,
            "after": list(after),
            "grant_digests": list(grant_digests),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return mint_token(
        payload,
        project="registry",
        endpoints=("registry",),
        operations=("read",),
        ttl_seconds=_CURSOR_TTL_SECONDS,
        actor_id=principal,
        tenant_id=tenant,
    )


def _decode_cursor(
    token: str,
    *,
    kind: str,
    query: str,
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...],
) -> tuple[str, str]:
    """Verify cursor integrity and bind it to this tenant/principal/filter."""

    from agent_utilities.security.run_token import TokenError, validate_token

    try:
        decoded = validate_token(token, endpoint="registry", operation="read")
        if decoded.tenant_id != tenant or decoded.actor_id != principal:
            raise TokenError("cursor authority mismatch")
        payload = json.loads(decoded.run_id)
        payload_grants = payload.get("grant_digests")
        if not isinstance(payload_grants, list) or not all(
            isinstance(value, str) and value for value in payload_grants
        ):
            raise TokenError("cursor grant binding is malformed")
        if (
            payload.get("kind") != kind
            or payload.get("query") != query
            or tuple(sorted(set(payload_grants))) != tuple(sorted(set(grant_digests)))
        ):
            raise TokenError("cursor query mismatch")
        after = payload.get("after")
        if (
            not isinstance(after, list)
            or len(after) != 2
            or any(not isinstance(value, str) for value in after)
        ):
            raise TokenError("cursor position is malformed")
        return after[0], after[1]
    except (TokenError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid registry cursor") from exc


def _redact_text(value: Any) -> Any:
    """Apply the existing persistence privacy boundary to catalog prose."""

    if value is None:
        return ""
    if not isinstance(value, str):
        # Preserve malformed source types so the strict public model rejects
        # them and the route can return an explicit unavailable response.
        return value
    try:
        from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

        safe, _ = PersistencePrivacyGuard().sanitize_text(value)
        return safe
    except Exception:  # noqa: BLE001 - response redaction remains conservative
        return ""


def _safe_url(value: Any) -> str:
    """Return only the host; path segments may carry opaque credentials."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise CatalogUnavailable("authoritative catalog URL has an invalid type")
    raw = value
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return ""
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        # The path is deliberately omitted.  A path segment can be an opaque
        # bearer/API credential even when userinfo/query/fragment are absent.
        return urlunsplit((parsed.scheme, host, "", "", ""))
    except (TypeError, ValueError):
        return ""


def _normalize_row(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a catalog row without exposing tenant/principal or raw secrets."""

    fields = _KIND_SPECS[kind].model.model_fields
    result = {field: row[field] for field in fields if field in row}
    if kind == "servers":
        result["url"] = _safe_url(result.get("url"))
    if kind in {"tools", "prompts", "resources", "skills"}:
        if "description" in result:
            result["description"] = _redact_text(result.get("description"))
        if "uri" in result:
            result["uri"] = _redact_text(result.get("uri"))
    if kind == "discoveries":
        # Discovery failures are intentionally classified, not echoed: the
        # stored message may contain host paths or connector details.
        error = result.get("last_error")
        if error is None or error == "":
            result["last_error"] = ""
        elif isinstance(error, str):
            result["last_error"] = "unavailable"
        result.pop("discovery_principal", None)
    return result


def _validate_item(
    kind: str, model: type[BaseModel], row: Mapping[str, Any]
) -> BaseModel:
    """Convert one catalog row or collapse malformed shape to safe unavailability."""

    try:
        return model.model_validate(_normalize_row(kind, row))
    except (CatalogUnavailable, ValidationError, TypeError, ValueError) as exc:
        # Do not retain Pydantic's value-bearing diagnostics in the public
        # response or logs; the route reports only the generic unavailable
        # state while preserving the cause for exception chaining/debugging.
        raise CatalogUnavailable(
            "authoritative catalog response shape is invalid"
        ) from exc


def _rows_from_engine(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        if raw.get("error"):
            raise CatalogUnavailable("catalog query returned an error")
        raw = raw.get("rows", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CatalogUnavailable("catalog query returned an invalid shape")
    rows: list[dict[str, Any]] = []
    for row in raw:
        if isinstance(row, Mapping):
            rows.append(dict(row))
        else:
            raise CatalogUnavailable("catalog query returned an invalid row")
    return rows


def _catalog_service_session() -> Any:
    """The fixed identity this deployment's catalog SQL RPC must run as.

    ROOT CAUSE (measured live 2026-08-25, D-catalog-503-human): the engine's
    native ``Method::Sql`` RPC (``epistemic-graph``
    ``src/server/handlers/query.rs``) resolves an **owner-scoped** redb
    table store keyed by the CALLING actor's own verified tenant+principal
    (``src/server/sql_tables.rs::user_table_store`` /
    ``owner_filename`` — "every owner receives a distinct redb database").
    There is no catalog shared across actors on this path — that sharing
    (``src/server/sql_catalog_acl.rs``, CONCEPT:NE-003) exists only for the
    wire-protocol adapters (pgwire/mysql/sqlite) that delegate through
    ``WireSession``; the native RPC AU's Python client uses never opts into
    it. ``fleet_catalog_tables.py`` writes the ``mcp_servers``/``mcp_tools``/
    ``skills``/... tables once, under this process's own fixed
    ``automated_service`` identity (whatever actor is ambient when the
    scheduled ``fleet-tool-schema-sync`` job runs — which resolves through
    the SAME :func:`~agent_utilities.security.request_identity.
    system_write_session` this function calls). Any OTHER verified actor —
    including a fully authorized human carrying ``kg:admin`` — therefore
    opens its OWN, always-empty private catalog on read and gets
    ``SQL error: ... table 'X' not found``, which was surfacing as a bare,
    cause-less 503 (see :func:`_log_catalog_exception`). Confirmed live:
    the writer's own identity reads the table fine (count=66); a distinct,
    fully-admitted actor gets ``table not found`` on the identical query.

    This module's own ``tenant_id``/principal predicate (:func:`_build_where`,
    built from the REAL calling actor via :func:`_require_catalog_authority`)
    is already the entire authorization boundary for this store — the engine
    enforces no RLS of its own on a plain user table (see
    ``fleet_catalog_tables`` module docstring: "gated only on authentication
    ... never on the named-graph Read/Write Pattern grant"). So running the
    already-correctly-scoped SQL under the fixed writer identity changes only
    WHICH physical catalog file is opened, never WHAT rows a caller may see.

    ``suspend_session()`` is required, not a bare call to
    ``system_write_session()``: that helper *prefers* an already-bound
    ambient session (by design, for the different "attribute an
    unauthenticated background write" problem it was built for, BUG-033/
    BUG-039) — inside a served request there always IS one (the caller's
    own), so an unguarded call would just hand back the caller's own session
    and fix nothing.
    """
    from agent_utilities.knowledge_graph.core.session import suspend_session
    from agent_utilities.security.request_identity import system_write_session

    with suspend_session():
        return system_write_session()


def _log_catalog_exception(action: str, exc: BaseException) -> None:
    """Log a catalog RPC failure with its real message and full cause chain.

    Previously these call sites logged only ``type(exc).__name__`` (e.g. a
    bare ``RuntimeError``), discarding the engine's own error text — exactly
    the detail that distinguishes WHY a call failed (a missing-table SQL
    plan error vs. an ACCESS_DENIED principal/graph mismatch vs. a network
    fault) from merely THAT it failed. That gap is what made the
    owner-scoped-SQL-catalog root cause (see :func:`_catalog_service_session`)
    take a live in-pod repro to uncover instead of one log line. Server-side
    log only — the HTTP response stays the generic ``catalog_unavailable``
    body; this never reaches the client.
    """
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    logger.warning(
        "authoritative registry catalog %s failed: %s", action, " <- ".join(chain)
    )


def _require_sql_exec(engine: Any) -> Callable[[str], Any]:
    """Resolve the engine's write-capable SQL surface or fail closed.

    The returned callable executes under this deployment's fixed
    catalog-service identity (:func:`_catalog_service_session`), never the
    caller's own ambient session — see that function's docstring for why.
    """

    graph_compute = getattr(engine, "graph_compute", None)
    sql_exec = getattr(graph_compute, "sql_exec", None)
    if not callable(sql_exec):
        raise CatalogUnavailable("authoritative catalog SQL is unavailable")

    def _run_as_catalog_service(statement: str) -> Any:
        from agent_utilities.knowledge_graph.core.session import use_session

        with use_session(_catalog_service_session()):
            return sql_exec(statement)

    return _run_as_catalog_service


def _search_columns(spec: _KindSpec) -> list[str]:
    """Columns eligible for the ``q`` substring filter, in ``_matches`` order."""

    candidates = (spec.name_column, "name", "server_name", "description")
    found: list[str] = []
    for column in candidates:
        if column in spec.columns and column not in found:
            found.append(column)
    return found


def _build_where(
    spec: _KindSpec,
    *,
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...],
    query: str,
) -> str:
    """Compose the tenant/authorization/filter predicate for one catalog read.

    These identifiers are module constants validated at construction. The
    only interpolated values are escaped SQL literals (:func:`_sql_literal`);
    caller-supplied filter text never enters the statement unescaped.
    """

    where = f"tenant_id = {_sql_literal(tenant)}"
    if spec.authority_column and spec.principal_column and spec.grant_column:
        local_scope = (
            f"({spec.authority_column} = "
            f"{_sql_literal(DISCOVERY_AUTHORITY_TENANT_LOCAL)} AND "
            f"{spec.principal_column} = {_sql_literal('')} AND "
            f"{spec.grant_column} = {_sql_literal('')})"
        )
        scope_terms = [local_scope]
        if grant_digests:
            grants_sql = ", ".join(_sql_literal(digest) for digest in grant_digests)
            scope_terms.append(
                f"({spec.authority_column} = "
                f"{_sql_literal(DISCOVERY_AUTHORITY_OAUTH_GRANT)} AND "
                f"{spec.principal_column} = {_sql_literal(principal)} AND "
                f"{spec.grant_column} IN ({grants_sql}))"
            )
        where += " AND (" + " OR ".join(scope_terms) + ")"
    if query:
        search_columns = _search_columns(spec)
        if not search_columns:
            # No searchable column exists for this kind; an unmatchable
            # predicate keeps the count and page pushdown honest instead of
            # silently ignoring the caller's filter.
            return where + " AND FALSE"
        needle = _sql_literal(query)
        # strpos(...) > 0 is a plain case-insensitive substring test (the
        # exact `needle in haystack` semantics `_matches` used to apply in
        # Python) with no LIKE wildcard-escaping pitfall for a `%`/`_` in
        # the caller's filter text.
        terms = " OR ".join(
            f"strpos(LOWER({column}), LOWER({needle})) > 0" for column in search_columns
        )
        where += f" AND ({terms})"
    return where


def _keyset_predicate(spec: _KindSpec, after: tuple[str, str]) -> str:
    """The keyset-pagination predicate for rows strictly after ``after``.

    Mirrors the ``(casefold(name), id)`` ordering :func:`_row_key` already
    encodes into the cursor. ``LOWER()`` is SQL's nearest portable
    equivalent to Python's ``str.casefold()`` — not byte-identical on every
    Unicode edge case, but the two agree on the ASCII identifiers this
    catalog's names/ids are drawn from.
    """

    after_name, after_id = after
    name_literal = _sql_literal(after_name)
    id_literal = _sql_literal(after_id)
    return (
        f"(LOWER({spec.name_column}) > LOWER({name_literal}) OR "
        f"(LOWER({spec.name_column}) = LOWER({name_literal}) AND "
        f"id > {id_literal}))"
    )


def _validate_scope(
    spec: _KindSpec,
    rows: list[dict[str, Any]],
    *,
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...],
) -> None:
    """Defence in depth: reject any row a misconfigured engine projection
    returned outside the tenant/principal contract the WHERE clause already
    encodes, before it reaches filtering, ordering, or response shaping."""

    required_columns = set(spec.columns)
    for row in rows:
        if not required_columns.issubset(row):
            raise CatalogUnavailable("authoritative catalog row is malformed")
        row_tenant = row.get("tenant_id")
        if not isinstance(row_tenant, str) or row_tenant != tenant:
            raise CatalogUnavailable("authoritative catalog scope is malformed")
        if spec.authority_column and spec.principal_column and spec.grant_column:
            row_authority = row.get(spec.authority_column)
            row_principal = row.get(spec.principal_column)
            row_grant = row.get(spec.grant_column)
            if row_authority == DISCOVERY_AUTHORITY_TENANT_LOCAL:
                if row_principal != "" or row_grant != "":
                    raise CatalogUnavailable("authoritative catalog scope is malformed")
            elif row_authority == DISCOVERY_AUTHORITY_OAUTH_GRANT:
                if (
                    not isinstance(row_principal, str)
                    or row_principal != principal
                    or not isinstance(row_grant, str)
                    or row_grant not in grant_digests
                ):
                    raise CatalogUnavailable("authoritative catalog scope is malformed")
            else:
                raise CatalogUnavailable("authoritative catalog scope is malformed")


def _authorized_count(
    kind: str,
    *,
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...],
    query: str,
    engine: Any,
) -> int:
    """``SELECT COUNT(*)`` for the total matching the same predicate as the
    page read, instead of counting a materialized Python list."""

    spec = _KIND_SPECS[kind]
    sql_exec = _require_sql_exec(engine)
    where = _build_where(
        spec,
        tenant=tenant,
        principal=principal,
        grant_digests=grant_digests,
        query=query,
    )
    statement = f"SELECT COUNT(*) AS row_count FROM {spec.table} WHERE {where}"
    try:
        rows = _rows_from_engine(sql_exec(statement))
    except CatalogUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - explicit unavailable response
        _log_catalog_exception("count", exc)
        raise CatalogUnavailable("authoritative catalog read failed") from exc
    if len(rows) != 1:
        raise CatalogUnavailable("authoritative catalog count is malformed")
    value = rows[0].get("row_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogUnavailable("authoritative catalog count is malformed")
    return value


def _authorized_page(
    kind: str,
    *,
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...],
    query: str,
    after: tuple[str, str] | None,
    limit: int,
    engine: Any,
) -> list[dict[str, Any]]:
    """Read one keyset-paginated page: LIMIT/keyset/filter/authz all pushed
    into SQL, so a page of N rows transfers N rows over the wire, never the
    whole table."""

    spec = _KIND_SPECS[kind]
    sql_exec = _require_sql_exec(engine)
    where = _build_where(
        spec,
        tenant=tenant,
        principal=principal,
        grant_digests=grant_digests,
        query=query,
    )
    if after is not None:
        where += f" AND {_keyset_predicate(spec, after)}"
    # Fetch one extra row to detect "there is a next page" without a second
    # round trip. `_MAX_CATALOG_ROWS` remains a defence-in-depth ceiling on
    # the fetch itself (unreachable in practice since `_parse_request` already
    # bounds `limit` to `_MAX_LIMIT`) so a pathological request still cannot
    # pull the whole table even if that bound were ever raised.
    fetch = min(min(limit, _MAX_LIMIT) + 1, _MAX_CATALOG_ROWS + 1)
    columns = ", ".join(spec.columns)
    statement = (
        f"SELECT {columns} FROM {spec.table} WHERE {where} "
        f"ORDER BY LOWER({spec.name_column}), id LIMIT {fetch}"
    )
    try:
        rows = _rows_from_engine(sql_exec(statement))
    except CatalogUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - explicit unavailable response
        _log_catalog_exception("page read", exc)
        raise CatalogUnavailable("authoritative catalog read failed") from exc
    if len(rows) > fetch:
        raise CatalogUnavailable(
            "authoritative catalog page exceeds the requested bound"
        )
    _validate_scope(
        spec, rows, tenant=tenant, principal=principal, grant_digests=grant_digests
    )
    return rows


def _authorized_item(
    kind: str,
    *,
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...],
    item_id: str,
    engine: Any,
) -> dict[str, Any] | None:
    """Read at most one row by id, with the id predicate pushed into SQL
    rather than fetching the authorized set and filtering it in Python."""

    spec = _KIND_SPECS[kind]
    sql_exec = _require_sql_exec(engine)
    where = _build_where(
        spec, tenant=tenant, principal=principal, grant_digests=grant_digests, query=""
    )
    where += f" AND id = {_sql_literal(item_id)}"
    columns = ", ".join(spec.columns)
    statement = f"SELECT {columns} FROM {spec.table} WHERE {where} LIMIT 1"
    try:
        rows = _rows_from_engine(sql_exec(statement))
    except CatalogUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - explicit unavailable response
        _log_catalog_exception("item read", exc)
        raise CatalogUnavailable("authoritative catalog read failed") from exc
    if len(rows) > 1:
        raise CatalogUnavailable("authoritative catalog item lookup is malformed")
    if not rows:
        # The same response is used for an absent row and another tenant's
        # row (the tenant predicate is already embedded in `where`).
        return None
    _validate_scope(
        spec, rows, tenant=tenant, principal=principal, grant_digests=grant_digests
    )
    return rows[0]


def _row_key(spec: _KindSpec, row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get(spec.name_column) or "").casefold(),
        str(row.get("id") or ""),
    )


async def _list_kind(
    request: Request,
    *,
    kind: str,
    model: type[BaseModel],
) -> RegistryPage[Any] | JSONResponse:
    """Return the typed page envelope, or a bare ``JSONResponse`` on the
    catalog-unavailable path (FastAPI honors a returned ``Response`` over
    the declared ``response_model``, so the 503 ``{"status":
    "unavailable", ...}`` body below is served exactly as constructed;
    this annotation only makes that documented, doubly-typed return honest
    for static analysis — it changes no wire behavior)."""
    limit, query, cursor = _parse_request(request)
    try:
        tenant, principal, grant_digests = _require_catalog_authority(
            require_discovery_binding=_KIND_SPECS[kind].principal_column is not None
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="registry access denied") from exc
    after: tuple[str, str] | None = None
    if cursor:
        after = _decode_cursor(
            cursor,
            kind=kind,
            query=query,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
        )
    spec = _KIND_SPECS[kind]
    engine = _get_catalog_engine()
    try:
        total = await _offload_catalog_call(
            _authorized_count,
            kind,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
            query=query,
            engine=engine,
        )
        rows = await _offload_catalog_call(
            _authorized_page,
            kind,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
            query=query,
            after=after,
            limit=limit,
            engine=engine,
        )
    except CatalogUnavailable as exc:
        logger.warning("registry %s unavailable: %s", kind, exc)
        return JSONResponse(
            {"status": "unavailable", "reason": "catalog_unavailable"},
            status_code=503,
        )
    except Exception as exc:  # noqa: BLE001 - backend unavailability is privacy-safe
        logger.warning("registry %s unavailable: %s", kind, exc)
        return JSONResponse(
            {"status": "unavailable", "reason": "catalog_unavailable"},
            status_code=503,
        )
    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_cursor = None
    if has_more and page_rows:
        last = _row_key(spec, page_rows[-1])
        next_cursor = _cursor_token(
            kind=kind,
            query=query,
            after=last,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
        )
    try:
        items = [_validate_item(kind, model, row) for row in page_rows]
    except CatalogUnavailable as exc:
        logger.warning("registry %s response shape unavailable: %s", kind, exc)
        return JSONResponse(
            {"status": "unavailable", "reason": "catalog_unavailable"},
            status_code=503,
        )
    return RegistryPage[Any](
        kind=kind, items=items, count=total, next_cursor=next_cursor
    )


async def _get_kind(
    request: Request,
    *,
    kind: str,
    model: type[BaseModel],
    item_id: str,
) -> RegistryItemEnvelope[Any] | JSONResponse:
    """Return the typed item envelope, or a bare ``JSONResponse`` on the
    catalog-unavailable path (FastAPI honors a returned ``Response`` over
    the declared ``response_model``, so the 503 ``{"status":
    "unavailable", ...}`` body below is served exactly as constructed;
    this annotation only makes that documented, doubly-typed return honest
    for static analysis — it changes no wire behavior)."""
    try:
        tenant, principal, grant_digests = _require_catalog_authority(
            require_discovery_binding=_KIND_SPECS[kind].principal_column is not None
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="registry access denied") from exc
    try:
        row = await _offload_catalog_call(
            _authorized_item,
            kind,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
            item_id=item_id,
            engine=_get_catalog_engine(),
        )
    except CatalogUnavailable as exc:
        logger.warning("registry %s unavailable: %s", kind, exc)
        return JSONResponse(
            {"status": "unavailable", "reason": "catalog_unavailable"},
            status_code=503,
        )
    except Exception as exc:  # noqa: BLE001 - backend unavailability is privacy-safe
        logger.warning("registry %s unavailable: %s", kind, exc)
        return JSONResponse(
            {"status": "unavailable", "reason": "catalog_unavailable"},
            status_code=503,
        )
    if row is None:
        # The same response is used for an absent row and another tenant's row.
        raise HTTPException(status_code=404, detail="registry item not found")
    try:
        item = _validate_item(kind, model, row)
    except CatalogUnavailable as exc:
        logger.warning("registry %s response shape unavailable: %s", kind, exc)
        return JSONResponse(
            {"status": "unavailable", "reason": "catalog_unavailable"},
            status_code=503,
        )
    return RegistryItemEnvelope[Any](item=item)


# Toggle-store item_type/key convention per kind, evidenced from the two
# places that already write/read this preference store today:
# ``agent_utilities.mcp.kg_server._build_tools_payload_sync`` (servers,
# keyed by name — ``("mcp_server", name)``; skills, keyed by the skill's
# frontmatter name, which is the same value as this catalog's ``name``
# column — ``("skill", name)``) and
# ``agent_webui.api_extensions``'s per-tool inventory enrichment (tools,
# keyed by ``f"{server_name}:{tool_name}"`` — ``("mcp_tool",
# f"{server_name}:{name}")"``). ``discoveries``/``prompts``/``resources``
# have no evidenced toggle convention and no ``enabled`` field on their
# models, so they are intentionally absent here — ``include=toggle`` is a
# no-op for those kinds rather than a guess.
_TOGGLE_KEY_BUILDERS: dict[str, Callable[[dict[str, Any]], tuple[str, str]]] = {
    "servers": lambda item: ("mcp_server", str(item.get("name") or "")),
    "tools": lambda item: (
        "mcp_tool",
        f"{item.get('server_name') or ''}:{item.get('name') or ''}",
    ),
    "skills": lambda item: ("skill", str(item.get("name") or "")),
}


async def _merge_toggle_states(
    kinds_map: dict[str, RegistryKindResult], *, engine: Any
) -> None:
    """Merge live user-toggle preference into every ``enabled`` item field,
    across every requested kind, in ONE batched engine round trip total —
    never one call per item (the exact N+1 pattern that cost 350+ sequential
    round trips in production; see ``get_toggle_states_batch``'s own
    docstring) and never one call per kind either.

    Deliberately reads from ``get_toggle_states_batch`` — the SAME
    Preference-node store ``/api/enhanced/tools`` and ``POST
    /api/tools/toggle`` already read and write — NOT this catalog's own
    ``enabled`` column. The catalog's ``enabled`` reflects CONFIG (a server
    marked disabled, a skill disabled in its source), not this per-user
    runtime toggle preference; substituting one for the other would make
    toggling an item in the UI silently stop being reflected on the next
    GET. A config-level disable still wins over an "on" toggle preference
    (ANDed below), matching ``_build_tools_payload_sync``'s existing
    precedent for ``mcp_tools``/servers.
    """

    from agent_utilities.mcp.kg_server import get_toggle_states_batch

    keys: list[tuple[str, str]] = []
    for kind, result in kinds_map.items():
        builder = _TOGGLE_KEY_BUILDERS.get(kind)
        if builder is None or result.status != "ok":
            continue
        for item in result.items:
            if "enabled" in item:
                keys.append(builder(item))
    if not keys:
        return

    toggle_states = await _offload_catalog_call(get_toggle_states_batch, engine, keys)

    for kind, result in kinds_map.items():
        builder = _TOGGLE_KEY_BUILDERS.get(kind)
        if builder is None or result.status != "ok":
            continue
        for item in result.items:
            if "enabled" not in item:
                continue
            key = builder(item)
            toggled = bool(toggle_states.get(key, True))
            item["enabled"] = toggled and bool(item.get("enabled", True))


async def _list_multi_kind(request: Request) -> RegistryMultiKindPage:
    """``GET /api/registry?kinds=a,b,c[&limit=][&q=][&cursor_<kind>=][&include=toggle]``.

    Reuses ``_authorized_page``/``_authorized_count``/``_build_where`` — the
    EXACT same predicate the single-kind ``GET /api/registry/{kind}`` route
    uses — via the same ``_offload_catalog_call`` off-loop wrapper, so there
    is exactly one predicate implementation for both surfaces (no forked
    second copy to drift out of sync).

    Every requested kind is read CONCURRENTLY, each on its own worker thread
    (``asyncio.gather`` over per-kind coroutines that each call
    ``_offload_catalog_call``): N kinds cost roughly as much wall-clock time
    as the single slowest kind, not N sequential ~1.2-2.8s round trips, and
    the event loop is never blocked by any of them.

    One kind's catalog failure never discards another's success: a failing
    kind reports ``RegistryKindResult(status="unavailable", reason=...)``,
    matching ``agent_webui.api_extensions._read_fleet_catalog``'s own
    per-kind degrade contract. The top-level ``status`` stays ``"ok"``
    whenever ANY per-kind read was attempted; only a total authority failure
    (no verified session/scope at all) 403s the whole request, before any
    per-kind read is attempted — the same authority gate the single-kind
    route enforces.
    """

    kinds, limit, query, cursors, include_toggle = _parse_multi_kind_request(request)
    require_discovery_binding = any(
        _KIND_SPECS[kind].principal_column is not None for kind in kinds
    )
    try:
        tenant, principal, grant_digests = _require_catalog_authority(
            require_discovery_binding=require_discovery_binding
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="registry access denied") from exc

    # Cursors are decoded up front (before any dispatch) so a tampered or
    # expired cursor for one kind 400s the whole request early, exactly like
    # the single-kind route -- a caller cannot silently keep paginating past
    # a rejected cursor for just that one kind.
    afters: dict[str, tuple[str, str] | None] = {}
    for kind in kinds:
        cursor = cursors.get(kind)
        afters[kind] = (
            _decode_cursor(
                cursor,
                kind=kind,
                query=query,
                tenant=tenant,
                principal=principal,
                grant_digests=grant_digests,
            )
            if cursor
            else None
        )

    engine = _get_catalog_engine()

    async def _read_one(kind: str) -> tuple[str, RegistryKindResult]:
        spec = _KIND_SPECS[kind]
        try:
            total = await _offload_catalog_call(
                _authorized_count,
                kind,
                tenant=tenant,
                principal=principal,
                grant_digests=grant_digests,
                query=query,
                engine=engine,
            )
            rows = await _offload_catalog_call(
                _authorized_page,
                kind,
                tenant=tenant,
                principal=principal,
                grant_digests=grant_digests,
                query=query,
                after=afters[kind],
                limit=limit,
                engine=engine,
            )
        except Exception as exc:  # noqa: BLE001 - explicit per-kind unavailable
            logger.warning("registry %s unavailable (multi-kind): %s", kind, exc)
            return kind, RegistryKindResult(
                status="unavailable", reason="catalog_unavailable"
            )

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor = None
        if has_more and page_rows:
            last = _row_key(spec, page_rows[-1])
            next_cursor = _cursor_token(
                kind=kind,
                query=query,
                after=last,
                tenant=tenant,
                principal=principal,
                grant_digests=grant_digests,
            )
        try:
            items = [_validate_item(kind, spec.model, row) for row in page_rows]
        except CatalogUnavailable as exc:
            logger.warning(
                "registry %s response shape unavailable (multi-kind): %s", kind, exc
            )
            return kind, RegistryKindResult(
                status="unavailable", reason="catalog_unavailable"
            )
        return kind, RegistryKindResult(
            items=[item.model_dump() for item in items],
            count=total,
            next_cursor=next_cursor,
        )

    results = await asyncio.gather(*(_read_one(kind) for kind in kinds))
    kinds_map: dict[str, RegistryKindResult] = dict(results)

    if include_toggle:
        await _merge_toggle_states(kinds_map, engine=engine)

    return RegistryMultiKindPage(kinds=kinds_map)


def _make_list_handler(kind: str, model: type[BaseModel]) -> Callable[..., Any]:
    async def handler(request: Request) -> Any:
        return await _list_kind(request, kind=kind, model=model)

    handler.__name__ = f"list_registry_{kind}"
    return handler


def _make_get_handler(kind: str, model: type[BaseModel]) -> Callable[..., Any]:
    async def handler(request: Request, item_id: str) -> Any:
        return await _get_kind(request, kind=kind, model=model, item_id=item_id)

    handler.__name__ = f"get_registry_{kind}"
    return handler


registry_router.add_api_route(
    "/registry",
    _list_multi_kind,
    methods=["GET"],
    response_model=RegistryMultiKindPage,
    name="list_registry_multi_kind",
)


for _kind, _spec in _KIND_SPECS.items():
    _model = _spec.model
    _page_response_model, _item_response_model = _RESPONSE_MODELS[_kind]
    _list_handler = _make_list_handler(_kind, _model)
    _get_handler = _make_get_handler(_kind, _model)
    registry_router.add_api_route(
        f"/registry/{_kind}",
        _list_handler,
        methods=["GET"],
        response_model=_page_response_model,
        name=f"list_registry_{_kind}",
    )
    registry_router.add_api_route(
        f"/registry/{_kind}/{{item_id}}",
        _get_handler,
        methods=["GET"],
        response_model=_item_response_model,
        name=f"get_registry_{_kind}",
    )


def register_registry_routes(app: Any, prefix: str = "/api") -> None:
    """Mount the typed GET-only registry routes on FastAPI or Starlette."""

    if hasattr(app, "include_router"):
        app.include_router(registry_router, prefix=prefix)
        return
    for route in registry_router.routes:
        if not isinstance(route, Route):
            continue
        app.add_route(
            prefix + route.path,
            route.endpoint,
            methods=list(route.methods or ["GET"]),
        )


__all__ = [
    "CatalogUnavailable",
    "RegistryDiscovery",
    "RegistryItemEnvelope",
    "RegistryKindResult",
    "RegistryMultiKindPage",
    "RegistryPage",
    "RegistryPrompt",
    "RegistryResource",
    "RegistryServer",
    "RegistrySkill",
    "RegistryTool",
    "register_registry_routes",
    "registry_router",
]
