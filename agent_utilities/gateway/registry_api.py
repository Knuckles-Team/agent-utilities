"""Tenant/principal-scoped read-only registry over EG's typed fleet catalog.

EH-345 (2026-09-22): the catalog writer used to be
:mod:`agent_utilities.knowledge_graph.core.fleet_catalog_tables` (an AU SQL
tier); it is now EG's own ``ServerRegistry``/``FleetCatalog`` contract
(``client.server_registry``/``client.fleet_catalog``), reached the same way
``core/engine_ingestion.py``'s reconciler and ``source_sync.py``'s writer
reach it. This module is deliberately a reader either way. It never starts
MCP children, probes a live server, writes a row, or falls back to a second
store. Authorization is resolved from the ambient verified
:class:`GraphSession` plus the process-owned broker's current exact
OAuth-grant fingerprints, then passed to EG as ``grant_digests`` — the
tenant/principal/grant visibility predicate is evaluated ENGINE-side now
(EG binds tenant/principal from the verified request context itself), not
built into a caller-side SQL WHERE clause the way it used to be.

CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables
CONCEPT:AU-OS.state.unified-durable-state-externalization
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

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


#
# PHASE A investigation — three sections `/api/enhanced/tools` still serves
# from the filesystem/KG (`builtin_tools`, `skill_graphs`, `skill_workflows`,
# see `agent_utilities.mcp.kg_server._build_tools_payload_sync`'s own
# docstring, "FIX LANE (collapse-tool-endpoints)") were evaluated as
# candidate additional catalog kinds here. None were added — same evidence as
# before EH-345 (a `builtin_tool` is never MCP-discovered/cataloged at all; a
# `skill_graph`/`skill_workflow` has real gaps — no automatic ingester for the
# former, no domain/tags fields for the latter — that switching sources would
# silently blank or empty). They remain reachable only through the existing
# filesystem/KG path (`GET /tools` / `/api/enhanced/tools`).
#
# EH-345 (2026-09-22): "servers" reads ServerRegistryClient; every other kind
# reads FleetCatalogClient (kinds: "discoveries"|"tools"|"prompts"|
# "resources"|"skills" — see AU-CUTOVER.md §0). Only the latter five carry a
# discovery/grant visibility predicate — "servers" (liveness/registration) is
# tenant-scoped by the engine but has no per-principal/grant dimension.
_KIND_MODELS: dict[str, type[BaseModel]] = {
    "servers": RegistryServer,
    "discoveries": RegistryDiscovery,
    "tools": RegistryTool,
    "prompts": RegistryPrompt,
    "resources": RegistryResource,
    "skills": RegistrySkill,
}
_DISCOVERY_BOUND_KINDS = frozenset(
    {"discoveries", "tools", "prompts", "resources", "skills"}
)

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

    ``_authorized_page``/``_authorized_item`` each make
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
        # when no provider OAuth grant is present. EG's own visibility
        # predicate (FleetCatalogListRequest.grant_digests — evaluated
        # engine-side, never a caller-built WHERE clause) keeps that
        # tenant-local scope disjoint from OAuth rows; an empty grant set
        # therefore removes only the provider-grant branch.
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
    comma-separated, validated against ``_KIND_MODELS``), a ``limit``/``q``
    shared across every requested kind (same bounds as the single-kind
    route), one cursor PER kind via ``cursor_<kind>`` query parameters (e.g.
    ``?kinds=tools,skills&cursor_tools=...&cursor_skills=...`` — never a
    single combined cursor, since each kind's keyset position is
    independent), and ``include=toggle``.
    """

    params = request.query_params
    kinds = _parse_registry_kinds(str(params.get("kinds", "") or "").strip())
    limit = _parse_registry_limit(params.get("limit", str(_DEFAULT_LIMIT)))
    query = _parse_registry_query(str(params.get("q", "") or "").strip())
    cursors = _parse_kind_cursors(params, kinds)

    include_raw = str(params.get("include", "") or "")
    include_toggle = "toggle" in {
        part.strip() for part in include_raw.split(",") if part.strip()
    }
    return kinds, limit, query, cursors, include_toggle


def _parse_registry_kinds(raw_kinds: str) -> list[str]:
    """Validate and dedupe the comma-separated ``kinds`` parameter against
    ``_KIND_MODELS``."""
    kinds: list[str] = []
    for token in raw_kinds.split(","):
        kind = token.strip()
        if not kind:
            continue
        if kind not in _KIND_MODELS:
            raise HTTPException(
                status_code=422, detail=f"unknown registry kind: {kind}"
            )
        if kind not in kinds:
            kinds.append(kind)
    if not kinds:
        raise HTTPException(status_code=422, detail="registry kinds is required")
    return kinds


def _parse_registry_limit(raw_limit: str) -> int:
    """Validate the shared ``limit`` bound (same bounds as the single-kind
    route)."""
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid registry limit") from exc
    if not 1 <= limit <= _MAX_LIMIT:
        raise HTTPException(status_code=422, detail="registry limit is out of bounds")
    return limit


def _parse_registry_query(query: str) -> str:
    """Bound the shared ``q`` filter's byte length."""
    if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
        raise HTTPException(status_code=422, detail="registry filter is too long")
    return query


def _parse_kind_cursors(params: Any, kinds: list[str]) -> dict[str, str]:
    """Parse one ``cursor_<kind>`` query parameter per requested kind --
    never a single combined cursor, since each kind's keyset position is
    independent."""
    cursors: dict[str, str] = {}
    for kind in kinds:
        cursor = params.get(f"cursor_{kind}") or None
        if cursor is None:
            continue
        if len(cursor.encode("utf-8")) > _MAX_CURSOR_BYTES:
            raise HTTPException(status_code=400, detail="invalid registry cursor")
        cursors[kind] = cursor
    return cursors


def _cursor_token(
    *,
    kind: str,
    query: str,
    after: Mapping[str, Any],
    tenant: str,
    principal: str,
    grant_digests: tuple[str, ...] = (),
    grant_digest: str | None = None,
) -> str:
    """Mint an existing HMAC run token bound to the registry read scope.

    EH-345: ``after`` is now EG's own opaque page-position object — a
    ``RegisteredServerCursor`` (``after_name``/``registry_digest``/
    ``registry_revision``) for ``kind="servers"``, a ``FleetCatalogCursor``
    (``after_name``/``after_id``/``snapshot_digest``) for every other kind —
    JSON-dumped verbatim (``model_dump(mode="json")``) rather than the old
    fixed 2-tuple keyset position, since the two cursor shapes differ. This
    HMAC wrapper still fences it to tenant/principal/kind/query/grants
    exactly as before; only what it opaquely carries changed.
    """

    from agent_utilities.security.run_token import mint_token

    if grant_digest is not None:
        grant_digests = (grant_digest,)
    grant_digests = tuple(sorted(set(grant_digests)))
    payload = json.dumps(
        {
            "kind": kind,
            "query": query,
            "after": dict(after),
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
) -> dict[str, Any]:
    """Verify cursor integrity and bind it to this tenant/principal/filter."""

    from agent_utilities.security.run_token import TokenError, validate_token

    try:
        decoded = validate_token(token, endpoint="registry", operation="read")
        _check_cursor_authority(decoded, tenant=tenant, principal=principal)
        payload = json.loads(decoded.run_id)
        _check_cursor_binding(
            payload, kind=kind, query=query, grant_digests=grant_digests
        )
        return _cursor_after_position(payload)
    except (TokenError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid registry cursor") from exc


def _check_cursor_authority(decoded: Any, *, tenant: str, principal: str) -> None:
    from agent_utilities.security.run_token import TokenError

    if decoded.tenant_id != tenant or decoded.actor_id != principal:
        raise TokenError("cursor authority mismatch")


def _check_cursor_binding(
    payload: dict[str, Any],
    *,
    kind: str,
    query: str,
    grant_digests: tuple[str, ...],
) -> None:
    from agent_utilities.security.run_token import TokenError

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


def _cursor_after_position(payload: dict[str, Any]) -> dict[str, Any]:
    from agent_utilities.security.run_token import TokenError

    after = payload.get("after")
    if (
        not isinstance(after, dict)
        or not after
        or any(not isinstance(value, (str, int)) for value in after.values())
    ):
        raise TokenError("cursor position is malformed")
    return dict(after)


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
        if not _url_safe_to_disclose(parsed):
            return ""
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        # The path is deliberately omitted.  A path segment can be an opaque
        # bearer/API credential even when userinfo/query/fragment are absent.
        return urlunsplit((parsed.scheme, host, "", "", ""))
    except (TypeError, ValueError):
        return ""


def _url_safe_to_disclose(parsed: Any) -> bool:
    """True when a parsed URL carries no userinfo/query/fragment and has an
    http(s) scheme + hostname -- the only shape whose bare host is safe to
    return."""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _normalize_row(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a catalog row without exposing tenant/principal or raw secrets."""

    fields = _KIND_MODELS[kind].model_fields
    result = {field: row[field] for field in fields if field in row}
    if kind == "servers":
        result["url"] = _safe_url(result.get("url"))
    if kind in {"tools", "prompts", "resources", "skills"}:
        _redact_prose_fields(result)
    if kind == "discoveries":
        _sanitize_discovery_row(result)
    return result


def _redact_prose_fields(result: dict[str, Any]) -> None:
    """Redact free-text fields (description/uri) via the privacy guard, in
    place."""
    if "description" in result:
        result["description"] = _redact_text(result.get("description"))
    if "uri" in result:
        result["uri"] = _redact_text(result.get("uri"))


def _sanitize_discovery_row(result: dict[str, Any]) -> None:
    """Discovery failures are intentionally classified, not echoed: the
    stored message may contain host paths or connector details."""
    error = result.get("last_error")
    if error is None or error == "":
        result["last_error"] = ""
    elif isinstance(error, str):
        result["last_error"] = "unavailable"
    result.pop("discovery_principal", None)


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


def _fleet_catalog_clients(engine: Any) -> tuple[Any, Any]:
    """``(server_registry, fleet_catalog)`` typed EG clients, or ``None``
    for either not yet available on this deployment's client.

    Same resolution path ``core/engine_ingestion.py``'s reconciler and
    ``source_sync.py``'s writer use — ``GraphComputeEngine.client`` is the
    process's own sync-wrapped ``SyncEpistemicGraphClient`` view.
    """
    gc = getattr(engine, "graph_compute", None)
    client = getattr(gc, "client", None)
    return (
        getattr(client, "server_registry", None),
        getattr(client, "fleet_catalog", None),
    )


def _row_from_server_view(view: Any) -> dict[str, Any]:
    """One ``RegisteredServerView`` as a plain dict matching ``RegistryServer``."""
    return {
        "id": f"mcp_server_{view.name}",
        "name": view.name,
        "transport": str(getattr(view, "transport", "") or ""),
        "url": str(getattr(view, "url", "") or ""),
        "enabled": str(getattr(view, "desired", "enabled")) != "disabled",
    }


def _discovery_outcome_fields(outcome: Any) -> tuple[bool, str]:
    """``(reachable, last_error)`` from one ``DiscoveryOutcome`` tagged union."""
    status = str(getattr(outcome, "status", "") or "")
    if status == "reachable":
        return True, ""
    return False, str(getattr(outcome, "error", "") or "")


def _row_from_fleet_row(entry: Any) -> dict[str, Any]:
    """One ``FleetCatalogRow`` (tagged union) as a plain dict matching this
    route's public field names.

    Field map (AU-CUTOVER.md §2.2): ``schema_digest`` <- ``input_schema_digest``
    (now ``sha256:<hex>`` of the served schema, not AU's own JSON digest);
    ``tool_mode``/``mime_type``(<- ``media_type``)/``classification``/``uri``/
    ``resource_kind``/``skill_type``/``enabled`` map by name or 1:1 rename.
    EG has no separate ``server_id`` -- ``server_name`` is the only join key
    to a registered server now, so ``server_id`` is populated from it too
    (a display-continuity best-effort, not a distinct identity).
    """
    body = getattr(entry, "row", None)
    if hasattr(body, "outcome"):  # FleetDiscoveryRow
        reachable, last_error = _discovery_outcome_fields(
            getattr(body, "outcome", None)
        )
        counts = getattr(body, "counts", None)
        return {
            "id": str(getattr(body, "id", "") or ""),
            "server_id": str(getattr(body, "server_name", "") or ""),
            "server_name": str(getattr(body, "server_name", "") or ""),
            "name": str(getattr(body, "server_name", "") or ""),
            "reachable": reachable,
            "last_error": last_error,
            "tool_count": int(getattr(counts, "tools", 0) or 0),
            "skill_count": int(getattr(counts, "skills", 0) or 0),
            "prompt_count": int(getattr(counts, "prompts", 0) or 0),
            "resource_count": int(getattr(counts, "resources", 0) or 0),
            "observed_at": str(getattr(body, "observed_at_ms", "") or ""),
            "tenant_id": str(
                getattr(getattr(body, "acl", None), "tenant_id", "") or ""
            ),
        }
    component = getattr(body, "component", None)
    row = {
        "id": str(getattr(component, "id", "") or ""),
        "server_id": str(getattr(component, "server_name", "") or ""),
        "server_name": str(getattr(component, "server_name", "") or ""),
        "name": str(getattr(component, "name", "") or ""),
        "description": str(getattr(component, "description", "") or ""),
        "enabled": bool(getattr(component, "enabled", True)),
        "tenant_id": str(
            getattr(getattr(component, "acl", None), "tenant_id", "") or ""
        ),
        "provider": str(getattr(component, "connector", "") or ""),
        "mcp_server": str(getattr(component, "server_name", "") or ""),
    }
    if hasattr(body, "tool_mode"):  # FleetToolRow
        row["schema_digest"] = str(getattr(body, "input_schema_digest", "") or "")
        row["tool_mode"] = str(getattr(body, "tool_mode", "") or "")
    elif hasattr(body, "resource_kind"):  # FleetResourceRow
        row["uri"] = str(getattr(body, "uri", "") or "")
        row["mime_type"] = str(getattr(body, "media_type", "") or "")
        row["resource_kind"] = str(getattr(body, "resource_kind", "") or "")
    elif hasattr(body, "skill_type"):  # FleetSkillRow
        row["uri"] = str(getattr(body, "uri", "") or "")
        row["skill_type"] = str(getattr(body, "skill_type", "") or "")
        row["classification"] = str(getattr(body, "classification", "") or "")
    elif hasattr(body, "uri"):  # FleetPromptRow
        row["uri"] = str(getattr(body, "uri", "") or "")
    return row


def _authorized_page(
    kind: str,
    *,
    grant_digests: tuple[str, ...],
    query: str,
    after: dict[str, Any] | None,
    limit: int,
    engine: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    """Read one page directly from EG: ``(rows, next_cursor_json, total)``.

    EH-345: tenant/principal/grant visibility, substring filtering, keyset
    ordering and the total are ALL computed engine-side now (one round trip)
    -- there is no local WHERE-clause construction, no local scope
    validation, and no separate untrusted-COUNT reconciliation to perform
    (the deleted ``_build_where``/``_validate_scope*``/``_authorized_count``/
    ``_page_total`` apparatus existed only because the old SQL tier's count
    and row-scope were each independently untrustworthy; EG's page read is
    the single source for both).
    """
    server_registry, fleet_catalog = _fleet_catalog_clients(engine)
    if kind == "servers":
        if server_registry is None:
            raise CatalogUnavailable("authoritative server registry is unavailable")
        cursor = None
        if after:
            from epistemic_graph.generated.server_registry import (
                RegisteredServerCursor,
            )

            cursor = RegisteredServerCursor(**after)
        page = server_registry.page(limit=limit, cursor=cursor)
        rows = [_row_from_server_view(view) for view in page.entries]
        next_cursor = (
            page.next_cursor.model_dump(mode="json") if page.next_cursor else None
        )
        return rows, next_cursor, int(page.total_live)
    if fleet_catalog is None:
        raise CatalogUnavailable("authoritative fleet catalog is unavailable")
    from epistemic_graph.generated.fleet_catalog import (
        FleetCatalogCursor,
        FleetCatalogListRequest,
    )

    cursor = FleetCatalogCursor(**after) if after else None
    try:
        page = fleet_catalog.page(
            FleetCatalogListRequest(
                kind=kind,
                query=query or None,
                grant_digests=list(grant_digests),
                limit=limit,
                cursor=cursor,
            )
        )
    except Exception as exc:  # noqa: BLE001 - explicit unavailable response
        logger.warning("authoritative registry catalog page read failed: %s", exc)
        raise CatalogUnavailable("authoritative catalog read failed") from exc
    rows = [_row_from_fleet_row(entry) for entry in page.rows]
    next_cursor = page.next_cursor.model_dump(mode="json") if page.next_cursor else None
    return rows, next_cursor, int(page.total)


def _authorized_item(
    kind: str,
    *,
    grant_digests: tuple[str, ...],
    item_id: str,
    engine: Any,
) -> dict[str, Any] | None:
    """Read at most one row by id."""

    server_registry, fleet_catalog = _fleet_catalog_clients(engine)
    if kind == "servers":
        if server_registry is None:
            raise CatalogUnavailable("authoritative server registry is unavailable")
        for view in server_registry.list_all():
            if _row_from_server_view(view)["id"] == item_id:
                return _row_from_server_view(view)
        return None
    if fleet_catalog is None:
        raise CatalogUnavailable("authoritative fleet catalog is unavailable")
    try:
        answer = fleet_catalog.lookup([item_id], grant_digests=grant_digests)
    except Exception as exc:  # noqa: BLE001 - explicit unavailable response
        logger.warning("authoritative registry catalog item read failed: %s", exc)
        raise CatalogUnavailable("authoritative catalog read failed") from exc
    for entry in answer.rows:
        row = _row_from_fleet_row(entry)
        if row.get("id") == item_id:
            return row
    # The same response is used for an absent row and another tenant's row
    # (the engine's own visibility predicate already filtered both alike).
    return None


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
            require_discovery_binding=kind in _DISCOVERY_BOUND_KINDS
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="registry access denied") from exc
    after: dict[str, Any] | None = None
    if cursor:
        after = _decode_cursor(
            cursor,
            kind=kind,
            query=query,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
        )
    engine = _get_catalog_engine()
    try:
        page_rows, eg_next_cursor, total = await _offload_catalog_call(
            _authorized_page,
            kind,
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
    next_cursor = None
    if eg_next_cursor is not None:
        next_cursor = _cursor_token(
            kind=kind,
            query=query,
            after=eg_next_cursor,
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
        _tenant, _principal, grant_digests = _require_catalog_authority(
            require_discovery_binding=kind in _DISCOVERY_BOUND_KINDS
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="registry access denied") from exc
    try:
        row = await _offload_catalog_call(
            _authorized_item,
            kind,
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

    toggleable = list(_iter_toggleable_items(kinds_map))
    if not toggleable:
        return

    keys = [builder(item) for builder, item in toggleable]
    toggle_states = await _offload_catalog_call(get_toggle_states_batch, engine, keys)

    for builder, item in toggleable:
        key = builder(item)
        toggled = bool(toggle_states.get(key, True))
        item["enabled"] = toggled and bool(item.get("enabled", True))


def _iter_toggleable_items(
    kinds_map: dict[str, RegistryKindResult],
) -> Iterator[tuple[Callable[[dict[str, Any]], tuple[str, str]], dict[str, Any]]]:
    """Yield ``(key_builder, item)`` for every item, across every requested
    kind, that carries an ``enabled`` field and has a toggle-key convention
    in ``_TOGGLE_KEY_BUILDERS``."""
    for kind, result in kinds_map.items():
        builder = _TOGGLE_KEY_BUILDERS.get(kind)
        if builder is None or result.status != "ok":
            continue
        for item in result.items:
            if "enabled" in item:
                yield builder, item


async def _list_multi_kind(request: Request) -> RegistryMultiKindPage:
    """``GET /api/registry?kinds=a,b,c[&limit=][&q=][&cursor_<kind>=][&include=toggle]``.

    Reuses ``_authorized_page`` — the EXACT same EG read the single-kind
    ``GET /api/registry/{kind}`` route uses — via the same
    ``_offload_catalog_call`` off-loop wrapper, so there is exactly one read
    implementation for both surfaces (no forked second copy to drift out of
    sync).

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
    require_discovery_binding = any(kind in _DISCOVERY_BOUND_KINDS for kind in kinds)
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
    afters: dict[str, dict[str, Any] | None] = {}
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
        try:
            page_rows, eg_next_cursor, total = await _offload_catalog_call(
                _authorized_page,
                kind,
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

        next_cursor = None
        if eg_next_cursor is not None:
            next_cursor = _cursor_token(
                kind=kind,
                query=query,
                after=eg_next_cursor,
                tenant=tenant,
                principal=principal,
                grant_digests=grant_digests,
            )
        try:
            items = [_validate_item(kind, _KIND_MODELS[kind], row) for row in page_rows]
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


for _kind, _model in _KIND_MODELS.items():
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
