"""REST twin of the MCP fleet catalog meta-tools (GOC-60-W03).

CONCEPT:AU-ECO.mcp.catalog-rest-surface

``agent_utilities/mcp/multiplexer.py`` already computes the fleet's
dispatchable truth for the ``find_tools``/``load_tools``/``list_catalog``/
``multiplexer_status`` MCP meta-tools (``list_catalog`` at ``:3728``,
``multiplexer_status`` at ``:4314``) — strictly more accurate than either a
nonexistent KG method or a static config file, since a tool's reported
``mounted`` state is derived from the SAME session-visibility predicate the
real dispatch gate (``SessionVisibilityMiddleware``) enforces.

It had **no REST route at all** — a violation of this repository's own
"Two surfaces by default" rule (every capability reachable via the gateway
AND MCP). This module is that REST route, dispatching into the SAME
``MCPMultiplexer.list_catalog``/``status_snapshot`` methods the MCP tools
call (via the process-wide standalone instance in
``agent_utilities.mcp.shared_multiplexer``) — never re-deriving the logic.

Authorization mirrors the MCP-tool-side gate
(``multiplexer._require_fleet_capability("discover")``) so the two surfaces
require the same capability, not merely serve the same payload shape.

A multiplexer/catalog failure is surfaced as a typed 503 payload with
``status: "DEGRADED"``, never silently downgraded to an empty list or a
generic 500 (GOC-60 lane authority/invariant 1).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)

__all__ = ["router"]

# Mirrors ``multiplexer._require_fleet_capability("discover")``'s scope set
# exactly (``{"mcp:discover", "mcp:delegate", *administrative}``) so the REST
# and MCP surfaces enforce the identical authorization requirement.
_DISCOVER_SCOPES = frozenset(
    {"mcp:discover", "mcp:delegate", "mcp:admin", "kg:admin", "admin"}
)


def _mcp_capabilities(request: Request) -> set[str] | None:
    """Resolve the caller's capabilities from verified identity.

    Same pattern as ``routers/enhanced.py``'s ``_enhanced_capabilities``:
    returns ``None`` (skip enforcement) only for the already-fully-trusted
    static-API-key path, matching the existing REST convention rather than
    inventing a new one here.
    """
    claims = getattr(request.state, "user_claims", None)
    if not claims or claims.get("auth_type") == "api_key":
        return None
    try:
        from agent_utilities.core.config import config
        from agent_utilities.security.identity import (
            base_capabilities,
            normalize_identity,
        )

        return set(
            base_capabilities(
                normalize_identity(claims), config.identity_group_capability_map
            )
        )
    except Exception:
        raise HTTPException(
            status_code=403, detail="MCP fleet capability required"
        ) from None


async def _require_mcp_discover(request: Request) -> None:
    capabilities = _mcp_capabilities(request)
    if capabilities is not None and not capabilities.intersection(_DISCOVER_SCOPES):
        raise HTTPException(
            status_code=403, detail="MCP fleet discover capability required"
        )


router = APIRouter(
    prefix="/api/mcp",
    tags=["MCP Fleet Catalog"],
    dependencies=[Depends(_require_mcp_discover)],
)


def _degraded(reason: str, exc: Exception) -> HTTPException:
    logger.error(
        "MCP fleet catalog degraded: %s (exception_type=%s)",
        reason,
        type(exc).__name__,
    )
    return HTTPException(
        status_code=503,
        detail={
            "status": "DEGRADED",
            "reason": reason,
            "detail": f"{type(exc).__name__}: {exc}"[:500],
        },
    )


async def _get_multiplexer_or_503():
    from agent_utilities.mcp.shared_multiplexer import get_shared_multiplexer

    try:
        return await get_shared_multiplexer()
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed 503 below, cause preserved via `from exc`
        raise _degraded("mcp_multiplexer_unavailable", exc) from exc


@router.get(
    "/catalog",
    summary="Browse the MCP fleet catalog (REST twin of the `list_catalog` MCP tool)",
)
async def get_mcp_catalog(
    server: str = "", include_tools: bool = True
) -> dict[str, Any]:
    """Mountable/mounted MCP servers with live per-tool dispatchable truth.

    ``server=""`` (default) returns the whole-fleet browse; a specific
    ``server`` name drills into that one server's full tool list. Identical
    contract to ``MCPMultiplexer.list_catalog`` because this calls it
    directly with no REST-side reshaping.
    """
    mux = await _get_multiplexer_or_503()
    try:
        result = await mux.list_catalog(server=server, include_tools=include_tools)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed 503 below, cause preserved via `from exc`
        raise _degraded("list_catalog_failed", exc) from exc
    if isinstance(result, dict) and result.get("error") and server:
        # A drill-down into an unknown server: `list_catalog` reports this as
        # a typed `{"error": ...}` payload rather than raising. Preserve that
        # as a typed 404 rather than a 200 the caller must sniff for errors.
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get(
    "/status",
    summary="Fleet health (REST twin of the `multiplexer_status` MCP tool)",
)
async def get_mcp_status() -> dict[str, Any]:
    """Per-child state (up/restarting/failed), restart counts, concurrency
    limits, in-flight/queued calls — identical to the ``multiplexer_status``
    MCP tool's ``mux.status_snapshot()`` call, no reshaping.
    """
    mux = await _get_multiplexer_or_503()
    try:
        return mux.status_snapshot()
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed 503 below, cause preserved via `from exc`
        raise _degraded("status_snapshot_failed", exc) from exc
