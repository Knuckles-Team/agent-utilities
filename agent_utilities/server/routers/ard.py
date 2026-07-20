"""ARD registry REST surface — the gateway twin of the graph-os custom routes.

CONCEPT:AU-ECO.mcp.eco-serves-two-ard / ECO-4.97. Serves the two ARD artifacts at the bare domain root so
external agents (and the ``hf discover`` CLI) discover our fleet:

* ``GET  /.well-known/ai-catalog.json`` — the static signed manifest.
* ``POST /search`` — the dynamic, ranked, federated registry API.

Both delegate into the single core (:mod:`ecosystem.ard_registry` /
:mod:`ecosystem.ard_federation`); the matching ``@mcp.custom_route`` handlers in
``mcp/kg_server.py`` call the same core, keeping the surfaces in lockstep. The router is
mounted with **no prefix** (in ``server/app.py``) and resolves before the optional SPA
``/`` mount, so the well-known path is reachable at the domain root.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agentic Resource Discovery"])


@router.get("/.well-known/ai-catalog.json", summary="ARD static capability manifest")
async def ai_catalog() -> dict:
    """Return the signed ``ai-catalog.json`` describing our discoverable resources."""
    from ...ecosystem.ard_registry import build_ai_catalog

    return build_ai_catalog()


@router.post("/search", summary="ARD dynamic registry search")
async def ard_search_endpoint(request: Request) -> dict:
    """Rank discoverable resources for an NL query (with optional federation).

    Accepts the ARD envelope ``{"query":{"text","filter":{"type":[...]}},"pageSize",
    "federationMode","via"}`` and returns ranked, media-type-filtered, optionally
    peer-merged results.
    """
    from ...ecosystem.ard_federation import ArdFederationRelay

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — a malformed body is an empty query, not a 500
        body = {}
    query = body.get("query") or {}
    text = str(query.get("text") or body.get("text") or "")
    types = ((query.get("filter") or {}).get("type")) or None
    page_size = int(body.get("pageSize") or 5)
    mode = body.get("federationMode")
    via = body.get("via") or []

    return ArdFederationRelay().federated_search(
        text, types=types, page_size=page_size, mode=mode, via=via
    )
