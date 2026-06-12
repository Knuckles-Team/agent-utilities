"""Local SPARQL endpoint router.

CONCEPT:KG-2.7 — a zero-dependency SPARQL surface over the OWL/RDF bridge
(rdflib materialization of the live LPG + OWL inferences). The canonical mount is
:func:`agent_utilities.gateway.graph_api.register_graph_routes` (``{prefix}/sparql``);
this standalone router is provided for apps that want to mount SPARQL on its own.

The previous implementation here was a broken stub that dumped every edge via a
non-existent ``_get_all_edges`` helper; it now delegates to the shared bridge.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sparql", tags=["sparql"])


@router.api_route("", methods=["GET", "POST"])
@router.api_route("/", methods=["GET", "POST"])
async def sparql(request: Request) -> JSONResponse:
    """Execute a SPARQL query over the local OWL/RDF bridge."""
    from agent_utilities.gateway.graph_api import _get_sparql_bridge

    query = request.query_params.get("query")
    if not query and request.method == "POST":
        try:
            body = await request.json()
            query = body.get("query") if isinstance(body, dict) else None
        except Exception:
            query = (await request.body()).decode("utf-8", "replace") or None
    if not query:
        return JSONResponse(
            {"status": "error", "message": "missing 'query'"}, status_code=400
        )

    bridge = _get_sparql_bridge()
    if bridge is None:
        return JSONResponse(
            {"status": "error", "message": "SPARQL layer unavailable"},
            status_code=503,
        )
    try:
        bindings = bridge.query_sparql(query)
        varnames = list(bindings[0].keys()) if bindings else []
        return JSONResponse(
            {
                "status": "success",
                "head": {"vars": varnames},
                "results": {"bindings": bindings},
            }
        )
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)
