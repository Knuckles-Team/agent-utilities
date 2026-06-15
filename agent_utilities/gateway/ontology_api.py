"""Granular, typed, OpenAPI-visible REST surface for the ontology/object layer.

The ontology capabilities are exposed over MCP as collapsed action-routed tools
(``ontology_*`` / ``object_*``) and as collapsed action-routed REST twins
(``POST /api/ontology/value-types`` + an ``action`` in the body). That is ideal
for agents (few tools, context-cheap) but opaque to HTTP/automation clients:
there is no ``GET /api/ontology/value-types/{name}``, no ``GET
/api/objects/{id}/history``, and none of it appears in ``/openapi.json``.

This module layers a thin **granular** surface on top — resource-style GET
routes with typed path/query params and a documented response envelope, mounted
as a FastAPI ``APIRouter`` so they show up in OpenAPI. Every handler is pure
sugar: it builds the ``action`` + params and dispatches through the SAME
``_execute_tool`` single source of truth the collapsed routes and MCP tools use
(no new business logic, no duplication). The collapsed routes stay for agents;
the parity contract test is unaffected.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

ontology_router = APIRouter(tags=["ontology"])


class OntologyEnvelope(BaseModel):
    """Uniform response envelope for the granular ontology reads."""

    status: str = Field(default="success")
    result: Any = Field(default=None, description="Tool result payload.")


async def _call(tool: str, **kwargs: Any) -> Any:
    """Dispatch through the shared in-process tool registry (single SoT)."""
    from agent_utilities.mcp.kg_server import _execute_tool, safe_json_load

    raw = await _execute_tool(tool, **kwargs)
    return safe_json_load(raw)


def _not_found_if_error(result: Any, detail: str) -> Any:
    """Map a tool's ``{"error": ...}`` payload to a 404 for resource GETs."""
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Value types (CONCEPT:KG-2.39) ───────────────────────────────────────────


@ontology_router.get("/ontology/value-types", response_model=OntologyEnvelope)
async def list_value_types() -> OntologyEnvelope:
    """List all constrained value-type names."""
    return OntologyEnvelope(result=await _call("ontology_value_types", action="list"))


@ontology_router.get("/ontology/value-types/{name}", response_model=OntologyEnvelope)
async def get_value_type(name: str) -> OntologyEnvelope:
    """Describe one value type (404 if unknown)."""
    res = await _call("ontology_value_types", action="describe", name=name)
    return OntologyEnvelope(result=_not_found_if_error(res, f"value type {name!r}"))


# ── Property types (CONCEPT:KG-2.47) ─────────────────────────────────────────


@ontology_router.get("/ontology/property-types", response_model=OntologyEnvelope)
async def list_property_types() -> OntologyEnvelope:
    """List all property-type names."""
    return OntologyEnvelope(
        result=await _call("ontology_property_types", action="list")
    )


@ontology_router.get(
    "/ontology/property-types/{type_ref:path}", response_model=OntologyEnvelope
)
async def describe_property_type(type_ref: str) -> OntologyEnvelope:
    """Describe a property type ref, e.g. ``array<string>`` (404 if unknown)."""
    res = await _call("ontology_property_types", action="describe", type_ref=type_ref)
    return OntologyEnvelope(result=_not_found_if_error(res, f"type {type_ref!r}"))


# ── Interfaces (CONCEPT:KG-2.38) ─────────────────────────────────────────────


@ontology_router.get("/ontology/interfaces", response_model=OntologyEnvelope)
async def list_interfaces(
    registry: str = Query("structural", description="'structural' or 'enterprise'."),
) -> OntologyEnvelope:
    """List interface names in the chosen registry."""
    return OntologyEnvelope(
        result=await _call("ontology_interface", action="list", registry=registry)
    )


@ontology_router.get("/ontology/interfaces/{name}", response_model=OntologyEnvelope)
async def get_interface_implementers(
    name: str,
    registry: str = Query("structural", description="'structural' or 'enterprise'."),
) -> OntologyEnvelope:
    """Resolve one interface/type to its concrete implementer types."""
    return OntologyEnvelope(
        result=await _call(
            "ontology_interface",
            action="implementers",
            name=name,
            registry=registry,
        )
    )


# ── Functions (CONCEPT:KG-2.41) ──────────────────────────────────────────────


@ontology_router.get("/ontology/functions", response_model=OntologyEnvelope)
async def list_functions() -> OntologyEnvelope:
    """List registered ontology functions with their typed signatures."""
    return OntologyEnvelope(result=await _call("ontology_function", action="list"))


@ontology_router.get("/ontology/functions/{name}", response_model=OntologyEnvelope)
async def get_function(name: str) -> OntologyEnvelope:
    """Get one function's signature by name (404 if unknown)."""
    listing = await _call("ontology_function", action="list")
    match = None
    if isinstance(listing, list):
        match = next((f for f in listing if f.get("name") == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"function {name!r}")
    return OntologyEnvelope(result=match)


# ── Objects: read + edit history (CONCEPT:KG-2.43/2.45) ──────────────────────


@ontology_router.get("/objects/{object_id}", response_model=OntologyEnvelope)
async def get_object(object_id: str) -> OntologyEnvelope:
    """Read a single object by id (via the object-set service)."""
    return OntologyEnvelope(
        result=await _call(
            "object_set", action="from_ids", ids_json=json.dumps([object_id])
        )
    )


@ontology_router.get("/objects/{object_id}/history", response_model=OntologyEnvelope)
async def get_object_history(object_id: str) -> OntologyEnvelope:
    """Per-object edit history / changelog (CONCEPT:KG-2.43)."""
    return OntologyEnvelope(
        result=await _call("object_edits", action="history", object_id=object_id)
    )


@ontology_router.get("/objects/{object_id}/as-of", response_model=OntologyEnvelope)
async def get_object_as_of(
    object_id: str,
    ts: float = Query(..., description="Unix timestamp for the point-in-time view."),
) -> OntologyEnvelope:
    """Bitemporal as-of snapshot of an object (CONCEPT:KG-2.43)."""
    return OntologyEnvelope(
        result=await _call("object_edits", action="as_of", object_id=object_id, ts=ts)
    )


# ── LeanIX metamodel sync (CONCEPT:KG-2.9) ──────────────────────────────────


@ontology_router.post("/ontology/leanix/sync", response_model=OntologyEnvelope)
async def sync_leanix_ontology_route(
    dry_run: bool = Query(
        default=True,
        description="Preview the generated ontology without writing (default). Set false to apply.",
    ),
) -> OntologyEnvelope:
    """Discover the live LeanIX metamodel and mirror it natively as OWL/RDF."""
    return OntologyEnvelope(
        result=await _call("ontology_leanix_sync", dry_run=dry_run)
    )


def register_ontology_routes(app, prefix: str = "/api") -> None:
    """Mount the granular typed ontology surface onto ``app``.

    On FastAPI this uses ``include_router`` so the routes appear in
    ``/openapi.json``; on a plain Starlette app it degrades to ``add_route``
    (no schema, but the endpoints still serve).
    """
    if hasattr(app, "include_router"):  # FastAPI
        app.include_router(ontology_router, prefix=prefix)
        return
    # Plain Starlette fallback: bridge each typed route to a Request handler.
    from starlette.responses import JSONResponse

    def _bridge(endpoint, param_names):
        async def _handler(request):  # noqa: ANN001
            kwargs = {p: request.path_params.get(p) for p in param_names}
            kwargs.update(dict(request.query_params))
            try:
                env = await endpoint(**kwargs)
                return JSONResponse(env.model_dump())
            except HTTPException as e:  # noqa: PERF203
                return JSONResponse(
                    {"status": "error", "message": e.detail}, status_code=e.status_code
                )

        return _handler

    for route in ontology_router.routes:
        param_names = list(getattr(route, "param_convertors", {}) or {})
        app.add_route(
            prefix + route.path,
            _bridge(route.endpoint, param_names),
            methods=list(route.methods or ["GET"]),
        )
