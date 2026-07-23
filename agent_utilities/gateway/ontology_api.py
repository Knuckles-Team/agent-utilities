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
from starlette.routing import Route

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


# ── Value types (CONCEPT:AU-KG.ontology.value-type-shacl-load) ───────────────────────────────────────────


@ontology_router.get("/ontology/value-types", response_model=OntologyEnvelope)
async def list_value_types() -> OntologyEnvelope:
    """List all constrained value-type names."""
    return OntologyEnvelope(result=await _call("ontology_value_types", action="list"))


@ontology_router.get("/ontology/value-types/{name}", response_model=OntologyEnvelope)
async def get_value_type(name: str) -> OntologyEnvelope:
    """Describe one value type (404 if unknown)."""
    res = await _call("ontology_value_types", action="describe", name=name)
    return OntologyEnvelope(result=_not_found_if_error(res, f"value type {name!r}"))


# ── Property types (CONCEPT:AU-KG.ontology.ontology-property-types) ─────────────────────────────────────────


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


# ── Interfaces (CONCEPT:AU-KG.ontology.conformance-check) ─────────────────────────────────────────────


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


# ── Schema graph / summary / lint (CONCEPT:AU-KG.ontology.schema-graph-visualization) ────


@ontology_router.get("/ontology/schema-graph", response_model=OntologyEnvelope)
async def get_ontology_schema_graph(
    registry: str = Query("structural", description="'structural' or 'enterprise'."),
) -> OntologyEnvelope:
    """The interface + link-type registries as a Cytoscape-style node/edge graph."""
    return OntologyEnvelope(
        result=await _call("ontology_interface", action="graph", registry=registry)
    )


@ontology_router.get("/ontology/schema-summary", response_model=OntologyEnvelope)
async def get_ontology_schema_summary(
    registry: str = Query("structural", description="'structural' or 'enterprise'."),
) -> OntologyEnvelope:
    """The same schema rendered as a Markdown document."""
    return OntologyEnvelope(
        result=await _call("ontology_interface", action="summary", registry=registry)
    )


@ontology_router.get("/ontology/lint", response_model=OntologyEnvelope)
async def get_ontology_lint(
    registry: str = Query("structural", description="'structural' or 'enterprise'."),
) -> OntologyEnvelope:
    """Naming-convention + typo findings for the interface registry."""
    return OntologyEnvelope(
        result=await _call("ontology_interface", action="lint", registry=registry)
    )


# ── From-scratch ontology generator (CONCEPT:AU-KG.ontology.standalone-generation, coverage row #13) ────────


class OntologyGenerateRequest(BaseModel):
    """Request body for ``POST /ontology/generate``."""

    sample_text: str = Field(
        default="",
        description="Representative document/business-scenario text to model.",
    )
    object_type: str = Field(
        default="", description="Optional domain hint, e.g. 'clinical_trial'."
    )


@ontology_router.get("/ontology/generate", response_model=OntologyEnvelope)
async def generate_ontology_get(
    sample_text: str = Query(
        "", description="Representative document/business-scenario text to model."
    ),
    object_type: str = Query(
        "", description="Optional domain hint, e.g. 'clinical_trial'."
    ),
) -> OntologyEnvelope:
    """From-scratch standalone Interface/LinkType proposal (GET, short samples).

    Same schema-discovery LLM path as ``discover_extensions``, run against an
    EMPTY base instead of a live-ontology diff. Always a human-reviewed
    proposal — never auto-applied/merged.
    """
    return OntologyEnvelope(
        result=await _call(
            "ontology_derive",
            action="generate",
            sample_text=sample_text,
            object_type=object_type,
        )
    )


@ontology_router.post("/ontology/generate", response_model=OntologyEnvelope)
async def generate_ontology_post(body: OntologyGenerateRequest) -> OntologyEnvelope:
    """From-scratch standalone Interface/LinkType proposal (POST, full document body).

    Same core as :func:`generate_ontology_get` — POST for a longer
    ``sample_text`` than comfortably fits a query string.
    """
    return OntologyEnvelope(
        result=await _call(
            "ontology_derive",
            action="generate",
            sample_text=body.sample_text,
            object_type=body.object_type,
        )
    )


# ── SHACL validation report (coverage row #9/#97 frontend gap) ──────────────


class OntologyValidateRequest(BaseModel):
    """Request body for ``POST /ontology/validate``."""

    source: str = Field(
        default="",
        description="A .ttl/OWL file path, HTTP(S) URL, or raw turtle/RDF text.",
    )
    source_type: str = Field(
        default="auto",
        description="How to read `source`: 'file' | 'url' | 'text' | 'auto'.",
    )


@ontology_router.post("/ontology/validate", response_model=OntologyEnvelope)
async def validate_ontology_candidate(
    body: OntologyValidateRequest,
) -> OntologyEnvelope:
    """Run the valid/connected/SHACL gate on a candidate WITHOUT committing it.

    Granular typed twin of ``graph_ontology(action='validate')`` — dispatches
    through the same tool/core (the ``lifecycle.py`` docstring already named
    this route; it was documented but never wired until now). The result
    includes a ``shacl_report`` (``conforms``/``text``/``turtle``) whenever
    pyshacl is installed and bundled shapes exist, so a caller gets the literal
    SHACL validation report, not just the derived valid/errors/warnings summary.
    """
    if not body.source:
        raise HTTPException(status_code=400, detail="source is required")
    return OntologyEnvelope(
        result=await _call(
            "graph_ontology",
            action="validate",
            source=body.source,
            source_type=body.source_type,
        )
    )


# ── Import / export (CONCEPT:AU-KG.ontology.import-export-rest-surface, coverage row #23) ───────────


class OntologyLoadRequest(BaseModel):
    """Request body for ``POST /ontology/load``."""

    source: str = Field(
        default="",
        description="A .ttl/OWL file path, HTTP(S) URL, or raw turtle/RDF text.",
    )
    source_type: str = Field(
        default="auto",
        description="How to read `source`: 'file' | 'url' | 'text' | 'auto' (sniff).",
    )
    iri: str = Field(
        default="",
        description="Optional IRI override (defaults to the ontology's own declared IRI).",
    )
    version: str = Field(
        default="", description="Optional version (defaults to '1.0.0')."
    )
    category: str = Field(
        default="", description="Optional catalogue category label, e.g. 'finance'."
    )
    tags: list[str] = Field(
        default_factory=list, description="Optional catalogue tags."
    )


@ontology_router.post("/ontology/load", response_model=OntologyEnvelope)
async def load_ontology(body: OntologyLoadRequest) -> OntologyEnvelope:
    """Parse + SHACL-validate + register + activate a hosted ontology.

    Granular typed twin of ``graph_ontology(action='load')`` — the route the
    agent-webui Import/Export modal POSTs a dropped/pasted ``.ttl``/RDF file
    (or a file path / URL) to. Same core as the collapsed MCP surface; no new
    business logic here.
    """
    if not body.source:
        raise HTTPException(status_code=400, detail="source is required")
    return OntologyEnvelope(
        result=await _call(
            "graph_ontology",
            action="load",
            source=body.source,
            source_type=body.source_type,
            iri=body.iri,
            version=body.version,
            category=body.category,
            tags_json=json.dumps(body.tags) if body.tags else "",
        )
    )


@ontology_router.get("/ontology/export", response_model=OntologyEnvelope)
async def export_ontology(
    iri: str = Query(..., description="Ontology IRI to export."),
    version: str = Query(
        "", description="Version to export (omit for the newest loaded)."
    ),
) -> OntologyEnvelope:
    """Re-serialize a hosted ontology to turtle (the Import/Export modal's Export button).

    Granular typed twin of ``graph_ontology(action='get', serialize=true)``.
    The response is the standard :class:`OntologyEnvelope` (turtle text lives at
    ``result.ontology.turtle``) — consistent with every other route in this
    module — the caller builds the downloadable file client-side from that string.
    """
    res = await _call(
        "graph_ontology", action="get", iri=iri, version=version, serialize=True
    )
    return OntologyEnvelope(
        result=_not_found_if_error(res, f"ontology not hosted: {iri!r}")
    )


# ── Catalogue (CONCEPT:AU-KG.ontology.catalogue-browse, coverage row #4) ─────


@ontology_router.get("/ontology/catalogue", response_model=OntologyEnvelope)
async def get_ontology_catalogue(
    search: str = Query(
        "", description="Case-insensitive substring filter over iri/version/source."
    ),
    category: str = Query(
        "", description="Filter to ontologies loaded with this catalogue category."
    ),
    source: str = Query(
        "",
        description="Filter by how the ontology was loaded: 'file' | 'url' | 'text'.",
    ),
    tag: str = Query(
        "", description="Filter to ontologies carrying this catalogue tag."
    ),
) -> OntologyEnvelope:
    """Browsable gallery over the hosted-ontology registry.

    Granular typed twin of ``graph_ontology(action='list')`` with search/facet
    filtering. The storage/lifecycle half already existed (``graph_ontology``
    hosts arbitrary named/versioned OWL/RDF ontologies); this is the curated-
    library browse surface Ontology-Playground's catalogue offers, scoped to
    what this platform actually hosts — see the coverage report's row #4
    rationale (one continuously-extended ontology library, not many
    interchangeable demo ontologies). Each entry already carries its
    #classes/#properties/#axioms counts (``OntologyLifecycle._public``).
    """
    return OntologyEnvelope(
        result=await _call(
            "graph_ontology",
            action="list",
            search=search,
            category=category,
            source_type=source,
            tag=tag,
        )
    )


# ── Sampling profiles (CONCEPT:AU-ORCH.routing.sampling-profile-selection / KG-2.94) ──────────────────────────


@ontology_router.get("/ontology/sampling-profiles", response_model=OntologyEnvelope)
async def list_sampling_profiles() -> OntologyEnvelope:
    """List the effective per-task-class sampling profiles."""
    return OntologyEnvelope(
        result=await _call("ontology_sampling_profile", action="list")
    )


@ontology_router.get(
    "/ontology/sampling-profiles/{task_class}", response_model=OntologyEnvelope
)
async def describe_sampling_profile(task_class: str) -> OntologyEnvelope:
    """Describe the sampling profile served for a task class."""
    return OntologyEnvelope(
        result=await _call(
            "ontology_sampling_profile", action="describe", task_class=task_class
        )
    )


# ── Functions (CONCEPT:AU-KG.ontology.default-runtime-bound-import) ──────────────────────────────────────────────


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
        raise HTTPException(status_code=404, detail="function not found")
    return OntologyEnvelope(result=match)


# ── Objects: read + edit history (CONCEPT:AU-KG.ontology.edit-ledger-writeback/2.45) ──────────────────────


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
    """Per-object edit history / changelog (CONCEPT:AU-KG.ontology.edit-ledger-writeback)."""
    return OntologyEnvelope(
        result=await _call("object_edits", action="history", object_id=object_id)
    )


@ontology_router.get("/objects/{object_id}/as-of", response_model=OntologyEnvelope)
async def get_object_as_of(
    object_id: str,
    ts: float = Query(..., description="Unix timestamp for the point-in-time view."),
) -> OntologyEnvelope:
    """Bitemporal as-of snapshot of an object (CONCEPT:AU-KG.ontology.edit-ledger-writeback)."""
    return OntologyEnvelope(
        result=await _call("object_edits", action="as_of", object_id=object_id, ts=ts)
    )


@ontology_router.get(
    "/objects/{source_id}/path/{target_id}", response_model=OntologyEnvelope
)
async def get_object_path(source_id: str, target_id: str) -> OntologyEnvelope:
    """Shortest path + hop-by-hop relationship chain between two objects (CONCEPT:AU-KG.ontology.object-path-finder)."""
    res = await _call(
        "object_set", action="path", source_id=source_id, target_id=target_id
    )
    return OntologyEnvelope(result=_not_found_if_error(res, "no path found"))


# ── LeanIX metamodel sync (CONCEPT:AU-KG.ingest.enterprise-source-extractor) ──────────────────────────────────


@ontology_router.post("/ontology/leanix/sync", response_model=OntologyEnvelope)
async def sync_leanix_ontology_route(
    dry_run: bool = Query(
        default=True,
        description="Preview the generated ontology without writing (default). Set false to apply.",
    ),
) -> OntologyEnvelope:
    """Discover the live LeanIX metamodel and mirror it natively as OWL/RDF."""
    return OntologyEnvelope(result=await _call("ontology_leanix_sync", dry_run=dry_run))


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
        if not isinstance(route, Route):
            continue  # pragma: no cover - APIRouter only emits Route entries here
        param_names = list(getattr(route, "param_convertors", {}) or {})
        app.add_route(
            prefix + route.path,
            _bridge(route.endpoint, param_names),
            methods=list(route.methods or ["GET"]),
        )
