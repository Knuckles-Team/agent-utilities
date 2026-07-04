#!/usr/bin/python
from __future__ import annotations

"""Granular, typed REST surface for Agent-Native Research Artifacts (CONCEPT:AU-KG.ontology.verified-by-implemented-by).

Mirrors :mod:`agent_utilities.gateway.ontology_api`: thin typed routes that dispatch
through the **same** in-process ``research_artifact`` MCP tool (single source of truth),
so the gateway and MCP surfaces can never drift. Exposes the one ontology-driven research
pipeline — OWL/RDF reasoning over the whole ecosystem (``reason``), ecosystem-grounded
ARA compilation (``compile``), OWL/SHACL-grounded review + certification (``review``),
live event capture with provenance (``capture``), plus reads.

Concept: research-api
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

research_router = APIRouter(tags=["research"])


class ResearchEnvelope(BaseModel):
    """Uniform response envelope for the research surface."""

    status: str = Field(default="success")
    result: Any = Field(default=None, description="ARA action result payload.")


async def _call(action: str, **kwargs: Any) -> Any:
    """Dispatch through the shared in-process tool registry (single SoT)."""
    from agent_utilities.mcp.kg_server import _execute_tool, safe_json_load

    raw = await _execute_tool("research_artifact", action=action, **kwargs)
    return safe_json_load(raw)


def _error_to_http(result: Any, status: int = 400) -> Any:
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=status, detail=result["error"])
    return result


# ── reasoning (the keystone) ────────────────────────────────────────────────


@research_router.post("/research/reason", response_model=ResearchEnvelope)
async def reason(query: str = "") -> ResearchEnvelope:
    """Reason over the whole ecosystem ontology and harvest extrapolated relationships."""
    return ResearchEnvelope(result=await _call("reason", query=query))


# ── compile / review ────────────────────────────────────────────────────────


@research_router.post("/research/compile", response_model=ResearchEnvelope)
async def compile_artifact(
    article_id: str, target_codebase: str = ""
) -> ResearchEnvelope:
    """Compile a paper into an ecosystem-grounded OWL-native ARA."""
    res = await _call("compile", article_id=article_id, target_codebase=target_codebase)
    return ResearchEnvelope(result=_error_to_http(res))


@research_router.post("/research/review", response_model=ResearchEnvelope)
async def review_artifact(article_id: str, level: str = "L1") -> ResearchEnvelope:
    """Run the ARA Seal (L1/L2/L3) over an artifact and certify it."""
    res = await _call("review", article_id=article_id, level=level)
    return ResearchEnvelope(result=_error_to_http(res))


# ── perspectival inquiry (STORM, native) ────────────────────────────────────


@research_router.post("/research/inquire", response_model=ResearchEnvelope)
async def inquire(topic: str, materialize: bool = True) -> ResearchEnvelope:
    """Run a native multi-perspective (STORM) inquiry over ``topic`` (CONCEPT:AU-KG.research.perspectival-inquiry)."""
    res = await _call("inquire", topic=topic, materialize=materialize)
    return ResearchEnvelope(result=_error_to_http(res))


# ── live capture ────────────────────────────────────────────────────────────


@research_router.post("/research/capture", response_model=ResearchEnvelope)
async def capture_event(
    article_id: str,
    text: str,
    provenance: str = "ai_executed",
    actor: str = "",
) -> ResearchEnvelope:
    """Capture a live research event (with provenance) and promote it to the graph."""
    res = await _call(
        "capture",
        article_id=article_id,
        text=text,
        provenance=provenance,
        actor=actor,
    )
    return ResearchEnvelope(result=_error_to_http(res))


# ── reads ───────────────────────────────────────────────────────────────────


@research_router.get("/research/artifacts", response_model=ResearchEnvelope)
async def list_artifacts(limit: int = 50) -> ResearchEnvelope:
    """List compiled research artifacts."""
    return ResearchEnvelope(result=await _call("list", limit=limit))


@research_router.get("/research/artifact/{article_id}", response_model=ResearchEnvelope)
async def get_artifact(article_id: str) -> ResearchEnvelope:
    """Fetch one compiled artifact and its claims (404 if absent)."""
    res = await _call("get", article_id=article_id)
    return ResearchEnvelope(result=_error_to_http(res, status=404))


def register_research_routes(app, prefix: str = "/api") -> None:
    """Mount the granular typed research surface onto ``app``.

    On FastAPI this uses ``include_router`` (routes appear in ``/openapi.json``); on a
    plain Starlette app it degrades to ``add_route`` (endpoints still serve). Mirrors
    :func:`agent_utilities.gateway.ontology_api.register_ontology_routes`.
    """
    if hasattr(app, "include_router"):  # FastAPI
        app.include_router(research_router, prefix=prefix)
        return
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
                    {"status": "error", "message": e.detail},
                    status_code=e.status_code,
                )

        return _handler

    for route in research_router.routes:
        param_names = list(getattr(route, "param_convertors", {}) or {})
        app.add_route(
            prefix + route.path,
            _bridge(route.endpoint, param_names),
            methods=list(route.methods or ["GET"]),
        )


__all__ = ["research_router", "register_research_routes", "ResearchEnvelope"]
