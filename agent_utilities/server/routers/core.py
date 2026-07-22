import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent_utilities.core.chat_persistence import (
    get_chat_from_disk,
    list_chats_from_disk,
)
from agent_utilities.security.error_surface import public_error_payload

from ..models import CodemapRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Core"])


@router.get("/models", summary="List Configured Models")
async def list_configured_models(request: Request) -> dict[str, Any]:
    """Return the configured model registry.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Consumers: web-UI model picker + cost table, terminal-UI
        ``/model list``, graph orchestrator's specialist spawner.
    """
    reg = getattr(request.app.state, "model_registry", None)
    if reg is None:
        return {"models": [], "default_id": None}
    return reg.to_api_payload()


@router.get("/health", summary="Health Check")
async def health_check(request: Request):
    """LIVENESS: always 200 — this process is up and answering.

    The JSON body carries the ONE truthful, shared health report (real engine
    reachability + circuit-breaker state, plus every configured co-service/
    dependency) from :func:`~agent_utilities.observability.runtime_health.collect_health`
    — the SAME core the graph-os MCP server's ``/health`` and
    ``graph_configure(action="health")`` dispatch into. A downstream
    dependency being down never flips this endpoint's HTTP status: killing/
    restarting this process over a dependency outage would only crash-loop a
    fine process. Use ``GET /health/ready`` for the readiness signal that DOES
    flip status code (CONCEPT:AU-OS.deployment.liveness-vs-readiness-split).
    """
    import asyncio

    from agent_utilities.observability.runtime_health import collect_health

    report = await asyncio.to_thread(collect_health)
    return JSONResponse(report, headers={"Cache-Control": "no-store"})


@router.get("/health/ready", summary="Readiness Check")
async def readiness_check(request: Request):
    """READINESS: the same health report, with HTTP 200/503 reflecting it.

    kubelet uses this to stop routing traffic to a genuinely unhealthy pod
    without restarting the process (CONCEPT:AU-OS.deployment.liveness-vs-readiness-split).
    """
    import asyncio

    from agent_utilities.observability.runtime_health import (
        collect_health,
        is_overall_healthy,
    )

    report = await asyncio.to_thread(collect_health)
    status_code = 200 if is_overall_healthy(report) else 503
    return JSONResponse(
        report, status_code=status_code, headers={"Cache-Control": "no-store"}
    )


@router.get("/chats", summary="List Chat History")
async def list_chats():
    """Returns a list of all stored chat sessions."""
    return list_chats_from_disk()


@router.get("/chats/{chat_id}", summary="Get Chat Details")
async def get_chat(chat_id: str):
    """Returns the full message history for a specific chat."""
    chat_data = get_chat_from_disk(chat_id)
    if not chat_data:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return chat_data


@router.get("/tools", summary="List Available Tools and Skills")
async def list_tools():
    """Returns a list of all tools and skills loaded in the Knowledge Graph."""
    from ...knowledge_graph.core.engine import IntelligenceGraphEngine

    kg = IntelligenceGraphEngine.get_active()
    if not kg or not kg.backend:
        return []

    # Query for Tools
    tool_query = "MATCH (t:Tool) RETURN t.id AS id, t.name AS name, t.description AS descriptionription, t.mcp_server AS source_name, 'tool' AS type"
    tools = kg.query_cypher(tool_query) or []

    # Query for Skills
    skill_query = "MATCH (s:Skill) RETURN s.id AS id, s.name AS name, s.description AS descriptionription, s.category AS source_name, 'skill' AS type"
    skills = kg.query_cypher(skill_query) or []

    return tools + skills


@router.post("/api/codemap", summary="Generate a codebase codemap")
async def generate_codemap_endpoint(payload: CodemapRequest):
    """Generate a task-specific hierarchical codemap artifact."""
    from ...knowledge_graph.core.codemaps import CodemapGenerator
    from ...knowledge_graph.core.engine import IntelligenceGraphEngine

    kg = IntelligenceGraphEngine.get_active()
    if not kg:
        return JSONResponse(
            {"status": "error", "message": "Knowledge Graph not initialized"},
            status_code=503,
        )

    generator = CodemapGenerator(kg)
    try:
        if payload.skeleton:
            # ORCH-1.48 — fast token-budgeted ranked-symbol skeleton (no LLM pass).
            text = await generator.skeleton(
                prompt=payload.prompt, max_tokens=payload.max_tokens
            )
            return {"status": "success", "skeleton": text}
        artifact = await generator.create(prompt=payload.prompt, mode=payload.mode)
        return {
            "status": "success",
            "codemap_id": artifact.id,
            "artifact": artifact.model_dump(),
        }
    except Exception as exc:
        return JSONResponse(
            public_error_payload(exc, logger=logger),
            status_code=500,
        )
