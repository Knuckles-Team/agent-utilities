import logging
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent_utilities.base_utilities import __version__
from agent_utilities.core.chat_persistence import (
    get_chat_from_disk,
    list_chats_from_disk,
)

from ..models import CodemapRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Core"])


@router.get("/models", summary="List Configured Models")
async def list_configured_models(request: Request) -> dict[str, Any]:
    """Return the configured model registry.

    Consumers: web-UI model picker + cost table, terminal-UI
    ``/model list``, graph orchestrator's specialist spawner.
    """
    reg = getattr(request.app.state, "model_registry", None)
    if reg is None:
        return {"models": [], "default_id": None}
    return reg.to_api_payload()


@router.get("/health", summary="Health Check")
async def health_check(request: Request):
    """Returns the current status of the agent server."""
    name = getattr(request.app.state, "agent_name", "agent")
    graph_bundle = getattr(request.app.state, "graph_bundle", None)

    health_info: dict[str, Any] = {
        "status": "OK",
        "agent": name,
        "version": __version__,
    }
    # Add graph info if available
    if graph_bundle:
        with suppress(Exception):
            from ...graph.config_helpers import get_discovery_registry

            registry = get_discovery_registry()
            skill_agents = [a for a in registry.agents if a.agent_type == "prompt"]
            mcp_agents = [a for a in registry.agents if a.agent_type == "mcp"]
            a2a_agents = [a for a in registry.agents if a.agent_type == "a2a"]

            health_info["graph"] = {
                "skill_agents": len(skill_agents),
                "mcp_agents": len(mcp_agents),
                "a2a_agents": len(a2a_agents),
                "mcp_tools": sum(len(a.tools) for a in registry.agents),
            }
    return health_info


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


@router.post("/api/codemap", summary="Generate a codebase codemap")
async def generate_codemap_endpoint(payload: CodemapRequest):
    """Generate a task-specific hierarchical codemap artifact."""
    from ...knowledge_graph.codemaps import CodemapGenerator
    from ...knowledge_graph.engine import IntelligenceGraphEngine

    kg = IntelligenceGraphEngine.get_active()
    if not kg:
        return JSONResponse(
            {"status": "error", "message": "Knowledge Graph not initialized"},
            status_code=503,
        )

    generator = CodemapGenerator(kg)
    try:
        artifact = await generator.create(prompt=payload.prompt, mode=payload.mode)
        return {
            "status": "success",
            "codemap_id": artifact.id,
            "artifact": artifact.model_dump(),
        }
    except Exception as e:
        logger.exception("Failed to generate codemap")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )
