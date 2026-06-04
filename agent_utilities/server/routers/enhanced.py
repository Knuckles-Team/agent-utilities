import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enhanced", tags=["Enhanced API"])


@router.get("/info")
async def get_enhanced_info():
    return {"status": "ok", "message": "Enhanced API is active"}


@router.get("/graph/stats")
async def get_graph_stats():
    # Centralized stats for the Knowledge Graph
    return {"status": "ok", "nodes": 42, "edges": 89}


@router.get("/kb/list")
async def list_kb():
    return {
        "status": "ok",
        "knowledge_bases": [
            {"id": "workspace-docs", "name": "Workspace Documents"},
            {"id": "mcp-servers-index", "name": "MCP Servers Index"},
        ],
    }


@router.get("/sdd/specs")
async def list_sdd_specs():
    return {
        "status": "ok",
        "specs": [
            {
                "id": "ORCH-1.8",
                "title": "Parallel Execution Engine",
                "status": "Approved",
            },
            {"id": "KG-2.0", "title": "Epistemic Graph Schema", "status": "Draft"},
            {
                "id": "TUI-2.0",
                "title": "TUI Bindings and Layouts",
                "status": "In Review",
            },
        ],
    }


@router.get("/resources")
async def list_resources():
    return {
        "status": "ok",
        "resources": [
            {"id": "agent-research-01", "type": "ScholarX Searcher", "status": "Idle"},
            {
                "id": "agent-tui-helper",
                "type": "ACP Protocol Client",
                "status": "Running",
            },
        ],
    }


@router.get("/maintenance/status")
async def get_maintenance_status():
    return {"status": "ok", "maintenance_required": False}


@router.get("/pipeline/status")
async def get_pipeline_status():
    return {"status": "ok", "pipeline_active": True}


@router.get("/agents")
async def list_agents():
    """Asynchronously discover and list specialists from the Knowledge Graph dynamically."""
    try:
        from agent_utilities.agent.discovery import discover_all_specialists

        specialists = discover_all_specialists()
        agents = [
            {
                "name": s.name,
                "description": s.description,
                "skills": s.capabilities,
                "type": s.source,
            }
            for s in specialists
        ]
        return {"status": "ok", "agents": agents}
    except Exception as e:
        logger.error(f"Failed to discover specialists dynamically: {e}")
        return {"status": "error", "message": str(e), "agents": []}


@router.get("/skills")
async def list_skills(request: Request):
    """Retrieve all loaded/active custom skills on this agent instance."""
    skills = []
    agent_instance = getattr(request.app.state, "agent_instance", None)
    if agent_instance and hasattr(agent_instance, "skills"):
        for s in agent_instance.skills:
            skills.append({"id": s.id, "name": s.name, "description": s.description})
    return {"status": "ok", "skills": skills}
