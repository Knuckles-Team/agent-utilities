import logging

from fastapi import APIRouter, Request

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enhanced", tags=["Enhanced API"])


def _active_engine():
    """Return the live IntelligenceGraphEngine if one is active, else None.

    Never constructs a fresh engine — these read-only status surfaces must
    report on the running process's engine (or honestly say it is cold),
    not spin up a side instance that would report misleading state.
    """
    try:
        from ...knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception as e:  # noqa: BLE001
        logger.debug("No active IntelligenceGraphEngine: %s", e)
        return None


@router.get("/info")
async def get_enhanced_info():
    return {"status": "ok", "message": "Enhanced API is active"}


@router.get("/graph/stats")
async def get_graph_stats():
    """Live node/edge counts from the active Knowledge Graph backend.

    Counts are queried from the running engine's backend (same access pattern
    as ``core.py:list_tools``). If no engine/backend is active, returns an
    honest ``unavailable`` status rather than fabricated counts.
    """
    engine = _active_engine()
    if not engine or not getattr(engine, "backend", None):
        return {
            "status": "unavailable",
            "message": "Knowledge Graph backend is not active in this process.",
        }
    backend = engine.backend
    try:
        node_rows = backend.execute("MATCH (n) RETURN count(n) AS c") or []
        edge_rows = backend.execute("MATCH ()-[r]->() RETURN count(r) AS c") or []
        nodes = int(node_rows[0]["c"]) if node_rows else 0
        edges = int(edge_rows[0]["c"]) if edge_rows else 0
    except Exception as e:  # noqa: BLE001
        logger.warning("Graph stats query failed: %s", e)
        return {
            "status": "error",
            "message": f"Graph stats query failed: {e}",
        }
    backend_name = type(backend).__name__
    return {"status": "ok", "nodes": nodes, "edges": edges, "backend": backend_name}


@router.get("/kb/list")
async def list_kb():
    """Enumerate the real KnowledgeBase nodes registered in the graph.

    Queries the active engine's backend for ``KnowledgeBase`` nodes. If no
    engine/backend is active, returns an empty list honestly.
    """
    engine = _active_engine()
    if not engine or not getattr(engine, "backend", None):
        return {
            "status": "unavailable",
            "message": "Knowledge Graph backend is not active in this process.",
            "knowledge_bases": [],
        }
    try:
        rows = (
            engine.backend.execute(
                "MATCH (kb:KnowledgeBase) "
                "RETURN kb.id AS id, kb.name AS name, kb.topic AS topic, "
                "kb.description AS description, kb.article_count AS article_count, "
                "kb.status AS status"
            )
            or []
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("KB list query failed: %s", e)
        return {"status": "error", "message": str(e), "knowledge_bases": []}
    knowledge_bases = [
        {
            "id": r.get("id"),
            "name": r.get("name") or r.get("id"),
            "topic": r.get("topic", ""),
            "description": r.get("description", ""),
            "article_count": r.get("article_count", 0),
            "status": r.get("status", "unknown"),
        }
        for r in rows
    ]
    return {"status": "ok", "knowledge_bases": knowledge_bases}


@router.get("/sdd/specs")
async def list_sdd_specs():
    """List real spec-driven specs from the ``.specify/specs`` store.

    Uses :class:`~agent_utilities.sdd.SDDManager`, the same SpecKit source the
    orchestration engine consumes. If the spec store is empty or unreachable,
    returns an empty list honestly.
    """
    try:
        import os

        from ...sdd import SDDManager

        workspace = setting("WORKSPACE_PATH") or os.getcwd()
        manager = SDDManager(workspace_path=workspace)
        raw_specs = manager.list_specs()
    except Exception as e:  # noqa: BLE001
        logger.warning("SDD spec listing failed: %s", e)
        return {"status": "error", "message": str(e), "specs": []}

    specs = [
        {
            "id": s.get("id"),
            "title": s.get("title", s.get("id")),
            "status": s.get("status", "unknown"),
        }
        for s in raw_specs
    ]
    return {"status": "ok", "specs": specs}


@router.get("/resources")
async def list_resources():
    """List live discovered specialist agents from the registry.

    Mirrors ``/agents`` (``discover_all_specialists``), which reads the real
    specialist registry from the Knowledge Graph. If none are registered,
    returns an empty list honestly.
    """
    try:
        from ...agent.discovery import discover_all_specialists

        specialists = discover_all_specialists()
    except Exception as e:  # noqa: BLE001
        logger.warning("Resource (specialist) discovery failed: %s", e)
        return {"status": "error", "message": str(e), "resources": []}

    resources = [
        {
            "id": s.name,
            "type": s.source or "specialist",
            "description": s.description,
            "mcp_server": s.mcp_server,
        }
        for s in specialists
    ]
    return {"status": "ok", "resources": resources}


@router.get("/maintenance/status")
async def get_maintenance_status():
    """Report real maintenance-scheduler state from the active engine.

    Surfaces the consolidated daemon's maintenance thread liveness and the
    registered maintenance jobs. If the scheduler is not running, says so
    honestly instead of asserting ``maintenance_required: false``.
    """
    engine = _active_engine()
    if not engine:
        return {
            "status": "unavailable",
            "message": "Maintenance scheduler is not running in this process.",
            "maintenance_running": False,
        }
    try:
        daemon = engine.unified_daemon_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("Maintenance status query failed: %s", e)
        return {"status": "error", "message": str(e)}
    threads = daemon.get("threads", {})
    maintenance_running = bool(threads.get("maintenance"))
    jobs = daemon.get("maintenance_jobs", [])
    return {
        "status": "ok",
        "maintenance_running": maintenance_running,
        "maintenance_jobs": jobs,
        "role": daemon.get("role"),
        "effective_role": daemon.get("effective_role"),
    }


@router.get("/pipeline/status")
async def get_pipeline_status():
    """Report the real ingestion pipeline / daemon state from the active engine.

    Reads the consolidated daemon status (submission + graph-writer threads and
    queue depth) from the active engine. If no engine is initialized, returns
    the true inactive state honestly.
    """
    engine = _active_engine()
    if not engine:
        return {
            "status": "unavailable",
            "message": "Ingestion pipeline is not initialized in this process.",
            "pipeline_active": False,
        }
    try:
        daemon = engine.unified_daemon_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("Pipeline status query failed: %s", e)
        return {"status": "error", "message": str(e)}
    threads = daemon.get("threads", {})
    pipeline_active = bool(
        threads.get("submission")
        or threads.get("graph_writer")
        or daemon.get("running")
    )
    result = {
        "status": "ok",
        "pipeline_active": pipeline_active,
        "threads": threads,
    }
    if "queue_depth" in daemon:
        result["queue_depth"] = daemon["queue_depth"]
    if "queue_backend" in daemon:
        result["queue_backend"] = daemon["queue_backend"]
    return result


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
