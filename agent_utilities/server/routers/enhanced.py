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


# --------------------------------------------------------------------------- #
# Document → knowledge-graph fact extraction (CONCEPT:ECO-4.43)
#
# The shared SSE/jobs/JSONL contract every frontend (agent-webui, agent-terminal-ui,
# geniusbot) consumes for the interactive extraction experience: submit a document
# (text/URL/file), stream facts as they generate, manage the GPU-slot job queue,
# and export JSONL. Backed by KG-2.64 (fact extractor) + KG-2.65 (slot scheduler).
# --------------------------------------------------------------------------- #

_EXTRACTION_MANAGER = None


def _extraction_manager():
    """The process-wide extraction job manager, or ``None`` if the engine is cold."""
    global _EXTRACTION_MANAGER
    engine = _active_engine()
    if engine is None:
        return None
    if _EXTRACTION_MANAGER is None:
        from ...knowledge_graph.extraction.job_manager import ExtractionJobManager

        _EXTRACTION_MANAGER = ExtractionJobManager(engine)
    return _EXTRACTION_MANAGER


@router.post("/extract/submit")
async def extract_submit(request: Request):
    """Submit a fact-extraction job. Body: ``{text|url, rounds?, dedup?,
    dedup_field?, dedup_threshold?}``. Returns ``{job_id}``."""
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable", "message": "Knowledge Graph engine is cold."}
    body = await request.json()
    text = body.get("text", "")
    url = body.get("url", "")
    if url and not text:
        text = _read_url(url)
    if not text.strip():
        return {"status": "error", "message": "provide non-empty 'text' or a 'url'"}
    job_id = await mgr.submit(
        text=text,
        rounds=max(1, min(10, int(body.get("rounds", 1)))),
        dedup=bool(body.get("dedup", True)),
        dedup_field=body.get("dedup_field", "triple"),
        dedup_threshold=float(body.get("dedup_threshold", 0.90)),
    )
    return {"status": "submitted", "job_id": job_id}


@router.get("/extract/stream/{job_id}")
async def extract_stream(job_id: str):
    """Server-Sent-Events stream of a job's extraction events (live + replay).

    Emits ``round_start | fact | metrics | round_end | file_start | file_end |
    done | job_done`` — the taxonomy all three frontends render."""
    import json as _json

    from fastapi.responses import StreamingResponse

    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable", "message": "Knowledge Graph engine is cold."}

    async def _gen():
        async for event in mgr.stream(job_id):
            yield f"data: {_json.dumps(event)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.get("/extract/jobs")
async def extract_jobs():
    """List all extraction jobs (queued/running/paused/held/done) for the queue panel."""
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable", "jobs": []}
    return {"status": "ok", "jobs": mgr.jobs()}


@router.get("/extract/status/{job_id}")
async def extract_status(job_id: str):
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable"}
    return mgr.status(job_id) or {"status": "not_found"}


@router.get("/extract/jsonl/{job_id}")
async def extract_jsonl(job_id: str):
    """Download a job's facts as newline-delimited JSON (upstream parity)."""
    from fastapi.responses import PlainTextResponse

    mgr = _extraction_manager()
    if mgr is None:
        return PlainTextResponse("", media_type="application/x-ndjson")
    return PlainTextResponse(
        mgr.jsonl(job_id) + "\n", media_type="application/x-ndjson"
    )


@router.post("/extract/pause/{job_id}")
async def extract_pause(job_id: str):
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable"}
    await mgr.pause(job_id)
    return {"status": "paused", "job_id": job_id}


@router.post("/extract/resume/{job_id}")
async def extract_resume(job_id: str):
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable"}
    await mgr.resume(job_id)
    return {"status": "resumed", "job_id": job_id}


def _read_url(url: str) -> str:
    """Read a URL to clean text via the readability ReaderConnector (KG-2.66)."""
    try:
        from ...protocols.source_connectors.registry import (
            discover,
            get_connector_class,
        )

        discover()
        cls = get_connector_class("reader")
        if cls is None:
            return ""
        docs = list(cls(url=url).load())
        return docs[0].text if docs else ""
    except Exception as e:  # noqa: BLE001 — a bad URL becomes an empty doc, not a 500
        logger.warning("reader fetch failed for %s: %s", url, e)
        return ""
