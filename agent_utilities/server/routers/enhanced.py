import logging
import math
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from agent_utilities.core.config import setting
from agent_utilities.security.error_surface import public_error_payload

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MAX_EXTRACT_TEXT_BYTES = 4 * 1024 * 1024


def _enhanced_capabilities(request: Request) -> set[str] | None:
    claims = getattr(request.state, "user_claims", None)
    if not claims or claims.get("auth_type") == "api_key":
        return None
    try:
        from agent_utilities.core.config import config
        from agent_utilities.security.identity import (
            base_capabilities,
            normalize_identity,
        )

        return set(
            base_capabilities(
                normalize_identity(claims), config.identity_group_capability_map
            )
        )
    except Exception:
        raise HTTPException(status_code=403, detail="enhanced API capability required") from None


async def _require_enhanced_read(request: Request) -> None:
    capabilities = _enhanced_capabilities(request)
    if capabilities is not None and not capabilities.intersection(
        {"enhanced:read", "enhanced:write", "kg:read", "kg:write", "kg:admin", "admin"}
    ):
        raise HTTPException(status_code=403, detail="enhanced API capability required")


def _require_enhanced_write(request: Request) -> None:
    capabilities = _enhanced_capabilities(request)
    if capabilities is not None and not capabilities.intersection(
        {"enhanced:write", "kg:write", "kg:admin", "admin"}
    ):
        raise HTTPException(status_code=403, detail="enhanced API write capability required")


def _request_owner(request: Request) -> str:
    claims: dict[str, Any] = getattr(request.state, "user_claims", None) or {}
    identity = "\x00".join(
        str(claims.get(key) or "")[:1024]
        for key in ("tenant_id", "tenant", "sub", "client_id", "auth_type")
    ) or "local"
    from agent_utilities.security.persistence_privacy import persistence_reference

    return persistence_reference("extract_owner", identity, namespace="enhanced-api")


def _validated_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="extraction job not found")
    return job_id


router = APIRouter(
    prefix="/api/enhanced",
    tags=["Enhanced API"],
    dependencies=[Depends(_require_enhanced_read)],
)


def _active_engine():
    """Return the live IntelligenceGraphEngine if one is active, else None.

    Never constructs a fresh engine — these read-only status surfaces must
    report on the running process's engine (or honestly say it is cold),
    not spin up a side instance that would report misleading state.
    """
    try:
        from ...knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "No active IntelligenceGraphEngine (exception_type=%s)",
            type(exc).__name__,
        )
        return None


@router.get("/info")
async def get_enhanced_info():
    return {"status": "ok", "message": "Enhanced API is active"}


@router.get("/graph/stats")
async def get_graph_stats():
    """Live node/edge counts from the active Knowledge Graph backend.

    Counts are queried through the guarded engine facade. If no engine is active, returns an
    honest ``unavailable`` status rather than fabricated counts.
    """
    engine = _active_engine()
    if not engine or not getattr(engine, "backend", None):
        return {
            "status": "unavailable",
            "message": "Knowledge Graph backend is not active in this process.",
        }
    try:
        node_rows = engine.query_cypher("MATCH (n) RETURN count(n) AS c") or []
        edge_rows = engine.query_cypher("MATCH ()-[r]->() RETURN count(r) AS c") or []
        nodes = int(node_rows[0]["c"]) if node_rows else 0
        edges = int(edge_rows[0]["c"]) if edge_rows else 0
    except Exception as exc:  # noqa: BLE001
        return public_error_payload(exc, logger=logger)
    backend_name = type(engine.backend).__name__
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
        rows = engine.query_cypher(
            "MATCH (kb:KnowledgeBase) "
            "RETURN kb.id AS id, kb.name AS name, kb.topic AS topic, "
            "kb.description AS description, kb.article_count AS article_count, "
            "kb.status AS status"
        ) or []
    except Exception as exc:  # noqa: BLE001
        return {
            **public_error_payload(exc, logger=logger),
            "knowledge_bases": [],
        }
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
    except Exception as exc:  # noqa: BLE001
        return {**public_error_payload(exc, logger=logger), "specs": []}

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
    except Exception as exc:  # noqa: BLE001
        return {**public_error_payload(exc, logger=logger), "resources": []}

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
    except Exception as exc:  # noqa: BLE001
        return public_error_payload(exc, logger=logger)
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
    except Exception as exc:  # noqa: BLE001
        return public_error_payload(exc, logger=logger)
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
    except Exception as exc:
        return {**public_error_payload(exc, logger=logger), "agents": []}


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
# Document → knowledge-graph fact extraction (CONCEPT:AU-ECO.connector.git-task-resolver)
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
    _require_enhanced_write(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="invalid JSON request") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="request body must be an object")
    text = body.get("text", "")
    url = body.get("url", "")
    if not isinstance(text, str) or not isinstance(url, str):
        raise HTTPException(status_code=422, detail="text and url must be strings")
    if len(text.encode("utf-8")) > _MAX_EXTRACT_TEXT_BYTES or len(url) > 8192:
        raise HTTPException(status_code=413, detail="extraction input too large")
    if url and not text:
        import asyncio

        text = await asyncio.to_thread(_read_url, url)
    if not text.strip():
        raise HTTPException(status_code=422, detail="non-empty text or url required")
    try:
        rounds = int(body.get("rounds", 1))
        threshold = float(body.get("dedup_threshold", 0.90))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="invalid extraction options") from None
    if not math.isfinite(threshold):
        raise HTTPException(status_code=422, detail="invalid extraction options")
    try:
        job_id = await mgr.submit(
            text=text,
            rounds=max(1, min(10, rounds)),
            dedup=bool(body.get("dedup", True)),
            dedup_field=str(body.get("dedup_field", "triple"))[:32],
            dedup_threshold=max(0.0, min(1.0, threshold)),
            owner_ref=_request_owner(request),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=429 if isinstance(exc, RuntimeError) else 422,
            detail="extraction request rejected",
        ) from None
    return {"status": "submitted", "job_id": job_id}


@router.get("/extract/stream/{job_id}")
async def extract_stream(job_id: str, request: Request):
    """Server-Sent-Events stream of a job's extraction events (live + replay).

    Emits ``round_start | fact | metrics | round_end | file_start | file_end |
    done | job_done`` — the taxonomy all three frontends render."""
    import json as _json

    from fastapi.responses import StreamingResponse

    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable", "message": "Knowledge Graph engine is cold."}

    job_id = _validated_job_id(job_id)
    owner_ref = _request_owner(request)
    if mgr.status(job_id, owner_ref=owner_ref) is None:
        raise HTTPException(status_code=404, detail="extraction job not found")

    async def _gen():
        async for event in mgr.stream(job_id, owner_ref=owner_ref):
            yield f"data: {_json.dumps(event)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/extract/jobs")
async def extract_jobs(request: Request):
    """List all extraction jobs (queued/running/paused/held/done) for the queue panel."""
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable", "jobs": []}
    return {"status": "ok", "jobs": mgr.jobs(owner_ref=_request_owner(request))}


@router.get("/extract/status/{job_id}")
async def extract_status(job_id: str, request: Request):
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable"}
    status = mgr.status(
        _validated_job_id(job_id), owner_ref=_request_owner(request)
    )
    if status is None:
        raise HTTPException(status_code=404, detail="extraction job not found")
    return status


@router.get("/extract/jsonl/{job_id}")
async def extract_jsonl(job_id: str, request: Request):
    """Download a job's facts as newline-delimited JSON (upstream parity)."""
    from fastapi.responses import PlainTextResponse

    mgr = _extraction_manager()
    if mgr is None:
        return PlainTextResponse("", media_type="application/x-ndjson")
    try:
        payload = mgr.jsonl(
            _validated_job_id(job_id), owner_ref=_request_owner(request)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="extraction job not found") from None
    return PlainTextResponse(
        payload + "\n",
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/extract/pause/{job_id}")
async def extract_pause(job_id: str, request: Request):
    _require_enhanced_write(request)
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable"}
    try:
        await mgr.pause(
            _validated_job_id(job_id), owner_ref=_request_owner(request)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="extraction job not found") from None
    return {"status": "paused", "job_id": job_id}


@router.post("/extract/resume/{job_id}")
async def extract_resume(job_id: str, request: Request):
    _require_enhanced_write(request)
    mgr = _extraction_manager()
    if mgr is None:
        return {"status": "unavailable"}
    try:
        await mgr.resume(
            _validated_job_id(job_id), owner_ref=_request_owner(request)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="extraction job not found") from None
    return {"status": "resumed", "job_id": job_id}


def _read_url(url: str) -> str:
    """Read a URL to clean text via the readability ReaderConnector (KG-2.66)."""
    try:
        from ...protocols.source_connectors.base import LoadConnector
        from ...protocols.source_connectors.registry import (
            discover,
            get_connector_class,
        )

        discover()
        cls = get_connector_class("reader")
        if cls is None or not issubclass(cls, LoadConnector):
            return ""
        docs = list(cls(url=url).load())
        return docs[0].text if docs else ""
    except Exception as e:  # noqa: BLE001 — a bad URL becomes an empty doc, not a 500
        logger.warning("reader fetch failed (exception_type=%s)", type(e).__name__)
        return ""
