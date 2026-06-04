"""FastAPI router for the Centralized Knowledge Graph Backend.

CONCEPT:ECO-4.0 — Knowledge Graph API Gateway

Exposes the internal Knowledge Graph engine over a resilient HTTP/WebSocket gateway,
eliminating direct Kuzu file lock contention by funnelling all graph requests
(from agent-terminal-ui, geniusbot, subagents, and ingestion scripts) through
this persistent daemon.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

graph_router = APIRouter(tags=["graph_backend"])

INGESTION_JOBS: dict[str, dict[str, Any]] = {}


def _get_engine():
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        # Initialize if not active
        engine = IntelligenceGraphEngine()
    return engine


@graph_router.post("/query")
async def graph_query(request: Request) -> dict[str, Any]:
    """Execute a read-only Cypher query against the Knowledge Graph."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    cypher = body.get("cypher", "")
    params = body.get("params", {})
    scope = body.get("scope", "local")
    reference_id = body.get("reference_id", "")

    engine = _get_engine()

    if scope == "federated":
        if not reference_id:
            raise HTTPException(
                status_code=400, detail="reference_id required for federated queries"
            )
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.execute_federated_query, reference_id, cypher, params
                ),
                timeout=30.0,
            )
            return {"status": "success", "result": results}
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Query timed out") from None
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Local query - block writes
    cypher_upper = cypher.upper().strip()
    for kw in ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP"]:
        if kw in cypher_upper:
            raise HTTPException(
                status_code=403,
                detail=f"Write operations ({kw}) are not allowed via /query endpoint. Use /write instead.",
            )

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(engine.query_cypher, cypher, params), timeout=30.0
        )
        return {"status": "success", "result": results}
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out") from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@graph_router.post("/search")
async def graph_search(request: Request) -> dict[str, Any]:
    """Search the Knowledge Graph using various modes."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = body.get("query", "")
    mode = body.get("mode", "hybrid")
    top_k = body.get("top_k", 10)

    engine = _get_engine()
    try:
        # Simplified routing - actual implementation would call the specific method
        results = []
        if mode == "hybrid":
            results = await asyncio.wait_for(
                asyncio.to_thread(engine.hybrid_search, query, top_k), timeout=30.0
            )
        elif mode == "concept":
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.query_cypher,
                    f"MATCH (n) WHERE n.id CONTAINS '{query}' RETURN n LIMIT {top_k}",
                ),
                timeout=30.0,
            )
        else:
            # Fallback for other modes
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.query_cypher,
                    f"MATCH (n) WHERE n.id CONTAINS '{query}' OR n.name CONTAINS '{query}' RETURN n LIMIT {top_k}",
                ),
                timeout=30.0,
            )

        return {"status": "success", "result": results}
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Search timed out") from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@graph_router.post("/write")
async def graph_write(request: Request) -> dict[str, Any]:
    """Write operations for the Knowledge Graph (adding nodes, edges, etc)."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    action = body.get("action", "")
    engine = _get_engine()

    try:
        if action == "add_node":
            node_id = body.get("id", "")
            node_type = body.get("node_type", "Concept")
            properties = body.get("properties", {})
            if isinstance(properties, str):
                import json

                properties = json.loads(properties)
            await asyncio.wait_for(
                asyncio.to_thread(engine.add_node, node_id, node_type, properties),
                timeout=30.0,
            )
            return {"status": "success", "result": f"Node {node_id} added"}

        elif action == "add_edge":
            source_id = body.get("source_id", "")
            target_id = body.get("target_id", "")
            rel_type = body.get("rel_type", "RELATED_TO")
            properties = body.get("properties", {})
            if isinstance(properties, str):
                import json

                properties = json.loads(properties)
            await asyncio.wait_for(
                asyncio.to_thread(
                    engine.add_edge, source_id, target_id, rel_type, properties
                ),
                timeout=30.0,
            )
            return {
                "status": "success",
                "result": f"Edge {source_id} -> {target_id} added",
            }

        elif action == "bulk_ingest":
            nodes = body.get("nodes", [])
            if isinstance(nodes, str):
                import json

                nodes = json.loads(nodes)

            # Simple implementation of bulk ingest loop
            def _bulk_insert():
                for node in nodes:
                    node_id = node.get("id", str(uuid.uuid4()))
                    node_type = node.get("type", "Concept")
                    props = {k: v for k, v in node.items() if k not in ("id", "type")}
                    engine.add_node(node_id, node_type, props)

            await asyncio.wait_for(asyncio.to_thread(_bulk_insert), timeout=30.0)
            return {"status": "success", "result": f"Bulk ingested {len(nodes)} items"}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")

    except TimeoutError:
        logger.error("Write operation timed out")
        raise HTTPException(
            status_code=504, detail="Write operation timed out"
        ) from None
    except Exception as e:
        logger.error(f"Write operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@graph_router.post("/ingest")
async def graph_ingest(request: Request) -> dict[str, Any]:
    """Trigger ingestion of codebases or documents."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None

    target_path = body.get("target_path")
    if not target_path:
        raise HTTPException(status_code=400, detail="target_path required")

    engine = _get_engine()

    from pathlib import Path

    target = Path(target_path)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Target path not found: {target}")

    is_codebase = False
    if target.is_dir():
        for indicator in [".git", "pyproject.toml", "package.json", "setup.py"]:
            if (target / indicator).exists():
                is_codebase = True
                break

    try:
        if is_codebase:
            job_id = f"job-{uuid.uuid4()}"
            INGESTION_JOBS[job_id] = {"status": "running", "target": str(target)}

            async def _run_pipeline():
                try:
                    from agent_utilities.core.paths import kg_db_path
                    from agent_utilities.knowledge_graph.pipeline import (
                        IntelligencePipeline,
                    )
                    from agent_utilities.models.knowledge_graph import PipelineConfig

                    config = PipelineConfig(
                        workspace_path=str(target),
                        ladybug_path=str(kg_db_path()),
                    )
                    pipeline = IntelligencePipeline(config, backend=engine.backend)
                    metadata = await pipeline.run()
                    INGESTION_JOBS[job_id] = {
                        "status": "success",
                        "result": f"Codebase {target.name} ingested",
                        "nodes": metadata.node_count,
                        "edges": metadata.edge_count,
                    }
                except Exception as exc:
                    logger.error(f"Background ingestion failed: {exc}", exc_info=True)
                    INGESTION_JOBS[job_id] = {"status": "error", "detail": str(exc)}

            asyncio.create_task(_run_pipeline())
            return {"status": "accepted", "job_id": job_id, "target": str(target)}

        elif target.is_file():
            # Delegate to the previous document ingest logic but keep it inside the daemon
            # We'll just call a helper or implement the logic directly
            # For simplicity, we can just say "document ingestion not fully moved to gateway"
            # or do the llama-index parsing here.
            import hashlib
            import json
            from datetime import UTC, datetime

            from llama_index.core import SimpleDirectoryReader

            from agent_utilities.core.embedding_utilities import create_embedding_model

            embed_model = create_embedding_model()
            docs = SimpleDirectoryReader(input_files=[str(target)]).load_data()

            created = []
            ingestion_timestamp = datetime.now(UTC).isoformat()

            for idx, doc in enumerate(docs):
                chunk_text = doc.text
                if not chunk_text.strip():
                    continue
                file_path = doc.metadata.get("file_path", str(target))
                raw_id = f"{file_path}::{chunk_text}".encode()
                nid = f"doc-{hashlib.sha256(raw_id).hexdigest()[:8]}"
                existing = engine.query_cypher(
                    "MATCH (n:Article {id: $nid}) RETURN n.id as id", {"nid": nid}
                )
                if existing:
                    engine.backend.execute(
                        "MATCH (n:Article {id: $nid}) SET n.last_seen_timestamp = $ts",
                        {"nid": nid, "ts": ingestion_timestamp},
                    )
                    created.append(nid)
                    continue
                embedding = embed_model.get_text_embedding(chunk_text)
                props = {
                    "content": chunk_text,
                    "embedding": embedding,
                    "metadata": json.dumps(doc.metadata),
                    "last_seen_timestamp": ingestion_timestamp,
                    "target_path": str(target),
                    "chunk_index": idx,
                }
                engine.add_node(nid, "Article", properties=props)
                created.append(nid)

            engine.backend.execute(
                "MATCH (n:Article) WHERE n.target_path = $target AND n.last_seen_timestamp < $ts DETACH DELETE n",
                {"target": str(target), "ts": ingestion_timestamp},
            )
            return {
                "status": "success",
                "result": f"Document {target.name} ingested",
                "chunks": len(created),
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Target is neither a codebase nor a file: {target}",
            )

    except Exception as e:
        logger.error(f"Ingestion setup failed for {target}: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@graph_router.get("/jobs/{job_id}/status")
async def graph_ingest_job_status(job_id: str) -> dict[str, Any]:
    """Poll the status of an ingestion job."""
    if job_id not in INGESTION_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **INGESTION_JOBS[job_id]}
