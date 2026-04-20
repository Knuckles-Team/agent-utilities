import logging
import time
from typing import Any, Dict
from ..types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)
from ...backends import create_backend

logger = logging.getLogger(__name__)


async def execute_memory(
    ctx: PipelineContext, deps: Dict[str, PhaseResult]
) -> Dict[str, Any]:
    """Phase 1: Hydrate existing state from the persistent backend."""
    if not ctx.config.persist_to_ladybug:
        return {"status": "skipped", "reason": "persistence disabled"}

    graph = ctx.nx_graph
    start_time = time.time()

    try:
        # Use the shared backend from context, or create one via factory
        db = ctx.backend
        if db is None:
            db_path = ctx.config.ladybug_path or "knowledge_graph.db"
            db = create_backend(db_path=db_path)
        if db is None:
            return {"status": "skipped", "reason": "graph backend not available"}

        # Retrieve all nodes
        results = db.execute("MATCH (n) RETURN n")
        for res in results:
            node_data = res["n"]
            if "id" in node_data:
                graph.add_node(node_data["id"], **node_data)

        # Retrieve all edges
        results = db.execute(
            "MATCH (a)-[r]->(b) RETURN a.id as u, b.id as v, type(r) as t"
        )
        for res in results:
            graph.add_edge(res["u"], res["v"], type=res["t"].lower())

        duration = (time.time() - start_time) * 1000
        return {
            "nodes_loaded": graph.number_of_nodes(),
            "edges_loaded": graph.number_of_edges(),
            "duration_ms": duration,
        }
    except Exception as e:
        logger.warning(f"Failed to hydrate from DB: {e}")
        return {"status": "failed", "error": str(e)}


memory_phase = PipelinePhase(name="memory", deps=[], execute_fn=execute_memory)
