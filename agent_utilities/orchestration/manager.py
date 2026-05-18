"""
Orchestration Manager — CONCEPT:ORCH-1.0
Handles multi-agent workflow dispatch, lifecycle, and consensus
across the Knowledge Graph.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Manages multi-agent tasks, dispatching them to the KG engine,
    and tracking their lifecycle and consensus state.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    async def dispatch_task(
        self, task_description: str, dependencies: list[str] | None = None
    ) -> str:
        """
        Dispatch a new task to the orchestration queue.
        """
        job_id = f"orch-{uuid.uuid4().hex[:8]}"

        # Register the task in the KG
        self.engine.add_node(
            node_id=job_id,
            node_type="Task",
            properties={
                "description": task_description,
                "status": "pending",
                "dependencies": dependencies or [],
            },
        )

        logger.info(f"Dispatched orchestration task: {job_id}")
        return job_id

    def get_task_status(self, job_id: str) -> dict[str, Any]:
        """
        Retrieve the current status of an orchestrated task.
        """
        # Try backend first
        if self.engine.backend:
            results = self.engine.query_cypher(
                "MATCH (t:Task {id: $job_id}) RETURN t.status as status, t.description as desc",
                {"job_id": job_id},
            )
            if results:
                return {
                    "job_id": job_id,
                    "status": results[0].get("status"),
                    "description": results[0].get("desc"),
                }

        # Fallback to NX graph
        if job_id in self.engine.graph.nodes:
            data = self.engine.graph.nodes[job_id]
            return {
                "job_id": job_id,
                "status": data.get("status"),
                "description": data.get("description"),
            }

        return {"error": f"Task {job_id} not found"}

    def grant_approval(self, job_id: str, status: str) -> str:
        """
        Grant or deny approval for a pending task.
        """
        if self.engine.backend:
            self.engine.query_cypher(
                "MATCH (t:Task {id: $job_id}) SET t.approval_status = $status",
                {"job_id": job_id, "status": status},
            )
        elif job_id in self.engine.graph.nodes:
            self.engine.graph.nodes[job_id]["approval_status"] = status

        return f"Task {job_id} approval set to {status}"
