"""Focused graph-os surface for durable orchestration jobs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


def register_job_tools(mcp: Any) -> None:
    """Register durable job dispatch and status operations."""

    @mcp.tool(
        name="graph_jobs",
        description=(
            "Submit and inspect durable orchestration WorkItems. Actions: 'dispatch' "
            "(enqueue a task and return its job handle), 'status' (read a WorkItem, "
            "RunTrace, or workflow-session trace by id)."
        ),
        tags=["graph-os", "jobs", "orchestration"],
    )
    async def graph_jobs(
        action: str = Field(default="status", description="dispatch | status"),
        task: str = Field(default="", description="Task to dispatch."),
        job_id: str = Field(default="", description="Job/run/session id for status."),
        agent_name: str = Field(
            default="", description="Optional agent hint for the queued turn."
        ),
        dependencies: str = Field(
            default="[]", description="JSON list of prerequisite WorkItem ids."
        ),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            from agent_utilities.orchestration.manager import Orchestrator

            orchestrator = Orchestrator(engine)
            if action == "dispatch":
                parsed = json.loads(dependencies) if dependencies else []
                if not isinstance(parsed, list):
                    raise ValueError("dependencies must decode to a list")
                dispatched_id = await orchestrator.dispatch_task(
                    task, [str(item) for item in parsed]
                )
                from agent_utilities.orchestration.agent_dispatch import (
                    KIND_ORCHESTRATOR_TASK,
                    AgentTurnEnvelope,
                    enqueue_agent_turn,
                )

                handle = enqueue_agent_turn(
                    AgentTurnEnvelope(
                        job_id=dispatched_id,
                        session_id=dispatched_id,
                        kind=KIND_ORCHESTRATOR_TASK,
                        payload_ref=dispatched_id,
                        agent_name=agent_name,
                    )
                )
                handle["status_url"] = "/api/graph/jobs"
                handle["status_request"] = {
                    "action": "status",
                    "job_id": dispatched_id,
                }
                return json.dumps(handle, default=str)
            if action == "status":
                if not job_id:
                    return "Error: job_id required"
                if job_id.startswith(("run:", "trace:")):
                    result = orchestrator.get_run_trace(job_id)
                elif job_id.startswith(("wf-", "session:")):
                    result = orchestrator.get_session_runs(job_id)
                else:
                    result = orchestrator.get_task_status(job_id)
                return json.dumps(result, default=str)
            return f"Error: Unknown graph_jobs action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_jobs"] = graph_jobs
    kg_server.ACTION_TOOL_ROUTES["graph_jobs"] = "/graph/jobs"
