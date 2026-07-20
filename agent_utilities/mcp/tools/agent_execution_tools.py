"""Focused graph-os agent, swarm, GUI, and runtime-org execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


async def _run_swarm(
    engine: Any,
    task: str,
    *,
    context: str,
    context_ref: str,
    max_fan_out: int,
) -> dict[str, Any]:
    from agent_utilities.core.config import DEFAULT_KG_MODEL_ID, DEFAULT_LLM_PROVIDER
    from agent_utilities.core.model_factory import create_model
    from agent_utilities.graph.parallel_engine import ParallelEngine
    from agent_utilities.graph.planning import Planner
    from agent_utilities.messaging.bus import swarm_topic
    from agent_utilities.models.execution_manifest import ExecutionManifest

    try:
        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
    except Exception:
        model = None
    plan = await Planner(model=model).decompose(task)
    manifest = ExecutionManifest.from_graph_plan(plan, name="swarm", query=task)

    swarm_context = context
    if not swarm_context and context_ref:
        try:
            rows = engine.query_cypher(
                "MATCH (c:ContextBlob) WHERE c.id = $id RETURN c.content AS content",
                {"id": context_ref},
            )
            if rows and rows[0].get("content"):
                swarm_context = str(rows[0]["content"])
        except Exception:
            swarm_context = ""

    topic = swarm_topic(hashlib.sha256(task.encode()).hexdigest()[:16])
    coordination = (
        "You are one agent in a swarm on the same overall task. Coordinate with peers over "
        f"the AgentBus topic '{topic}': announce ownership before work, share findings, and "
        "check peer messages before duplicating effort."
    )
    manifest.context = "\n\n".join(
        part for part in (manifest.context, swarm_context, coordination) if part
    )
    manifest.metadata["verify"] = True
    manifest.metadata["max_retries"] = 2
    manifest.max_concurrency = max(1, int(max_fan_out))
    for agent in manifest.agents:
        if not agent.success_criteria:
            agent.success_criteria = (
                "Output must substantively address: "
                f"{(agent.task_template or task)[:240]}"
            )

    result = await ParallelEngine(engine=engine).execute(manifest)
    return {
        "deliverable": result.synthesis_output,
        "agent_count": result.agent_count,
        "wave_count": result.wave_count,
        "critical_path_length": result.critical_path_length,
        "parallelism_ratio": result.parallelism_ratio,
        "verification": result.verification,
        "telemetry": result.telemetry,
        "execution_id": result.execution_id,
        "success": result.success,
        "mermaid": result.mermaid,
    }


def register_agent_execution_tools(mcp: Any) -> None:
    """Register focused multi-agent execution capabilities."""

    @mcp.tool(
        name="graph_agents",
        description=(
            "Execute graph-grounded agent collectives. Actions: 'swarm' performs governed "
            "goal decomposition and parallel-wave synthesis; 'computer_use' drives a GUI "
            "sandbox; 'synthesize_org' builds a staffed runtime org; 'run_org' executes its "
            "WorkItem DAG. Single-agent delegation remains graph_orchestrate."
        ),
        tags=["graph-os", "agents", "swarm", "computer-use", "org"],
    )
    async def graph_agents(
        action: str = Field(
            default="swarm",
            description="swarm | computer_use | synthesize_org | run_org",
        ),
        task: str = Field(default="", description="Goal or GUI task."),
        context: str = Field(default="", description="Curated swarm context."),
        context_ref: str = Field(
            default="", description="Persisted ContextBlob id for the swarm."
        ),
        max_fan_out: int = Field(default=5, ge=1),
        max_steps: int = Field(default=30, ge=1),
        host: str = Field(default="", description="Computer-use inventory host alias."),
        container_id: str = Field(
            default="", description="Existing GUI sandbox container id."
        ),
        options_json: str = Field(
            default="{}",
            description="JSON options; org actions accept {domains:[...]}.",
        ),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action == "swarm":
                if not task:
                    raise ValueError("task is required for swarm")
                return json.dumps(
                    await _run_swarm(
                        engine,
                        task,
                        context=context,
                        context_ref=context_ref,
                        max_fan_out=max_fan_out,
                    ),
                    default=str,
                )

            if action == "computer_use":
                from agent_utilities.orchestration.computer_use_agent import (
                    provision_and_run_computer_use,
                    run_computer_use_task,
                )

                if container_id:
                    return await run_computer_use_task(
                        task, container_id, host=host or None, engine=engine
                    )
                return await provision_and_run_computer_use(
                    task, host=host or None, engine=engine
                )

            options = json.loads(options_json) if options_json else {}
            if not isinstance(options, dict):
                raise ValueError("options_json must decode to an object")
            domains = options.get("domains")
            if action == "synthesize_org":
                from agent_utilities.orchestration.org_runtime import Recruiter

                if not task:
                    raise ValueError("task is required for synthesize_org")
                return json.dumps(
                    Recruiter(engine).synthesize_org(task, domains=domains).to_dict(),
                    default=str,
                )
            if action == "run_org":
                from agent_utilities.orchestration.org_runtime import OrgRuntime

                if not task:
                    raise ValueError("task is required for run_org")
                return json.dumps(
                    await OrgRuntime(engine, max_steps=max_steps).run(
                        task, domains=domains
                    ),
                    default=str,
                )
            return f"Error: Unknown graph_agents action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_agents"] = graph_agents
    kg_server.ACTION_TOOL_ROUTES["graph_agents"] = "/graph/agents"
