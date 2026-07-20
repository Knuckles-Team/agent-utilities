"""CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Manifest Generators.

Strangled by graph/planning/ (Plan 03 Step 4): re-exported via
``agent_utilities.graph.planning.manifest``. This module remains the live
implementation.

Functions that generate ``ExecutionManifest`` objects from various sources,
providing the single conversion layer between legacy/external formats
and the ``ParallelEngine`` input contract.

Generators:
    - ``manifest_from_planner``: From HTN planner ``GraphPlan``
    - ``manifest_from_teamconfig``: From KG ``TeamComposition``
    - ``manifest_from_workflow``: From Skill workflow ``GraphPlan``
    - ``manifest_from_heavy_thinking``: For parallel reasoning + deliberation
    - ``manifest_from_preset``: From KG-stored SwarmTemplate
    - ``manifest_from_department``: From OWL-materialized company department
    - ``manifest_for_enterprise``: Full enterprise manifest (all departments)

See docs/pillars/1_graph_orchestration/ORCH-1.8-Parallel_Engine.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..models.execution_manifest import (
    AgentSpec,
    ExecutionManifest,
    SynthesisSpec,
)

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..models import GraphPlan
    from ..models.knowledge_graph import TeamComposition

logger = logging.getLogger(__name__)


# ── From HTN Planner ────────────────────────────────────────────────


def manifest_from_planner(
    plan: GraphPlan,
    query: str,
    engine: IntelligenceGraphEngine | None = None,
) -> ExecutionManifest:
    """Generate manifest from the HTN planner output.

    CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

    Converts ``GraphPlan`` steps into ``AgentSpec`` entries, preserving
    dependencies and parallel batching from the planner.

    Args:
        plan: The HTN planner ``GraphPlan``.
        query: Original user query.
        engine: Optional KG engine for agent enrichment.

    Returns:
        An ``ExecutionManifest`` ready for the ``ParallelEngine``.
    """
    agents: list[AgentSpec] = []

    for step in plan.steps:
        agents.append(
            AgentSpec(
                agent_id=step.id,
                role=step.id,
                task_template=step.refined_subtask or query,
                depends_on=step.depends_on,
                timeout=step.timeout if step.timeout != 3600.0 else None,
            )
        )

    return ExecutionManifest(
        name=f"Planner: {query[:50]}",
        agents=agents,
        query=query,
        source="planner",
        metadata=plan.metadata,
    )


# ── From TeamConfig ─────────────────────────────────────────────────


def manifest_from_teamconfig(
    team: TeamComposition,
    query: str,
) -> ExecutionManifest:
    """Generate manifest from a KG ``TeamComposition``.

    CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

    Converts the team's agent roster, execution mode, and parallel
    groups into an ``ExecutionManifest``.

    Args:
        team: The ``TeamComposition`` from the KG.
        query: Original user query.

    Returns:
        An ``ExecutionManifest`` with proper DAG edges from parallel groups.
    """
    agents: list[AgentSpec] = []

    for agent_config in team.adaptive_agent_router:
        agents.append(
            AgentSpec(
                agent_id=str(agent_config.get("agent_id", "")),
                role=str(agent_config.get("role", "")),
                tools=agent_config.get("tools", []),
                model_id=str(agent_config.get("model_id", "")),
                system_prompt=str(agent_config.get("system_prompt", "")),
                memory_channels=agent_config.get("memory_channels", ["episodic"]),
            )
        )

    # Map execution mode
    mode = team.execution_mode
    if mode not in ("sequential", "parallel", "mixed", "wave"):
        mode = "auto"

    return ExecutionManifest(
        name=f"Team: {team.team_id}",
        agents=agents,
        query=query,
        execution_mode=mode,  # type: ignore[arg-type]
        source="teamconfig",
        metadata={
            "team_id": team.team_id,
            "topology_template_id": team.topology_template_id,
            "confidence": team.confidence,
        },
        kg_template_id=team.topology_template_id,
    )


# ── From Workflow ───────────────────────────────────────────────────


def manifest_from_workflow(
    workflow: GraphPlan,
    query: str,
) -> ExecutionManifest:
    """Generate manifest from a Skill workflow ``GraphPlan``.

    CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

    Converts workflow steps into agents, preserving wave structure
    from the original workflow definition.

    Args:
        workflow: The workflow ``GraphPlan``.
        query: Original user query.

    Returns:
        An ``ExecutionManifest`` with wave-based execution.
    """
    agents: list[AgentSpec] = []

    for step in workflow.steps:
        agents.append(
            AgentSpec(
                agent_id=step.id,
                role=step.id,
                task_template=step.refined_subtask or query,
                depends_on=step.depends_on,
            )
        )

    return ExecutionManifest(
        name=f"Workflow: {query[:50]}",
        agents=agents,
        query=query,
        execution_mode="wave",
        source="workflow",
        metadata=workflow.metadata,
    )


# ── From KG Preset ──────────────────────────────────────────────────


def manifest_from_preset(
    preset_name: str,
    partitions: list[str],
    query: str,
) -> ExecutionManifest:
    """Generate manifest from a KG-stored SwarmTemplate preset.

    CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine

    Looks up a named preset in the KG and materializes it as a manifest.
    If the preset is not found, falls back to a single-agent manifest.

    Args:
        preset_name: Name of the preset (e.g., "fan_out_research").
        partitions: Data partitions for fan-out execution.
        query: Original user query.

    Returns:
        An ``ExecutionManifest`` from the preset definition.
    """
    # Preset registry — common swarm topologies
    presets: dict[str, dict[str, Any]] = {
        "fan_out_research": {
            "role": "researcher",
            "system_prompt": "You are a research specialist. Investigate the assigned topic thoroughly.",
        },
        "fan_out_audit": {
            "role": "auditor",
            "system_prompt": "You are an audit specialist. Review the assigned item for compliance and quality.",
        },
        "fan_out_code_review": {
            "role": "code_reviewer",
            "system_prompt": "You are a code review specialist. Analyze the assigned code for quality, security, and best practices.",
        },
    }

    preset = presets.get(preset_name, {})
    role = preset.get("role", preset_name)
    system_prompt = preset.get("system_prompt", f"You are a {role} specialist.")

    agent = AgentSpec(
        agent_id=preset_name,
        role=role,
        system_prompt=system_prompt,
        task_template=f"{query}\n\nFocus on: {{{{partition}}}}",
        partitions=partitions,
    )

    return ExecutionManifest(
        name=f"Preset({preset_name}): {query[:40]}",
        agents=[agent],
        query=query,
        source="preset",
        metadata={"preset_name": preset_name},
    )


# ── From Company Department ────────────────────────────────────────


def manifest_from_department(
    department: str,
    task: str,
    engine: IntelligenceGraphEngine,
) -> ExecutionManifest:
    """Generate manifest from an OWL-materialized company department.

    CONCEPT:AU-ORCH.execution.autonomous-department-orchestration — Autonomous Department Orchestration

    Queries the KG for all agents in a department, their tools,
    their reporting hierarchy, and generates a manifest with proper
    ``depends_on`` edges matching the org chart.

    Args:
        department: Department name (e.g., "Infrastructure", "Research").
        task: Task description.
        engine: The Intelligence Graph Engine.

    Returns:
        An ``ExecutionManifest`` with department agents and hierarchy edges.
    """
    agents: list[AgentSpec] = []

    # Query KG for department agents
    if engine.backend:
        try:
            results = engine.backend.execute(
                "MATCH (d:Department {name: $dept})-[:HAS_AGENT_ROLE]->(r:AgentRole) "
                "OPTIONAL MATCH (r)-[:USES_TOOL]->(t:MCPServer) "
                "OPTIONAL MATCH (r)-[:REPORTS_TO]->(mgr:AgentRole) "
                "RETURN r.id AS agent_id, r.role AS role, "
                "       collect(DISTINCT t.name) AS tools, "
                "       mgr.id AS reports_to",
                {"dept": department},
            )

            for row in results:
                deps = []
                if row.get("reports_to"):
                    deps = [str(row["reports_to"])]
                agents.append(
                    AgentSpec(
                        agent_id=str(row.get("agent_id", "")),
                        role=str(row.get("role", "")),
                        department=department,
                        tools=row.get("tools", []),
                        depends_on=deps,
                        task_template=task,
                    )
                )
        except Exception as e:
            logger.warning(
                "[CONCEPT:AU-ORCH.execution.autonomous-department-orchestration] Department KG query failed: %s",
                e,
            )

    if not agents:
        # Fallback: single executor agent
        agents = [
            AgentSpec(
                agent_id=f"{department.lower()}_executor",
                role="executor",
                department=department,
                task_template=task,
            )
        ]

    return ExecutionManifest(
        name=f"Department({department}): {task[:40]}",
        agents=agents,
        query=task,
        execution_mode="wave",
        source="department",
        metadata={"department": department},
    )


# ── Full Enterprise Manifest ───────────────────────────────────────


def manifest_for_enterprise(
    task: str,
    engine: IntelligenceGraphEngine,
) -> ExecutionManifest:
    """Generate a full enterprise manifest — ALL agents across ALL departments.

    CONCEPT:AU-ORCH.execution.autonomous-department-orchestration — Autonomous Department Orchestration

    This is the 300-agent case: every agent, every MCP server,
    organized hierarchically by department with inter-department
    dependencies.

    Args:
        task: Task description.
        engine: The Intelligence Graph Engine.

    Returns:
        An ``ExecutionManifest`` with all enterprise agents.
    """
    all_agents: list[AgentSpec] = []
    departments: list[str] = []

    # Query KG for all departments
    if engine.backend:
        try:
            dept_results = engine.backend.execute(
                "MATCH (d:Department) RETURN d.name AS name ORDER BY d.name",
                {},
            )
            departments = [
                str(r.get("name", "")) for r in dept_results if r.get("name")
            ]
        except Exception as e:
            logger.warning(
                "[CONCEPT:AU-ORCH.execution.autonomous-department-orchestration] Enterprise department query failed: %s",
                e,
            )

    if not departments:
        departments = [
            "Infrastructure",
            "IT",
            "Finance",
            "Research",
            "Admin",
            "Media",
            "Communications",
            "Wellness",
        ]

    # Define inter-department DAG dependencies
    dept_deps = {
        "Engineering": ["Product", "Research"],
        "QA": ["Engineering"],
        "Compliance": ["QA", "Finance", "Operations"],
        "Finance": ["Engineering", "Product"],
        "Operations": ["Engineering"],
    }

    dept_agents: dict[str, list[str]] = {}

    # Generate department manifests and merge
    for dept in departments:
        dept_manifest = manifest_from_department(dept, task, engine)
        dept_agents[dept] = [a.agent_id for a in dept_manifest.agents]
        all_agents.extend(dept_manifest.agents)

    # Apply inter-department DAG edges
    for agent in all_agents:
        if agent.department in dept_deps:
            for dep_dept in dept_deps[agent.department]:
                if dep_dept in dept_agents:
                    if not agent.depends_on:
                        agent.depends_on = []
                    agent.depends_on.extend(dept_agents[dep_dept])

    return ExecutionManifest(
        name=f"Enterprise: {task[:40]}",
        agents=all_agents,
        query=task,
        execution_mode="wave",
        synthesis=SynthesisSpec(strategy="hierarchical", ratio=10),
        source="enterprise",
        metadata={
            "departments": departments,
            "total_agents": len(all_agents),
        },
    )
