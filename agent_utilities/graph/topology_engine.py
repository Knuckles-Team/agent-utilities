#!/usr/bin/python
from __future__ import annotations

"""Dynamic Topology Engine (CONCEPT:AU-ORCH.execution.dynamic-topology-materialization).

Replaces static ``create_graph_agent()`` topology with KG-driven dynamic
graph materialization.  Instead of all execution paths existing simultaneously,
the engine selects and materializes only the relevant subgraph based on:

    - Task domain (general, finance, medical, legal, government)
    - Task complexity (1-5 scale)
    - KG-stored ``TopologyTemplateNode`` success rates
    - Available adaptive_agent_router and tools

The engine supports all pydantic-graph execution patterns:
    - **Sequential**: A → B → C (simple pipeline)
    - **Parallel**: [A, B] → C (fan-out/fan-in)
    - **Mixed**: A → [B, C] → D → [E, F] → G (arbitrary DAG)
    - **Fan-out**: A → [B₁, B₂, ..., Bₙ] (scatter)
    - **Fan-in**: [B₁, B₂, ..., Bₙ] → C (gather)

Each materialized topology creates isolated subgraph instances with:
    - Per-specialist system prompts
    - Per-specialist MCP tool assignments
    - Per-specialist model selection
    - Shared KG memory channels for P2P communication
"""


import logging
import time
from typing import TYPE_CHECKING, Any

from ..models.knowledge_graph import (
    TeamComposition,
)

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class TopologyEngine:
    """Materializes KG-stored topology templates into executable graphs.

    CONCEPT:AU-ORCH.execution.dynamic-topology-materialization — Dynamic Topology Materialization

    The engine is the bridge between the KG's declarative topology
    descriptions and the pydantic-graph's runtime execution model.

    Usage::

        engine = TopologyEngine(knowledge_engine)

        # Materialize a topology for a task
        materialized = engine.materialize(
            team_composition=composer.compose_team(query),
            session_id="sess:abc123",
        )

        # Get execution plan
        plan = materialized["execution_plan"]
        # => [{"step": 0, "role": "router", "mode": "sequential"}, ...]

    Args:
        engine: The IntelligenceGraphEngine for KG queries.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    def materialize(
        self,
        team_composition: TeamComposition,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Materialize a team composition into an executable topology.

        Converts the declarative ``TeamComposition`` into an ordered
        execution plan that the pydantic-graph runner can consume.

        Args:
            team_composition: The composed team to materialize.
            session_id: Session ID for provenance tracking.

        Returns:
            A dict with:
                - ``execution_plan``: Ordered list of execution steps
                - ``specialist_configs``: Per-specialist configuration
                - ``memory_channels``: Shared KG channels
                - ``topology_id``: Template ID for tracking
        """
        adaptive_agent_router = team_composition.adaptive_agent_router
        mode = team_composition.execution_mode
        parallel_groups = team_composition.parallel_groups

        # Build execution plan
        execution_plan = self._build_execution_plan(
            adaptive_agent_router, mode, parallel_groups
        )

        # Build per-specialist configs with full context
        specialist_configs = self._build_specialist_configs(adaptive_agent_router)

        result = {
            "execution_plan": execution_plan,
            "specialist_configs": specialist_configs,
            "memory_channels": team_composition.memory_channels,
            "topology_id": team_composition.topology_template_id,
            "team_id": team_composition.team_id,
            "session_id": session_id,
            "execution_mode": mode,
            "materialized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Track in KG
        if self.engine:
            self._record_materialization(result)

        logger.info(
            "[CONCEPT:AU-ORCH.execution.dynamic-topology-materialization] Materialized topology: %d steps, "
            "%d adaptive_agent_router, mode=%s",
            len(execution_plan),
            len(specialist_configs),
            mode,
        )

        return result

    def _build_execution_plan(
        self,
        adaptive_agent_router: list[dict[str, Any]],
        mode: str,
        parallel_groups: list[list[str]],
    ) -> list[dict[str, Any]]:
        """Build an ordered execution plan from adaptive_agent_router and mode.

        Returns a list of execution steps, where each step may contain:
        - A single specialist (sequential)
        - Multiple adaptive_agent_router (parallel group)
        """
        if mode == "sequential":
            return [
                {
                    "step": i,
                    "roles": [s["role"]],
                    "mode": "sequential",
                    "agent_ids": [s.get("agent_id", s["role"])],
                }
                for i, s in enumerate(adaptive_agent_router)
            ]

        if mode == "parallel":
            # All adaptive_agent_router execute in parallel, then join
            return [
                {
                    "step": 0,
                    "roles": [s["role"] for s in adaptive_agent_router],
                    "mode": "parallel",
                    "agent_ids": [
                        s.get("agent_id", s["role"]) for s in adaptive_agent_router
                    ],
                }
            ]

        if mode == "fan_out":
            # First specialist fans out to the rest
            if len(adaptive_agent_router) < 2:
                return self._build_execution_plan(
                    adaptive_agent_router, "sequential", []
                )

            steps = [
                {
                    "step": 0,
                    "roles": [adaptive_agent_router[0]["role"]],
                    "mode": "sequential",
                    "agent_ids": [
                        adaptive_agent_router[0].get(
                            "agent_id", adaptive_agent_router[0]["role"]
                        )
                    ],
                },
                {
                    "step": 1,
                    "roles": [s["role"] for s in adaptive_agent_router[1:]],
                    "mode": "parallel",
                    "agent_ids": [
                        s.get("agent_id", s["role"]) for s in adaptive_agent_router[1:]
                    ],
                },
            ]
            return steps

        if mode == "fan_in":
            # All but last execute in parallel, then last gathers
            if len(adaptive_agent_router) < 2:
                return self._build_execution_plan(
                    adaptive_agent_router, "sequential", []
                )

            steps = [
                {
                    "step": 0,
                    "roles": [s["role"] for s in adaptive_agent_router[:-1]],
                    "mode": "parallel",
                    "agent_ids": [
                        s.get("agent_id", s["role"]) for s in adaptive_agent_router[:-1]
                    ],
                },
                {
                    "step": 1,
                    "roles": [adaptive_agent_router[-1]["role"]],
                    "mode": "sequential",
                    "agent_ids": [
                        adaptive_agent_router[-1].get(
                            "agent_id", adaptive_agent_router[-1]["role"]
                        )
                    ],
                },
            ]
            return steps

        if mode == "mixed":
            return self._build_mixed_plan(adaptive_agent_router, parallel_groups)

        # Fallback: sequential
        return self._build_execution_plan(adaptive_agent_router, "sequential", [])

    def _build_mixed_plan(
        self,
        adaptive_agent_router: list[dict[str, Any]],
        parallel_groups: list[list[str]],
    ) -> list[dict[str, Any]]:
        """Build a mixed sequential/parallel execution plan.

        Specialists in parallel_groups execute concurrently.
        Others execute sequentially in order.
        """
        # Create a set of roles that are in parallel groups
        parallel_roles: set[str] = set()
        for group in parallel_groups:
            parallel_roles.update(group)

        steps: list[dict[str, Any]] = []
        step_idx = 0
        role_to_spec = {s["role"]: s for s in adaptive_agent_router}

        # Track which roles have been scheduled
        scheduled: set[str] = set()

        for specialist in adaptive_agent_router:
            role = specialist["role"]
            if role in scheduled:
                continue

            if role in parallel_roles:
                # Find which parallel group this role belongs to
                for group in parallel_groups:
                    if role in group and not all(r in scheduled for r in group):
                        group_specs = [
                            role_to_spec[r]
                            for r in group
                            if r in role_to_spec and r not in scheduled
                        ]
                        if group_specs:
                            steps.append(
                                {
                                    "step": step_idx,
                                    "roles": [s["role"] for s in group_specs],
                                    "mode": "parallel",
                                    "agent_ids": [
                                        s.get("agent_id", s["role"])
                                        for s in group_specs
                                    ],
                                }
                            )
                            step_idx += 1
                            scheduled.update(s["role"] for s in group_specs)
                        break
            else:
                # Sequential step
                steps.append(
                    {
                        "step": step_idx,
                        "roles": [role],
                        "mode": "sequential",
                        "agent_ids": [specialist.get("agent_id", role)],
                    }
                )
                step_idx += 1
                scheduled.add(role)

        return steps

    def _build_specialist_configs(
        self, adaptive_agent_router: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Build per-specialist configuration from the team composition."""
        configs: dict[str, dict[str, Any]] = {}

        for spec in adaptive_agent_router:
            role = spec["role"]
            configs[role] = {
                "agent_id": spec.get("agent_id", role),
                "model_id": spec.get("model_id", ""),
                "tools": spec.get("tools", []),
                "system_prompt": spec.get("system_prompt", ""),
                "memory_channels": spec.get("memory_channels", ["episodic"]),
                "role": role,
            }

        return configs

    def _record_materialization(self, result: dict[str, Any]) -> None:
        """Record the materialization event in the KG for provenance."""
        if not self.engine:
            return

        try:
            node_id = f"mat:{result.get('team_id', '')}"
            self.engine.add_node(
                node_id,
                "topology_materialization",
                {
                    "name": f"Materialization: {result.get('execution_mode', '')}",
                    "topology_id": result.get("topology_id", ""),
                    "session_id": result.get("session_id", ""),
                    "specialist_count": len(result.get("specialist_configs", {})),
                    "execution_mode": result.get("execution_mode", ""),
                    "materialized_at": result.get("materialized_at", ""),
                },
            )
        except Exception as e:
            logger.debug("Failed to record materialization: %s", e)

    def record_outcome(
        self,
        topology_id: str,
        success: bool,
        quality_score: float = 0.5,
    ) -> None:
        """Record execution outcome to update topology template success rates.

        This is the evolutionary feedback loop — successful topologies get
        higher success_rate and are preferred in future selections.

        Args:
            topology_id: The TopologyTemplate ID.
            success: Whether execution succeeded.
            quality_score: Quality of the result (0-1).
        """
        if not self.engine or not topology_id:
            return

        if self.engine.backend:
            try:
                # Update rolling success rate with exponential moving average
                alpha = 0.15
                score = quality_score if success else 0.0

                self.engine.backend.execute(
                    "MATCH (t:TopologyTemplate) WHERE t.id = $tid "
                    "SET t.success_rate = (1 - $alpha) * t.success_rate + $alpha * $score, "
                    "t.usage_count = t.usage_count + 1",
                    {"tid": topology_id, "alpha": alpha, "score": score},
                )
                logger.info(
                    "[CONCEPT:AU-ORCH.execution.dynamic-topology-materialization] Updated topology '%s': success=%s, quality=%.2f",
                    topology_id,
                    success,
                    quality_score,
                )
            except Exception as e:
                logger.debug("Failed to record topology outcome: %s", e)

    def get_topology_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all topology templates.

        Returns:
            List of topology stats (id, name, success_rate, usage_count).
        """
        stats: list[dict[str, Any]] = []

        if self.engine and self.engine.backend:
            try:
                results = self.engine.backend.execute(
                    "MATCH (t:TopologyTemplate) "
                    "RETURN t.id AS id, t.name AS name, "
                    "t.success_rate AS success_rate, "
                    "t.usage_count AS usage_count, "
                    "t.execution_mode AS mode "
                    "ORDER BY t.success_rate DESC",
                    {},
                )
                for r in results:
                    stats.append(
                        {
                            "id": r.get("id", ""),
                            "name": r.get("name", ""),
                            "success_rate": r.get("success_rate", 0),
                            "usage_count": r.get("usage_count", 0),
                            "mode": r.get("mode", ""),
                        }
                    )
            except Exception:
                pass  # nosec

        return stats
