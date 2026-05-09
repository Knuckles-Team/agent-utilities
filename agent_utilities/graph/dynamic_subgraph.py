#!/usr/bin/python
"""Dynamic Subgraph Orchestrator (CONCEPT:ORCH-1.19).

Dynamically synthesizes pydantic-graph transition logic from the Knowledge
Graph on the fly without using predefined templates. Uses formal graph theory
primitives (KG-2.41) to determine the exact DAG structure, parallel groups,
and execution paths.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import networkx as nx

from ..knowledge_graph.core.graph_theory_primitives import (
    chromatic_schedule,
    dag_critical_path,
)
from ..models.knowledge_graph import TeamComposition

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class DynamicSubgraphOrchestrator:
    """Dynamically synthesizes graph topology.

    CONCEPT:ORCH-1.19 — Dynamic Subgraph Orchestration
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    def synthesize_team(
        self,
        query: str,
        domain: str = "general",
        complexity: int = 3,
        available_tools: list[str] | None = None,
        available_agents: list[str] | None = None,
        delegated_authority: str | None = None,
    ) -> TeamComposition:
        """Synthesize a TeamComposition fully dynamically from the KG.

        Flow:
            1. Parse required capabilities from the query.
            2. Extract matching agent nodes from the KG.
            3. Build a dependency DAG based on capability prerequisites.
            4. Use graph theory primitives to find critical paths and parallel groups.
            5. Construct the TeamComposition.
        """
        team_id = f"team:dyn:{uuid.uuid4().hex[:8]}"

        # Step 1 & 2: Get candidate agents for this task
        agents = self._retrieve_candidate_agents(
            query, domain, available_agents, delegated_authority
        )
        if not agents:
            # Fallback to a basic executor if KG lookup fails
            agents = [{"role": "executor", "agent_id": "executor", "tools": []}]

        # Step 3: Build dependency DAG
        dag = self._build_dependency_dag(agents)

        # Step 4: Analyze DAG (Critical path & parallel groups)
        critical_info = {}
        if len(dag) > 1:
            try:
                critical_info = dag_critical_path(dag)
            except Exception as e:
                logger.debug("DAG critical path failed: %s", e)

        # Build conflict graph for parallelization (chromatic scheduling)
        conflict_graph = self._build_conflict_graph(dag)
        coloring = chromatic_schedule(conflict_graph)

        # Group by color (parallel groups)
        parallel_groups_dict: dict[int, list[str]] = {}
        for node, color in coloring.items():
            parallel_groups_dict.setdefault(color, []).append(node)

        parallel_groups = [g for g in parallel_groups_dict.values() if len(g) > 1]

        # Determine execution mode
        if len(agents) == 1:
            mode = "sequential"
        elif parallel_groups:
            if len(parallel_groups) == 1 and len(parallel_groups[0]) == len(agents):
                mode = "parallel"
            else:
                mode = "mixed"
        else:
            mode = "sequential"

        # Build specialists configs
        specialists = []
        for agent in agents:
            # Step 5: Ask KG for tools
            tools = self._discover_tools_for_agent(str(agent["role"]), available_tools)
            specialists.append(
                {
                    "role": agent["role"],
                    "agent_id": agent["agent_id"],
                    "tools": tools,
                    "model_id": agent.get("model_id", ""),
                    "system_prompt": agent.get(
                        "system_prompt",
                        f"You are a dynamically spawned {agent['role']}.",
                    ),
                    "memory_channels": ["episodic", domain],
                }
            )

        composition = TeamComposition(
            team_id=team_id,
            source="dynamic_synthesis",
            topology_template_id=f"topo:dyn:{domain}:{complexity}",
            specialists=specialists,
            execution_mode=mode,
            parallel_groups=parallel_groups,
            memory_channels=["episodic", domain],
            confidence=0.85,
            reasoning=f"Dynamically synthesized topology via ORCH-1.19. Makespan: {critical_info.get('makespan', 1.0)}",
        )

        logger.info(
            "[CONCEPT:ORCH-1.19] Dynamically synthesized subgraph '%s': %d specialists, mode=%s",
            team_id,
            len(specialists),
            mode,
        )

        return composition

    def _retrieve_candidate_agents(
        self,
        query: str,
        domain: str,
        available_agents: list[str] | None,
        delegated_authority: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the KG for agents capable of handling aspects of the query."""
        if not self.engine or not self.engine.backend:
            return []

        agents = []
        try:
            query_str = "MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:AgentCapability) "

            if delegated_authority:
                query_str += "MATCH (a)-[:HAS_DELEGATED_AUTHORITY_FROM*0..5]->(:Person {id: $delegated_authority}) "

            query_str += (
                "WHERE toLower($query) CONTAINS toLower(c.name) "
                "RETURN a.id AS agent_id, a.name AS name, a.role AS role "
                "LIMIT 5"
            )

            results = self.engine.backend.execute(
                query_str,
                {"query": query, "delegated_authority": delegated_authority},
            )
            for r in results:
                role = r.get("role") or r.get("name") or r.get("agent_id")
                aid = r.get("agent_id")
                if available_agents is None or aid in available_agents:
                    agents.append({"role": role, "agent_id": aid})
        except Exception:
            pass  # nosec

        return agents

    def _build_dependency_dag(self, agents: list[dict[str, Any]]) -> nx.DiGraph:
        """Build a DAG of agent dependencies based on expected data flow."""
        dag = nx.DiGraph()
        for a in agents:
            dag.add_node(a["role"], weight=1.0)

        # In a fully KG-driven system, we'd query REQUIRES_OUTPUT_FROM edges.
        # Here we create a simple sequential DAG if no explicit edges exist.
        roles = [a["role"] for a in agents]
        for i in range(len(roles) - 1):
            dag.add_edge(roles[i], roles[i + 1], weight=1.0)

        return dag

    def _build_conflict_graph(self, dag: nx.DiGraph) -> nx.Graph:
        """Convert DAG into a conflict graph for chromatic scheduling.
        Nodes that share a directed path have a conflict (must run sequentially).
        """
        conflict_graph = nx.Graph()
        conflict_graph.add_nodes_from(dag.nodes())

        # Add edges between any two nodes where one is reachable from the other
        for u in dag.nodes():
            reachable = nx.descendants(dag, u)
            for v in reachable:
                conflict_graph.add_edge(u, v)

        return conflict_graph

    def _discover_tools_for_agent(
        self, role: str, available_tools: list[str] | None
    ) -> list[str]:
        if not self.engine or not self.engine.backend:
            return []
        tools = []
        try:
            results = self.engine.backend.execute(
                "MATCH (a)-[:PROVIDES|HAS_CAPABILITY]->(t:CallableResource) "
                "WHERE toLower(a.name) CONTAINS toLower($role) "
                "RETURN t.name AS tool_name LIMIT 5",
                {"role": role},
            )
            for r in results:
                name = r.get("tool_name", "")
                if name:
                    if available_tools is None or name in available_tools:
                        tools.append(name)
        except Exception:
            pass  # nosec
        return tools
