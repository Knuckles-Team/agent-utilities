from __future__ import annotations

"""CONCEPT:ECO-4.0 — A2A-Native PlannerAgent (Graph-Backed Skill).

Graph-backed A2A Skill that delegates directly to ``execute_graph()``.

This replaces the flat ``PydanticAI Agent → run_graph_flow tool`` pattern
with a direct ``Skill`` handler, eliminating one LLM call per A2A request.
External A2A agents still see the same JSON-RPC interface
(``message/send``, ``tasks/get``).  The only change is internal: we skip
the LLM wrapper hop.  The agent card stays identical.

See docs/pillars/architecture_c4.md §CONCEPT:ECO-4.0
"""


import logging
from typing import Any

logger = logging.getLogger(__name__)


class PlannerGraphSkill:
    """A2A Skill backed by the full pydantic graph pipeline.

    CONCEPT:ECO-4.0 — A2A-Native PlannerAgent

    External A2A agents see the same JSON-RPC interface.  Internally,
    queries go directly through router→dispatcher→adaptive_agent_router→verifier
    without an intermediary LLM deciding to call ``run_graph_flow``.

    Args:
        graph: The pydantic graph instance.
        graph_config: Configuration dict for graph execution.
        mcp_toolsets: Optional list of MCP toolsets for specialist binding.
        skill_id: Skill identifier for the agent card.
        name: Human-readable name for the skill.
        description: Description for the agent card.
        tags: Tags for discoverability.
        input_modes: Supported input MIME types.
        output_modes: Supported output MIME types.
    """

    def __init__(
        self,
        graph: Any,
        graph_config: Any,
        mcp_toolsets: list[Any] | None = None,
        skill_id: str = "planner",
        name: str = "Planner",
        description: str = "Graph-backed planning agent",
        tags: list[str] | None = None,
        input_modes: list[str] | None = None,
        output_modes: list[str] | None = None,
    ) -> None:
        self.graph = graph
        self.graph_config = graph_config
        self.mcp_toolsets = mcp_toolsets or []
        self.id = skill_id
        self.name = name
        self.description = description
        self.tags = tags or ["agent", "planner"]
        self.input_modes = input_modes or ["text"]
        self.output_modes = output_modes or ["text"]

    async def run(self, messages: list[dict[str, Any]], context: Any = None) -> str:
        """Execute the graph pipeline directly for an A2A request.

        CONCEPT:ECO-4.0 — A2A-Native PlannerAgent

        Args:
            messages: A2A message list (JSON-RPC format).
            context: Optional A2A context metadata.

        Returns:
            The graph pipeline output as a string.
        """
        from agent_utilities.graph.protocol_agnostic_execution import execute_graph

        query = self._extract_query(messages)
        if not query:
            return "No user query found in the messages."

        logger.info(
            "[A2A:GRAPH_SKILL] Executing graph pipeline directly for query: '%s...'",
            query[:80],
        )

        try:
            result = await execute_graph(
                graph=self.graph,
                config=self.graph_config,
                query=query,
                mcp_toolsets=self.mcp_toolsets,
            )
            output = result.get("results", {}).get("output", "")
            if not output:
                output = str(result.get("results", {}))
            return output
        except Exception as e:
            logger.exception("[A2A:GRAPH_SKILL] Graph execution failed: %s", e)
            return f"Graph execution failed: {e}"

    @staticmethod
    def _extract_query(messages: list[dict[str, Any]]) -> str:
        """Extract the latest user query from A2A message format.

        Walks messages in reverse to find the most recent user text part.

        Args:
            messages: A2A message list.

        Returns:
            The extracted query string, or empty if not found.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                for part in msg.get("parts", []):
                    if part.get("kind") == "text":
                        return part["text"]
        # Fallback: try content field
        for msg in reversed(messages):
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""


class CoordinatedGraphSkill(PlannerGraphSkill):
    """A2A Skill with coordination protocol negotiation.

    CONCEPT:ECO-4.0 — Research: 2605.03310v1, social_theory

    Extends ``PlannerGraphSkill`` with an explicit coordination phase
    before graph execution. When an A2A request arrives:

    1. Identify participating agents from the request context.
    2. Select an appropriate coordination protocol based on agent count,
       task type, and historical success from the KG.
    3. Apply the protocol (setting up coordination metadata).
    4. Execute the graph pipeline with coordination context attached.
    5. Log the coordination trace to the KG.

    This ensures coordination quality is tracked even for A2A requests,
    not just orchestrator-initiated tasks.

    Args:
        graph: The pydantic graph instance.
        graph_config: Configuration dict for graph execution.
        engine: Optional KG engine for coordination lookup/traces.
        mcp_toolsets: Optional list of MCP toolsets.
        skill_id: Skill identifier for the agent card.
        name: Human-readable name.
        description: Description for the agent card.
        tags: Tags for discoverability.
        input_modes: Supported input MIME types.
        output_modes: Supported output MIME types.
    """

    def __init__(
        self,
        graph: Any,
        graph_config: Any,
        engine: Any = None,
        mcp_toolsets: list[Any] | None = None,
        skill_id: str = "coordinated_planner",
        name: str = "Coordinated Planner",
        description: str = "Graph-backed planning agent with coordination protocol negotiation",
        tags: list[str] | None = None,
        input_modes: list[str] | None = None,
        output_modes: list[str] | None = None,
    ) -> None:
        super().__init__(
            graph=graph,
            graph_config=graph_config,
            mcp_toolsets=mcp_toolsets,
            skill_id=skill_id,
            name=name,
            description=description,
            tags=tags or ["agent", "planner", "coordinated"],
            input_modes=input_modes,
            output_modes=output_modes,
        )
        self.engine = engine
        # Lazily initialize coordination layer
        self._coordination_layer: Any = None

    @property
    def coordination_layer(self) -> Any:
        """Lazily initialize the coordination layer."""
        if self._coordination_layer is None:
            from agent_utilities.graph.coordination import CoordinationLayer

            self._coordination_layer = CoordinationLayer(engine=self.engine)
        return self._coordination_layer

    async def run(self, messages: list[dict[str, Any]], context: Any = None) -> str:
        """Execute the graph pipeline with coordination negotiation.

        CONCEPT:ECO-4.0 — Research: 2605.03310v1

        Args:
            messages: A2A message list (JSON-RPC format).
            context: Optional A2A context metadata (may include agent info).

        Returns:
            The graph pipeline output as a string.
        """
        from agent_utilities.graph.protocol_agnostic_execution import execute_graph

        query = self._extract_query(messages)
        if not query:
            return "No user query found in the messages."

        # Extract participating agents from context
        agent_ids = self._extract_agent_ids(messages, context)
        task_type = self._infer_task_type(query)

        # CONCEPT:ORCH-1.3 — Coordination negotiation before execution
        protocol = self.coordination_layer.select_protocol(
            agent_count=len(agent_ids),
            task_type=task_type,
            execution_mode="sequential",
        )
        coord_result = self.coordination_layer.apply_protocol(
            protocol=protocol,
            agent_ids=agent_ids,
            task=query,
            task_type=task_type,
        )

        logger.info(
            "[A2A:COORDINATED_SKILL] Protocol '%s' selected for %d agents "
            "(quality=%.2f), executing graph pipeline...",
            protocol.name,
            len(agent_ids),
            coord_result.quality_score,
        )

        try:
            result = await execute_graph(
                graph=self.graph,
                config=self.graph_config,
                query=query,
                mcp_toolsets=self.mcp_toolsets,
            )

            # Log coordination trace to KG
            self.coordination_layer.log_coordination_trace(coord_result)

            output = result.get("results", {}).get("output", "")
            if not output:
                output = str(result.get("results", {}))
            return output
        except Exception as e:
            logger.exception("[A2A:COORDINATED_SKILL] Graph execution failed: %s", e)
            # Still log the trace (with failure metadata)
            coord_result.converged = False
            coord_result.metadata["error"] = str(e)
            self.coordination_layer.log_coordination_trace(coord_result)
            return f"Graph execution failed: {e}"

    @staticmethod
    def _extract_agent_ids(
        messages: list[dict[str, Any]],
        context: Any = None,
    ) -> list[str]:
        """Extract participating agent IDs from A2A messages and context.

        Args:
            messages: A2A message list.
            context: Optional context metadata.

        Returns:
            List of agent ID strings. Defaults to ["self"] if none found.
        """
        agents: list[str] = []

        # Check context for agent info
        if context and hasattr(context, "agents"):
            agents.extend(str(a) for a in context.agents)
        elif isinstance(context, dict):
            ctx_agents = context.get("agents") or context.get("agent_ids") or []
            agents.extend(str(a) for a in ctx_agents)

        # Check message metadata for sender info
        for msg in messages:
            sender = msg.get("sender", msg.get("agent_id", ""))
            if sender and sender not in agents:
                agents.append(str(sender))

        return agents or ["self"]

    @staticmethod
    def _infer_task_type(query: str) -> str:
        """Infer task type from query for protocol selection.

        Simple keyword-based classification. The coordination layer
        uses this for historical protocol lookup in the KG.

        Args:
            query: The user query.

        Returns:
            Task type string.
        """
        query_lower = query.lower()
        if any(w in query_lower for w in ("code", "implement", "fix", "bug", "test")):
            return "code"
        if any(w in query_lower for w in ("research", "paper", "analyze", "study")):
            return "research"
        if any(w in query_lower for w in ("plan", "design", "architect")):
            return "planning"
        if any(w in query_lower for w in ("deploy", "release", "ship")):
            return "operations"
        return "general"
