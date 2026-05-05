"""CONCEPT:ECO-4.2 — A2A-Native PlannerAgent (Graph-Backed Skill).

Graph-backed A2A Skill that delegates directly to ``execute_graph()``.

This replaces the flat ``PydanticAI Agent → run_graph_flow tool`` pattern
with a direct ``Skill`` handler, eliminating one LLM call per A2A request.
External A2A agents still see the same JSON-RPC interface
(``message/send``, ``tasks/get``).  The only change is internal: we skip
the LLM wrapper hop.  The agent card stays identical.

See docs/emergent-architecture.md §AU-027.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PlannerGraphSkill:
    """A2A Skill backed by the full pydantic graph pipeline.

    CONCEPT:ECO-4.2 — A2A-Native PlannerAgent

    External A2A agents see the same JSON-RPC interface.  Internally,
    queries go directly through router→dispatcher→specialists→verifier
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

        CONCEPT:ECO-4.2 — A2A-Native PlannerAgent

        Args:
            messages: A2A message list (JSON-RPC format).
            context: Optional A2A context metadata.

        Returns:
            The graph pipeline output as a string.
        """
        from agent_utilities.graph.unified import execute_graph

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
