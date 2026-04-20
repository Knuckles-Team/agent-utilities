import logging
from typing import Any, Dict
from ..types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)
from ....models.knowledge_graph import (
    AgentNode,
    ToolNode,
    RegistryEdgeType,
)

logger = logging.getLogger(__name__)


async def execute_registry(
    ctx: PipelineContext, deps: Dict[str, PhaseResult]
) -> Dict[str, Any]:
    """Phase 4: Load agents and tools from the graph backend into memory."""
    from ....graph.config_helpers import get_discovery_registry

    # Load from the Knowledge Graph backend
    registry = get_discovery_registry()
    graph = ctx.nx_graph

    # Add Agent Nodes
    for agent in registry.agents:
        node = AgentNode(
            id=agent.name,
            name=agent.name,
            description=agent.description,
            agent_type=agent.agent_type,
            system_prompt=agent.system_prompt,
            endpoint_url=agent.endpoint_url,
            tool_count=agent.tool_count,
        )
        graph.add_node(node.id, **node.model_dump())

    # Add Tool Nodes and Relationships
    for tool in registry.tools:
        node = ToolNode(
            id=f"tool:{tool.name}",
            name=tool.name,
            description=tool.description,
            mcp_server=tool.mcp_server,
            relevance_score=tool.relevance_score,
            requires_approval=tool.requires_approval,
        )
        graph.add_node(node.id, **node.model_dump())

        # Link Tool to its source Server/Agent
        if tool.mcp_server:
            graph.add_edge(
                tool.mcp_server,
                node.id,
                type=RegistryEdgeType.PROVIDES,
                weight=1.0,
            )

    return {"agents": len(registry.agents), "tools": len(registry.tools)}


registry_phase = PipelinePhase(name="registry", deps=[], execute_fn=execute_registry)
