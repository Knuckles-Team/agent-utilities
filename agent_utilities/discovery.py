#!/usr/bin/python
"""Agent Discovery Module.

This module provides functionality for discovering local specialists, MCP tools,
and remote A2A peer agents. All discovery is performed via the Knowledge Graph,
which serves as the unified specialist registry for the graph orchestrator.
"""

from typing import Any

from .models import DiscoveredSpecialist


def discover_agents(
    include_packages: list[str] | None = None,
    exclude_packages: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Discover agents from the Knowledge Graph.

    Returns:
        A dictionary mapping domain tags to agent metadata.
    """
    from .graph.config_helpers import get_discovery_registry

    agent_descriptions: dict[str, dict[str, Any]] = {}

    # Read from the Knowledge Graph
    registry = get_discovery_registry()
    for agent in registry.agents:
        # Filtering
        if include_packages and agent.name not in include_packages:
            continue
        if exclude_packages and agent.name in exclude_packages:
            continue

        # Type-specific mapping
        if agent.agent_type == "prompt":
            agent_descriptions[agent.name] = {
                "description": agent.description,
                "name": agent.name,
                "type": "prompt",
                "skills": agent.capabilities,
            }
        elif agent.agent_type == "mcp":
            agent_descriptions[agent.name] = {
                "package": agent.name,
                "description": agent.description,
                "name": agent.name,
                "type": "local_mcp",
            }
        elif agent.agent_type == "a2a":
            agent_descriptions[agent.name.lower()] = {
                "url": agent.endpoint_url,
                "name": agent.name,
                "description": agent.description,
                "capabilities": ", ".join(agent.capabilities),
                "type": "remote_a2a",
            }

    return agent_descriptions


def discover_all_specialists() -> list[DiscoveredSpecialist]:
    """Discover all specialist agents from the Knowledge Graph.

    Returns:
        A list of DiscoveredSpecialist objects.
    """
    from .graph.config_helpers import get_discovery_registry

    specialists: list[DiscoveredSpecialist] = []
    registry = get_discovery_registry()

    for agent in registry.agents:
        specialists.append(
            DiscoveredSpecialist(
                tag=agent.name,
                name=agent.name,
                description=agent.description or "",
                source=agent.agent_type,
                mcp_server=agent.mcp_server or "",
                url=agent.endpoint_url or "",
                capabilities=agent.capabilities,
            )
        )

    return specialists
