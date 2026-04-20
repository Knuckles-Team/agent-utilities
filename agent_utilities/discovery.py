#!/usr/bin/python
# coding: utf-8
"""Agent Discovery Module.

This module provides functionality for discovering local MCP agents and remote
A2A peer agents. it consolidates information from NODE_AGENTS.md and
A2A_AGENTS.md into a unified specialist registry for the graph orchestrator.
"""

from typing import Any, List

from .models import DiscoveredSpecialist
from .workspace import CORE_FILES, load_workspace_file


def discover_agents() -> dict[str, dict[str, Any]]:
    """Discover agents from the Knowledge Graph.

    Returns:
        A dictionary mapping domain tags to agent metadata.
    """
    from .graph.config_helpers import get_discovery_registry

    agent_descriptions = {}

    # Read from the Knowledge Graph
    registry = get_discovery_registry()
    for agent in registry.agents:
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

    # Also read A2A_AGENTS for backward compatibility if it exists
    a2a_content = ""
    try:
        a2a_file = CORE_FILES.get("A2A_AGENTS", "A2A_AGENTS.md")
        a2a_content = load_workspace_file(a2a_file)
    except Exception:
        pass

    if a2a_content:
        from .a2a import parse_a2a_registry

        a2a_registry = parse_a2a_registry(a2a_content)
        for agent in a2a_registry.peers:
            if agent.name.lower() not in agent_descriptions:
                agent_descriptions[agent.name.lower()] = {
                    "url": agent.url,
                    "name": agent.name,
                    "description": agent.description,
                    "capabilities": ", ".join(agent.capabilities),
                    "type": "remote_a2a",
                }

    return agent_descriptions


def discover_all_specialists() -> List[DiscoveredSpecialist]:
    """Discover all specialist agents from the Knowledge Graph.

    Returns:
        A list of DiscoveredSpecialist objects.
    """
    from .graph.config_helpers import get_discovery_registry

    specialists: List[DiscoveredSpecialist] = []
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
