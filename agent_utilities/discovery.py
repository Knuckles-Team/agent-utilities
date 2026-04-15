#!/usr/bin/python
# coding: utf-8
"""Agent Discovery Module.

This module provides functionality for discovering local MCP agents and remote
A2A peer agents. it consolidates information from NODE_AGENTS.md and
A2A_AGENTS.md into a unified specialist registry for the graph orchestrator.
"""

from typing import Any, List

from .a2a import A2AClient, load_a2a_peers
from .models import DiscoveredSpecialist
from .workspace import CORE_FILES, load_workspace_file


def discover_agents() -> dict[str, dict[str, Any]]:
    """Discover local MCP agents and remote A2A peers.

    This function scans both the NODE_AGENTS.md (local) and A2A_AGENTS.md
    (remote) registries to build a unified map of specialists for the
    graph orchestrator.

    Returns:
        A dictionary mapping domain tags to agent metadata (package, type, etc.).

    """
    from .workspace import parse_node_registry

    agent_descriptions = {}

    # 1. Discover local MCP specialist agents from NODE_AGENTS.md
    mcp_agents_content = load_workspace_file(CORE_FILES["NODE_AGENTS"])
    if mcp_agents_content:
        mcp_registry = parse_node_registry(mcp_agents_content)
        for agent in mcp_registry.agents:
            if agent.tag:
                agent_descriptions[agent.tag] = {
                    "package": agent.name,
                    "description": agent.description,
                    "name": agent.name,
                    "type": "local_mcp",
                }

    # 2. Remote Discovery from A2A_AGENTS.md
    # We fetch these fresh every time as per user feedback
    registry = load_a2a_peers()
    if registry.peers:
        client = A2AClient()
        for peer in registry.peers:
            tag = peer.name.lower().replace(" ", "_")
            if tag in agent_descriptions:
                continue

            # Attempt to fetch agent card for rich metadata
            card = client.fetch_card_sync(peer.url)
            if card:
                description = card.get("description", peer.description)
                display_name = card.get("name", peer.name)
                capabilities = card.get("capabilities", peer.capabilities)
            else:
                description = peer.description
                display_name = peer.name
                capabilities = peer.capabilities

            agent_descriptions[tag] = {
                "url": peer.url,
                "name": display_name,
                "description": description,
                "capabilities": capabilities,
                "type": "remote_a2a",
            }

    return agent_descriptions


def discover_all_specialists() -> List[DiscoveredSpecialist]:
    """Discover all specialist agents from all sources.

    This function provides a unified list of specialists (local MCP and remote A2A)
    represented as DiscoveredSpecialist models, which is used by the graph orchestrator
    for registration and routing.

    Returns:
        A list of DiscoveredSpecialist objects.

    """
    specialists: List[DiscoveredSpecialist] = []

    # 1. Local Specialists (MCP)
    from .workspace import parse_node_registry

    mcp_agents_content = load_workspace_file(CORE_FILES["NODE_AGENTS"])
    if mcp_agents_content:
        mcp_registry = parse_node_registry(mcp_agents_content)
        for agent in mcp_registry.agents:
            specialists.append(
                DiscoveredSpecialist(
                    tag=agent.tag or agent.name.lower().replace(" ", "_"),
                    name=agent.name,
                    description=agent.description or "",
                    source="mcp",
                    mcp_server=agent.mcp_server,
                    tools=agent.tools or [],
                )
            )

    # 2. Remote Specialists (A2A)
    registry = load_a2a_peers()
    if registry.peers:
        for peer in registry.peers:
            specialists.append(
                DiscoveredSpecialist(
                    tag=peer.name.lower().replace(" ", "_"),
                    name=peer.name,
                    description=peer.description or "",
                    source="a2a",
                    url=peer.url,
                    capabilities=peer.capabilities or "",
                )
            )

    return specialists
