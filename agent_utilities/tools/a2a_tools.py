#!/usr/bin/python
# coding: utf-8
"""A2A Tools Module.

This module provides tools for discovering, registering, and managing
Model Context Protocol (MCP) and A2A peer agents.
"""

import logging
from typing import Any
from pydantic_ai import RunContext
from ..models import A2ARegistryModel
from ..a2a import (
    register_a2a_peer as register_a2a_peer_util,
    delete_a2a_peer as delete_a2a_peer_util,
    list_a2a_peers as list_a2a_peers_util,
)

logger = logging.getLogger(__name__)


async def list_a2a_peers(ctx: RunContext[Any]) -> A2ARegistryModel:
    """List all known A2A peer agents in the workspace registry.

    Args:
        ctx: The agent run context.

    Returns:
        A model containing the list of known A2A peers.

    """
    return list_a2a_peers_util()


async def register_a2a_peer(
    ctx: RunContext[Any],
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
) -> str:
    """Register or update an A2A agent for delegated task execution.

    Args:
        ctx: The agent run context.
        name: The unique identifier for the peer agent.
        url: The connection URL for the peer service.
        description: A brief summary of the peer's purpose.
        capabilities: A comma-separated list of peer specialties.
        auth: Authentication type (e.g., 'none', 'bearer').

    Returns:
        A confirmation message indicating success.

    """
    return register_a2a_peer_util(name, url, description, capabilities, auth)


async def delete_a2a_peer(ctx: RunContext[Any], name: str) -> str:
    """Remove an A2A peer agent from the local workspace registry.

    Args:
        ctx: The agent run context.
        name: The name of the peer to remove.

    Returns:
        A confirmation message indicating success.

    """
    return delete_a2a_peer_util(name)


# Tool grouping for registration
a2a_tools = [
    list_a2a_peers,
    register_a2a_peer,
    delete_a2a_peer,
]
