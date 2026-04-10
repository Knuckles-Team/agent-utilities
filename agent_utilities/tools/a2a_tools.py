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
    """List all known A2A peer agents."""
    return list_a2a_peers_util()


async def register_a2a_peer(
    ctx: RunContext[Any],
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
) -> str:
    """Register or update another A2A agent this agent can call."""
    return register_a2a_peer_util(name, url, description, capabilities, auth)


async def delete_a2a_peer(ctx: RunContext[Any], name: str) -> str:
    """Remove an A2A peer agent from the local registry."""
    return delete_a2a_peer_util(name)


# Tool grouping for registration
a2a_tools = [
    list_a2a_peers,
    register_a2a_peer,
    delete_a2a_peer,
]
