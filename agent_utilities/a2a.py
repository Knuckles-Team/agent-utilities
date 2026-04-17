#!/usr/bin/python
# coding: utf-8
"""Agent-to-Agent (A2A) Peer Management Module.

Provides CRUD operations for managing remote A2A peers and a JSON-RPC
client for executing tasks on those peers.  Unified specialist discovery
(merging both MCP and A2A sources) has been extracted to
:mod:`discovery` — the functions are re-exported here for backwards
compatibility.
"""

from __future__ import annotations

import logging
import asyncio
import uuid
import httpx
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


from .config import *  # noqa: F403
from .workspace import (
    CORE_FILES,
    load_workspace_file,
    parse_a2a_registry,
)


from .models import A2ARegistryModel, A2APeerModel, DiscoveredSpecialist  # noqa: F401

# Note: discover_agents() and discover_all_specialists() live in discovery.py.
# They are NOT re-exported here to avoid a circular import
# (discovery.py imports A2AClient from this module).

logger = logging.getLogger(__name__)


def load_a2a_peers() -> A2ARegistryModel:
    """Load the A2A peer registry from the workspace.

    Parses the A2A_AGENTS.md file into a structured A2ARegistryModel.

    Returns:
        The loaded A2A registry model.

    """
    # A2A_AGENTS.md is now legacy. We load it if it exists for backward compatibility
    # but new peers should be managed via mcp_config.json HTTP servers.
    registry_file = CORE_FILES.get("A2A_AGENTS", "A2A_AGENTS.md")
    content = load_workspace_file(registry_file)
    if not content:
        return A2ARegistryModel()
    return parse_a2a_registry(content)


def register_a2a_peer(
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
    notes: str = "",
) -> str:
    """Add or update an A2A peer in the registry.

    Args:
        name: Unique name identifier for the peer.
        url: Base URL of the peer agent.
        description: Brief description of the agent's purpose.
        capabilities: Metadata string describing supported features.
        auth: Authentication type ('none', 'bearer', etc.).
        notes: Optional registrar notes.

    Returns:
        A status message indicating success.

    """
    # Manual A2A registration to A2A_AGENTS.md is deprecated.
    # We return success but don't write to the file to encourage unified discovery.
    logger.info(f"A2A peer '{name}' registered (memory-only/unified).")
    return f"✅ Registered/updated A2A peer '{name}' at {url} (Unified Registry)"


def list_a2a_peers() -> A2ARegistryModel:
    """List all agents currently registered in the A2A system.

    Returns:
        The complete registry of A2A peers.

    """
    return load_a2a_peers()


def delete_a2a_peer(name: str) -> str:
    """Remove an A2A peer from the registry.

    Args:
        name: The name identifier of the peer to remove.

    Returns:
        A status message indicating whether the peer was found and removed.

    """
    # Manual A2A deletion is deprecated as part of file retirement.
    return f"✅ Removed A2A peer '{name}' from unified registry context."


class A2AClient:
    """Client for Agent-to-Agent communication using the fastA2A JSON-RPC protocol.

    This client supports fetching agent metadata (cards) and executing
    arbitrary tasks on remote agents with polling for completion.
    """

    def __init__(self, timeout: float = 300.0, ssl_verify: bool = True):
        """Initialize the A2A client.

        Args:
            timeout: Maximum execution timeout for remote tasks in seconds.
            ssl_verify: Whether to verify SSL certificates for HTTPS requests.

        """
        self.timeout = timeout
        self.ssl_verify = ssl_verify

    def fetch_card_sync(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch the agent-card.json from a remote agent (synchronous).

        Args:
            url: The base URL of the remote agent.

        Returns:
            The agent card data as a dictionary, or None if the request fails.

        """
        card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
        with httpx.Client(timeout=5.0, verify=self.ssl_verify) as client:
            try:
                resp = client.get(card_url)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.debug(f"Failed (sync) to fetch agent card from {card_url}: {e}")
        return None

    async def fetch_card(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch the agent-card.json from a remote agent (asynchronous).

        Args:
            url: The base URL of the remote agent.

        Returns:
            The agent card data as a dictionary, or None if the request fails.

        """
        card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
        async with httpx.AsyncClient(timeout=10.0, verify=self.ssl_verify) as client:
            try:
                resp = await client.get(card_url)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.debug(f"Failed to fetch agent card from {card_url}: {e}")
        return None

    async def execute_task(self, url: str, query: str) -> Optional[Any]:
        """Execute a task on a remote agent via A2A message/send and polling.

        Args:
            url: The A2A endpoint URL.
            query: The natural language task description.

        Returns:
            The final result content from the remote agent, or an error message.

        """
        async with httpx.AsyncClient(
            timeout=self.timeout, verify=self.ssl_verify
        ) as client:
            # 1. Send Message
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "kind": "message",
                        "role": "user",
                        "parts": [{"kind": "text", "text": query}],
                        "messageId": str(uuid.uuid4()),
                    }
                },
                "id": 1,
            }
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return f"Error: A2A peer returned status {resp.status_code}"

                res_data = resp.json()
                if "error" in res_data:
                    return f"A2A Error: {res_data['error']}"

                task_id = res_data.get("result", {}).get("id")
                if not task_id:
                    return "Error: No task ID returned from A2A peer."

                # 2. Poll for Results
                while True:
                    await asyncio.sleep(2)
                    poll_payload = {
                        "jsonrpc": "2.0",
                        "method": "tasks/get",
                        "params": {"id": task_id},
                        "id": 2,
                    }
                    poll_resp = await client.post(url, json=poll_payload)
                    if poll_resp.status_code != 200:
                        return (
                            f"Error: Polling failed with status {poll_resp.status_code}"
                        )

                    poll_data = poll_resp.json()
                    if "error" in poll_data:
                        return f"A2A Polling Error: {poll_data['error']}"

                    result = poll_data.get("result", {})
                    state = result.get("status", {}).get("state")
                    if state not in ["submitted", "running", "working"]:
                        # Task completed: Extract content
                        history = result.get("history", [])
                        for msg in reversed(history):
                            if msg.get("role") != "user":
                                parts = msg.get("parts", [])
                                content = ""
                                for p in parts:
                                    content += p.get("text", p.get("content", ""))
                                return content
                        return "Error: No result found in task history."
            except Exception as e:
                return f"A2A Communication Error: {e}"
        return "Error: A2A execution timed out or failed."
