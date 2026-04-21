"""Agent-to-Agent (A2A) Peer Management Module.

This module provides a JSON-RPC client for executing tasks on remote A2A peer agents.
All discovery and registration of peers is now handled via the Knowledge Graph.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    pass


from .config import *  # noqa: F403
from .models import A2APeerModel, A2ARegistryModel, DiscoveredSpecialist  # noqa: F401

logger = logging.getLogger(__name__)


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

    def fetch_card_sync(self, url: str) -> dict[str, Any] | None:
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

    async def fetch_card(self, url: str) -> dict[str, Any] | None:
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

    async def execute_task(self, url: str, query: str) -> Any | None:
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


# --- Registry Utilities (Graph-Native Fallbacks) ---


def register_a2a_peer(
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
) -> str:
    """Register or update an A2A agent in the Knowledge Graph.

    Args:
        name: The unique identifier for the peer agent.
        url: The connection URL for the peer service.
        description: A brief summary of the peer's purpose.
        capabilities: A comma-separated list of peer specialties.
        auth: Authentication type.

    Returns:
        A confirmation message.
    """
    import time

    from .knowledge_graph.engine import IntelligenceGraphEngine
    from .models.knowledge_graph import AgentNode, RegistryNodeType

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    agent_id = f"a2a:{name.lower().replace(' ', '_')}"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    node = AgentNode(
        id=agent_id,
        name=name,
        type=RegistryNodeType.AGENT,
        agent_type="a2a",
        description=description,
        endpoint_url=url,
        timestamp=ts,
        metadata={"capabilities": capabilities.split(","), "auth": auth},
    )

    try:
        engine.graph.add_node(node.id, **node.model_dump())
        return f"A2A peer '{name}' registered successfully in the Knowledge Graph."
    except Exception as e:
        return f"Error registering A2A peer: {e}"


def delete_a2a_peer(name: str) -> str:
    """Remove an A2A peer agent from the Knowledge Graph."""
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    agent_id = f"a2a:{name.lower().replace(' ', '_')}"

    try:
        if agent_id in engine.graph:
            engine.graph.remove_node(agent_id)
            return f"A2A peer '{name}' removed from the Knowledge Graph."
        else:
            return f"Error: A2A peer '{name}' not found."
    except Exception as e:
        return f"Error removing A2A peer: {e}"


def list_a2a_peers() -> Any:
    """List all known A2A peer agents from the Knowledge Graph."""
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return A2ARegistryModel(peers=[])

    query = "MATCH (a:Agent {agent_type: 'a2a'}) RETURN a.name as name, a.description as description, a.endpoint_url as url, a.metadata as meta"
    results = engine.query_cypher(query)

    peers = {}
    for r in results:
        peers[r["name"]] = A2APeerModel(
            name=r["name"],
            url=r["url"],
            description=r["description"],
            capabilities=",".join(r.get("meta", {}).get("capabilities", [])),
        )

    return A2ARegistryModel(peers=list(peers.values()))
