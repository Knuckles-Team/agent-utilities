#!/usr/bin/python

from __future__ import annotations

import logging
import asyncio
import uuid
import httpx
from typing import Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass


from .config import *  # noqa: F403
from .workspace import (
    CORE_FILES,
    get_workspace_path,
    load_workspace_file,
    parse_a2a_registry,
    serialize_a2a_registry,
)


from .models import A2ARegistryModel, A2APeerModel

logger = logging.getLogger(__name__)


def load_a2a_peers() -> A2ARegistryModel:
    """Parse A2A_AGENTS.md table into A2ARegistryModel."""
    content = load_workspace_file(CORE_FILES["A2A_AGENTS"])
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
    """Add or update a peer in A2A_AGENTS.md table."""
    registry = load_a2a_peers()

    updated = False
    for p in registry.peers:
        if p.name.lower() == name.lower():
            p.url = url
            p.description = description
            p.capabilities = capabilities
            p.auth = auth
            p.notes = notes or datetime.now().strftime("%Y-%m-%d")
            updated = True
            break

    if not updated:
        registry.peers.append(
            A2APeerModel(
                name=name,
                url=url,
                description=description,
                capabilities=capabilities,
                auth=auth,
                notes=notes or datetime.now().strftime("%Y-%m-%d"),
            )
        )

    content = serialize_a2a_registry(registry)
    path = get_workspace_path(CORE_FILES["A2A_AGENTS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"✅ Registered/updated A2A peer '{name}' at {url}"


def list_a2a_peers() -> A2ARegistryModel:
    """List all registered A2A peers."""
    return load_a2a_peers()


def delete_a2a_peer(name: str) -> str:
    """Remove a peer from A2A_AGENTS.md registry."""
    registry = load_a2a_peers()
    original_count = len(registry.peers)
    registry.peers = [p for p in registry.peers if p.name.lower() != name.lower()]

    if len(registry.peers) < original_count:
        content = serialize_a2a_registry(registry)
        path = get_workspace_path(CORE_FILES["A2A_AGENTS"])
        path.write_text(content, encoding="utf-8")
        return f"✅ Removed A2A peer '{name}' from registry."
    return f"ℹ️ A2A peer '{name}' not found in registry."


class A2AClient:
    """
    Client for Agent-to-Agent (A2A) communication following the JSON-RPC spec.
    """

    def __init__(self, timeout: float = 300.0, ssl_verify: bool = True):
        self.timeout = timeout
        self.ssl_verify = ssl_verify

    def fetch_card_sync(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the well-known agent card from a remote agent (Synchronous version).
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
        """
        Fetches the well-known agent card from a remote agent.
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
        """
        Executes a task on a remote agent via A2A protocol (message/send + polling).
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


def discover_agents() -> dict[str, dict[str, str]]:
    """Discovers available agent packages from A2A_AGENTS.md and NODE_AGENTS.md registries.

    Returns:
        dict: {tag: {"package": package_name, "description": desc, "name": display_name, "type": "local" | "remote_a2a"}}
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
