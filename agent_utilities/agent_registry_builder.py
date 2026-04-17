#!/usr/bin/python
# coding: utf-8
"""Agent Registry Builder Module.

This module provides logic to rebuild the unified NODE_AGENTS.md registry
by merging definitions from prompt frontmatter, MCP configurations,
and remote A2A peers.
"""

import json
import yaml
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from .workspace import (
    CORE_FILES,
    get_agent_workspace,
    get_workspace_path,
    parse_node_registry,
    serialize_node_registry,
)
from .models import MCPAgent, MCPAgentRegistryModel, MCPToolInfo
from .mcp_agent_manager import (
    extract_tool_metadata,
    partition_tools,
    generate_system_prompt,
    score_tools,
    compute_agent_metadata_score,
)
from .a2a import A2AClient

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter from a markdown string."""
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
    return None


async def rebuild_node_agents_md():
    """Consolidate all agent sources into the unified NODE_AGENTS.md registry."""
    workspace = get_agent_workspace()
    registry_path = get_workspace_path(CORE_FILES["NODE_AGENTS"])

    # 1. Load existing registry to preserve user edits (capabilities, extra_config)
    existing_agents = {}
    if registry_path.exists():
        try:
            old_registry = parse_node_registry(
                registry_path.read_text(encoding="utf-8")
            )
            existing_agents = {a.name: a for a in old_registry.agents}
        except Exception as e:
            logger.warning(f"Could not parse existing registry: {e}")

    new_agents: List[MCPAgent] = []

    # 2. Process Prompt Agents (prompts/*.md)
    # Filter out reserved core node IDs that are registered statically in builder.py
    RESERVED_CORE_NODES = {
        "usage_guard",
        "router",
        "dispatcher",
        "synthesizer",
        "onboarding",
        "error_recovery",
        "parallel_batch_processor",
        "expert_executor",
        "research_joiner",
        "execution_joiner",
        "mcp_router",
        "mcp_server_execution",
        "approval_gate",
        "memory_selection",
        "expert_dispatch",
    }
    prompts_dir = Path(__file__).parent / "prompts"
    if prompts_dir.exists():
        for pfile in prompts_dir.glob("*.md"):
            name = pfile.stem
            if pfile.name.startswith("_") or name in RESERVED_CORE_NODES:
                continue

            content = pfile.read_text(encoding="utf-8")
            metadata = parse_frontmatter(content) or {}

            name = metadata.get("name", pfile.stem)
            description = metadata.get("description", "")
            if not description:
                # Fallback: extract first paragraph or heading
                d_match = re.search(r"^#.*?\n+(.*?)\n", content, re.MULTILINE)
                if d_match:
                    description = d_match.group(1).strip()

            agent = MCPAgent(
                name=name,
                agent_type="prompt",
                prompt_file=f"prompts/{pfile.name}",
                description=description,
                capabilities=metadata.get("skills", metadata.get("capabilities", [])),
                is_custom=False,
                avg_relevance_score=compute_agent_metadata_score(
                    description,
                    metadata.get("skills", metadata.get("capabilities", [])),
                ),
            )

            # Merge existing user-added capabilities if any
            if name in existing_agents:
                existing = existing_agents[name]
                if existing.capabilities:
                    # Union of skills
                    agent.capabilities = list(
                        set(agent.capabilities + existing.capabilities)
                    )
                if existing.extra_config:
                    agent.extra_config.update(existing.extra_config)

            new_agents.append(agent)

    # 3. Process MCP Agents (mcp_config.json)
    mcp_config_path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    all_discovered_tools: List[MCPToolInfo] = []

    # Try to reuse existing tools if probing fails
    existing_tools = (
        {t.name: t for t in old_registry.tools} if "old_registry" in locals() else {}
    )

    if mcp_config_path.exists():
        try:
            # PROBE MCP Servers for real tools and tags
            logger.info("Probing MCP servers for tool metadata and partitioning...")
            tools_inventory = await extract_tool_metadata(mcp_config_path)
            score_tools(tools_inventory)  # Enable relevance scoring
            all_discovered_tools = tools_inventory

            # Fetch A2A Cards for SSE/HTTP servers
            mcp_data = json.loads(mcp_config_path.read_text(encoding="utf-8"))
            servers_config = mcp_data.get("mcpServers", {})
            a2a_client = A2AClient()
            a2a_metadata = {}

            for sname, scfg in servers_config.items():
                url = scfg.get("url")
                if url and "http" in url:
                    logger.info(f"Fetching A2A card for {sname} at {url}")
                    card = await a2a_client.fetch_card(url)
                    if card:
                        a2a_metadata[sname] = card

            if tools_inventory:
                partitions = await partition_tools(tools_inventory)
                for tag, group in partitions.items():
                    server_name = group[0].mcp_server
                    clean_server = (
                        server_name.replace("-mcp", "")
                        .replace("-agent", "")
                        .replace("_", " ")
                        .title()
                    )
                    clean_tag = tag.replace("_", " ").title()

                    if clean_server.lower() in clean_tag.lower():
                        agent_name = f"{clean_tag} Specialist"
                    else:
                        agent_name = f"{clean_server} {clean_tag} Specialist"

                    description = f"Expert specialist for {tag} domain tasks."
                    system_prompt = await generate_system_prompt(
                        agent_name, group, tag, server_name
                    )

                    # If this is an A2A server, use card description if available
                    skills = []
                    score = (
                        (sum(t.relevance_score for t in group) // len(group))
                        if group
                        else 0
                    )
                    if server_name in a2a_metadata:
                        card = a2a_metadata[server_name]
                        description = card.get("description", description)
                        skills = card.get("skills", [])
                        mscore = compute_agent_metadata_score(description, skills)
                        score = (score + mscore) // 2

                    new_agents.append(
                        MCPAgent(
                            name=agent_name,
                            agent_type=(
                                "a2a"
                                if "http"
                                in servers_config.get(server_name, {}).get("url", "")
                                else "mcp"
                            ),
                            endpoint_url=servers_config.get(server_name, {}).get(
                                "url", "stdio"
                            ),
                            description=description,
                            system_prompt=system_prompt,
                            mcp_server=server_name,
                            mcp_tools=tag,
                            capabilities=skills,
                            tool_count=len(group),
                            avg_relevance_score=score,
                            is_custom=False,
                        )
                    )
            else:
                # Fallback to simple server-based agents if no tools found (e.g. servers offline)
                logger.warning(
                    "No tools discovered during probe, falling back to basic MCP server mapping."
                )
                for sname, scfg in servers_config.items():
                    url = scfg.get("url", "-")
                    atype = "a2a" if "http" in url else "mcp"

                    description = f"Specialist agent for {sname} ({atype.upper()})"
                    skills = []
                    score = 0
                    if sname in a2a_metadata:
                        card = a2a_metadata[sname]
                        description = card.get("description", description)
                        skills = card.get("skills", [])
                        score = compute_agent_metadata_score(description, skills)
                    elif atype == "a2a":
                        score = 40  # Moderate score for unprobed but identified A2A

                    new_agents.append(
                        MCPAgent(
                            name=sname,
                            agent_type=atype,
                            endpoint_url="stdio" if "command" in scfg else url,
                            description=description,
                            mcp_server=sname,
                            capabilities=skills,
                            avg_relevance_score=score,
                            is_custom=False,
                        )
                    )
                # Recover tool inventory from cache if available
                all_discovered_tools = list(existing_tools.values())

        except Exception as e:
            logger.warning(f"Failed to process MCP config for registry: {e}")
            all_discovered_tools = list(existing_tools.values())

    # 4. Save unified registry with Tool Inventory
    registry_model = MCPAgentRegistryModel(
        agents=new_agents, tools=all_discovered_tools
    )
    new_content = serialize_node_registry(registry_model)
    registry_path.write_text(new_content, encoding="utf-8")

    logger.info(f"Registry rebuilt: {len(new_agents)} agents specialized.")
    return registry_model


if __name__ == "__main__":
    import asyncio

    # Setup basic logging to see the output when run directly
    logging.basicConfig(level=logging.INFO)
    asyncio.run(rebuild_node_agents_md())
