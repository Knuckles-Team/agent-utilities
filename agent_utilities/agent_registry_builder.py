#!/usr/bin/python
# coding: utf-8
"""Agent Registry Builder Module.

This module provides logic to rebuild the unified NODE_AGENTS.md registry
by merging definitions from prompt frontmatter, MCP configurations,
and remote A2A peers.
"""

import yaml
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from .workspace import (
    get_agent_workspace,
)

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
    """Ingest local prompt files into the Knowledge Graph as PromptNodes.

    (Kept name for backwards compatibility, but it no longer writes NODE_AGENTS.md)
    """
    workspace = get_agent_workspace()

    from .knowledge_graph.backends import create_backend

    backend = create_backend(db_path=str(workspace / "knowledge_graph.db"))
    if backend is None:
        logger.warning("Graph backend not available, skipping prompt ingestion.")
        return None

    # Process Prompt Agents (prompts/*.md)
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

            agent_name = metadata.get("name", pfile.stem)
            description = metadata.get("description", "")
            if not description:
                d_match = re.search(r"^#.*?\n+(.*?)\n", content, re.MULTILINE)
                if d_match:
                    description = d_match.group(1).strip()

            capabilities = metadata.get("skills", metadata.get("capabilities", []))

            # Store in Knowledge Graph as PromptNode
            query = """
            MERGE (p:Prompt {id: $id})
            SET p.name = $name,
                p.desc = $desc,
                p.content = $content,
                p.capabilities = $capabilities
            """
            backend.execute(
                query,
                {
                    "id": f"prompt:{agent_name}",
                    "name": agent_name,
                    "desc": description,
                    "content": content,
                    "capabilities": capabilities,
                },
            )

    logger.info("Local prompt files ingested into the Knowledge Graph.")
    return None


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(rebuild_node_agents_md())
