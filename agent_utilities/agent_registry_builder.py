#!/usr/bin/python
"""Agent Registry Builder Module.

This module provides logic to ingest specialized prompt metadata from
markdown frontmatter directly into the Knowledge Graph as PromptNodes.
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .workspace import (
    get_agent_workspace,
)

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter from a markdown string."""
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
    return None


async def ingest_prompts_to_graph():
    """Ingest local prompt files into the Knowledge Graph as PromptNodes.

    This replaces the legacy NODE_AGENTS.md registry with a dynamic KG-native
    discovery mechanism.
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
            data = {
                "id": f"prompt:{agent_name}",
                "name": agent_name,
                "description": description,
                "system_prompt": content,
                "capabilities": capabilities,
                "type": "prompt",
            }
            # Ladybug compatibility: use MATCH/SET then CREATE
            set_clause = ", ".join([f"p.{k} = ${k}" for k in data.keys() if k != "id"])
            update_query = (
                f"MATCH (p:Prompt) WHERE p.id = $id SET {set_clause} RETURN p.id"
            )
            res = backend.execute(update_query, data)

            if not res:
                cols = ", ".join([f"{k}: ${k}" for k in data.keys()])
                create_query = f"CREATE (p:Prompt {{{cols}}})"
                backend.execute(create_query, data)

    logger.info("Local prompt files ingested into the Knowledge Graph.")
    return None


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(ingest_prompts_to_graph())
