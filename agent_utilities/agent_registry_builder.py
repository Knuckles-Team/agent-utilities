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
    from .knowledge_graph.engine import IntelligenceGraphEngine
    import networkx as nx

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        workspace = get_agent_workspace()
        db_path = str(workspace / "knowledge_graph.db")
        engine = IntelligenceGraphEngine(graph=nx.MultiDiGraph(), db_path=db_path)
    
    if engine.backend:
        engine.backend.create_schema()

    backend = engine.backend
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
        json_files = list(prompts_dir.glob("*.json"))
        print(f"DEBUG: Found {len(json_files)} prompt files in {prompts_dir}")
        for pfile in json_files:
            name = pfile.stem
            print(f"DEBUG: Processing file: {pfile.name}, stem={name}")
            if pfile.name.startswith("_") or name in RESERVED_CORE_NODES:
                print(f"DEBUG: Skipping reserved/internal node: {name}")
                continue

            try:
                import json

                from .structured_prompts import StructuredPrompt

                raw_data = json.loads(pfile.read_text(encoding="utf-8"))
                prompt_obj = StructuredPrompt.model_validate(raw_data)

                agent_name = prompt_obj.task or pfile.stem
                description = getattr(prompt_obj, "description", "")
                if not description:
                    description = prompt_obj.goal or ""

                capabilities = (
                    getattr(prompt_obj, "tools", None)
                    or getattr(prompt_obj, "skills", None)
                    or getattr(prompt_obj, "capabilities", [])
                )

                import json
                # Store in Knowledge Graph as PromptNode
                from .models.knowledge_graph import PromptNode
                import time

                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                node = PromptNode(
                    id=f"prompt:{agent_name}",
                    name=agent_name,
                    description=description,
                    system_prompt=prompt_obj.render(),
                    json_blueprint=raw_data,  # Pass as dict, engine will serialize
                    capabilities=capabilities,
                    type="prompt",
                    timestamp=ts,
                    is_permanent=True
                )

                print(f"DEBUG: Ingesting prompt {agent_name} via engine.upsert")
                serialized_data = engine._serialize_node(node, label="Prompt")
                engine._upsert_node("Prompt", node.id, serialized_data)
            except Exception as e:
                print(f"ERROR: Failed to ingest prompt {pfile.name}: {e}")
                logger.warning(f"Failed to ingest prompt {pfile.name}: {e}")

    logger.info("Local prompt files ingested into the Knowledge Graph.")
    return None


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(ingest_prompts_to_graph())
