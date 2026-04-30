#!/usr/bin/python
"""Agent Registry Builder Module.

This module provides logic to ingest specialized prompt metadata into the
Knowledge Graph as ``PromptNode`` entries. Prompts are JSON blueprints that
live under ``agent_utilities/prompts/*.json``; each file is read as a dict
and its top-level keys (``name``, ``description``, ``capabilities``,
``tags``, ``content``) are projected directly onto a ``PromptNode``.

Two JSON schemas are accepted in ``_resolve_fields`` because the repository
currently mixes two valid shapes:

* The modern blueprint schema: ``{name, description, capabilities, content}``.
* The ``StructuredPrompt`` schema still used by ~40 prompts:
  ``{task, goal, tools, input}``.

Markdown fallbacks have been removed. Only ``*.json`` files under
``prompts/`` are ingested.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from agent_utilities.core.workspace import (
    get_agent_workspace,
)
from agent_utilities.models.knowledge_graph import RegistryNodeType

logger = logging.getLogger(__name__)


RESERVED_CORE_NODES: set[str] = {
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


def _load_prompt_metadata(pfile: Path) -> dict[str, Any] | None:
    """Load a single JSON prompt file and return a unified metadata dict.

    Only the modern JSON blueprint schema (``*.json`` with top-level
    ``name``/``description``/``capabilities``/``content`` keys, or the
    ``StructuredPrompt`` variant with ``task``/``goal``/``tools``/``input``)
    is supported. Returns ``None`` if the file cannot be parsed.
    """
    if pfile.suffix != ".json":
        logger.debug("Skipping unsupported prompt file: %s", pfile.name)
        return None

    try:
        raw = pfile.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"Failed to read prompt file {pfile.name}: {e}")
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse prompt JSON {pfile.name}: {e}")
        return None

    if not isinstance(data, dict):
        logger.warning(f"Prompt {pfile.name} is not a JSON object")
        return None

    data.setdefault("name", pfile.stem)
    data.setdefault("type", "prompt")
    return data


def _resolve_fields(data: dict[str, Any], stem: str) -> tuple[str, str, list[str], str]:
    """Extract ``(name, description, capabilities, system_prompt)`` from data.

    Accepts both the modern blueprint schema
    (``{name, description, capabilities, content}``) and the
    ``StructuredPrompt`` schema (``{task, goal, tools, input}``) because the
    repository currently ships both valid JSON shapes side-by-side.
    """
    name = str(data.get("name") or data.get("task") or stem)
    description = str(data.get("description") or data.get("goal") or "")
    capabilities_raw: Any = (
        data.get("capabilities") or data.get("tools") or data.get("skills") or []
    )
    if isinstance(capabilities_raw, list):
        capabilities = [str(c) for c in capabilities_raw]
    elif isinstance(capabilities_raw, str):
        capabilities = [capabilities_raw]
    else:
        capabilities = []

    system_prompt = data.get("content") or data.get("input") or ""
    if not isinstance(system_prompt, str):
        system_prompt = json.dumps(system_prompt)
    return name, description, capabilities, system_prompt


async def ingest_prompts_to_graph():
    """Ingest local JSON prompt files into the Knowledge Graph.

    This replaces the legacy NODE_AGENTS.md registry with a dynamic KG-native
    discovery mechanism. Only ``*.json`` blueprints under
    ``agent_utilities/prompts/`` are ingested.
    """
    import networkx as nx
    from filelock import FileLock, Timeout

    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
    from agent_utilities.models.knowledge_graph import PromptNode

    workspace = get_agent_workspace()
    lock_path = workspace / ".prompts.sync.lock"
    lock = FileLock(str(lock_path), timeout=0)

    try:
        with lock.acquire(timeout=0):
            engine = IntelligenceGraphEngine.get_active()
            if not engine:
                db_path = str(workspace / "knowledge_graph.db")
                engine = IntelligenceGraphEngine(
                    graph=nx.MultiDiGraph(), db_path=db_path
                )

            if engine.backend:
                engine.backend.create_schema()

            backend = engine.backend
            if backend is None:
                logger.warning(
                    "Graph backend not available, skipping prompt ingestion."
                )
                return None

            prompts_dir = Path(__file__).parent / "prompts"
            if not prompts_dir.exists():
                logger.info("No prompts directory found at %s", prompts_dir)
                return None

            prompt_files: list[Path] = list(prompts_dir.glob("*.json"))
            logger.debug(f"Found {len(prompt_files)} prompt files in {prompts_dir}")

            for pfile in prompt_files:
                stem = pfile.stem
                if pfile.name.startswith("_") or stem in RESERVED_CORE_NODES:
                    logger.debug(f"Skipping reserved/internal node: {stem}")
                    continue

                data = _load_prompt_metadata(pfile)
                if data is None:
                    continue

                try:
                    name, description, capabilities, system_prompt = _resolve_fields(
                        data, stem
                    )
                    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    node = PromptNode(
                        id=f"prompt:{name}",
                        name=name,
                        description=description,
                        system_prompt=system_prompt,
                        json_blueprint=data,
                        capabilities=capabilities,
                        type=RegistryNodeType.PROMPT,
                        timestamp=ts,
                        is_permanent=True,
                    )
                    logger.debug(f"Ingesting prompt {name} via engine.upsert")
                    serialized_data = engine._serialize_node(node, label="Prompt")
                    engine._upsert_node("Prompt", node.id, serialized_data)
                except Exception as e:
                    logger.warning(f"Failed to ingest prompt {pfile.name}: {e}")

            logger.info("Local prompt files ingested into the Knowledge Graph.")
    except Timeout:
        logger.info("Another process is currently ingesting prompts. Skipping...")
    return None


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(ingest_prompts_to_graph())
