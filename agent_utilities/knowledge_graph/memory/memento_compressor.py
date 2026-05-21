#!/usr/bin/python
from __future__ import annotations

"""Generalized Memento Context Compressor.

CONCEPT:KG-2.1 -- Observational Memory Bridge (Extension)

Provides generalized LLM-based state compression for long-running
agents. Takes a block of conversation history and generates a dense
memento preserving formulas, intermediate values, and strategic decisions.
"""

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

MEMENTO_SYSTEM_PROMPT = """You are a state-compression Memento generator for an autonomous agent.
Your task is to take a block of reasoning and conversation history and compress it into a dense Memento.

## Strict Rules:
1. You are NOT summarizing for a human. You are compressing state for an LLM to reason forward from.
2. You MUST extract exact formulas, key intermediate values, commands executed, and their precise outcomes.
3. Keep the strategic decisions and the current execution state (what succeeded, what failed, what is next).
4. Do NOT hallucinate or add outside knowledge.
5. Provide a terse, information-dense output that can act as a drop-in replacement for the raw block.
6. Output ONLY the memento text.
"""


def compress_to_memento(
    engine: IntelligenceGraphEngine,
    messages: list[dict[str, str]],
    *,
    source: str = "agent_runner",
    dry_run: bool = False,
) -> str | None:
    """Compress a block of messages into a dense memento and persist it.

    Args:
        engine: IntelligenceGraphEngine instance.
        messages: The block of raw messages to compress.
        source: The source agent or component name.
        dry_run: If True, do not persist to the KG.

    Returns:
        The generated memento string, or None if compression failed.
    """
    if not messages:
        return None

    # Format block for compression
    transcript_lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        transcript_lines.append(f"[{role}]: {content}")
    block_text = "\n\n".join(transcript_lines)

    try:
        from pydantic_ai import Agent

        from agent_utilities.core.config import (
            DEFAULT_KG_MODEL_ID,
            DEFAULT_LLM_PROVIDER,
        )
        from agent_utilities.core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = Agent(model, system_prompt=MEMENTO_SYSTEM_PROMPT)

        import nest_asyncio

        nest_asyncio.apply()

        user_content = (
            f"## Compress the following block into a Memento:\n\n{block_text}"
        )
        result = agent.run_sync(user_content)
        memento_text = str(result.data).strip()
    except Exception as e:
        logger.warning("Memento compression failed: %s", e)
        return None

    if dry_run:
        return memento_text

    _persist_memento(engine, memento_text, source=source)
    return memento_text


def _persist_memento(
    engine: IntelligenceGraphEngine,
    memento_text: str,
    *,
    source: str = "unknown",
) -> None:
    """Persist the generated memento to the Knowledge Graph."""
    if not engine or not engine.backend:
        return

    memento_id = f"mem_{hashlib.md5(memento_text.encode(), usedforsecurity=False).hexdigest()[:10]}"
    current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    props: dict[str, Any] = {
        "name": f"Memento: {current_time}",
        "content": memento_text,
        "source": source,
        "timestamp": current_time,
        "type": "MementoBlock",
    }

    try:
        engine.add_node(memento_id, "Memento", properties=props)
        logger.info("[KG-2.10] Persisted Memento context block (%s)", memento_id)
    except Exception as e:
        logger.debug("Failed to persist Memento: %s", e)


def get_recent_mementos(
    engine: IntelligenceGraphEngine,
    source: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve the most recent mementos for a given source."""
    if not engine or not engine.backend:
        return []

    try:
        rows = engine.backend.execute(
            "MATCH (m:Memento {source: $source}) "
            "RETURN m.content AS content "
            "ORDER BY m.timestamp ASC LIMIT $limit",
            {"source": source, "limit": limit},
        )
        return [r.get("content", "") for r in rows if r.get("content")]
    except Exception as e:
        logger.debug("Failed to retrieve Mementos: %s", e)
        return []
