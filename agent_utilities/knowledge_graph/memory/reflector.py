#!/usr/bin/python
from __future__ import annotations

"""LLM-Powered Reflection Condenser.

CONCEPT:KG-2.10 -- Observational Memory Bridge

Condenses observations into durable long-term reflections using LLM.
Wired into the existing ConsolidationEngine (KG-2.4) pipeline.

Pipeline: ObservationNodes -> LLM Reflector -> ReflectionNode/PreferenceNode (KG)
"""

import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM_PROMPT = """You are the Reflector for the agent-utilities Knowledge Graph.
Your job is to condense observations into a stable long-term memory document.

## Tasks
1. MERGE duplicate or overlapping observations into single entries
2. PROMOTE frequently-seen patterns from \U0001f7e1 to \U0001f534
3. DEMOTE stale or one-off observations from \U0001f534 to \U0001f7e1 or \U0001f7e2
4. ARCHIVE observations that are no longer relevant
5. EXTRACT preferences, principles, and identity facts

## Output Sections
Produce the complete reflections document with these sections:
- ## Core Identity (name, role, communication style, working hours)
- ## Preferences & Opinions (categorized, with priority markers)
- ## Key Facts & Context (important background facts)
- ## Active Projects (current work)
- ## Recent Themes (recurring patterns)

## Rules
- Preserve exact technical details
- Use bullet points, keep entries concise
- Include a *Last updated: YYYY-MM-DD HH:MM UTC* line after the title
- Include a *Last reflected: YYYY-MM-DD* line for the latest observation date processed
"""

_REFLECTOR_MAX_OUTPUT_TOKENS = 8192


def run_reflector(
    engine: IntelligenceGraphEngine,
    *,
    dry_run: bool = False,
) -> str | None:
    """Read observations, condense into reflections, persist to KG.

    Args:
        engine: IntelligenceGraphEngine instance.
        dry_run: If True, return result without persisting.

    Returns:
        The new reflections text, or None if nothing to reflect on.
    """
    # Gather observations from KG
    observations = _gather_observations(engine)
    if not observations:
        return None

    # Gather existing reflections
    reflections = _gather_reflections(engine)

    # Build LLM prompt
    obs_text = "\n".join(
        f"- {o.get('content', o.get('description', ''))}" for o in observations
    )
    ref_text = (
        "\n".join(
            f"- {r.get('content', r.get('description', ''))}" for r in reflections
        )
        if reflections
        else "(no existing reflections)"
    )

    user_content = (
        f"## Current reflections\n\n{ref_text}\n\n"
        f"---\n\n"
        f"## Current observations\n\n{obs_text}"
    )

    # Call LLM
    try:
        from pydantic_ai import Agent

        from ...core.config import DEFAULT_KG_MODEL_ID, DEFAULT_LLM_PROVIDER
        from ...core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = Agent(model, system_prompt=REFLECTOR_SYSTEM_PROMPT)

        import nest_asyncio

        nest_asyncio.apply()

        result = agent.run_sync(user_content)
        result_text = str(result.data)
    except Exception as e:
        logger.warning("Reflector LLM call failed: %s", e)
        return None

    # Stamp timestamps
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    result_text = _stamp_timestamps(result_text, now_utc)

    if dry_run:
        return result_text

    # Persist reflections to KG
    _persist_reflections(engine, result_text)

    # Trigger materialization
    from .memory_materializer import materialize_memory

    try:
        materialize_memory(engine)
    except Exception as e:
        logger.debug("Post-reflection materialization failed: %s", e)

    return result_text


def _gather_observations(
    engine: IntelligenceGraphEngine,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Gather observation nodes from the KG."""
    if not engine.backend:
        return [
            dict(a)
            for _, a in engine.graph.nodes(data=True)
            if a.get("type") == "observation"
        ][:limit]
    try:
        res = engine.backend.execute(
            "MATCH (n:Observation) RETURN n ORDER BY n.timestamp DESC LIMIT $limit",
            {"limit": limit},
        )
        return [r["n"] for r in res if "n" in r]
    except Exception:
        return [
            dict(a)
            for _, a in engine.graph.nodes(data=True)
            if a.get("type") == "observation"
        ][:limit]


def _gather_reflections(
    engine: IntelligenceGraphEngine,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Gather existing reflection nodes from the KG."""
    if not engine.backend:
        return [
            dict(a)
            for _, a in engine.graph.nodes(data=True)
            if a.get("type") == "reflection"
        ][:limit]
    try:
        res = engine.backend.execute(
            "MATCH (n:Reflection) RETURN n ORDER BY n.timestamp DESC LIMIT $limit",
            {"limit": limit},
        )
        return [r["n"] for r in res if "n" in r]
    except Exception:
        return []


def _persist_reflections(engine: IntelligenceGraphEngine, text: str) -> int:
    """Parse reflection markdown and create/update KG nodes."""
    count = 0
    current_section = ""

    for line in text.splitlines():
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue

        bullet_match = re.match(r"^- (?:[\U0001f534\U0001f7e1\U0001f7e2] )?(.+)$", line)
        if not bullet_match:
            continue

        content = bullet_match.group(1).strip()
        if not content or content.startswith("*"):
            continue

        node_id = f"ref_{hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:10]}"
        category = current_section.lower().replace(" & ", "_").replace(" ", "_")

        # Detect if this is a preference
        is_preference = (
            "preference" in current_section.lower()
            or "opinion" in current_section.lower()
        )

        if is_preference:
            engine.add_node(
                node_id,
                "preference",
                {
                    "name": content[:80],
                    "value": content,
                    "category": category,
                    "description": content,
                    "importance_score": 0.7,
                },
            )
        else:
            engine.add_node(
                node_id,
                "reflection",
                {
                    "name": content[:80],
                    "content": content,
                    "category": category,
                    "description": content,
                    "confidence": 0.8,
                    "importance_score": 0.6,
                },
            )
        count += 1

    logger.info("[KG-2.10] Persisted %d reflection entries", count)
    return count


def _stamp_timestamps(text: str, updated: str) -> str:
    """Ensure reflections have correct timestamp lines."""
    updated_line = f"*Last updated: {updated}*"
    if "*Last updated:" in text:
        text = re.sub(r"\*Last updated:.*?\*", updated_line, text, count=1)
    else:
        title_match = re.match(r"(#[^\n]*\n)", text)
        if title_match:
            pos = title_match.end()
            text = text[:pos] + f"\n{updated_line}\n" + text[pos:]
    return text
