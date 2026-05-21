#!/usr/bin/python
from __future__ import annotations

"""LLM-Powered Transcript Observer.

CONCEPT:KG-2.1 -- Observational Memory Bridge

Compresses agent session transcripts into structured observations using LLM,
then persists them as ObservationNode entries in the Knowledge Graph.

Pipeline: Raw transcript -> LLM Observer -> ObservationNode (KG) -> materialize()
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

OBSERVER_SYSTEM_PROMPT = """You are the Observer for the agent-utilities Knowledge Graph.
Your job is to compress conversation transcripts into dense, prioritized observation notes.

## Output Format

Use dated sections with priority markers:
- \U0001f534 CRITICAL: Decisions made, preferences stated, corrections issued
- \U0001f7e1 IMPORTANT: Useful context, project status, approach changes
- \U0001f7e2 NORMAL: Nice-to-know facts, background context

## Rules
1. Extract ONLY new information not already in existing observations
2. Preserve exact technical details (names, paths, versions, commands)
3. Attribute observations to the source agent when known
4. Use bullet points, keep each observation to 1-2 lines
5. Output ONLY the new observations section in markdown format
6. Start with ## YYYY-MM-DD header for today's date

## Example Output
## 2026-05-17
- \U0001f534 User decided to use KG-first architecture for memory bridge
- \U0001f7e1 Codex session focused on fixing CI pipeline for agent-utilities
- \U0001f7e2 User mentioned preferring dark mode in terminal tools
"""


def observe_transcript(
    engine: IntelligenceGraphEngine,
    messages: list[dict[str, str]],
    *,
    source: str = "unknown",
    dry_run: bool = False,
) -> str | None:
    """Compress transcript messages into observations and persist to KG.

    Args:
        engine: IntelligenceGraphEngine instance.
        messages: List of dicts with 'role' and 'content' keys.
        source: Source agent name (claude, codex, grok, etc.).
        dry_run: If True, return observations without persisting.

    Returns:
        The new observations text, or None if below threshold.
    """
    if len(messages) < 3:
        return None

    # Format transcript for LLM
    transcript_lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:2000]
        ts = msg.get("timestamp", "")[:19]
        transcript_lines.append(f"[{ts}] {role} [{source}]: {content}")
    transcript_text = "\n\n".join(transcript_lines)

    # Call LLM to extract observations
    try:
        from pydantic_ai import Agent

        from ...core.config import DEFAULT_KG_MODEL_ID, DEFAULT_LLM_PROVIDER
        from ...core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = Agent(model, system_prompt=OBSERVER_SYSTEM_PROMPT)

        import nest_asyncio

        nest_asyncio.apply()

        user_content = f"## New transcript to process\n\n{transcript_text}"
        result = agent.run_sync(user_content)
        observations_text = str(result.data)
    except Exception as e:
        logger.warning("Observer LLM call failed: %s", e)
        return None

    if dry_run:
        return observations_text

    # Parse and persist observations as KG nodes
    _persist_observations(engine, observations_text, source=source)

    # Trigger Memento compression for the block as well
    if len(messages) >= 5:
        from .memento_compressor import compress_to_memento

        try:
            compress_to_memento(engine, messages, source=source, dry_run=dry_run)
        except Exception as e:
            logger.warning("Failed to compress Memento: %s", e)

    return observations_text


def observe_from_file(
    engine: IntelligenceGraphEngine,
    transcript_path: Path,
    *,
    source: str = "unknown",
    dry_run: bool = False,
) -> str | None:
    """Observe a transcript file (JSONL format).

    Supports Claude Code, Codex, and Grok Build JSONL transcript formats.
    Uses cursor-based incremental processing to avoid re-processing.
    """
    if not transcript_path.exists():
        return None

    # Load cursor
    cursor = _load_cursor(engine)
    cursor_key = str(transcript_path)
    after_line = cursor.get(cursor_key, 0)

    # Parse JSONL transcript
    messages = []
    line_count = 0
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        line_count += 1
        if line_count <= after_line:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            role = entry.get("role", entry.get("type", ""))
            content = entry.get("content", entry.get("message", ""))
            if role in ("user", "assistant", "human") and content:
                messages.append(
                    {
                        "role": role,
                        "content": content[:2000],
                        "timestamp": entry.get(
                            "timestamp", entry.get("created_at", "")
                        ),
                    }
                )
        except json.JSONDecodeError:
            continue

    if not messages:
        return None

    result = observe_transcript(engine, messages, source=source, dry_run=dry_run)

    if result and not dry_run:
        cursor[cursor_key] = line_count
        _save_cursor(engine, cursor)

    return result


def _persist_observations(
    engine: IntelligenceGraphEngine,
    observations_text: str,
    *,
    source: str = "unknown",
) -> int:
    """Parse observation markdown and create KG nodes."""
    import re

    count = 0
    current_date = time.strftime("%Y-%m-%d", time.gmtime())

    for line in observations_text.splitlines():
        date_match = re.match(r"^## (\d{4}-\d{2}-\d{2})$", line)
        if date_match:
            current_date = date_match.group(1)
            continue

        obs_match = re.match(r"^- ([\U0001f534\U0001f7e1\U0001f7e2]) (.+)$", line)
        if obs_match:
            emoji = obs_match.group(1)
            content = obs_match.group(2).strip()
            priority = {"\\U0001f534": "critical", "\U0001f7e1": "important"}.get(
                emoji, "normal"
            )

            obs_id = f"obs_{hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:10]}"
            engine.add_node(
                obs_id,
                "observation",
                {
                    "name": content[:80],
                    "content": content,
                    "description": content,
                    "priority": priority,
                    "source": source,
                    "timestamp": f"{current_date}T{time.strftime('%H:%M:%SZ', time.gmtime())}",
                    "importance_score": {"critical": 0.9, "important": 0.6}.get(
                        priority, 0.3
                    ),
                },
            )
            count += 1

    logger.info("[KG-2.10] Persisted %d observations from %s", count, source)
    return count


def _load_cursor(engine: IntelligenceGraphEngine) -> dict[str, Any]:
    """Load observer cursor from KG or fallback file."""
    from .memory_materializer import memory_dir

    cursor_path = memory_dir() / ".observer_cursor.json"
    if cursor_path.exists():
        try:
            return json.loads(cursor_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cursor(engine: IntelligenceGraphEngine, cursor: dict[str, Any]) -> None:
    """Save observer cursor."""
    from .memory_materializer import memory_dir

    cursor_path = memory_dir() / ".observer_cursor.json"
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
