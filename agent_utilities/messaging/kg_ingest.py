"""KG Auto-Ingest for Messaging Events (CONCEPT:ECO-4.0 + KG-2.1).

Auto-ingests inbound and outbound messages into the Knowledge Graph as
``ChatMessage`` memory nodes with Ebbinghaus decay. This creates a
searchable, cross-platform conversational memory that agents can query
via ``recall_memory()``.

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
CONCEPT:KG-2.1 — Tiered Memory & Context
CONCEPT:KG-2.3 — Auto-Similarity Memory Graph

See Also:
    - ``knowledge_graph/core/engine_memory.py`` for ``store_memory()``
    - ``models/knowledge_graph.py`` for ``MemoryDecayConfig``
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent_utilities.messaging.models import (
    InboundEvent,
    Message,
)

logger = logging.getLogger(__name__)


async def ingest_message_to_kg(
    event: InboundEvent,
    knowledge_engine: Any = None,
    agent_id: str = "messaging_router",
) -> str | None:
    """Ingest a messaging event into the Knowledge Graph as a memory node.

    CONCEPT:ECO-4.0 + CONCEPT:KG-2.1

    Creates a tiered memory node with:
    - ``memory_type="episodic"`` (conversation memories decay over time)
    - ``trust_score=0.7`` (user messages are trusted but not authoritative)
    - Tags for platform, channel, and user for filtered recall

    Args:
        event: The inbound messaging event.
        knowledge_engine: The ``IntelligenceGraphEngine`` instance. If None,
            attempts to load from the default workspace.
        agent_id: Agent identifier for provenance tracking.

    Returns:
        The memory node ID if ingested, None if skipped or failed.
    """
    if not event.content and not (event.message and event.message.content):
        return None

    content = event.content or (event.message.content if event.message else "")
    if not content or len(content) < 3:
        return None  # Skip trivial messages

    # Resolve engine
    engine = knowledge_engine
    if engine is None:
        try:
            engine = _get_default_engine()
        except Exception:
            logger.debug("[CONCEPT:ECO-4.0] No KG engine available, skipping ingest.")
            return None

    if engine is None:
        return None

    # Build memory content with rich metadata
    platform = str(event.platform)
    user = event.user_name or event.user_id or "unknown"
    channel = event.channel_id or "unknown"
    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    memory_content = (
        f"[{platform.upper()}] Message from {user} in #{channel}:\n{content}"
    )

    tags = [
        f"platform:{platform}",
        f"channel:{channel}",
        f"user:{user}",
        "messaging",
        "conversation",
    ]

    if event.thread_id:
        tags.append(f"thread:{event.thread_id}")

    try:
        # CONCEPT:ECO-4.72 — store_memory is a BLOCKING call (graph write + embedding).
        # Run it off the event loop so ingest never stalls the messaging/reply loop.
        # CONCEPT:ECO-4.76 — stamp a flat, indexable channel_key + role so the reply
        # path can recall the last N turns for THIS channel via a cheap exact-match
        # query (recall_recent_messages) instead of the heavy semantic recall_memory.
        memory_id = await asyncio.to_thread(
            engine.store_memory,
            content=memory_content,
            memory_type="episodic",
            name=f"Chat: {user} on {platform}",
            tags=tags,
            trust_score=0.7,
            agent_id=agent_id,
            extra_props={
                "channel_key": f"{platform}:{channel}",
                "chat_role": "user",
                "chat_user": user,
                "chat_text": content,
            },
        )

        logger.debug(
            "[CONCEPT:ECO-4.0] Ingested message to KG: %s (platform=%s, user=%s)",
            memory_id,
            platform,
            user,
        )
        return memory_id

    except Exception as e:
        logger.warning("[CONCEPT:ECO-4.0] Failed to ingest message to KG: %s", e)
        return None


async def ingest_outbound_to_kg(
    message: Message,
    knowledge_engine: Any = None,
    agent_id: str = "messaging_router",
) -> str | None:
    """Ingest an outbound message into the KG for full conversation tracking.

    CONCEPT:ECO-4.0 + CONCEPT:KG-2.1

    Stores agent-sent messages as semantic memories (longer half-life
    than episodic) for future context retrieval.

    Args:
        message: The outbound ``Message`` being sent.
        knowledge_engine: The ``IntelligenceGraphEngine`` instance.
        agent_id: Agent identifier for provenance.

    Returns:
        Memory node ID if ingested, None otherwise.
    """
    if not message.content:
        return None

    engine = knowledge_engine
    if engine is None:
        try:
            engine = _get_default_engine()
        except Exception:
            return None

    if engine is None:
        return None

    platform = str(message.platform)
    channel = message.channel_id or "unknown"

    memory_content = (
        f"[{platform.upper()}] Agent response in #{channel}:\n{message.content}"
    )

    tags = [
        f"platform:{platform}",
        f"channel:{channel}",
        "messaging",
        "agent_response",
    ]

    try:
        memory_id = engine.store_memory(
            content=memory_content,
            memory_type="semantic",  # Agent responses have longer half-life
            name=f"Agent response on {platform}",
            tags=tags,
            trust_score=0.9,  # Agent's own output is highly trusted
            agent_id=agent_id,
            extra_props={
                "channel_key": f"{platform}:{channel}",
                "chat_role": "assistant",
                "chat_text": message.content,
            },
        )

        logger.debug(
            "[CONCEPT:ECO-4.0] Ingested outbound to KG: %s (platform=%s)",
            memory_id,
            platform,
        )
        return memory_id

    except Exception as e:
        logger.warning("[CONCEPT:ECO-4.0] Failed to ingest outbound to KG: %s", e)
        return None


def recall_recent_messages(
    platform: str,
    channel_id: str,
    limit: int = 8,
    knowledge_engine: Any = None,
) -> list[dict[str, str]]:
    """Recall the last ``limit`` chat turns for ONE channel, most-recent-last.

    CONCEPT:ECO-4.76 — bounded, FAST conversation history for reply continuity. This is a
    cheap exact-match recency query on the flat ``channel_key`` scalar stamped at ingest
    (``ingest_message_to_kg`` / ``ingest_outbound_to_kg``) — NOT the heavy semantic
    ``recall_memory`` (HNSW + cross-encoder) that was removed from the reply path because it
    blocked replies. Both user and assistant turns are returned so the agent has continuity.

    Args:
        platform: The messaging platform (e.g. ``telegram``).
        channel_id: The channel/chat id within that platform.
        limit: Max number of turns to return (most recent ``limit``).
        knowledge_engine: The live engine; falls back to the active served engine.

    Returns:
        A chronological (oldest→newest) list of ``{"role": ..., "text": ...}`` dicts.
        Empty on any failure — never raises, so the caller can proceed without history.
    """
    if not platform or not channel_id or limit <= 0:
        return []
    engine = knowledge_engine
    if engine is None:
        try:
            engine = _get_default_engine()
        except Exception:  # noqa: BLE001
            return []
    query = getattr(engine, "query_cypher", None)
    if not callable(query):
        return []
    channel_key = f"{platform}:{channel_id}"
    # Exact-match scalar filter (portable across backends) + timestamp ordering done in
    # Python (some backends — e.g. AGE/L1 — drop ORDER BY). Over-fetch a bounded window,
    # then sort + slice locally.
    try:
        rows = query(
            "MATCH (m:Memory {channel_key: $ck}) "
            "RETURN m.chat_role AS role, m.chat_text AS text, m.timestamp AS ts "
            "LIMIT 200",
            {"ck": channel_key},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.76] history recall failed: %s", exc)
        return []

    turns: list[tuple[str, str, str]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not text:
            continue
        role = str(row.get("role") or "user")
        ts = str(row.get("ts") or "")
        turns.append((ts, role, str(text)))
    # Most recent ``limit`` by timestamp, returned oldest→newest for natural reading.
    turns.sort(key=lambda t: t[0])
    recent = turns[-limit:]
    return [{"role": r, "text": t} for _ts, r, t in recent]


def _get_default_engine() -> Any | None:
    """Return the live served engine (the same one ``gateway/daemon.py`` uses).

    CONCEPT:ECO-4.0 — bind to ``IntelligenceGraphEngine.get_active()`` so ingested
    messages land in the running graph rather than a throwaway side database.

    Returns:
        The active IntelligenceGraphEngine instance, or None if none is running.
    """
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.0] Default engine load failed: %s", e)
        return None
