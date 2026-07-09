"""KG Auto-Ingest for Messaging Events (CONCEPT:AU-ECO.messaging.native-backend-abstraction + KG-2.1).

Auto-ingests inbound and outbound messages into the Knowledge Graph as
``ChatMessage`` memory nodes with Ebbinghaus decay. This creates a
searchable, cross-platform conversational memory that agents can query
via ``recall_memory()``.

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction
CONCEPT:AU-KG.memory.tiered-memory-caching — Tiered Memory & Context
CONCEPT:AU-KG.memory.auto-similarity-memory-graph — Auto-Similarity Memory Graph

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

    CONCEPT:AU-ECO.messaging.native-backend-abstraction + CONCEPT:AU-KG.memory.tiered-memory-caching

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
            logger.debug(
                "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] No KG engine available, skipping ingest."
            )
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
        # CONCEPT:AU-ECO.messaging.blocking-store-memory — store_memory is a BLOCKING call (graph write + embedding).
        # Run it off the event loop so ingest never stalls the messaging/reply loop.
        # CONCEPT:AU-ECO.messaging.episodic-memory-recall — this is episodic memory the universal recall reads on demand;
        # reply continuity comes from the per-session conversation memento, not a bespoke
        # per-channel history query, so no flat channel_key/role/text scaffolding is stamped.
        memory_id = await asyncio.to_thread(
            engine.store_memory,
            content=memory_content,
            memory_type="episodic",
            name=f"Chat: {user} on {platform}",
            tags=tags,
            trust_score=0.7,
            agent_id=agent_id,
        )

        logger.debug(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Ingested message to KG: %s (platform=%s, user=%s)",
            memory_id,
            platform,
            user,
        )
        return memory_id

    except Exception as e:
        logger.warning(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Failed to ingest message to KG: %s",
            e,
        )
        return None


async def ingest_outbound_to_kg(
    message: Message,
    knowledge_engine: Any = None,
    agent_id: str = "messaging_router",
) -> str | None:
    """Ingest an outbound message into the KG for full conversation tracking.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction + CONCEPT:AU-KG.memory.tiered-memory-caching

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
        )

        logger.debug(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Ingested outbound to KG: %s (platform=%s)",
            memory_id,
            platform,
        )
        return memory_id

    except Exception as e:
        logger.warning(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Failed to ingest outbound to KG: %s",
            e,
        )
        return None


def _get_default_engine() -> Any | None:
    """Return the live served engine (the same one ``gateway/daemon.py`` uses).

    CONCEPT:AU-ECO.messaging.native-backend-abstraction — bind to ``IntelligenceGraphEngine.get_active()`` so ingested
    messages land in the running graph rather than a throwaway side database.

    Returns:
        The active IntelligenceGraphEngine instance, or None if none is running.
    """
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Default engine load failed: %s",
            e,
        )
        return None
