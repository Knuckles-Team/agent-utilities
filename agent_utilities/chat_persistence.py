#!/usr/bin/python
# coding: utf-8
"""Chat Persistence Module.

This module handles the serialization and retrieval of chat histories from
the workspace. It provides utilities for saving messages to disk, pruning
large tool outputs to manage context window usage, and listing/deleting
stored conversations.
"""

from __future__ import annotations

import sys
import logging

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass


from pydantic_ai import Agent


from .config import *  # noqa: F403

logger = logging.getLogger(__name__)


def save_chat_to_disk(chat_id: str, messages: List[Dict[str, Any]]):
    """Save a chat conversation to the Knowledge Graph."""
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        logger.warning("Graph backend not available for chat persistence.")
        return

    ts = datetime.now().isoformat()
    # Create or update ThreadNode
    query_thread = """
    MERGE (t:Thread {id: $id})
    SET t.title = $title,
        t.created_at = $ts
    """
    first_msg = ""
    if messages:
        first_msg = str(messages[0].get("content", ""))[:100]

    engine.backend.execute(
        query_thread,
        {"id": chat_id, "title": first_msg or "Untitled Chat", "ts": ts},
    )

    # Create MessageNodes and link to Thread
    for i, msg in enumerate(messages):
        msg_id = f"msg:{chat_id}:{i}"
        query_msg = """
        MERGE (m:Message {id: $id})
        SET m.role = $role,
            m.content = $content,
            m.timestamp = $ts
        WITH m
        MATCH (t:Thread {id: $chat_id})
        MERGE (t)-[:CONTAINS]->(m)
        """
        engine.backend.execute(
            query_msg,
            {
                "id": msg_id,
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", "")),
                "ts": ts,
                "chat_id": chat_id,
            },
        )

    logger.debug(f"Saved chat {chat_id} to Knowledge Graph")


def list_chats_from_disk() -> List[Dict[str, Any]]:
    """List all chats stored in the Knowledge Graph."""
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return []

    try:
        res = engine.backend.execute(
            "MATCH (t:Thread) RETURN t.id, t.title, t.created_at ORDER BY t.created_at DESC"
        )
        chats = []
        for row in res:
            chats.append(
                {
                    "id": row.get("t.id", ""),
                    "firstMessage": row.get("t.title", ""),
                    "timestamp": row.get("t.created_at", ""),
                }
            )
        return chats
    except Exception as e:
        logger.debug(f"Failed to list chats from graph: {e}")
        return []


def get_chat_from_disk(chat_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific chat from the Knowledge Graph."""
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return None

    try:
        res_thread = engine.backend.execute(
            "MATCH (t:Thread {id: $id}) RETURN t.id, t.created_at", {"id": chat_id}
        )
        if not res_thread:
            return None

        thread = res_thread[0]
        res_msgs = engine.backend.execute(
            "MATCH (t:Thread {id: $id})-[:CONTAINS]->(m:Message) RETURN m.role, m.content, m.timestamp ORDER BY m.id",
            {"id": chat_id},
        )
        messages = []
        for rm in res_msgs:
            messages.append(
                {
                    "role": rm.get("m.role", ""),
                    "content": rm.get("m.content", ""),
                    "timestamp": rm.get("m.timestamp", ""),
                }
            )

        return {
            "id": thread.get("t.id"),
            "timestamp": thread.get("t.created_at"),
            "messages": messages,
        }
    except Exception as e:
        logger.error(f"Error reading chat {chat_id} from graph: {e}")
        return None


def delete_chat_from_disk(chat_id: str) -> bool:
    """Delete a chat and its messages from the Knowledge Graph."""
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return False

    try:
        # Detach and delete thread and its contained messages
        engine.backend.execute(
            "MATCH (t:Thread {id: $id})-[:CONTAINS]->(m:Message) DETACH DELETE m",
            {"id": chat_id},
        )
        engine.backend.execute(
            "MATCH (t:Thread {id: $id}) DETACH DELETE t", {"id": chat_id}
        )
        return True
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id} from graph: {e}")
        return False


def prune_large_messages(messages: list[Any], max_length: int = 5000) -> list[Any]:
    """Summarize large tool outputs in the message history to save context window.
    Keeps the most recent tool outputs intact if they are the very last message,
    but generally we want to prune history.
    """
    pruned_messages = []
    for i, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if isinstance(content, str) and len(content) > max_length:
            summary = (
                f"{content[:200]} ... "
                f"[Output truncated, original length {len(content)} characters] "
                f"... {content[-200:]}"
            )

            if isinstance(msg, dict):
                msg["content"] = summary
                pruned_messages.append(msg)
            elif hasattr(msg, "content"):
                try:
                    from copy import copy

                    new_msg = copy(msg)
                    new_msg.content = summary
                    pruned_messages.append(new_msg)
                except Exception:
                    pruned_messages.append(msg)
            else:
                pruned_messages.append(msg)
        else:
            pruned_messages.append(msg)

    return pruned_messages


async def chat(agent: Agent, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}", file=sys.stderr)


async def node_chat(agent: Agent, prompt: str) -> List:
    nodes = []
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            nodes.append(node)
            print(node, file=sys.stderr)
    return nodes


async def stream_chat(agent: Agent, prompt: str) -> None:
    async with agent.run_stream(prompt) as result:
        async for text_chunk in result.stream_text(delta=True):
            print(text_chunk, end="", flush=True, file=sys.stderr)
        print("\nDone!", file=sys.stderr)
