#!/usr/bin/python
from __future__ import annotations

"""Chat Persistence Module.

This module handles the serialization and retrieval of chat histories from
the workspace. It provides utilities for saving messages to disk, pruning
large tool outputs to manage context window usage, and listing/deleting
stored conversations.
"""


import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from pydantic import BaseModel, ConfigDict, Field

from .config import *  # noqa: F403

logger = logging.getLogger(__name__)


def save_chat_to_disk(chat_id: str, messages: list[dict[str, Any]]):
    """Save a chat conversation to the Knowledge Graph.

    BUG-033/BUG-039: this write path is a legitimate background writer (the
    cron scheduler persists a task's transcript with no request-bound actor
    in scope) as well as a request-triggered one (the WebUI's ``save_chat``
    helper). Either way it must run under a REAL, verified actor -- never
    silently proceed with none -- so the resulting ``Thread``/``Message``
    nodes are never left unowned by accident.
    :func:`~agent_utilities.security.request_identity.system_write_session`
    resolves the caller's own ambient session when one is already bound, and
    otherwise mints this process's own verified system identity through the
    existing process-authority convention (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.security.request_identity import system_write_session

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        logger.warning("Graph backend not available for chat persistence.")
        return

    session = system_write_session()

    with use_actor(session.actor), use_session(session):
        from agent_utilities.knowledge_graph.core.tenant_sharing import (
            stamp_classification,
            stamp_ownership,
        )

        ts = datetime.now().isoformat()
        first_msg = ""
        if messages:
            first_msg = str(messages[0].get("content", ""))[:100]

        # This MERGE is raw Cypher (the engine's typed ``add_node`` seam
        # doesn't fit its "keep the id stable, overwrite title/created_at"
        # upsert shape), so it bypasses the generic ``_upsert_node``
        # ownership-stamping seam entirely. Stamp explicitly here so the
        # Thread node carries the same governance properties every other
        # write path gets for free. A failed stamp under a just-bound,
        # verified actor is a real defect, not a best-effort skip -- let it
        # raise.
        governance_props: dict[str, Any] = {}
        stamp_ownership(governance_props)
        stamp_classification(governance_props, "Thread")

        thread_params: dict[str, Any] = {
            "id": chat_id,
            "title": first_msg or "Untitled Chat",
            "ts": ts,
            **governance_props,
        }
        # ``created_at`` is the stored property name; ``ts`` is only the bound
        # parameter name (kept from the original query so no behavior changes).
        set_clause = ", ".join(
            ["t.title = $title", "t.created_at = $ts"]
            + [f"t.{key} = ${key}" for key in governance_props]
        )
        engine.backend.execute(
            f"MERGE (t:Thread {{id: $id}}) SET {set_clause}",
            thread_params,
        )

        # Clean up orphan messages (for rollback strategy)
        engine.backend.execute(
            "MATCH (t:Thread {id: $id})-[:CONTAINS]->(m:Message) "
            "WHERE toInteger(split(m.id, ':')[2]) >= $max_len "
            "DETACH DELETE m",
            {"id": chat_id, "max_len": len(messages)},
        )

        # Create MessageNodes and link to Thread. The engine's native Cypher
        # write subset supports only one leading MATCH and no WITH between write
        # clauses (epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184), so
        # this can't be one "MERGE ... WITH ... MATCH ... MERGE" statement; the
        # typed engine API (CONCEPT:AU-KG.query.register-each-user-table) does the
        # node upsert + edge merge as two dispatched, backend-appropriate writes.
        for i, msg in enumerate(messages):
            msg_id = f"msg:{chat_id}:{i}"
            engine.add_node(
                msg_id,
                "Message",
                {
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", "")),
                    "timestamp": ts,
                },
            )
            engine.link_nodes(chat_id, msg_id, "CONTAINS")

    logger.debug(f"Saved chat {chat_id} to Knowledge Graph")


def list_chats_from_disk() -> list[dict[str, Any]]:
    """List all chats stored in the Knowledge Graph."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

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
    except Exception as e:  # noqa: BLE001 — best-effort read: on KG failure the UI/API (server/routers/core.py) shows an empty chat list rather than a 500; no state is written here, and a transient failure self-heals on the next poll
        logger.debug(f"Failed to list chats from graph: {e}")
        return []


def get_chat_from_disk(chat_id: str) -> dict[str, Any] | None:
    """Retrieve a specific chat from the Knowledge Graph."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

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
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

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


# ---------------------------------------------------------------------------
# Cross-Session Chat Recall (adapted from Goose's ChatHistorySearch)
# ---------------------------------------------------------------------------


class ChatRecallMessage(BaseModel):
    """A single message from a recalled chat session."""

    model_config = ConfigDict(from_attributes=True)

    role: str = ""
    content: str = ""
    timestamp: str = ""


class ChatRecallResult(BaseModel):
    """A recalled chat session with matching messages.

    Attributes:
        session_id: The chat session identifier.
        session_title: Title or first message of the session.
        last_activity: ISO timestamp of the most recent message.
        total_messages_in_session: Total messages in the full session.
        messages: Matched messages from this session.
        relevance_score: Relevance score from the retriever (0.0–1.0).
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: str = ""
    session_title: str = ""
    last_activity: str = ""
    total_messages_in_session: int = 0
    messages: list[ChatRecallMessage] = Field(default_factory=list)
    relevance_score: float = 0.0


class ChatRecallResults(BaseModel):
    """Aggregate results from a cross-session chat recall search.

    Attributes:
        results: List of recalled sessions with matching messages.
        total_matches: Total number of matching messages across all sessions.
        query: The original search query.
    """

    results: list[ChatRecallResult] = Field(default_factory=list)
    total_matches: int = 0
    query: str = ""


def search_chat_history(
    query: str,
    limit: int = 10,
    after_date: str | None = None,
    before_date: str | None = None,
    exclude_session_id: str | None = None,
) -> ChatRecallResults:
    """Search across stored chat sessions for matching messages.

    CONCEPT:AU-KG.memory.tiered-memory-caching — Cross-Session Chat Recall

    Adapted from Goose's ``ChatHistorySearch`` (Rust/SQLite). Uses
    the Knowledge Graph's Cypher backend for keyword-based search
    across ``Thread`` and ``Message`` nodes.

    Args:
        query: Search query (keywords separated by spaces).
        limit: Maximum number of matching messages to return.
        after_date: ISO date string — only return messages after this date.
        before_date: ISO date string — only return messages before this date.
        exclude_session_id: Session ID to exclude from results.

    Returns:
        ChatRecallResults with matching sessions and messages.

    Example::

        results = search_chat_history("kubernetes deployment error")
        for r in results.results:
            print(f"Session {r.session_id}: {len(r.messages)} matches")
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return ChatRecallResults(query=query)

    keywords = [w.strip() for w in query.lower().split() if w.strip()]
    if not keywords:
        return ChatRecallResults(query=query)

    try:
        # Build Cypher query for keyword matching
        keyword_conditions = " OR ".join(
            f"toLower(m.content) CONTAINS '{kw}'" for kw in keywords
        )

        cypher = (
            f"MATCH (t:Thread)-[:CONTAINS]->(m:Message) WHERE ({keyword_conditions})"
        )

        if exclude_session_id:
            cypher += f" AND t.id <> '{exclude_session_id}'"
        if after_date:
            cypher += f" AND m.timestamp >= '{after_date}'"
        if before_date:
            cypher += f" AND m.timestamp <= '{before_date}'"

        cypher += (
            " RETURN t.id AS session_id, t.title AS session_title, "
            "t.created_at AS created_at, "
            "m.role AS role, m.content AS content, m.timestamp AS timestamp "
            f"ORDER BY m.timestamp DESC LIMIT {limit}"
        )

        rows = engine.backend.execute(cypher)

        # Group messages by session
        sessions: dict[str, ChatRecallResult] = {}
        for row in rows:
            sid = row.get("session_id", "")
            if not sid:
                continue

            if sid not in sessions:
                sessions[sid] = ChatRecallResult(
                    session_id=sid,
                    session_title=row.get("session_title", ""),
                    last_activity=row.get("timestamp", ""),
                )

            sessions[sid].messages.append(
                ChatRecallMessage(
                    role=row.get("role", ""),
                    content=row.get("content", ""),
                    timestamp=row.get("timestamp", ""),
                )
            )

            # Update last_activity to most recent
            ts = row.get("timestamp", "")
            if ts > sessions[sid].last_activity:
                sessions[sid].last_activity = ts

        # Calculate relevance scores based on keyword hit density
        for result in sessions.values():
            total_hits = 0
            total_words = 0
            for msg in result.messages:
                words = msg.content.lower().split()
                total_words += len(words) if words else 1
                total_hits += sum(1 for w in words for kw in keywords if kw in w)
            result.relevance_score = min(1.0, total_hits / max(total_words, 1) * 10)
            result.total_messages_in_session = len(result.messages)

        # Sort by relevance
        sorted_results = sorted(
            sessions.values(),
            key=lambda r: r.relevance_score,
            reverse=True,
        )

        total_matches = sum(len(r.messages) for r in sorted_results)

        return ChatRecallResults(
            results=sorted_results,
            total_matches=total_matches,
            query=query,
        )

    except Exception as exc:  # noqa: BLE001 — best-effort read: chat_search.py's convenience wrapper treats an empty ChatRecallResults as "no matches"; nothing is written on this path
        logger.debug("Chat history search failed: %s", exc)
        return ChatRecallResults(query=query)


async def chat(agent: Any, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}", file=sys.stderr)


async def node_chat(agent: Any, prompt: str) -> list:
    nodes = []
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            nodes.append(node)
            print(node, file=sys.stderr)
    return nodes


async def stream_chat(agent: Any, prompt: str) -> None:
    async with agent.run_stream(prompt) as result:
        async for text_chunk in result.stream_text(delta=True):
            print(text_chunk, end="", flush=True, file=sys.stderr)
        print("\nDone!", file=sys.stderr)
