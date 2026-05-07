#!/usr/bin/python
"""Cross-Session Chat Recall — Facade Module.

CONCEPT:KG-2.13 — Cross-Session Chat Recall

This module provides a dedicated entry point for cross-session chat search
functionality.  It re-exports the keyword-based search implementation from
:mod:`agent_utilities.core.chat_persistence` so that downstream consumers
(including the ``overview.md`` conceptual registry) can import directly
from the ``knowledge_graph`` namespace::

    from agent_utilities.knowledge_graph.chat_search import (
        search_chat_history,
        ChatSearchResult,
    )

The underlying implementation uses the KG Cypher backend to query stored
``Thread`` and ``Message`` nodes, group results by session, and compute
keyword-hit-density relevance scores.

See Also:
    - :func:`agent_utilities.core.chat_persistence.search_chat_history`
    - :class:`agent_utilities.core.chat_persistence.ChatRecallResult`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from agent_utilities.core.chat_persistence import (
    ChatRecallMessage,
    ChatRecallResult,
    ChatRecallResults,
    search_chat_history,
)


@dataclass
class ChatSearchResult:
    """Simplified result container for chat search consumers.

    CONCEPT:KG-2.13 — Cross-Session Chat Recall

    Adapts :class:`ChatRecallResult` into a simpler flat dataclass for
    callers that do not need the full Pydantic model semantics.

    Attributes:
        session_id: The chat session identifier.
        session_title: Title or first message of the session.
        snippet: Best-matching message content preview (first 200 chars).
        relevance: Keyword-hit-density relevance score (0.0–1.0).
        last_activity: ISO timestamp of most recent message in the session.
        match_count: Number of messages that matched the search query.
    """

    session_id: str = ""
    session_title: str = ""
    snippet: str = ""
    relevance: float = 0.0
    last_activity: str = ""
    match_count: int = 0


def search_sessions(
    query: str,
    *,
    max_results: int = 20,
    date_from: datetime | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    exclude_session: str | None = None,
) -> list[ChatSearchResult]:
    """Search across stored chat sessions and return simplified results.

    CONCEPT:KG-2.13 — Cross-Session Chat Recall

    A convenience wrapper around :func:`search_chat_history` that returns
    :class:`ChatSearchResult` dataclass instances instead of the raw
    Pydantic models.

    Args:
        query: Search query (keywords separated by spaces).
        max_results: Maximum number of sessions to return.
        date_from: Only return messages after this datetime.
        date_after: ISO date string — only return messages after this date.
        date_before: ISO date string — only return messages before this date.
        exclude_session: Session ID to exclude from results.

    Returns:
        List of :class:`ChatSearchResult` ordered by relevance.

    Example::

        results = search_sessions("kubernetes deployment error")
        for r in results:
            print(f"Session {r.session_id}: {r.match_count} matches, "
                  f"relevance={r.relevance:.2f}")
    """
    after = date_after
    if date_from and not after:
        after = date_from.isoformat()

    recall = search_chat_history(
        query=query,
        limit=max_results * 5,  # over-fetch messages to cover max_results sessions
        after_date=after,
        before_date=date_before,
        exclude_session_id=exclude_session,
    )

    results: list[ChatSearchResult] = []
    for session in recall.results[:max_results]:
        snippet = ""
        if session.messages:
            snippet = session.messages[0].content[:200]

        results.append(
            ChatSearchResult(
                session_id=session.session_id,
                session_title=session.session_title,
                snippet=snippet,
                relevance=session.relevance_score,
                last_activity=session.last_activity,
                match_count=len(session.messages),
            )
        )

    return results


__all__ = [
    # Re-exports from chat_persistence
    "ChatRecallMessage",
    "ChatRecallResult",
    "ChatRecallResults",
    "search_chat_history",
    # Local additions
    "ChatSearchResult",
    "search_sessions",
]
