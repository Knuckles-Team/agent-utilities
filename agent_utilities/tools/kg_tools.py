"""First-class Knowledge Graph tools for every agent (CONCEPT:ECO-4.61).

Attached by default to every ``create_agent`` agent so the ONE shared OWL/RDF Knowledge
Graph is a first-class query + reason layer — the agent can search, recall, and query the
same live engine that every other agent shares. This is what makes an agent feel like an
"LLM wrapped around the knowledge graph": it answers from the graph, on demand, without
pre-loading everything into context. Opt out with ``AGENT_KG_TOOLS=0``.

NOTE: using these requires a tool-capable model — a model that can't emit tool calls can't
query the KG (it only gets passively-recalled context).

CONCEPT:ECO-4.61 — KG as a first-class shared knowledge/reasoning layer for all agents
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic_ai import RunContext

logger = logging.getLogger(__name__)

# Cypher that mutates — kg_query is strictly read-only.
_WRITE_CYPHER = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|LOAD\s+CSV|CALL\s+db\.)\b", re.I
)


def _active_engine() -> Any:
    """The one shared live engine (None if unavailable)."""
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ECO-4.61] no active engine: %s", exc)
        return None


def _fmt_result(r: Any) -> dict[str, Any]:
    """Best-effort compact view of a search result (objects or dicts)."""
    if isinstance(r, dict):
        return {
            k: r.get(k)
            for k in ("id", "name", "description", "score", "type")
            if r.get(k) is not None
        }
    return {
        k: getattr(r, k)
        for k in ("id", "name", "description", "score", "type")
        if getattr(r, k, None) is not None
    }


async def kg_search(ctx: RunContext[Any], query: str, top_k: int = 8) -> str:
    """Search the shared Knowledge Graph (semantic + keyword hybrid). CONCEPT:ECO-4.61

    Use this to ground answers in what the system actually knows — concepts, code, services,
    research, entities — instead of guessing.

    Args:
        ctx: Agent run context.
        query: What to look for.
        top_k: Max results (default 8).

    Returns:
        JSON list of matched nodes (id/name/description/score).
    """
    engine = _active_engine()
    if engine is None:
        return "Knowledge Graph engine not available."
    try:
        results = engine.search_hybrid(query=query, top_k=top_k)
        return json.dumps([_fmt_result(r) for r in (results or [])], default=str)[:4000]
    except Exception as e:  # noqa: BLE001
        return f"kg_search error: {e}"


async def kg_recall(ctx: RunContext[Any], query: str, top_k: int = 5) -> str:
    """Recall relevant memories (episodic/semantic/procedural) from the KG. CONCEPT:ECO-4.61

    Args:
        ctx: Agent run context.
        query: The topic to recall context about.
        top_k: Max memories (default 5).

    Returns:
        JSON list of recalled memories.
    """
    engine = _active_engine()
    if engine is None:
        return "Knowledge Graph engine not available."
    try:
        memories = engine.recall_memory(query=query, top_k=top_k)
        return json.dumps(memories or [], default=str)[:4000]
    except Exception as e:  # noqa: BLE001
        return f"kg_recall error: {e}"


async def kg_query(ctx: RunContext[Any], cypher: str) -> str:
    """Run a READ-ONLY Cypher query against the shared Knowledge Graph. CONCEPT:ECO-4.61

    For precise structured questions (counts, relationships, filters). Writes are rejected.

    Args:
        ctx: Agent run context.
        cypher: A read-only Cypher MATCH/RETURN query.

    Returns:
        JSON rows, or an error string.
    """
    if _WRITE_CYPHER.search(cypher or ""):
        return (
            "kg_query is read-only — use MATCH/RETURN (no CREATE/MERGE/DELETE/SET/...)."
        )
    engine = _active_engine()
    if engine is None:
        return "Knowledge Graph engine not available."
    try:
        rows = engine.query_cypher(cypher, {})
        return json.dumps(rows or [], default=str)[:4000]
    except Exception as e:  # noqa: BLE001
        return f"kg_query error: {e}"


# Curated first-class KG toolset attached to every agent by default.
kg_tools = [kg_search, kg_recall, kg_query]
