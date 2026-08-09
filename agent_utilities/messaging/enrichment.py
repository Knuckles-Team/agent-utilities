"""Post-conversation KG enrichment for messaging (CONCEPT:AU-ECO.messaging.post-conversation-enrichment).

After a conversation turn, mine the chat for ``Concept`` nodes and link them into the shared
Knowledge Graph (``MENTIONS``), reusing the same extractor the IDE-conversation ingestion
uses (``extract_text_concepts`` + the lite LLM). This turns chat history into durable,
queryable graph knowledge that interweaves with code/docs/research — so what you discuss
makes the agent smarter over time, not just a transcript.

Runs in the background (off the reply path) and is best-effort. Disable with
``MESSAGING_ENRICH=0``.

CONCEPT:AU-ECO.messaging.post-conversation-enrichment — Post-conversation concept extraction + KG linking for chat
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return str(setting("MESSAGING_ENRICH", "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def enrich_conversation(
    engine: Any, text: str, *, platform: str, channel_id: str, title: str = ""
) -> int:
    """Extract concepts from a chat turn and link them into the KG. Returns concept count.

    CONCEPT:AU-ECO.messaging.post-conversation-enrichment — synchronous + best-effort; call via ``asyncio.to_thread`` so it never
    blocks the reply. Mirrors ``conversation_ingestion``'s per-thread concept write.

    BUG-033/BUG-039: runs per chat turn on every messaging platform (incl.
    Telegram) over real conversation content. Bind a REAL verified actor for
    the whole enrichment pass so the ``Thread``/``Concept``/``Goal``/``Spec``
    nodes it writes are never left unowned because nothing happened to be
    ambient in this particular thread/task. ``asyncio.to_thread`` (the
    caller, ``messaging/router.py``) copies the calling task's context, so
    when that context already carries a verified session (the messaging
    daemon/co-service bound one at startup) it is reused as-is; otherwise
    this mints the process's own system identity once.
    """
    if not _enabled() or engine is None or not (text and text.strip()):
        return 0
    add_node = getattr(engine, "add_node", None)
    link_nodes = getattr(engine, "link_nodes", None)
    if not callable(add_node) or not callable(link_nodes):
        return 0

    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.security.request_identity import system_write_session

    session = system_write_session()
    with use_actor(session.actor), use_session(session):
        return _enrich_conversation_body(
            engine,
            text,
            add_node=add_node,
            link_nodes=link_nodes,
            platform=platform,
            channel_id=channel_id,
            title=title,
        )


def _enrich_conversation_body(
    engine: Any,
    text: str,
    *,
    add_node: Any,
    link_nodes: Any,
    platform: str,
    channel_id: str,
    title: str,
) -> int:
    """Body of :func:`enrich_conversation`, run under a bound actor."""
    try:
        from agent_utilities.knowledge_graph.enrichment.cards import make_lite_llm_fn
        from agent_utilities.knowledge_graph.enrichment.extractors.text import (
            extract_text_concepts,
        )

        llm_fn = make_lite_llm_fn()
    except Exception as exc:  # noqa: BLE001 — best-effort background enrichment; import failure short-circuits to 0 concepts, caller discards the return value anyway
        logger.debug("[ECO-4.65] enrichment unavailable: %s", exc)
        return 0
    if llm_fn is None:
        return 0

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    source_id = f"chatturn:{platform}:{channel_id}:{digest}"
    try:
        add_node(
            node_id=source_id,
            node_type="Thread",
            properties={
                "id": source_id,
                "platform": platform,
                "channel_id": channel_id,
                "kind": "chat_turn",
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    except Exception as exc:  # noqa: BLE001 — best-effort chat-turn node write; extraction proceeds regardless, no downstream code checks this write succeeded
        logger.debug("[ECO-4.65] chat-turn node write failed: %s", exc)

    try:
        concepts, _edges = extract_text_concepts(
            text, source_id, llm_fn, source_type="chat", title=title or "chat"
        )
    except Exception as exc:  # noqa: BLE001 — best-effort concept extraction; failure returns 0, caller discards the count
        logger.debug("[ECO-4.65] concept extraction failed: %s", exc)
        return 0

    written = 0
    for c in concepts:
        try:
            add_node(
                node_id=c.id,
                node_type="Concept",
                properties={
                    "name": c.name,
                    "kind": c.kind,
                    "summary": c.summary,
                    "source_ids": json.dumps(c.source_ids),
                },
            )
            link_nodes(
                source_id=source_id,
                target_id=c.id,
                rel_type="MENTIONS",
                properties={"source": "chat"},
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 — best-effort per-concept KG write; skips this concept only, written count is informational-only
            logger.debug("[ECO-4.65] concept write failed: %s", exc)
    if written:
        logger.info(
            "[CONCEPT:AU-ECO.messaging.post-conversation-enrichment] Enriched chat turn on %s with %d concept(s).",
            platform,
            written,
        )

    # Surface goals / SDD-worthy specs from the turn (CONCEPT:AU-ECO.messaging.surfaced).
    _surface_intents(engine, text, source_id, llm_fn)
    return written


def _surface_intents(engine: Any, text: str, source_id: str, llm_fn: Any) -> None:
    """Detect a goal/SDD-worthy spec in the turn → surface as Goal/Spec nodes (ECO-4.70).

    Surfaces (does not auto-execute): a ``Goal`` or ``Spec`` node linked from the chat-turn
    so the loop engine / SDD tooling can pick it up. Best-effort + opt-out
    (``MESSAGING_GOALS=0``). Re-detection is idempotent (stable hashed node ids).
    """
    if str(setting("MESSAGING_GOALS", "1")).strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    prompt = (
        "From this chat, extract any concrete GOAL the user wants accomplished and any "
        "feature/spec worth a spec-driven-development plan. Reply JSON only: "
        '{"goal": "<one line or empty>", "spec": "<one line or empty>"}.\n\n'
        + text[:4000]
    )
    try:
        raw = llm_fn(prompt)
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception as exc:  # noqa: BLE001 — best-effort goal/spec detection; failure aborts surfacing for this turn, no state was written yet
        logger.debug("[ECO-4.70] intent detection failed: %s", exc)
        return
    for label, key in (("Goal", "goal"), ("Spec", "spec")):
        desc = str((data or {}).get(key, "") or "").strip()
        if not desc:
            continue
        digest = hashlib.sha256((source_id + desc).encode()).hexdigest()[:32]
        nid = f"{label.lower()}:{digest}"
        try:
            engine.add_node(
                node_id=nid,
                node_type=label,
                properties={
                    "id": nid,
                    "name": desc[:120],
                    "description": desc,
                    "status": "surfaced",
                    "origin": "chat",
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            engine.link_nodes(
                source_id=source_id,
                target_id=nid,
                rel_type="HAS_GOAL" if label == "Goal" else "PROPOSES_SPEC",
                properties={"source": "chat"},
            )
            logger.info(
                "[CONCEPT:AU-ECO.messaging.surfaced] Surfaced %s from chat: %s",
                label,
                desc[:60],
            )
        except Exception as exc:  # noqa: BLE001 — best-effort Goal/Spec surfacing; node ids are content-hashed so a later turn safely retries
            logger.debug("[ECO-4.70] %s surfacing failed: %s", label, exc)
