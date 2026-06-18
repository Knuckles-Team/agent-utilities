"""Post-conversation KG enrichment for messaging (CONCEPT:ECO-4.65).

After a conversation turn, mine the chat for ``Concept`` nodes and link them into the shared
Knowledge Graph (``MENTIONS``), reusing the same extractor the IDE-conversation ingestion
uses (``extract_text_concepts`` + the lite LLM). This turns chat history into durable,
queryable graph knowledge that interweaves with code/docs/research — so what you discuss
makes the agent smarter over time, not just a transcript.

Runs in the background (off the reply path) and is best-effort. Disable with
``MESSAGING_ENRICH=0``.

CONCEPT:ECO-4.65 — Post-conversation concept extraction + KG linking for chat
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
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

    CONCEPT:ECO-4.65 — synchronous + best-effort; call via ``asyncio.to_thread`` so it never
    blocks the reply. Mirrors ``conversation_ingestion``'s per-thread concept write.
    """
    if not _enabled() or engine is None or not (text and text.strip()):
        return 0
    add_node = getattr(engine, "add_node", None)
    link_nodes = getattr(engine, "link_nodes", None)
    if not callable(add_node) or not callable(link_nodes):
        return 0
    try:
        from agent_utilities.knowledge_graph.enrichment.cards import make_lite_llm_fn
        from agent_utilities.knowledge_graph.enrichment.extractors.text import (
            extract_text_concepts,
        )

        llm_fn = make_lite_llm_fn()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ECO-4.65] enrichment unavailable: %s", exc)
        return 0
    if llm_fn is None:
        return 0

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]  # noqa: S324 — id only
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
                "created_at": time.time(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ECO-4.65] chat-turn node write failed: %s", exc)

    try:
        concepts, _edges = extract_text_concepts(
            text, source_id, llm_fn, source_type="chat", title=title or "chat"
        )
    except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.65] concept write failed: %s", exc)
    if written:
        logger.info(
            "[CONCEPT:ECO-4.65] Enriched chat turn on %s with %d concept(s).",
            platform,
            written,
        )

    # Surface goals / SDD-worthy specs from the turn (CONCEPT:ECO-4.70).
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
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ECO-4.70] intent detection failed: %s", exc)
        return
    for label, key in (("Goal", "goal"), ("Spec", "spec")):
        desc = str((data or {}).get(key, "") or "").strip()
        if not desc:
            continue
        nid = f"{label.lower()}:{hashlib.sha1((source_id + desc).encode()).hexdigest()[:10]}"  # noqa: S324
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
                    "created_at": time.time(),
                },
            )
            engine.link_nodes(
                source_id=source_id,
                target_id=nid,
                rel_type="HAS_GOAL" if label == "Goal" else "PROPOSES_SPEC",
                properties={"source": "chat"},
            )
            logger.info(
                "[CONCEPT:ECO-4.70] Surfaced %s from chat: %s", label, desc[:60]
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.70] %s surfacing failed: %s", label, exc)
