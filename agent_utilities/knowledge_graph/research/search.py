"""Research acquisition — find sources that address a KG topic.

CONCEPT:KG-2.7 — Research assimilation (acquire stage of the golden loop).

For a given topic, gather candidate "addressing" sources. The reliable,
always-available substrate is **semantic search over the local KG** (existing
Documents/Code/Concepts/Threads — searchable now that embeddings are backfilled).
External acquisition (X / SearXNG / scholarx) is optional and gated by
``KG_RESEARCH_EXTERNAL`` since it needs live MCP/network; the golden loop stays
useful (propose-only) without it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Node labels that count as substantive "sources" addressing a topic.
_SOURCE_TYPES = {"Document", "Code", "Feature", "Article", "Thread", "Skill"}


def acquire_for_topic(
    engine: Any,
    topic: dict[str, Any],
    *,
    top_k: int = 5,
) -> list[str]:
    """Return source-node ids that semantically address ``topic``.

    ``topic`` is ``{"id", "name"}``. Uses the configured embedding model +
    backend vector search; returns the ids of related source nodes (excluding
    the topic itself). Best-effort — returns ``[]`` if embeddings/search are
    unavailable.
    """
    name = str(topic.get("name") or topic.get("id") or "").strip()
    if not name:
        return []
    backend = getattr(engine, "backend", None)
    search = getattr(backend, "semantic_search", None)
    if not callable(search):
        return []
    try:
        from ..enrichment.semantic import make_embed_fn

        vec = make_embed_fn()([name])[0]
    except Exception as e:  # noqa: BLE001
        logger.debug("acquire embed failed for %s: %s", name, e)
        return []

    ids: list[str] = []
    seen: set[str] = set()
    try:
        # Search a wider pool, then keep the best source-typed matches.
        for r in search(vec, max(top_k * 6, 30)) or []:
            if len(ids) >= top_k:
                break
            rtype = str(
                r.get("type") or r.get("node_type") or r.get("_table_label") or ""
            )
            sid = r.get("id")
            if rtype in _SOURCE_TYPES and sid and sid != topic.get("id"):
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
    except Exception as e:  # noqa: BLE001
        logger.debug("acquire search failed for %s: %s", name, e)

    if os.getenv("KG_RESEARCH_EXTERNAL", "0") == "1":
        ids.extend(_acquire_external(engine, name, top_k))
    return ids


def _acquire_external(engine: Any, query: str, top_k: int) -> list[str]:
    """Optional external acquisition (X / SearXNG / scholarx) — best-effort.

    Gated by ``KG_RESEARCH_EXTERNAL=1``. Ingests discovered items via the unified
    engine so they become first-class KG sources (concepts via B1). Returns any
    new source ids. Kept intentionally minimal/defensive: external connectors
    vary by deployment, so failures degrade to ``[]``.
    """
    try:
        from ..kb import x_ingestion  # noqa: F401  (presence check)
    except Exception:  # noqa: BLE001
        return []
    # Connector wiring is deployment-specific; the local-KG path above is the
    # guaranteed substrate. External ingestion hooks land here when configured.
    logger.debug("external acquisition requested for %r (top_k=%d)", query, top_k)
    return []
