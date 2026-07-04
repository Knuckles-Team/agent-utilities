"""Research acquisition — find sources that address a KG topic.

CONCEPT:AU-KG.research.self-evolution-convergence — Research assimilation (acquire stage of the golden loop).

For a given topic, gather candidate "addressing" sources. The reliable,
always-available substrate is **semantic search over the local KG** (existing
Documents/Code/Concepts/Threads — searchable now that embeddings are backfilled).
External acquisition (X / SearXNG / scholarx) is optional and gated by
``KG_RESEARCH_EXTERNAL`` since it needs live MCP/network; the golden loop stays
useful (propose-only) without it.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Node labels that count as substantive "sources" addressing a topic.
_SOURCE_TYPES = {"Document", "Code", "Feature", "Article", "Thread", "Skill"}

# Wall-clock bound for a single embed during acquisition. The embedding model
# talks to a remote endpoint (default ``vllm-embed.arpa``); when that endpoint
# is slow or down the OpenAI client retries with backoff against a 300s timeout,
# which would stall the golden loop for minutes per topic. Bounding each embed
# means a dead endpoint degrades the acquire stage in seconds, not minutes.
_ACQUIRE_TIMEOUT_S = 8.0


def bounded_embed(embed_fn: Any, text: str, timeout: float) -> list[float] | None:
    """Embed ``text`` via ``embed_fn`` but never block longer than ``timeout`` s.

    Returns the embedding vector, or ``None`` if the embedding endpoint is
    unreachable/slow (timeout) or errors. The worker thread is abandoned
    (not cancellable) on timeout — it finishes in the background — but control
    returns to the caller immediately so the loop never hangs.
    """
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(lambda: embed_fn([text])[0]).result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "embedding endpoint did not respond within %.1fs — treating as unavailable",
            timeout,
        )
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("bounded_embed failed: %s", e)
        return None
    finally:
        ex.shutdown(wait=False)


def acquire_for_topic(
    engine: Any,
    topic: dict[str, Any],
    *,
    top_k: int = 5,
    embed_fn: Any = None,
    timeout: float | None = None,
) -> list[str]:
    """Return source-node ids that semantically address ``topic``.

    ``topic`` is ``{"id", "name"}``. Uses the configured embedding model +
    backend vector search; returns the ids of related source nodes (excluding
    the topic itself). Best-effort — returns ``[]`` if embeddings/search are
    unavailable.

    ``embed_fn`` lets the caller build the embedding fn ONCE and reuse it across
    topics (the golden loop does this) instead of re-creating the model per call.
    The embed is wall-clock bounded by ``timeout`` (default ``_ACQUIRE_TIMEOUT_S``)
    so an unreachable embedding endpoint degrades in seconds rather than stalling
    the loop on client-side retries.
    """
    name = str(topic.get("name") or topic.get("id") or "").strip()
    if not name:
        return []
    backend = getattr(engine, "backend", None)
    search = getattr(backend, "semantic_search", None)
    if not callable(search):
        return []
    if embed_fn is None:
        from ..enrichment.semantic import make_embed_fn

        embed_fn = make_embed_fn()
    vec = bounded_embed(
        embed_fn, name, _ACQUIRE_TIMEOUT_S if timeout is None else timeout
    )
    if vec is None:
        logger.debug("acquire embed unavailable for %s — skipping", name)
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

    if setting("KG_RESEARCH_EXTERNAL", "0") == "1":
        ids.extend(_acquire_external(engine, name, top_k))
    return ids


def acquire_for_topic_perspectival(
    engine: Any,
    topic: dict[str, Any],
    *,
    top_k: int = 5,
    embed_fn: Any = None,
    timeout: float | None = None,
) -> list[str]:
    """Multi-perspective acquire — STORM made native (CONCEPT:AU-KG.research.perspectival-inquiry).

    Instead of one semantic probe of the topic name, fan :func:`acquire_for_topic`
    across questions asked from several expert lenses, derive the contradiction /
    agreement / blind-spot map and a self-critique, materialize them as KG nodes, and
    submit the frontier question as the next research loop. Returns the **union** of
    source ids every lens surfaced, so the loop's ``mark_addressed`` still converges
    the topic. Falls back to the single-lens probe when the fan-out finds nothing (e.g.
    embeddings unavailable), so behaviour never regresses.
    """
    from .perspective import PerspectiveEngine

    tid = str(topic.get("id") or "")

    def _probe(question: str) -> list[str]:
        return acquire_for_topic(
            engine,
            {"id": tid, "name": question},
            top_k=max(2, top_k // 2),
            embed_fn=embed_fn,
            timeout=timeout,
        )

    eng = PerspectiveEngine(engine)
    inquiry = eng.inquire({"id": tid, "name": str(topic.get("name") or tid)}, _probe)
    eng.materialize(inquiry)
    return inquiry.all_source_ids() or acquire_for_topic(
        engine, topic, top_k=top_k, embed_fn=embed_fn, timeout=timeout
    )


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
