"""Unified feed-source bridge + first-class feed registry (CONCEPT:AU-KG.ingest.rss-feed-connector/2.122).

Two jobs, one place:

1. **ScholarX arXiv as a feed source** — ``scholarx_feed_documents`` maps ScholarX
   ``Paper`` objects (its specialized arXiv RSS parser stays inside scholarx) onto
   the SAME ``SourceDocument`` shape the native ``rss`` connector and the FreshRSS
   preset emit, with a ``metadata["record"]`` whose ``origin.streamId`` marks it as
   research — so it flows through the one ``WorldModelPipelineRunner`` gate and takes
   the research branch.

2. **First-class feed registry** — ``register_feed_nodes`` materializes every
   configured feed (native RSS URLs, FreshRSS, ScholarX categories) as a durable
   ``:FeedSource``/``:RssFeed`` node in the KG (the long-missing "presets→KG" wiring),
   so feeds are first-class citizens that ``graph_feeds`` lists/adds/removes. Each
   ingested item links ``:ingestedFrom`` its feed source.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_utilities.knowledge_graph.enrichment.provenance import stamp_source
from agent_utilities.protocols.source_connectors.base import (
    ExternalAccess,
    SourceDocument,
)

logger = logging.getLogger(__name__)

_FEED_LABEL = "FeedSource"


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


# ── ScholarX arXiv → unified SourceDocument ──────────────────────────────────
def scholarx_feed_documents(
    categories: list[str] | None = None, days: int = 1
) -> list[SourceDocument]:
    """ScholarX arXiv RSS items as unified ``SourceDocument``s (CONCEPT:AU-KG.ingest.rss-feed-connector).

    No-op (``[]``) when ScholarX is not installed. The ``origin.streamId`` is set to
    ``scholarx:arxiv`` so ``WorldModelPipelineRunner._is_research`` routes each item
    to the research path; ``id`` is the canonical ``arxiv:<id>`` so it converges with
    the same paper arriving via FreshRSS.
    """
    try:
        from scholarx.api_client import ScholarXClient
    except ImportError:
        logger.info("ScholarX not installed — scholarx feed source is a no-op")
        return []

    def _attr(obj: Any, name: str, default: Any) -> Any:
        return getattr(obj, name, default)

    client = ScholarXClient()
    result = _run(client.get_recent_papers(categories=categories, days=days))
    papers = getattr(result, "papers", None)
    if papers is None:
        papers = result if isinstance(result, list) else []
    out: list[SourceDocument] = []
    for p in papers:
        aid = str(_attr(p, "id", "") or "")
        if not aid:
            continue
        title = _attr(p, "title", "") or ""
        abstract = _attr(p, "abstract", "") or ""
        url = _attr(p, "url", "") or ""
        published = str(_attr(p, "published_date", "") or "")
        record = {
            "id": aid,
            "title": title,
            "published": published,
            "categories": list(_attr(p, "categories", []) or []),
            "authors": list(_attr(p, "authors", []) or []),
            "pdf_url": _attr(p, "pdf_url", "") or "",
            "url": url,
            "canonical": [{"href": url}] if url else [],
            "origin": {
                "htmlUrl": url,
                "streamId": "scholarx:arxiv",
                "title": "ScholarX arXiv",
            },
        }
        out.append(
            SourceDocument(
                id=aid,
                source_uri=url,
                title=title[:300],
                text=abstract,
                doc_type="paper",
                updated_at=published,
                metadata={"record": record, "source_system": "scholarx"},
                external_access=ExternalAccess.public(),
            )
        )
    return out


# ── First-class feed registry (presets → KG, CONCEPT:AU-KG.compute.first-class-rss-atom) ───────────────
def _feed_node_id(source_system: str, key: str) -> str:
    safe = str(key).replace(":", "-").replace("/", "-")
    return f"feed:{source_system}:{safe[:160]}"


def upsert_feed_source(
    engine: Any,
    *,
    key: str,
    source_system: str,
    feed_url: str = "",
    kind: str = "RssFeed",
    name: str = "",
    enabled: bool = True,
) -> str:
    """Materialize one configured feed as a durable :FeedSource/:RssFeed node.

    The long-missing "presets→KG" wiring (CONCEPT:AU-KG.compute.first-class-rss-atom): a feed is a first-class
    KG citizen, not just declarative config. Returns the node id.
    """
    node_id = _feed_node_id(source_system, key)
    props: dict[str, Any] = {
        "name": name or key,
        "feed_url": feed_url,
        "enabled": bool(enabled),
        # One flat LPG label ``FeedSource``; ``kind`` ("RssFeed"|"FeedSource")
        # carries the OWL refinement (:RssFeed rdfs:subClassOf :FeedSource).
        "kind": kind,
    }
    stamp_source(props, source_system)
    try:
        engine.add_node(node_id, _FEED_LABEL, properties=props)
    except Exception as e:  # noqa: BLE001 — registry write is best-effort
        logger.debug("upsert_feed_source failed for %s: %s", node_id, e)
    return node_id


def register_feed_nodes(
    engine: Any,
    *,
    native_urls: list[str] | None = None,
    scholarx_categories: list[str] | None = None,
    freshrss_configured: bool = False,
) -> list[str]:
    """Upsert a :FeedSource node per configured feed (called on the live sweep path)."""
    ids: list[str] = []
    for url in native_urls or []:
        ids.append(
            upsert_feed_source(
                engine, key=url, source_system="rss", feed_url=url, kind="RssFeed"
            )
        )
    for cat in scholarx_categories or []:
        ids.append(
            upsert_feed_source(
                engine,
                key=cat,
                source_system="scholarx",
                feed_url=f"https://rss.arxiv.org/rss/{cat}",
                kind="RssFeed",
                name=f"arXiv {cat}",
            )
        )
    if freshrss_configured:
        ids.append(
            upsert_feed_source(
                engine,
                key="freshrss",
                source_system="freshrss",
                kind="FeedSource",
                name="FreshRSS",
            )
        )
    return ids


def remove_feed_source(engine: Any, *, key: str, source_system: str = "rss") -> bool:
    """Tombstone a registered feed by its url/key (CONCEPT:AU-KG.compute.first-class-rss-atom). Best-effort."""
    backend = getattr(engine, "backend", None)
    if backend is None:
        return False
    node_id = _feed_node_id(source_system, key)
    try:
        backend.execute(
            "MATCH (f:FeedSource {id: $id}) DETACH DELETE f", {"id": node_id}
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("remove_feed_source failed for %s: %s", node_id, e)
        return False


def list_feed_sources(engine: Any) -> list[dict[str, Any]]:
    """Return the registered feed-source nodes for the graph_feeds surface."""
    backend = getattr(engine, "backend", None)
    if backend is None:
        return []
    rows = backend.execute(
        "MATCH (f:FeedSource) RETURN f.id as id, f.name as name, "
        "f.feed_url as feed_url, f.source_system as source_system, "
        "f.kind as kind, f.enabled as enabled"
    )
    return [r for r in (rows or []) if isinstance(r, dict)]
