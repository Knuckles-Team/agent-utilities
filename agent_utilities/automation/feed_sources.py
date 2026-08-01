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
import hashlib
import json
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
def _scholarx_mcp_configured() -> bool:
    """Whether a ``scholarx-mcp``/``scholarx`` server is reachable in ``mcp_config``."""
    try:
        from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
            _load_mcp_config,
        )

        servers = _load_mcp_config() or {}
        return "scholarx-mcp" in servers or "scholarx" in servers
    except Exception:  # noqa: BLE001 — best-effort discovery
        return False


def _scholarx_mcp_documents(
    categories: list[str] | None, days: int
) -> list[SourceDocument]:
    """Drive the ``scholarx-mcp`` fleet server via the ``scholarx-papers`` preset.

    The fallback path when the ``scholarx`` python package is not installed but the
    fleet's scholarx MCP server is reachable (CONCEPT:AU-KG.ingest.research-connector-presets/7.3) —
    "prefer driving the fleet server over a new HTTP client". The generic
    ``mcp_tool`` connector returns the tool's raw record shape (no ``origin``/
    ``canonical`` envelope), so each drained document is re-shaped here into the
    EXACT SAME record envelope the direct-import branch below produces, so
    ``WorldModelPipelineRunner._is_research``/``_arxiv_id`` route and dedup it
    identically regardless of which path fetched it.
    """
    from agent_utilities.protocols.source_connectors.registry import build_connector

    params: dict[str, Any] = {}
    if categories:
        params["query"] = " OR ".join(f"cat:{c}" for c in categories)
    try:
        conn = build_connector(
            "mcp_tool", {"preset": "scholarx-papers", "params": params}
        )
        raw_docs = (
            list(conn.poll_all())  # type: ignore[attr-defined]
            if hasattr(conn, "poll_all")
            else list(conn.load())  # type: ignore[attr-defined]
        )
    except Exception:  # noqa: BLE001 — an unreachable fleet server is a no-op, not a crash
        logger.info(
            "scholarx-mcp not reachable — scholarx MCP feed source is a no-op",
            exc_info=True,
        )
        return []

    out: list[SourceDocument] = []
    for raw in raw_docs:
        rec = dict(raw.metadata.get("record") or {})
        aid = str(rec.get("id") or raw.id or "")
        if not aid:
            continue
        url = str(rec.get("url") or raw.source_uri or "")
        published = raw.updated_at or str(rec.get("published_date") or "")
        record = {
            "id": aid,
            "title": raw.title,
            "published": published,
            "categories": list(rec.get("categories") or categories or []),
            "authors": list(rec.get("authors") or []),
            "pdf_url": str(rec.get("pdf_url") or ""),
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
                title=(raw.title or "")[:300],
                text=raw.text,
                doc_type="paper",
                updated_at=published,
                metadata={"record": record, "source_system": "scholarx"},
                external_access=ExternalAccess.public(),
            )
        )
    return out


def scholarx_feed_documents(
    categories: list[str] | None = None, days: int = 1
) -> list[SourceDocument]:
    """ScholarX arXiv RSS items as unified ``SourceDocument``s (CONCEPT:AU-KG.ingest.rss-feed-connector).

    Prefers the local ``scholarx`` python package (its specialized arXiv parser);
    when that is not installed, falls back to driving the fleet's ``scholarx-mcp``
    server (CONCEPT:AU-KG.ingest.research-connector-presets); no-op (``[]``) only when NEITHER is
    available. The ``origin.streamId`` is set to ``scholarx:arxiv`` so
    ``WorldModelPipelineRunner._is_research`` routes each item to the research path;
    ``id`` is the canonical ``arxiv:<id>`` so it converges with the same paper
    arriving via FreshRSS.
    """
    try:
        from scholarx.api_client import ScholarXClient
    except ImportError:
        if _scholarx_mcp_configured():
            return _scholarx_mcp_documents(categories, days)
        logger.info(
            "ScholarX not installed and scholarx-mcp not configured — "
            "scholarx feed source is a no-op"
        )
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
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:32]
    return f"feed:{source_system}:{digest}"


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
    from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
    from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
        ingest_envelope,
    )

    record = {"id": node_id, "type": _FEED_LABEL, **props}
    record["updatedAt"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    env = ChangeEnvelope.from_connector_record(
        record,
        connector=source_system,
        id_field="id",
        version_field="updatedAt",
        source_acl=ExternalAccess.public(),
    )
    applied = ingest_envelope(engine, env)
    if applied.get("status") not in {"success", "skipped"}:
        raise RuntimeError("native FeedSource ChangeEnvelope failed")
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
    node_id = _feed_node_id(source_system, key)
    from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
    from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
        ingest_envelope,
    )

    env = ChangeEnvelope(
        connector=source_system,
        operation="delete",
        source_object_id=node_id,
        source_version="deleted",
    )
    applied = ingest_envelope(engine, env)
    return applied.get("status") in {"success", "skipped"}


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
