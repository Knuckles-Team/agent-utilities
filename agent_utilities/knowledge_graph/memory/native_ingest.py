"""Native connector ingestion primitives — the fleet's one push-into-the-KG path.

CONCEPT:AU-KG.ingest.enterprise-source-extractor. Every ``agents/*`` connector should
natively push its data into the ONE epistemic-graph engine **from its own code**, in every
modality that applies (the "maximum ingestion" bar):

* **typed nodes** — structured records → OWL ``:Class`` nodes + links (``ingest_entities``)
* **documents** — text worth semantic search → ``:Document`` nodes carrying the text +
  ``source_uri`` (``ingest_documents``); hub-side enrichment chunks/embeds them
* **blobs** — raw bytes (files, attachments, scans, media) → ``:Blob`` + ``:MediaAsset``
  via :class:`MediaStore` (``media_store``)

All three ride the **lightweight engine client** (``GraphComputeEngine()._client`` + ``txn``) —
the same fast client :class:`MediaStore` uses. The heavy ``IntelligenceGraphEngine`` is NOT
constructible inside a connector, so these helpers deliberately avoid it. Everything is
dependency-/engine-guarded: with no KG stack or no reachable engine every entry point **no-ops**
(returns ``None``), so a connector runs with zero KG infrastructure. Node ids follow
``<domain>:<class>:<externalId>``; ``type`` on each entity must match a class the package's
``ontology_providers`` ``.ttl`` federates.

This is the ONE implementation of the txn write path — connectors ship only a thin mapper
(records → entity/document dicts) and call these, never re-implementing the txn dance.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("agent_utilities.native_ingest")

_DEFAULT_GRAPH = "__commons__"


def native_client() -> tuple[Any | None, str]:
    """Return ``(engine_client, graph_name)`` or ``(None, "")`` when unavailable.

    Builds the lightweight :class:`GraphComputeEngine` and hands back its fast
    ``_client`` (carrying ``.txn``/``.edges``/``.nodes``/``.blob``). Never raises.
    """
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
    except Exception as e:  # noqa: BLE001 — KG stack absent
        logger.debug("native ingest unavailable (import): %s", e)
        return None, ""
    try:
        engine = GraphComputeEngine()
        client = getattr(engine, "_client", None)
        if client is None:
            return None, ""
        return client, (getattr(engine, "graph_name", None) or _DEFAULT_GRAPH)
    except Exception as e:  # noqa: BLE001 — engine unreachable
        logger.debug("native ingest: engine unreachable: %s", e)
        return None, ""


def _write_nodes(
    client: Any,
    graph: str,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None,
    *,
    source: str,
    domain: str,
) -> dict[str, int] | None:
    """Stamp provenance, MERGE the nodes in one txn, then add the edges."""
    nodes = [n for n in nodes if n.get("id")]
    if not nodes:
        return None
    try:
        txn = client.txn.begin(graph=graph)
        for node in nodes:
            props = {k: v for k, v in node.items() if k != "id" and v is not None}
            props.setdefault("source", source)
            props.setdefault("domain", domain)
            client.txn.add_node(txn, node["id"], props)
        committed = client.txn.commit(txn)
    except Exception as e:  # noqa: BLE001 — engine/txn failure is non-fatal
        logger.warning("native ingest: txn failed: %s", e)
        return None
    if not committed:
        logger.warning("native ingest: txn not committed (conflict)")
        return None

    edges = 0
    for rel in relationships or []:
        try:
            client.edges.add(
                rel["source"], rel["target"], {"type": rel.get("type", "RELATED")}
            )
            edges += 1
        except Exception as e:  # noqa: BLE001 — pure edge link, best-effort
            logger.debug("native ingest: edge skipped: %s", e)

    logger.info(
        "native ingest[%s]: wrote %d nodes, %d edges", domain, len(nodes), edges
    )
    return {"nodes": len(nodes), "edges": edges}


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str,
    domain: str,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write typed OWL nodes (+ edges) into the engine.

    ``entities``: ``[{"id":..., "type":<owl:Class>, ...props}]``.
    ``relationships``: ``[{"source":id, "target":id, "type":<link>}]``.
    Returns ``{"nodes":n, "edges":m}`` or ``None``. ``client``/``graph`` may be
    injected (tests); otherwise resolved via :func:`native_client`.
    """
    if not entities:
        return None
    if client is None:
        client, graph = native_client()
    if client is None:
        return None
    return _write_nodes(
        client,
        graph or _DEFAULT_GRAPH,
        entities,
        relationships,
        source=source,
        domain=domain,
    )


def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    source: str,
    domain: str,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write text records as ``:Document`` nodes (semantic-search fodder).

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    The ``:Document`` carries the text + full provenance (``source_uri``, ``title``,
    ``fetched_at``, ``backend``, ``content_hash``). A connector runs OUTSIDE the
    hub process (no heavy engine available here), so it cannot itself chunk/embed/
    concept-extract/topic-classify — every write is stamped ``needs_enrichment=true``
    so the hub-side catch-up sweep (:func:`enrich_pending_documents`, CONCEPT:AU-KG.enrichment.topic-classification-topology)
    can find it and run it through the SAME unified DocumentProcessor + central
    ``_enrich_text`` seam every other ingestion path gets — so a searxng/RSS/etc.
    connector document is never a permanently-shallower write than a directly
    ingested one. Returns ``{"nodes":n, "edges":0}`` or ``None``.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nodes: list[dict[str, Any]] = []
    for doc in documents or []:
        did = doc.get("id")
        text = doc.get("text") or doc.get("content")
        if not did or not text:
            continue
        node = {k: v for k, v in doc.items() if k not in ("content",) and v is not None}
        node["id"] = did
        node["type"] = "Document"
        node["text"] = text
        node.setdefault("created_at", now)
        node.setdefault("fetched_at", now)
        node.setdefault("backend", "native_ingest")
        if "content_hash" not in node:
            import hashlib

            node["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        node["needs_enrichment"] = True
        nodes.append(node)
    if not nodes:
        return None
    if client is None:
        client, graph = native_client()
    if client is None:
        return None
    return _write_nodes(
        client, graph or _DEFAULT_GRAPH, nodes, None, source=source, domain=domain
    )


async def enrich_pending_documents(engine: Any, *, limit: int = 200) -> dict[str, int]:
    """Hub-side catch-up sweep: run every ``needs_enrichment`` ``:Document`` through
    the SAME unified pipeline a direct ingest gets (CONCEPT:AU-KG.enrichment.topic-classification-topology).

    Connector-written documents (:func:`ingest_documents` — searxng-mcp results,
    any future ``native_ingest`` producer) land as a raw ``:Document`` node with
    just text + provenance, because the connector process has no heavy engine to
    chunk/embed/enrich with. This sweep — run hub-side, where the full
    ``IntelligenceGraphEngine`` IS available — finds those pending nodes, runs
    :class:`~agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor`
    (chunk + contextual-retrieval enrichment) and the central
    ``IngestionEngine._enrich_text`` seam (concepts + facts + WorldView topic
    classification) over each, then clears the flag so it isn't reprocessed.
    Best-effort per document — one bad document never aborts the sweep. Callable
    via ``graph_ingest action=enrich_pending_documents`` (MCP + REST twin).
    """
    result = {"scanned": 0, "enriched": 0, "failed": 0}
    query_cypher = getattr(engine, "query_cypher", None)
    backend = getattr(engine, "backend", None)
    if not callable(query_cypher) or backend is None:
        return result
    try:
        rows = query_cypher(
            "MATCH (d:Document) WHERE d.needs_enrichment = true "
            f"RETURN d LIMIT {int(limit)}"
        )
    except Exception as exc:  # noqa: BLE001 — sweep never raises
        logger.warning("enrich_pending_documents: query failed: %s", exc)
        return result
    if not rows:
        return result

    from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine
    from agent_utilities.knowledge_graph.ontology.document_processing import (
        ChunkingConfig,
        DocumentProcessor,
    )

    proc = DocumentProcessor(backend, chunking=ChunkingConfig(), contextual=True)
    ing = IngestionEngine(kg_engine=engine)
    add_node = getattr(backend, "add_node", None)

    for row in rows:
        d = row.get("d") if isinstance(row, dict) else None
        if not isinstance(d, dict) or not d.get("id"):
            continue
        result["scanned"] += 1
        doc_id = d["id"]
        text = d.get("text") or ""
        title = d.get("title") or d.get("name") or doc_id
        source_uri = d.get("source_uri") or d.get("source") or ""
        try:
            proc.process(
                text,
                document_id=doc_id,
                title=title,
                doc_type=d.get("doc_type") or "document",
                source=source_uri,
                persist=True,
            )
            await ing._enrich_text(doc_id, text, d.get("domain") or "document", title)
            if callable(add_node):
                add_node(doc_id, type="Document", needs_enrichment=False)
            result["enriched"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad document never aborts the sweep
            result["failed"] += 1
            logger.debug(
                "enrich_pending_documents: %s failed: %s", doc_id, exc, exc_info=True
            )
    return result


def media_store() -> Any | None:
    """Return a :class:`MediaStore` over a live engine (for raw-blob ingestion), or ``None``."""
    client, _ = native_client()
    if client is None:
        return None
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
        from agent_utilities.knowledge_graph.memory.media_store import MediaStore

        return MediaStore(GraphComputeEngine())
    except Exception as e:  # noqa: BLE001
        logger.debug("native ingest: media_store unavailable: %s", e)
        return None
