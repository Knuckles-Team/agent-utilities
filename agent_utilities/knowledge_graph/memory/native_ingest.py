"""Native connector ingestion primitives — the fleet's one push-into-the-KG path.

CONCEPT:AU-KG.ingest.enterprise-source-extractor. Every ``agents/*`` connector should
natively push its data into the ONE epistemic-graph engine **from its own code**, in every
modality that applies (the "maximum ingestion" bar):

* **typed nodes** — structured records → OWL ``:Class`` nodes + links (``ingest_entities``)
* **documents** — text worth semantic search → ``:Document`` nodes carrying the text +
  ``source_uri`` (``ingest_documents``); hub-side enrichment chunks/embeds them
* **blobs** — raw bytes (files, attachments, scans, media) → ``:Blob`` + ``:AssetOccurrence``
  via :class:`MediaStore` (``media_store``)

Typed nodes and documents resolve the process-owned Epistemic Graph authority and
commit through the engine-native ``ApplyChangeEnvelope`` boundary.  That one
governed transaction carries graph rows, policy, lineage, content version, and
outbox state; connector code never opens the engine's control-plane transaction
surface directly.  The heavy ``IntelligenceGraphEngine`` is not constructible
inside a connector. Engine failures are explicit and never acknowledged as
successful ingestion. Node ids follow
``<domain>:<class>:<externalId>``; ``node_type`` on each entity must match a class the package's
``ontology_providers`` ``.ttl`` federates.

This is the one connector write path: connectors ship only a thin mapper
(records → entity/document dicts) and call these, never re-implementing a
transaction protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agent_utilities.native_ingest")


class NativeIngestError(RuntimeError):
    """A connector record could not reach the authoritative engine transaction."""


def native_authority() -> Any:
    """Return the process-owned governed Epistemic Graph authority.

    The returned :class:`GraphComputeEngine` exposes the first-class
    ``ApplyChangeEnvelope`` client used by :func:`ingest_graph_slice`.
    """
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
    except Exception as exc:  # noqa: BLE001 - public error is type-only
        raise NativeIngestError("native ingest engine client is unavailable") from exc
    try:
        engine = GraphComputeEngine.get_or_create()
        client = getattr(engine, "client", None)
        if client is None or getattr(client, "changes", None) is None:
            raise NativeIngestError("native ingest authority is unavailable")
        return engine
    except NativeIngestError:
        raise
    except Exception as exc:  # noqa: BLE001 - public error is type-only
        raise NativeIngestError("native ingest engine client is unavailable") from exc


@dataclass(frozen=True)
class _InjectedChangeEnvelopeAuthority:
    """GraphCompute-shaped view for a first-class injected native client."""

    client: Any
    graph_name: str
    catalog_epoch: int = 0
    placement_group: int | None = None

    def for_graph(self, graph: str) -> _InjectedChangeEnvelopeAuthority:
        if graph != self.graph_name:
            raise PermissionError("injected native ingest authority cannot retarget")
        return self


def _change_envelope_authority(client: Any | None, graph: str | None) -> Any:
    """Resolve production authority or validate one injected test dependency.

    An injected dependency must be the generated ChangeEnvelope-capable client;
    the retired raw ``txn``-only fake is deliberately rejected. Graph selection
    remains bound to the ambient verified session and cannot be supplied by a
    connector.
    """

    if client is None:
        return native_authority()

    from agent_utilities.knowledge_graph.core.session import resolve_session

    session = resolve_session(required_scope="kg:write")
    if graph is not None and graph != session.graph:
        raise NativeIngestError("injected native ingest graph is unauthorized")
    required = (
        getattr(client, "changes", None),
        getattr(client, "nodes", None),
        getattr(client, "rdf", None),
        getattr(client, "supports", None),
    )
    if any(value is None for value in required) or not callable(required[-1]):
        raise NativeIngestError(
            "injected native ingest client must support ChangeEnvelope"
        )
    return _InjectedChangeEnvelopeAuthority(client, session.graph)


def _write_nodes(
    client: Any | None,
    graph: str | None,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None,
    *,
    source: str,
    domain: str,
) -> dict[str, int]:
    """Stamp provenance and commit nodes plus edges in one ChangeEnvelope."""
    nodes = [n for n in nodes if n.get("id")]
    if not nodes:
        raise NativeIngestError("native ingest requires at least one identified node")
    for node in nodes:
        if "type" in node or not node.get("node_type"):
            raise NativeIngestError("native ingest nodes require canonical node_type")
    for rel in relationships or []:
        if (
            "type" in rel
            or not rel.get("relationship")
            or not rel.get("source")
            or not rel.get("target")
        ):
            raise NativeIngestError(
                "native ingest edges require source, target, and canonical relationship"
            )
    prepared = []
    for node in nodes:
        props = {key: value for key, value in node.items() if value is not None}
        props.setdefault("source", source)
        props.setdefault("domain", domain)
        prepared.append(props)
    authority = _change_envelope_authority(client, graph)
    try:
        from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
            ingest_graph_slice,
        )

        result = ingest_graph_slice(
            authority,
            domain,
            prepared,
            [dict(relationship) for relationship in relationships or []],
            source_instance=source,
        )
    except NativeIngestError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve retryable cause privately
        raise NativeIngestError("native ChangeEnvelope ingestion failed") from exc
    if result.get("status") not in {"success", "skipped"}:
        raise NativeIngestError("native ChangeEnvelope ingestion failed")

    logger.info(
        "native ingest committed %d nodes and %d edges",
        len(nodes),
        len(relationships or []),
    )
    return {"nodes": len(nodes), "edges": len(relationships or [])}


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str,
    domain: str,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write typed OWL nodes (+ edges) into the engine.

    ``entities``: ``[{"id":..., "node_type":<owl:Class>, ...props}]``.
    ``relationships``: ``[{"source":id, "target":id, "relationship":<link>}]``.
    Returns ``{"nodes":n, "edges":m}``. ``client`` may inject a generated,
    ChangeEnvelope-capable client for tests; ``graph`` may only repeat the
    ambient verified graph. Production resolves :func:`native_authority`.
    """
    if not entities:
        raise NativeIngestError("native ingest requires at least one entity")
    return _write_nodes(
        client,
        graph,
        entities,
        relationships,
        source=source,
        domain=domain,
    )


def ingest_documents(
    documents: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str,
    domain: str,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write text records as ``:Document`` nodes (semantic-search fodder).

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    ``relationships`` uses the same canonical ``source``/``target``/``relationship``
    shape as :func:`ingest_entities`; document nodes and their links commit together.
    The ``:Document`` carries the text + full provenance (``source_uri``, ``title``,
    a source-provided ``fetched_at``, ``backend``, ``content_hash``). This helper
    does not invent wall-clock properties: generated operational timestamps would
    make an otherwise-identical redelivery produce a different content identity.
    A connector runs OUTSIDE the
    hub process (no heavy engine available here), so it cannot itself chunk/embed/
    concept-extract/topic-classify — every write is stamped ``needs_enrichment=true``
    so the hub-side catch-up sweep (:func:`enrich_pending_documents`, CONCEPT:AU-KG.enrichment.topic-classification-topology)
    can find it and run it through the SAME unified DocumentProcessor + central
    ``_enrich_text`` seam every other ingestion path gets — so a searxng/RSS/etc.
    connector document is never a permanently-shallower write than a directly
    ingested one. Returns the committed node and edge counts.
    """
    nodes: list[dict[str, Any]] = []
    for doc in documents or []:
        did = doc.get("id")
        text = doc.get("text") or doc.get("content")
        if not did or not text:
            continue
        node = {k: v for k, v in doc.items() if k not in ("content",) and v is not None}
        node["id"] = did
        node["node_type"] = "Document"
        node["text"] = text
        node.setdefault("backend", "native_ingest")
        if "content_hash" not in node:
            import hashlib

            node["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        node["needs_enrichment"] = True
        nodes.append(node)
    if not nodes:
        raise NativeIngestError("native ingest requires at least one document")
    return _write_nodes(
        client,
        graph,
        nodes,
        relationships,
        source=source,
        domain=domain,
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

    proc = DocumentProcessor(
        backend,
        engine=engine,
        chunking=ChunkingConfig(),
        contextual=True,
    )
    ing = IngestionEngine(kg_engine=engine)

    for row in rows:
        d = row.get("d") if isinstance(row, dict) else None
        if not isinstance(d, dict) or not d.get("id"):
            continue
        result["scanned"] += 1
        doc_id = d["id"]
        text = d.get("text") or ""
        title = d.get("title") or d.get("name") or doc_id
        try:
            proc.process(
                text,
                document_id=doc_id,
                title=title,
                doc_type=d.get("doc_type") or "document",
                source="pending-document",
                metadata={"needs_enrichment": False},
                persist=True,
                connector="document-enrichment",
                source_instance="pending-document-sweep",
            )
            await ing._enrich_text(doc_id, text, d.get("domain") or "document", title)
            result["enriched"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad document never aborts the sweep
            result["failed"] += 1
            logger.debug(
                "enrich_pending_documents: %s failed: %s", doc_id, exc, exc_info=True
            )
    return result


def media_store() -> Any:
    """Return a :class:`MediaStore` over the authoritative engine."""
    try:
        from agent_utilities.knowledge_graph.memory.media_store import MediaStore

        return MediaStore(native_authority())
    except Exception as exc:  # noqa: BLE001 - preserve cause without leaking it
        raise NativeIngestError("native media store is unavailable") from exc
