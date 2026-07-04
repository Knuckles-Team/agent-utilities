#!/usr/bin/python
from __future__ import annotations

"""Multi-source ingest adapters + granular idempotency (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

Brings documents (PRD/BRD/SOW/tasks → `Requirement` nodes) and conversations /
SDD transcripts (→ `Decision` nodes) into the assimilation graph, with the
content-addressed idempotency that keeps cost growing with the *delta*, not the
*corpus*:

* :func:`canonical_source_id` — collapses the same source ingested twice from
  different URIs (arxiv abs/pdf/version, DOI variants, URL, file path) onto one id,
  so duplicates never create duplicate nodes.
* :func:`content_fingerprint` — a per-item content hash; re-ingesting an unchanged
  item is a **no-op** (skipped), and a changed item updates in place.

Granular: each item is hashed independently, so a changed batch skips its unchanged
members instead of re-processing the whole batch (per-paper skip).

Concept: ingest-adapters
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryNodeType
from .dedup import iter_typed_nodes

_REQUIREMENT = RegistryNodeType.REQUIREMENT.value
_DECISION = RegistryNodeType.DECISION.value
_CONCEPT = RegistryNodeType.CONCEPT.value
# Gate matching gap_analysis._concept_key — a real concept id is letters-then-digit.
_CONCEPT_ID_GATE = re.compile(r"^[A-Z]{2}-(?:ORCH|KG|AHE|ECO|OS|GBOT)\.")  # OKF-CIS (OS-5.77)

_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)(?:v\d+)?", re.IGNORECASE)
_DOI = re.compile(r"(?:doi\.org/|doi:)\s*(10\.\S+)", re.IGNORECASE)
_WS = re.compile(r"\s+")
# Concept ids referenced in a doc (KG-2.7 / AHE-3.12 / ORCH-1.3b) — the exact
# signal the gap matcher (auto_satisfy) uses to recognize already-built features.
_CONCEPT_REF = re.compile(r"\b([A-Z]{2,6}-\d+(?:\.\d+[a-z]?|-\d+)?)\b")


@dataclass
class IngestReport:
    ingested: int = 0  # new nodes created
    updated: int = 0  # existing node, changed content
    skipped: int = 0  # unchanged (idempotent no-op)
    node_ids: list[str] = field(default_factory=list)


def canonical_source_id(uri: str) -> str:
    """Canonicalize a source URI so equivalent references collapse to one id.

    arxiv abs/pdf/versioned → ``arxiv:<id>``; DOI variants → ``doi:<id>``; other
    URLs → ``url:<host/path>`` (trailing slash + scheme stripped); file paths →
    ``file:<normalized path>``.
    """
    u = (uri or "").strip()
    if not u:
        return ""
    m = _ARXIV.search(u)
    if m:
        return f"arxiv:{m.group(1)}"
    m = _DOI.search(u)
    if m:
        return f"doi:{m.group(1).rstrip('/')}"
    if u.startswith(("http://", "https://")):
        rest = u.split("://", 1)[1].rstrip("/").lower()
        return f"url:{rest}"
    return f"file:{u.rstrip('/')}"


def content_fingerprint(text: str) -> str:
    """Stable per-item content hash (whitespace-normalized SHA-256, 16 hex)."""
    norm = _WS.sub(" ", (text or "").strip()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _get_node(engine: Any, node_id: str) -> dict[str, Any] | None:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return None
    try:
        for nid, data in graph.nodes(data=True):
            if nid == node_id and isinstance(data, dict):
                return data
    except TypeError:  # pragma: no cover
        return None
    return None


def _ingest_items(
    engine: Any,
    items: list[dict[str, Any]],
    *,
    node_type: str,
    id_prefix: str,
) -> IngestReport:
    report = IngestReport()
    for item in items:
        uri = str(item.get("uri") or item.get("id") or "").strip()
        text = str(item.get("text") or item.get("content") or "")
        if not uri:
            continue
        node_id = f"{id_prefix}:{canonical_source_id(uri)}"
        fp = content_fingerprint(text)
        existing = _get_node(engine, node_id)
        if existing is not None and existing.get("content_hash") == fp:
            report.skipped += 1  # unchanged → idempotent no-op
            continue
        name = str(item.get("title") or item.get("name") or uri)
        props = {
            "name": name,
            "content": text,
            "content_hash": fp,
            "source_uri": uri,
            "kind": str(item.get("kind", "") or ""),
            "status": "open",
            # The concept(s) this item DECLARES (from its title/identity), so the
            # gap matcher can recognize it as already-built. Title only — not body
            # prose, which cites many concepts the item merely relates to.
            "concept_ids": sorted(set(_CONCEPT_REF.findall(name.upper()))),
        }
        engine.add_node(node_id, node_type, properties=props)
        report.node_ids.append(node_id)
        if existing is not None:
            report.updated += 1
        else:
            report.ingested += 1
    return report


def ingest_documents(engine: Any, docs: list[dict[str, Any]]) -> IngestReport:
    """Ingest PRD/BRD/SOW/tasks docs as `Requirement` nodes (idempotent)."""
    return _ingest_items(engine, docs, node_type=_REQUIREMENT, id_prefix="req")


def ingest_conversations(engine: Any, convos: list[dict[str, Any]]) -> IngestReport:
    """Ingest chat / SDD transcripts as `Decision` nodes (idempotent)."""
    return _ingest_items(engine, convos, node_type=_DECISION, id_prefix="conv")


def ingest_concepts(engine: Any, concepts: list[dict[str, Any]]) -> IngestReport:
    """Ingest the ecosystem capability registry as *built* ``Concept`` nodes.

    These are the "already-built" side the golden-loop gap matcher compares
    research against: :func:`gap_analysis.auto_satisfy` matches each research
    ``Article`` / ``sdd_feature`` against ``Concept`` nodes, so **without** them
    every paper looks like an open gap. Source = ``docs/concepts.yaml`` registries
    + ``CONCEPT:<ID>`` code markers, keyed by canonical concept id (``KG-2.7``).
    Idempotent via ``content_hash``; embeddings are filled by the daemon's
    embed-backfill (so the embedding-fallback match works once backfilled, while
    the explicit-id match works immediately). (CONCEPT:AU-KG.query.vendor-agnostic-traversal)

    Each item: ``{"id": "AU-KG.query.vendor-agnostic-traversal", "name": "...", "pillar": "EG-KG.compute.backend", "status": "live",
    "source": "..."}`` (only ``id`` is required).
    """
    report = IngestReport()
    seen: set[str] = set()
    for c in concepts:
        cid = str(c.get("id") or "").strip().upper()
        if not _CONCEPT_ID_GATE.match(cid) or cid in seen:
            continue
        seen.add(cid)
        name = str(c.get("name") or c.get("doc") or cid)
        pillar = str(c.get("pillar") or "")
        status = str(c.get("status") or "live")
        # Rich, embeddable identity text (id + name + description + pillar): a terse
        # "id — name" sits below the embedding noise floor vs verbose paper
        # abstracts, so the description carries the semantic signal the matcher's
        # retrieval needs; the id still drives the high-precision explicit match.
        description = str(c.get("description") or c.get("doc") or "").strip()
        body = " — ".join(p for p in (cid, name, description, pillar) if p)
        fp = content_fingerprint(f"{body}|{status}")
        node_id = f"concept:{cid}"
        existing = _get_node(engine, node_id)
        if existing is not None and existing.get("content_hash") == fp:
            report.skipped += 1
            continue
        props = {
            "name": name,
            "concept_id": cid,
            "concept_ids": [cid],
            "content": body,
            "description": description,
            "content_hash": fp,
            "pillar": pillar,
            "status": status,
            "source": str(c.get("source") or "ecosystem-registry"),
        }
        engine.add_node(node_id, _CONCEPT, properties=props)
        report.node_ids.append(node_id)
        if existing is not None:
            report.updated += 1
        else:
            report.ingested += 1
    return report


def enrich_concepts(
    engine: Any,
    *,
    embed_fn: Any = None,
    concept_types: tuple[str, ...] = (_CONCEPT,),
    batch: int = 256,
) -> int:
    """Embed ``Concept`` nodes that have no vector yet, so the matcher works *now*.

    The daemon's embed-backfill fills concept embeddings eventually; but a fresh
    concept-bridge run leaves them vectorless, and the matcher's embedding-recall
    stage is then blind until backfill catches up. This fills them on demand from
    each concept's rich text via the shared embedder (no second model), writing
    through ``backend.add_embedding``. Idempotent: concepts that already carry an
    embedding are skipped. Returns the number embedded. (CONCEPT:AU-KG.ingest.world-model-gate)
    """
    graph = getattr(engine, "graph", None)
    backend = getattr(engine, "backend", None)
    if graph is None or backend is None or not hasattr(backend, "add_embedding"):
        return 0
    # BOUNDED per-label fetch (CONCEPT:EG-KG.txn.per-graph-write-isolation/2.264) — never a whole-graph
    # ``GetNodes`` dump (refused as RESULT_TOO_LARGE on a large engine).
    pending: list[tuple[str, str]] = []
    for nid, data in iter_typed_nodes(graph, concept_types):
        if data.get("embedding"):
            continue
        text = str(data.get("content") or data.get("name") or nid)
        pending.append((nid, text))
    if not pending:
        return 0
    if embed_fn is None:
        from ..enrichment.semantic import make_embed_fn

        embed_fn = make_embed_fn()
    embedded = 0
    for i in range(0, len(pending), batch):
        chunk = pending[i : i + batch]
        vecs = embed_fn([t for _, t in chunk])
        for (nid, _), vec in zip(chunk, vecs, strict=False):
            if not vec or len(vec) < 2:  # skip the degraded [[0.0]] no-embedder result
                continue
            try:
                backend.add_embedding(nid, list(vec))
                embedded += 1
            except Exception:  # pragma: no cover - best-effort
                pass
    return embedded


__all__ = [
    "IngestReport",
    "canonical_source_id",
    "content_fingerprint",
    "ingest_documents",
    "ingest_conversations",
    "ingest_concepts",
    "enrich_concepts",
]
