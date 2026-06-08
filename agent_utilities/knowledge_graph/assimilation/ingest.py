#!/usr/bin/python
from __future__ import annotations

"""Multi-source ingest adapters + granular idempotency (CONCEPT:KG-2.7).

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

_REQUIREMENT = RegistryNodeType.REQUIREMENT.value
_DECISION = RegistryNodeType.DECISION.value

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


__all__ = [
    "IngestReport",
    "canonical_source_id",
    "content_fingerprint",
    "ingest_documents",
    "ingest_conversations",
]
