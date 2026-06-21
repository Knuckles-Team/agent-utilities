"""Embedding + semantic cross-linking (CONCEPT:KG-2.8 Phase 3).

Embeds entities into one space and uses the **engine's vector search** (HNSW/
cosine — the compute layer) to discover cross-category relationships: a paper
``Concept`` that a code symbol ``REALIZES``, or anything ``RELATES_TO`` a topic/
goal. ``embed_fn``/``search_fn`` are injectable so the logic is testable without a
live model or daemon.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .models import Concept, EnrichmentEdge

logger = logging.getLogger(__name__)

# texts -> embeddings (batched)
EmbedFn = Callable[[list[str]], list[list[float]]]
# (query_vec, k) -> list of {id, type, _similarity, ...}
SearchFn = Callable[[list[float], int], list[dict[str, Any]]]


def make_embed_fn(batch_size: int = 64) -> EmbedFn:
    """Batched embedding fn backed by the configured embedding model (bge-m3).

    Batches are fanned out CONCURRENTLY up to the embedding model's declared
    parallel-call capacity (``parallel_instances × max_parallel_calls``) via the
    shared concurrency controller (CONCEPT:KG-2.143). Capacity ``1`` (the safe
    default) is byte-for-byte the historical sequential for-loop; capacity ``K``
    runs up to ``K`` batches in flight. Batch boundaries and output order are
    preserved, so the same vectors come out in the same order.
    """
    try:
        from agent_utilities.core.embedding_utilities import create_embedding_model
        from agent_utilities.core.model_concurrency import (
            map_concurrent_sync,
            resolve_capacity,
        )

        model = create_embedding_model()

        def _fn(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            chunks = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
            capacity = resolve_capacity("embedding")
            # Fan out per-batch embedding up to capacity; order preserved, so
            # flattening the per-chunk results reproduces the input order.
            chunk_results = map_concurrent_sync(
                chunks,
                model.get_text_embedding_batch,
                model="embedding",
                capacity=capacity,
            )
            out: list[list[float]] = []
            for vecs in chunk_results:
                out.extend(vecs)
            return out

        return _fn
    except Exception as e:  # pragma: no cover
        logger.warning("make_embed_fn unavailable (%s)", e)
        return lambda texts: [[0.0] for _ in texts]


def make_search_fn(backend: Any) -> SearchFn:
    """Vector search over the backend's embedding store (engine HNSW/cosine)."""

    def _fn(query_vec: list[float], k: int) -> list[dict[str, Any]]:
        try:
            return backend.semantic_search(query_vec, k)
        except Exception:
            return []

    return _fn


def entity_text(node_type: str, name: str, summary: str = "", extra: str = "") -> str:
    """Compose the text used to embed an entity."""
    parts = [name]
    if summary:
        parts.append(summary)
    if extra:
        parts.append(extra)
    return " — ".join(p for p in parts if p)


def embed_and_store(
    backend: Any, items: list[tuple[str, str]], embed_fn: EmbedFn
) -> int:
    """Embed (id, text) pairs and store vectors on the backend. Returns count."""
    if not items:
        return 0
    vecs = embed_fn([t for _, t in items])
    n = 0
    for (nid, _), vec in zip(items, vecs, strict=False):
        try:
            backend.add_embedding(nid, vec)
            n += 1
        except Exception:
            pass
    return n


def _result_type(r: dict[str, Any]) -> str:
    # ``_table_label`` is set by the PostgreSQL/L3 vector search (per-label node
    # tables); ``type``/``node_type`` by other backends.
    return str(r.get("type") or r.get("node_type") or r.get("_table_label") or "")


def link_concepts_to_code(
    concepts: list[Concept],
    embed_fn: EmbedFn,
    search_fn: SearchFn,
    top_k: int = 5,
    relates_threshold: float = 0.55,
    realizes_threshold: float = 0.78,
) -> list[EnrichmentEdge]:
    """Link concepts to the code/features that relate to or realize them.

    Uses vector similarity: above ``relates_threshold`` → ``RELATES_TO``; above
    ``realizes_threshold`` → ``REALIZES`` (the code implements the concept).
    """
    if not concepts:
        return []
    vecs = embed_fn([entity_text("concept", c.name, c.summary) for c in concepts])
    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str, str]] = set()
    # Search a WIDER pool than top_k: vector search mixes all node labels, so
    # Code/Feature candidates can be crowded out of a small top_k by other
    # densely-embedded labels (Skill/Agent/Message). Fetch more, then keep the
    # best ``top_k`` Code/Feature matches per concept. (CONCEPT:KG-2.8)
    search_k = max(top_k * 6, 30)
    for c, vec in zip(concepts, vecs, strict=False):
        kept = 0
        for r in search_fn(vec, search_k):
            if kept >= top_k:
                break
            if _result_type(r) not in ("Code", "Feature"):
                continue
            score = float(r.get("_similarity", 0.0) or 0.0)
            tgt = r.get("id")
            if not tgt or score < relates_threshold:
                continue
            rel = "REALIZES" if score >= realizes_threshold else "RELATES_TO"
            key = (c.id, tgt, rel)
            if key not in seen:
                seen.add(key)
                edges.append(EnrichmentEdge(source=c.id, target=tgt, rel_type=rel))
                kept += 1
    return edges


def find_related(
    topic: str,
    embed_fn: EmbedFn,
    search_fn: SearchFn,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Goal/topic-driven cross-ingestion discovery: nearest entities to a topic."""
    vec = embed_fn([topic])[0]
    results = search_fn(vec, top_k)
    out = []
    for r in results:
        out.append(
            {
                "id": r.get("id"),
                "type": _result_type(r),
                "name": r.get("name", ""),
                "summary": r.get("summary", ""),
                "similarity": round(float(r.get("_similarity", 0.0) or 0.0), 3),
            }
        )
    return out
