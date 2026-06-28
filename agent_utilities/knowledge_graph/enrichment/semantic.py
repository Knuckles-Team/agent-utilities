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

# bge-m3 (the deployed embedder) handles large batches per request, so we send a
# big LIST of inputs in ONE ``/v1/embeddings`` POST rather than re-chunking it into
# tiny sub-requests. This caps a single POST's payload (and is also the value we
# pin on the llama-index model's ``embed_batch_size`` so it stops splitting our
# chunk into ``DEFAULT_EMBED_BATCH_SIZE``-sized POSTs). (CONCEPT:KG-2.280)
_EMBED_MAX_BATCH = 256

# Memoized cpu/load-derived embed concurrency (computed once; cheap to reuse).
_EMBED_CONCURRENCY: int | None = None


def _embed_concurrency() -> int:
    """Auto-sized number of embed requests to keep in flight concurrently.

    Reuses the *shared* cpu/memory/load sizing anchor (``compute_ingest_worker_count``
    — the same ~36%-of-cores, Pi-OOM-capped budget the ingest pools use) rather than
    inventing a new knob. Embedding is network/GPU-bound (the bge-m3 vLLM endpoint
    services many requests at once), not local-cpu-bound, so we allow ~2× that anchor,
    capped at 16. The model's *declared* parallel capacity (CONCEPT:KG-2.143) is the
    floor, so an explicitly-configured higher capacity always wins. Never below 1.
    (CONCEPT:KG-2.280)
    """
    global _EMBED_CONCURRENCY
    if _EMBED_CONCURRENCY is not None:
        return _EMBED_CONCURRENCY
    ceiling = 16
    try:
        from agent_utilities.core.model_concurrency import (
            resolve_capacity,
            server_ceiling,
        )

        declared = max(1, resolve_capacity("embedding"))
        # CONCEPT:ORCH-1.102 — the embed fan-out is local-cpu-derived (~2× the ingest
        # anchor); never let it exceed the embedder SERVER's real capacity ceiling.
        ceiling = max(1, server_ceiling("embedding"))
    except Exception:  # noqa: BLE001 — capacity is best-effort
        declared = 1
    try:
        from agent_utilities.knowledge_graph.core.engine_tasks import (
            compute_ingest_worker_count,
        )

        anchor = max(1, compute_ingest_worker_count())
    except Exception:  # noqa: BLE001 — sizing is best-effort
        anchor = 4
    # cpu/load-derived (≤16), floored at the model's declared capacity, then HARD-
    # capped at the server ceiling so it can never oversubscribe the embedder.
    _EMBED_CONCURRENCY = min(max(declared, min(anchor * 2, 16)), ceiling)
    return _EMBED_CONCURRENCY


def _auto_batch(n_texts: int, concurrency: int) -> int:
    """Batch size that makes POSTs big BUT leaves enough chunks to fill the lanes.

    A single huge batch would serialize the whole job on one POST; tiny batches
    waste round-trips. Aim for ~``concurrency`` chunks, each a big LIST, clamped to
    ``[32, _EMBED_MAX_BATCH]``. (CONCEPT:KG-2.280)
    """
    if n_texts <= 0:
        return 1
    import math

    per = math.ceil(n_texts / max(1, concurrency))
    return max(32, min(per, _EMBED_MAX_BATCH))


def make_embed_fn(batch_size: int | None = None) -> EmbedFn:
    """Batched + concurrent embedding fn backed by the configured embedder (bge-m3).

    Two compounding throughput wins over the historical one-text-per-request loop
    (CONCEPT:KG-2.280, applying the AGENTS.md *batch-never-per-element* rule to
    embeddings):

    * **BATCH** — every request carries a big LIST of inputs (auto-sized up to
      :data:`_EMBED_MAX_BATCH`), and the underlying llama-index model's
      ``embed_batch_size`` is pinned so it issues ONE POST per chunk instead of
      re-splitting it into ``DEFAULT_EMBED_BATCH_SIZE`` (=10) sub-POSTs.
    * **CONCURRENCY** — chunks are fanned out CONCURRENTLY up to
      :func:`_embed_concurrency` (cpu/load-derived, ≥ the model's declared
      capacity) via the shared controller (CONCEPT:KG-2.143), so the ENRICH stage
      is never one-request-in-flight.

    ``batch_size`` pins the per-request batch explicitly (mainly for tests);
    ``None`` (the default) auto-sizes it per call from the input length and the
    resolved concurrency. Batch boundaries and output order are preserved, so the
    same vectors come out in the same order. The fail-loud KG-2.3 contract below is
    unchanged: a missing/unreachable embedder raises rather than returning a stub.
    """
    try:
        from agent_utilities.core.embedding_utilities import create_embedding_model
        from agent_utilities.core.model_concurrency import map_concurrent_sync

        model = create_embedding_model()
        # Pin the model's internal batch so a chunk we hand it is ONE POST, not a
        # fan of DEFAULT_EMBED_BATCH_SIZE-sized sub-POSTs (the serial-POST symptom).
        try:
            current = int(getattr(model, "embed_batch_size", 0) or 0)
            model.embed_batch_size = max(current, _EMBED_MAX_BATCH)
        except Exception:  # noqa: BLE001 — model may not expose the attr; harmless
            pass

        def _fn(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            concurrency = _embed_concurrency()
            bs = batch_size or _auto_batch(len(texts), concurrency)
            chunks = [texts[i : i + bs] for i in range(0, len(texts), bs)]
            # Cap concurrency at the chunk count — never spin idle workers.
            capacity = max(1, min(concurrency, len(chunks)))
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
            # Record embed usage into the active ingest profile (OS-5.69) — embedding
            # endpoints rarely return token counts, so estimate from text length.
            from ..core.ingest_profile import record_embed_usage

            record_embed_usage(texts=texts)
            return out

        return _fn
    except Exception as e:
        # Zero-Stub compliance (AGENTS.md): NEVER return a degenerate stub that
        # silently yields 1-dim ``[0.0]`` vectors. That stub previously masked a
        # missing-embedder deployment (the serving plane shipped bare ``embeddings``
        # without ``embeddings-openai`` → ``No module named 'llama_index.embeddings'``):
        # enrichment "succeeded" while writing garbage vectors into a 1024-dim store,
        # so the failure was invisible (embed_calls=0, no real embeddings) instead of
        # loud. Fail loud here; every production caller wraps embedding as best-effort
        # in try/except, so this degrades to "no enrichment edges" — observable and
        # safe — rather than silent vector-store corruption. (CONCEPT:KG-2.3)
        logger.error("make_embed_fn unavailable (%s)", e)
        raise RuntimeError(
            f"embedding model unavailable: {e}. The KG embedding plane requires the "
            "'embeddings-openai' extra (llama-index-embeddings-openai) and a reachable "
            "bge-m3 vLLM endpoint."
        ) from e


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
