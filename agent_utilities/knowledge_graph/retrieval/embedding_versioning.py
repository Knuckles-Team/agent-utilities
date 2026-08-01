#!/usr/bin/python
from __future__ import annotations

"""Embedding versioning — a vector from model A must never be silently compared
against model B's (CONCEPT:AU-KG.retrieval.embedding-version-identity).

**The problem this closes.** `CapabilityIndex` (the HNSW/numpy ANN index behind
`designate`/`retrieve_hybrid`) previously stored *only* the raw vector per id —
no record of which embedding model produced it. Re-pointing
`default_embedding_model` at a new model (a routine operational change — model
upgrade, provider migration, dimension change) would silently start comparing
old-model vectors against new-model query embeddings via plain cosine
similarity. Two different embedding spaces rank as noise against each other:
this is not a crash, it is a **silent retrieval-quality regression** — the
worst kind, because nothing fails, results just quietly get worse.

**The fix.** Every embedding carries an explicit :class:`EmbeddingVersion`
(``provider:model``, the identity of the space it lives in — dimensionality is
already guarded separately by `CapabilityIndex`'s existing dim-mismatch check).
`CapabilityIndex` is single-version-per-instance: the first vector `add()`ed
pins the index's version, and every subsequent `add()`/`designate()` call is
checked against it. A mismatch raises :class:`EmbeddingVersionMismatchError`
loudly instead of silently inserting or ranking a foreign-space vector.

**Re-indexing.** Because an index is pinned to one version, a model change
cannot be applied in place — it is a **generation swap**: build a new
`CapabilityIndex` tagged with the new version, re-embed and populate it (via
the same `create_embedding_model`/`make_embed_fn` path — never a second
embedding factory), and cut retrieval traffic over once it is populated. A
**partial** re-index is the same swap scoped to the subset of documents whose
content changed since the last run — this reuses the ingestion layer's
existing incremental change-detection (`knowledge_graph/ingestion/coverage.py`,
`connector_coverage.py`, `checkpoint.py`: idempotency keys + content hashes
already used to skip unchanged documents on delta syncs) rather than a new
mechanism: "which documents need a vector under the new version" is the same
question as "which documents changed," just re-triggered by a version bump
instead of a source poll. `is_stale` below is the one-line predicate a
reindexing job polls to decide whether an index generation needs replacing.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingVersion",
    "EmbeddingVersionMismatchError",
    "EmbeddingVersionUnresolvedError",
    "resolve_current_embedding_version",
    "is_stale",
]


class EmbeddingVersionUnresolvedError(RuntimeError):
    """No embedding model is configured, so no version can be resolved.

    Distinct from :class:`EmbeddingVersionMismatchError` — this is a
    configuration gap (nothing to compare against), not a detected mismatch.
    """


class EmbeddingVersionMismatchError(RuntimeError):
    """A vector's embedding version does not match the expected/index version.

    Raised **loudly** (never swallowed into a silent fallback) by
    ``CapabilityIndex.add``/``designate`` when a caller supplies a vector or
    query produced by a different embedding model than the index is pinned
    to (CONCEPT:AU-KG.retrieval.embedding-version-identity). The fix is a
    re-index (see module docstring), never a forced comparison.
    """

    def __init__(self, *, expected: str, actual: str, context: str = "") -> None:
        self.expected = expected
        self.actual = actual
        msg = (
            f"Embedding version mismatch{f' ({context})' if context else ''}: "
            f"expected {expected!r}, got {actual!r}. Vectors from different "
            "embedding models are not comparable — re-index instead of "
            "forcing this comparison."
        )
        super().__init__(msg)


@dataclass(frozen=True)
class EmbeddingVersion:
    """The identity of the embedding space a vector lives in.

    ``provider``/``model`` are the same values ``create_embedding_model``
    resolves and builds its client cache key from — this is deliberately NOT a
    second source of truth, just the subset of that identity worth persisting
    alongside a vector. Dimensionality is intentionally excluded: two models
    can coincidentally share a dimension while producing incomparable spaces,
    and `CapabilityIndex` already has an independent, unconditional dim-mismatch
    guard in `add()` — conflating the two would weaken both checks.
    """

    provider: str
    model: str

    @property
    def id(self) -> str:
        """The stable version key stored alongside every vector."""
        return f"{self.provider}:{self.model}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.id


def resolve_current_embedding_version() -> EmbeddingVersion:
    """Resolve the embedding version that is ACTIVE right now.

    Mirrors the exact resolution `create_embedding_model()` uses for its
    no-arg ("give me the default embedder") call — including automatic
    failover (CONCEPT:AU-KG.enrichment.each-call-resolves-active) — without
    constructing the (expensive) client itself, so ingest/query hot paths can
    cheaply tag a vector or check a version without paying for a second client
    build.

    Raises:
        EmbeddingVersionUnresolvedError: no embedding model is configured at
            all. Callers on a hot path should let this propagate — an
            unresolvable version is a configuration bug, not something to
            paper over with an "unversioned" sentinel that would defeat the
            whole point of this module.
    """
    from agent_utilities.core.embedding_failover import active_embedding_endpoint

    try:
        endpoint = active_embedding_endpoint()
    except Exception as exc:  # noqa: BLE001 — surfaced below as a typed error
        raise EmbeddingVersionUnresolvedError(
            "Could not resolve the active embedding endpoint"
        ) from exc
    if not endpoint.provider or not endpoint.model_id:
        raise EmbeddingVersionUnresolvedError(
            "No embedding model is configured; set one in AgentConfig before "
            "indexing or querying vectors."
        )
    return EmbeddingVersion(provider=endpoint.provider, model=endpoint.model_id)


def is_stale(indexed_version: str, *, current: EmbeddingVersion | None = None) -> bool:
    """Whether an index (or a single vector) tagged ``indexed_version`` is stale.

    A re-indexing job polls this per index generation: ``True`` means the
    configured embedding model has moved on since this index/vector was built,
    so it needs the generation-swap re-index described in the module
    docstring rather than continuing to serve comparisons in a space nothing
    else is being embedded into anymore.
    """
    current = current or resolve_current_embedding_version()
    return indexed_version != current.id
