# Design Document: A vector always carries its embedding-space identity, and a retrieval chunk always cites the addressable evidence it was cut from — neither can silently drift

CONCEPT:AU-KG.retrieval.embedding-version-identity ·
CONCEPT:AU-KG.retrieval.embedding-alignment-diagnostics ·
CONCEPT:AU-KG.retrieval.fragment-cited-chunk

> `agent_utilities/knowledge_graph/retrieval/embedding_versioning.py` (primary
> decision), `agent_utilities/knowledge_graph/memory/
> optimization_engine.py:497-534` (CKA diagnostics, formerly
> `embedding_diagnostics.py`), `agent_utilities/knowledge_graph/ontology/
> document_processing.py:315-371` (`DocumentChunk.fragment_ids`/
> `embedding_version`).

## Decision — `CapabilityIndex` is single-version-per-instance; a mismatch raises loudly instead of silently ranking a foreign-space vector

`CONCEPT:AU-KG.retrieval.embedding-version-identity`

`embedding_versioning.py:4-39` names the failure mode this closes precisely:
`CapabilityIndex` (the HNSW/numpy ANN index behind `designate`/
`retrieve_hybrid`) previously stored only the raw vector per id — "no record
of which embedding model produced it." Re-pointing `default_embedding_model`
at a new model (a routine operational change) "would silently start comparing
old-model vectors against new-model query embeddings via plain cosine
similarity. Two different embedding spaces rank as noise against each other:
this is not a crash, it is a **silent retrieval-quality regression** — the
worst kind, because nothing fails, results just quietly get worse." **The
rejected alternative is exactly that prior state**: no version tracking, so a
model swap degrades ranking quality with zero signal that anything changed.

The fix: every embedding carries an explicit `EmbeddingVersion`
(`provider:model`); the first vector `add()`ed pins the index's version, and
every later `add()`/`designate()` call is checked against it — a mismatch
raises `EmbeddingVersionMismatchError`. Because an index is pinned to one
version, a model change is a **generation swap** (build a new index tagged
with the new version, re-embed, cut traffic over), reusing the ingestion
layer's existing incremental change-detection (idempotency keys + content
hashes) rather than inventing a second "what changed" mechanism — the
docstring frames it precisely: "which documents need a vector under the new
version" is the same question as "which documents changed."

### Pointer — `CONCEPT:AU-KG.retrieval.embedding-alignment-diagnostics`

`optimization_engine.py:497-534` (`compute_cka`, now consolidated from the
standalone `embedding_diagnostics.py` into `MemoryOptimizationEngine`). Where
version-identity HARD-BLOCKS comparing two spaces outright, this diagnostic
MEASURES how misaligned two spaces actually are when a migration is being
planned — Centered Kernel Alignment (`cka_score`) alongside raw mean cosine
similarity, with `alignment_ratio = mean_cosine / cka_score` deciding
`needs_transformation` (true when the ratio drops below 0.5). **What this
concretely adds**: raw cosine similarity alone can look deceptively healthy —
two spaces can have a high mean cosine yet a low CKA score, meaning they are
superficially similar-looking but *structurally* misaligned. Relying on raw
cosine alone (the rejected alternative) would greenlight a generation swap
that looks fine on a spot check but produces bad rankings once live. This is
the informational half of the re-indexing decision above: it tells a
migration whether a spatial transformation is needed before cutting traffic
over, not just whether the versions differ.

### Pointer — `CONCEPT:AU-KG.retrieval.fragment-cited-chunk`

`document_processing.py:315-371`. `DocumentChunk.fragment_ids` links the
retrieval unit (a fixed-size, overlapping `ChunkSpan`) to the citation unit (a
structurally addressed evidence-spine `Fragment`) by comparing character
spans — both derived from the SAME extracted text, so their offsets are
directly comparable rather than independently re-derived (which would risk
alignment drift between the two). `embedding_version` rides alongside
`fragment_ids` on the same `DocumentChunk` model, from the same commit: a
chunk's embedding claims a version, and a chunk's text claims a citation — a
chunk that cannot cite a fragment is, per the code comment, "not citable":
"retrieval consumers must resolve a result back to real, addressable
evidence, never a bare span of characters." **The rejected alternative**: a
retrieval result whose only provenance is its own raw character offsets into
whatever the source document happened to be at ingest time — brittle the
moment the source document changes, and not resolvable back to the
evidence-spine's own drift/staleness detection
(`CONCEPT:AU-KG.retrieval.source-to-claim-lineage`).

## Risk Assessment

- **Blast Radius**: `embedding_versioning.py`, `capability_index.py`
  (`CapabilityIndex`), `optimization_engine.py`, `document_processing.py`
  (`DocumentChunk`), `scripts/check_citation_lineage.py`.
- **Backward Compatible**: An existing index built before version-tracking
  existed has no recorded version — the first `add()`/`designate()` call
  after upgrade pins whatever version is passed, which is a one-time
  compatibility gap, not an ongoing one.
- **Known weak point**: `needs_transformation`'s 0.5 alignment-ratio threshold
  is a fixed heuristic, not derived from measured retrieval-quality impact per
  threshold value — a migration could sit just above the threshold and still
  carry meaningfully degraded ranking quality.
