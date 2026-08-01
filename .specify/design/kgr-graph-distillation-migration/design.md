# Design Document: Pre-compute similarity shortcuts across KG nodes so retrieval is O(degree), not O(N) full-index scans

CONCEPT:AU-KG.retrieval.graph-distillation-migration

> `agent_utilities/knowledge_graph/retrieval/semantic_retrieval_engine.py:640-769`
> (`GraphDistillationMigrator`).

## Decision — similarity edges are pre-computed and stored on the graph, then walked at query time, instead of scanning the full embedding index per query

`semantic_retrieval_engine.py:643-648` states the trade directly:
`GraphDistillationMigrator` "pre-computes similarity edges across KG nodes to
enable O(degree) retrieval instead of O(N) full-index scans." A batch of
nodes is distilled incrementally (`distill_batch`, skipping already-distilled
nodes by default) — each new node compares against prior nodes only
(`predecessors = embeddable[:i]`), edges above a similarity threshold are
created and indexed, and stale edges are pruned in the same pass
(`_linker.prune_stale_edges`) so the index does not grow unbounded with
one-time or superseded similarity relationships. The migrator additionally
tracks `coverage_ratio` (distilled nodes / embeddable nodes) and
`avg_edges_per_node`, and exposes `coverage_report()` for index-health
monitoring rather than leaving distillation coverage unobservable.

**The rejected alternative is named directly in the docstring**: "instead of
O(N) full-index scans." A query-time full scan of the embedding index (the
prior/default RAG pattern) re-pays the full-corpus similarity computation on
every query, regardless of how many times the same or a related query has
been asked before. Pre-computing the similarity edges moves that cost to
ingestion/batch time (paid once, incrementally, as nodes are distilled) so
`distilled_retrieve` at query time walks the graph's own degree-bounded
neighborhood — the graph itself becomes the index, rather than a separate
structure that must be kept in sync with it. The `incremental=True` default
on `distill_batch` is itself a smaller instance of the same principle: a
node's similarity edges, once computed, are not recomputed on every
subsequent batch unless explicitly forced (`incremental=False`), so the
distillation cost is genuinely amortized rather than repeated per run.

## Risk Assessment

- **Blast Radius**: `semantic_retrieval_engine.py`
  (`GraphDistillationMigrator`, `AutoSimilarityLinker`,
  `KGNativeRetrievalRetriever`).
- **Backward Compatible**: Yes — distillation is an additive index built
  alongside the existing embedding store; `distilled_retrieve` is a new
  retrieval path, not a replacement of the existing ones.
- **Known weak point**: coverage is only as good as the batches actually run
  through `distill_batch` — a node ingested but never distilled has no
  similarity edges and is invisible to `distilled_retrieve` until the next
  distillation pass reaches it, which `coverage_report()` can surface but
  does not automatically remediate.
