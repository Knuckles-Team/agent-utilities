# Design Document: Relevance rankings are pre-computed by a periodic sweep and persisted as graph edges, not recomputed live per query

CONCEPT:AU-KG.retrieval.per-item-relevance-ranking

> `agent_utilities/knowledge_graph/core/engine_tasks.py:2730-2736,5190-5199,5608-5619`
> (`_tick_evolution`, `_run_relevance_sweep`, `query_relevance_rankings`).

## Decision — a periodic sweep scores every ingested paper/repo against a target codebase and persists the scores as `RELEVANCE_SCORED` edges; reads are then a pre-computed lookup

`engine_tasks.py:5190-5198` (`_run_relevance_sweep`) groups `Article` nodes by
source paper and `Code` nodes by repository, computes a composite relevance
score, and persists it as a `RELEVANCE_SCORED` edge in the KG.
`query_relevance_rankings` (`5608-5619`) then answers "what's relevant to
this codebase" by reading those pre-computed edges back via Cypher, rather
than re-scoring the corpus for every caller that asks.

**The rejected alternative** is scoring relevance at query time, on demand:
every call to "what's relevant to X" would re-run the same heavy
computation (per-item embeddings + composite scoring across every ingested
paper/repo) that the sweep already paid for on the prior cycle. Persisting
the score as an edge amortizes that cost across every subsequent read, at
the cost of the ranking being only as fresh as the last completed sweep
cycle rather than always current. The sweep is explicitly deferred while a
bulk ingest is in flight (`engine_tasks.py:5199-5203`) — "it's periodic, so
skipping a cycle is cheap — the maintenance scheduler re-enqueues it once the
ingest drains" — which is itself a smaller instance of the same trade:
correctness-under-load (never contending with a bulk write) is worth more
here than always running on schedule. A per-item edge-persistence failure
inside the sweep's loop is logged and skipped rather than aborting the whole
sweep (`engine_tasks.py:5605`, "a failed write is simply absent from the next
query... it does not falsely appear scored") — the rejected alternative there
is an all-or-nothing sweep that a single bad item's write failure would
abort entirely, discarding every score the sweep had already computed that
cycle.

## Risk Assessment

- **Blast Radius**: `engine_tasks.py` (`_tick_evolution`,
  `_run_relevance_sweep`, `query_relevance_rankings`), the `RELEVANCE_SCORED`
  edge type.
- **Backward Compatible**: Yes — an additive scheduled sweep and a new read
  query over its output; nothing else changes shape.
- **Known weak point**: `query_relevance_rankings` reads whatever the last
  completed sweep wrote — a caller has no built-in signal of how stale the
  ranking is relative to the most recent ingest, short of separately checking
  when the evolution cycle last ran.
