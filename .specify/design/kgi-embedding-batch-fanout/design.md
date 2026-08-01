# Design Document: Embedding applies the AGENTS.md "batch, never per-element" rule — big LISTs per request, fanned out concurrently

CONCEPT:AU-KG.ingest.applying-agents-md-batch

> `agent_utilities/knowledge_graph/enrichment/semantic.py`
> (`make_embed_fn`, `_embed_concurrency`, `_auto_batch`, `_joint_budget_cap`),
> pinned by `tests/unit/knowledge_graph/test_embed_throughput.py`.

> Note on the marker id: the tool triage flagged this as `[review]` —
> "marker text truncated by grammar tool; id reads like a real name." Having
> read the code, it is not truncated garbage: `applying-agents-md-batch` is
> literally the module's own description of itself — "applying the
> AGENTS.md *batch-never-per-element* rule to embeddings" (`semantic.py:126`).
> It names a real, well-grounded decision and belongs documented, not retired.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.ingest.generalized-cross-lane-parallelization` | the SAME "bounded concurrent fan-out over serial per-item work" shape, applied to LLM enrichment windows / document-dir ingest / paper ingest instead of embedding requests | 0.55 | KG |
| `AU-KG.compute.concurrency-controller-sizing` | the model's declared parallel-capacity ceiling this fan-out is clamped under | 0.40 | KG |

### Extension Analysis

- **Primary Extension Point**: `_embed_concurrency()` (the sizing anchor) and
  `_auto_batch()` (the per-request batch size) — both are pure sizing
  functions `make_embed_fn` composes.
- **Extension Strategy**: augment — a new embedder with different batch
  economics changes the anchor/ceiling inputs, not the batch+concurrency
  shape itself.
- **New Concept Required?**: No.

## Decision — every embed call sends a big LIST per HTTP request, and requests overlap under a sized semaphore, instead of one text per round-trip

`CONCEPT:AU-KG.ingest.applying-agents-md-batch`

`test_embed_throughput.py:1-9` names the trigger directly: a north-star e2e
run showed embeds "issued one HTTP round-trip at a time (the host log showed
one `POST /v1/embeddings` every ~2-3s), dropping `parallelism_factor` to
~1.83." `make_embed_fn` (`semantic.py:122-140`) fixes this with two
compounding changes, both attributed to the same principle in the code
comment: "applying the AGENTS.md *batch-never-per-element* rule to
embeddings" (`semantic.py:126`).

1. **BATCH** — every request carries a big LIST of inputs, auto-sized up to
   `_EMBED_MAX_BATCH = 256` (`semantic.py:25-30`), and the underlying
   llama-index model's `embed_batch_size` is pinned to the same value so it
   stops silently re-splitting one chunk into `DEFAULT_EMBED_BATCH_SIZE`
   (=10)-sized sub-POSTs — the exact per-element leak the AGENTS.md rule
   targets (`semantic.py:129-132`). `_auto_batch` (`semantic.py:107-119`)
   sizes each request to land ~`concurrency` chunks total, each chunk a big
   LIST, clamped to `[32, _EMBED_MAX_BATCH]` — big enough to amortize a POST,
   small enough to leave enough chunks to fill the concurrent lanes.
2. **CONCURRENCY** — chunks are fanned out concurrently up to
   `_embed_concurrency()` (`semantic.py:33-69`), so the enrich stage is never
   one-request-in-flight even with batching alone.

**The rejected alternative is the one-text-per-request loop the fix
replaces** — every text embedded as its own HTTP round-trip, serially. The
regression the fix repairs is literally that shape (confirmed live via the
host log cadence in the test docstring), and it is rejected on cost grounds
that are architectural, not incidental: bge-m3 (the deployed embedder)
"handles large batches per request" (`semantic.py:25-26`), so paying one
round-trip per text wastes exactly the throughput headroom the server
already offers.

**A second, narrower rejected alternative is uncapped fan-out.** The
concurrency anchor is deliberately triple-bounded, not just "as fast as
possible": (a) the model's own *declared* parallel capacity
(`resolve_capacity("embedding")`) is a hard per-model ceiling this local
fan-out may never exceed (`semantic.py:40-41`, 51); (b) the embedder
server's real capacity ceiling additionally caps it
(`server_ceiling("embedding")`, `semantic.py:52-54`); (c) while
failed-over onto a GPU shared with the generator, `_joint_budget_cap`
clamps the fan-out further to the shared-accelerator's joint budget so bulk
embeds "cannot OOM the host" (`semantic.py:72-104`) — a no-op when the
primary embedder has its own dedicated endpoint (`semantic.py:100-101`).
Concurrency here is sized "big enough to fix the regression," not "as wide
as the box allows."

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/enrichment/semantic.py`
  (`make_embed_fn` and its three sizing helpers). Callers see the same
  `EmbedFn` signature — batching/concurrency are internal.
- **Backward Compatible**: Yes — `batch_size` still pins an explicit
  per-request size (mainly for tests); batch boundaries and output order are
  preserved (`semantic.py:138-140`).
- **Known weak point**: `_embed_concurrency()` re-resolves capacity on every
  call (config reloads, adaptive-controller state, endpoint failover can all
  change the safe width), which is the intended behavior, but it also means
  the effective concurrency for two calls issued moments apart can silently
  differ with no log line explaining why throughput changed.
