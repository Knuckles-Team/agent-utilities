# Design Document: Independent enrichment/ingest units fan out under ONE bounded semaphore, generalized across call sites — never a serial per-item loop

CONCEPT:AU-KG.ingest.generalized-cross-lane-parallelization

> `agent_utilities/knowledge_graph/ingestion/engine.py:1490-1529`
> (LLM concept/fact-extraction windows), `agent_utilities/knowledge_graph/
> ingestion/engine.py:2484-2529` (`_ingest_document_dir`, a directory of
> documents), `agent_utilities/automation/research_pipeline.py:1092-1100`
> (paper discovery/ingest), all three sharing the same
> `asyncio.Semaphore(...)` + `asyncio.gather(...)` shape.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.ingest.staged` | fans different STAGES of one item's pipeline out concurrently; this document fans independent ITEMS out concurrently WITHIN one stage/call — orthogonal, composable axes | 0.35 | KG |
| `AU-KG.ingest.applying-agents-md-batch` | the same "bounded concurrent fan-out replaces a serial per-element loop" shape, specialized to embedding HTTP requests specifically | 0.55 | KG |
| `AU-ORCH.execution.reserved-inference-slots` | the interactive-slot-reserving capacity ceiling this fan-out's semaphore is sized under at the LLM-enrichment site | 0.40 | ORCH |

### Extension Analysis

- **Primary Extension Point**: the `asyncio.Semaphore(max(1,
  compute_ingest_worker_count()))` (or the LLM-capacity variant) +
  `asyncio.gather(*(_one(x) for x in items), return_exceptions=True)` pair,
  repeated at each site.
- **Extension Strategy**: augment — a new independent-unit fan-out site
  copies this same sized-semaphore + gather shape rather than inventing a
  new concurrency primitive; there is no shared helper function today (see
  Known weak point).
- **New Concept Required?**: No.

## Decision — replace "N independent units processed one at a time" with "N units fanned out under a bounded semaphore," as a generalized pattern applied at each independent-unit site

`CONCEPT:AU-KG.ingest.generalized-cross-lane-parallelization`

Three call sites independently apply the same shape to three different
kinds of independent work:

1. **LLM enrichment windows** (`engine.py:1490-1529`) — a document's text is
   split into bounded windows; concept extraction fans each window out under
   a semaphore sized to `KG_LLM_CONCURRENCY` minus the reserved interactive
   slot (`engine.py:1493-1509`), replacing what the inline comment on the
   handler names directly: `_concepts_for` "was a serial per-window loop"
   (`engine.py:1513-1517`). vLLM batches server-side, so an N-window document
   costs ~N/concurrency instead of N sequential round-trips
   (`engine.py:1494-1496`).
2. **A directory of documents** (`_ingest_document_dir`,
   `engine.py:2484-2529`) — each file's sync `_ingest_document_file` is
   off-threaded and fanned out under `asyncio.Semaphore(max(1,
   compute_ingest_worker_count()))` (`engine.py:2516-2526`), so "a directory
   of N docs costs ~N/concurrency instead of N sequential read+LLM passes...
   Covers every lane that ingests a directory — downloaded papers, crawled
   web pages, a repo's docs" (`engine.py:2490-2495`).
3. **Research-paper scoring/ingest** (`research_pipeline.py:1080-1100`) —
   "Score + ingest every paper CONCURRENTLY... Each paper's full/abstract
   ingest (the heavy PDF + LLM work) is independent, so they fan out under a
   bounded semaphore sized to the ingest worker count; vLLM batches
   server-side, turning an N-paper run from N sequential ingests into
   ~N/concurrency." The comment adds the correctness argument for why this
   is safe: "the per-paper helper is pure (returns its record, mutates no
   shared state), so counters are tallied race-free after the gather"
   (`research_pipeline.py:1096-1099`).

**The rejected alternative, named explicitly at the first site, is the
serial per-item loop each of these replaced** — process unit 1 fully, then
unit 2, then unit 3. That shape is correct but leaves the dominant cost
(an LLM round-trip, or a PDF fetch+parse+LLM pass) fully serialized even
though the units share no state and the backing server (vLLM, the embedder)
is built to batch/parallelize server-side. The pattern is deliberately
GENERALIZED rather than fixed once at the enrichment-window site: it
recurs at the document-directory site and the paper-ingest site as the
SAME shape — a sized semaphore plus `asyncio.gather(...,
return_exceptions=True)` — because each is, independently, "N independent
units, one dominant blocking cost per unit, no shared mutable state."

**The rejected alternative to unbounded fan-out** (just `asyncio.gather` with
no semaphore) is also explicit at every site: the semaphore is always sized
to a real ceiling — `compute_ingest_worker_count()` (the shared cpu/mem
sizing anchor with the Pi-OOM cap) for CPU/IO-bound directory and paper
fan-out, or `KG_LLM_CONCURRENCY` minus `RESERVED_INTERACTIVE_INSTANCES` for
LLM-bound window fan-out — "so this background sweep can never starve the
slot the messaging responder / graph-os-spawned agents need to answer"
(`engine.py:1496-1499`). Unbounded concurrency would maximize throughput on
paper but starve interactive work or exceed a GPU's real capacity, exactly
the failure the reserved-slot subtraction and the shared cpu anchor both
guard against.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/engine.py`
  (`_enrich_document` / `_ingest_document_dir`),
  `agent_utilities/automation/research_pipeline.py`.
- **Backward Compatible**: Yes — each site's public method signature is
  unchanged; only the internal fan-out shape changed from serial to
  concurrent-bounded.
- **Known weak point**: the sized-semaphore-plus-gather shape is
  **duplicated at each site** rather than factored into one shared helper —
  there is no single `fan_out_bounded(items, worker_count, fn)` utility
  today. A fourth independent-unit site can (and, per the module comments
  above, is expected to) copy the pattern by hand, which means a future
  sizing-policy change (e.g. how the interactive reservation is subtracted)
  has to be applied at each call site individually rather than in one place.
