# Document → Knowledge-Graph Fact Extraction

**Concepts:** KG-2.64 (fact extractor), KG-2.65 (GPU-slot scheduler), KG-2.66
(readability reader), AU-ECO.connector.git-task-resolver (three-frontend extraction UI).

Turn any document, URL, or pasted text into an interactive knowledge graph: the
LLM emits atomic `(subject) --[predicate]--> (object)` facts — each carrying a
title, description, verbatim evidence span, confidence, and tags — which stream
live onto a force-directed graph, dedup semantically, and persist as engine
edges. The capability was assimilated from the open-source
`knowledge-graph-extractor` and re-grounded on our own embedder, engine edge
model, durable queue, and three frontends — **no new always-on service**.

## Why it exists

We already had claim/entity extraction (`kb/`), multi-backend graph storage,
OWL/RDF, and an ontology layer. What we lacked was the *extraction craft* that
makes a graph **connect**: a prompt that forces short canonical entity/value
nodes (so the same entity comes out identical every time and edges join instead
of becoming prose dead-ends), per-fact evidence + confidence, live streaming, and
a single-GPU job discipline. Those are the four pieces this subsystem adds.

## System flow

<div class="admonition architecture" markdown>
<p class="admonition-title">System flow: input to persisted edges, streamed live to three frontends</p>

Input (pasted text, a URL, or a local file) reaches `extract_facts`
(`fact_extractor`, KG-2.64) — a URL first passes through `ReaderConnector`
(KG-2.66, Jina/trafilatura/strip). `extract_facts` drives
`make_streaming_extract_fn` (vLLM, JSON-schema, sampling profile), whose
token deltas feed `parse_facts_incremental` (brace-match partial
stream), which feeds `FactDeduper` (vectorized cosine, shared embedder):
unique facts are kept, duplicates are marked `is_duplicate`.

Kept facts reach `persist_facts`, which writes engine edges
(confidence/evidence/tags/source, variant node-keys merged) into the
epistemic-graph durable backends, and are also written to
`/extract/jsonl` -> `facts.jsonl`.

The same extraction emits events over
`/api/enhanced/extract/stream` (`round_start`/`fact`/`metrics`/
`round_end`/`done`/`job_done`), consumed live by all three frontends:
`agent-webui` (Sigma.js graph), `agent-terminal-ui` (fact rows), and
`geniusbot` (QGraphicsView force graph).
</div>

The same `extract_facts` async generator drives **both** persistence and the SSE
stream, so what a user watches stream in is exactly what lands in the graph.

## The extraction contract

Each fact is wire-compatible with the upstream `facts.jsonl` schema so our export
is byte-comparable:

| Field | Meaning |
|-------|---------|
| `subject` / `object` | **canonical** short entity or atomic value — the graph nodes |
| `predicate` | precise `snake_case` relation (free-form, coined per fact) |
| `title` / `description` | natural-language statement + evidence-bearing context |
| `evidence_span` | verbatim substring of the source (provenance/grounding) |
| `confidence` | 0–100 (persisted as 0–1 on the edge) |
| `tags` | lowercase topic/entity/year tags |
| `is_duplicate` / `source_file` | our streaming + multi-file provenance |

Node identity uses `ExtractedFact.normalize_key` (NFKC + lowercase + punctuation
strip) so `"The Jina AI team"` and `"jina ai"` collapse to one node — in the
graph **and** in every frontend (each reimplements the same `normKey`).

## GPU-slot scheduler (KG-2.65)

The durable Kafka queue (KG-2.55–2.57) fans work across hosts; the slot scheduler
governs the **one** GPU inference slot a host contends on. A fresh foreground
submission preempts the running job, which is checkpointed and auto-resumes.

<div class="admonition architecture" markdown>
<p class="admonition-title">GPU-slot scheduler: preemptible, checkpointed, auto-resuming</p>

A job is `submit`ted to `queued`, moves to `running` when the slot frees
(FIFO). From `running`, a new foreground submission or a user hold moves
it to `pausing`, which resolves to either `paused` (a cooperative
checkpoint at a file boundary) or `held` (a user hold). `paused` jobs
auto-backfill back to `queued` (preempted tier first); `held` jobs return
to `queued` only on an explicit resume. `running` otherwise ends in
`done` (finished) or `failed` (error).

A `paused` job's checkpoint (`done_files`/round) is persisted
graph-natively (`GraphCheckpointStore`), so it survives a redeploy.
</div>

The runner (`ExtractionJobManager._run_job`) checks `scheduler.should_pause()` at
each file boundary, persists its checkpoint, and yields. On restart,
`reconcile_on_startup` demotes any mid-flight job to `paused` (preempted tier) so
it rejoins backfill — nothing is lost across a deploy.

## Live streaming sequence

<div class="admonition architecture" markdown>
<p class="admonition-title">A late subscriber replays history before tailing live events</p>

The frontend `POST`s `/extract/submit {text|url}`, which the gateway
forwards to `ExtractionJobManager.submit()` (returning a `job_id`); the
manager enqueues it on `GpuSlotScheduler` (preempting a running job if
this one is foreground). The frontend then opens
`GET /extract/stream/{job_id}` (SSE), and the gateway calls
`manager.stream(job_id)`, which replays the buffer before tailing live.

Once the scheduler runs the job on the slot, the manager streams each
fact: a streaming completion request to vLLM (JSON schema) returns token
deltas, parsed incrementally (`parse_facts_incremental`) and sent to the
frontend as `data: {type:"fact", fact:{...}}`. A final
`data: {type:"job_done"}` event lets the frontend fetch
`GET /extract/jsonl/{job_id}` for `facts.jsonl`.
</div>

A late subscriber misses nothing: the manager keeps a bounded per-job event
buffer and replays history before tailing live events.

## Three-frontend topology (AU-ECO.connector.git-task-resolver)

One gateway contract, three native renderings:

<div class="admonition architecture" markdown>
<p class="admonition-title">One gateway contract, three native renderings</p>

The gateway (`/api/enhanced/extract/*`) serves three frontends directly,
each with its own native rendering: `agent-webui` (React) —
`ExtractionView` rendering a Sigma.js + ForceAtlas2 graph with edge-fact
hover cards, longest-path, and JSONL export; `agent-terminal-ui`
(Textual) — an `/ingest` command driving live colorized fact rows with
JSONL export; `geniusbot` (Qt) — `ExtractionCockpitPanel` rendering a
native QGraphicsView force graph with `relax_layout`, edge cards, and
JSONL export.
</div>

| Frontend | Graph | Edge metadata | Live stream | Export |
|----------|-------|---------------|-------------|--------|
| agent-webui | Sigma.js / ForceAtlas2 | hover card + dup badge | EventSource | JSONL + RDF |
| agent-terminal-ui | colorized fact table | inline per row | `client.stream_extraction` | JSONL |
| geniusbot | native QGraphicsView (`relax_layout`) | click → fact card | worker `progress` signal | JSONL |

## Serving discipline

Extraction is long structured-JSON generation on a contended GPU. The tuning that
matters — low-bit k-quant (~3.5 bpw floor), MTP-style speculative decoding,
single-stream, uniform-KV flash-attn — is documented in
[Single-GPU LLM serving](../guides/single-gpu-llm-serving.md). The extractor
already sets the matching sampling profile (temp 0.7, top_p 0.8, top_k 20,
presence_penalty 1.5, `enable_thinking=False`) and a strict JSON-schema response
format.

## Watched directories — auto-ingest a document corpus (KG-2.8)

The file-watcher (`sdd/watcher.py`, a leader-only maintenance tick every 5s)
auto-ingests documents that land in a set of **watched directories**, so a
corpus stays mirrored into the KG with **no manual ingest call**:

- **Built-in:** the ScholarX / research download dirs
  (`~/.local/share/scholarx/papers`, `~/.local/share/agent-utilities/research`,
  plus `SCHOLARX_PAPERS_DIR` / `AGENT_RESEARCH_DIR`) — scanned top-level,
  provenance `watcher_scholarx`.
- **Operator-configured:** any directories in **`KG_WATCH_DIRS`** — scanned
  **recursively** (noise dirs like `.git`/`node_modules` skipped), provenance
  `watcher_documents`. Point it at e.g. `~/Documents`.

Both flow through the **same** unified resolver
`get_watched_directories() -> [(path, recursive, source)]` and the same
per-file submit (`process_watched_file` → `engine.submit_task(task_type=
"document")`), then the standard document adaptor + this fact/concept extraction.

**New / modified / unchanged** are handled by the content-hash `DeltaManifest`
(`knowledge_graph/ingestion/manifest.py`): a **new** file is ingested, a
**modified** file (new hash) is re-ingested, and an **unchanged** file is
delta-skipped — validated at ~0.15s/file for a 150 MB PDF. So you can drop or
edit files in the directory and the KG converges on the next tick.

**Configure it** (deployment) via `config.json`:

```json
{ "kg_watch_dirs": "${HOME}/Documents" }
```

or env `KG_WATCH_DIRS=~/Documents` (JSON array or `os.pathsep`/comma list for
multiple). Supported file types: `.pdf .docx .doc .txt .md` — PDFs use the single
bounded pypdf path in a killable spawned worker. File, page, extracted-character, and
IPC limits are enforced; Unix workers also receive CPU/address-space limits, while every
platform enforces a wall deadline and terminates the worker on expiry. See
`docs/architecture/configuration.md` (`KG_WATCH_DIRS`).

## Entry points

- **MCP:** `graph_ingest action=fact_extract` (inline) or
  `extract_submit|extract_jobs|extract_status|extract_pause|extract_resume|extract_jsonl`
  (GPU-slot-scheduled).
- **REST:** `/api/enhanced/extract/{submit,stream/{id},jobs,status/{id},jsonl/{id},pause/{id},resume/{id}}`.
- **Code:** `agent_utilities.knowledge_graph.extraction` —
  `extract_facts`, `FactDeduper`, `persist_facts`, `facts_to_jsonl`,
  `ExtractionJobManager`; scheduler in
  `knowledge_graph/ingestion/gpu_slot_scheduler.py`; reader in
  `protocols/source_connectors/connectors/reader.py`.
- **Second-brain sync** (CONCEPT:AU-KG.enrichment.second-brain-note-sync,
  `graph_ingest action=sync_second_brain`,
  `knowledge_graph/extraction/second_brain_sync.py`) is a thin one-call
  composition over a personal notes corpus: `extract_facts`/`persist_facts`
  per note, then `kb.entity_claim_extractor.EntityClaimExtractor` (claims
  PROPOSED into the governed `ClaimFlywheel`) and
  `adaptation.contradiction_detector.ContradictionDetector` (a finding
  persists as a propose-only `:BeliefRevisionProposal`). See the
  `second-brain-sync` universal-skill for the full workflow.

## Design choices (and what we deliberately did not copy)

- **Reuse the existing embedder** for dedup — no second model in memory.
- **Recursive separator chunker** (`ontology/document_processing.py`) already
  exists; we did not port the upstream backoff chunker.
- **vLLM**, not llama.cpp — we assimilate the serving *knowledge*, not the runtime.
- **Sigma.js** stays the webui engine; geniusbot uses a native `QGraphicsView`
  force layout (no QWebEngine dependency) whose math is a pure, unit-tested
  `relax_layout`.
