# Design Document: GraphRAG local/global search is a gap-fill over existing engine primitives, and its prompts are packaged JSON, never a second inline copy

CONCEPT:AU-KG.retrieval.graph-engineering-canonical-prompts

> `agent_utilities/knowledge_graph/retrieval/graph_engineering.py` (the
> retrieval surface), `agent_utilities/prompts/canonical.py` (the shared
> prompt loader both sites cite under this one concept id).

## Decision — reuse the Rust engine's community detection and the existing ContextCompiler; add only the two GraphRAG query modes and their prompts

`graph_engineering.py:3-40` frames this as a **gap-fill, not a from-scratch
implementation**: community detection already runs natively in the engine
(`eg-compute::mining::community` — Louvain + Label Propagation) and
community-*report* generation already exists as an ingest-time pipeline
phase. What was missing was (1) an on-demand path to (re)build those reports
over an already-ingested graph with an embedding for semantic findability,
and (2) the two GraphRAG query modes that read them back: `local_search`
(entity + relationship-path neighborhood, seeded by a bounded parameterized
Cypher match guided by the `kg_graph_query` canonical prompt, falling back to
semantic search) and `global_search` (bounded map-reduce over
`:CommunityReport` nodes — MAP ranks reports by embedding-cosine similarity
then a bounded per-report LLM call scores a partial answer; REDUCE synthesizes
the top-scored partials).

**The rejected alternative**, stated as a dependency discipline directly: "no
new ML dependency — embeddings reuse `core.embedding_utilities.
create_embedding_model`... community detection stays 100% inside the Rust
engine; every native-Cypher call goes through the engine backend's existing
parameterized/escaped `execute_read` (never a hand-inlined literal)." Building
a second, GraphRAG-specific community detector or a second Cypher-execution
path would duplicate machinery the engine and the ingestion pipeline already
own — and, for the Cypher path specifically, would reopen an injection
surface the existing parameterized `execute_read` already closed. Both search
modes hand their final synthesis to the EXISTING `ContextCompiler` (policy
enforcement, MMR diversity, budget-fit, citations, proof graph — "all reused,
nothing reimplemented") via a small static-candidate adapter, rather than
each query mode assembling and citing its own answer independently.

**Canonical prompts, one source of truth.** `canonical.py:3-24` is the second
site under this same concept id: it loads one of the five packaged canonical
KG-operation prompt blueprints (`kg_extraction`, `kg_normalization`,
`kg_graph_query`, `kg_grounded_answer`, `kg_graph_maintenance`) by name and
renders it with `string.Template` — explicitly NOT `str.format`, because
several prompts embed literal JSON examples whose braces would collide with
`str.format`'s field syntax. **The rejected alternative** is named directly:
"callers... load their prompt text from here rather than keeping a second
inline copy, so the packaged JSON is genuinely wired, not a parallel unused
document" — the failure mode a packaged-but-uncalled prompt file represents
elsewhere in the codebase. Every reader degrades to a caller-supplied
`fallback` on any load error, so a packaging edge case never breaks a live
extraction/query/answer path.

## Risk Assessment

- **Blast Radius**: `graph_engineering.py`, `canonical.py`,
  `context_compiler.py` (reused, not modified), `pipeline/phases/
  community_reports.py`, `mcp/tools/graph_engineering_tools.py`.
- **Backward Compatible**: Yes — both query modes are additive read paths
  over existing engine/pipeline state.
- **Known weak point**: `global_search`'s MAP stage bounds the per-report LLM
  call count, which means a query whose true answer is scattered across more
  communities than the bound allows can miss supporting evidence the REDUCE
  stage never sees — a recall ceiling inherent to bounded map-reduce, not a
  bug.
