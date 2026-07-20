---
name: kg-graphrag
skill_type: skill
description: >-
  GraphRAG-style Graph Engineering: entity-neighborhood local search, bounded
  map-reduce global search over community reports, and on-demand community-
  report (re)building. Gap-fill over existing capability -- community
  DETECTION already runs natively in the Rust engine
  (eg-compute::mining::community, Louvain/label-propagation) and report
  summarization reuses the exact prompt already used at ingest time; this
  skill adds the on-demand build path plus the two GraphRAG query modes. Use
  for "what's connected to X and how", "who/what does X relate to", "what are
  the main themes/topics in this graph", "summarize the communities", "give
  me a grounded answer with citations from the graph".
license: MIT
tags: [graph-os, graphrag, retrieval, community, local-search, global-search, context-compiler]
tier: core
wraps: [graph_engineering, graph_query, graph_search]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-graphrag

> **Condensed intent-surface note (Seam 8).** Under the default intent surface
> (`MCP_TOOL_MODE=intent`), `graph_engineering` is held back from the default
> tool list (nothing removed — REST + `_execute_tool` still reach it exactly
> as documented below). Two ways to use this skill unchanged: (1)
> `load_tools(tools=["graph_engineering"])` once per session (as below), then
> proceed exactly as documented; or (2) call the `why`/`find` intent verb with
> the same natural-language request — the resolver routes to
> `graph_engineering` for you. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both`
> to expose the granular tool eagerly instead.

`graph_engineering` is the GraphRAG (Microsoft GraphRAG method) query surface
over the knowledge graph. It does not reimplement anything the platform
already has — community DETECTION is the Rust engine's existing Louvain/
label-propagation mining family (`eg-compute::mining::community`), and report
summarization reuses the exact prompt already used at ingest time
(`pipeline/phases/community_reports.py`). What this skill adds is the
on-demand build path (so you can (re)build reports over an already-ingested
graph, not just at ingest time) plus the two GraphRAG query modes:

Actions:
- **`local_search`** — one entity + its relationship-path neighborhood. Give
  it either an explicit `node_id` or a free-text `query`; seed resolution
  tries a bounded, parameterized native-Cypher exact match first (guided by
  the `kg_graph_query` canonical prompt when an LLM is configured, and always
  falling back to trying the raw query text itself as a literal name — so it
  still works with no LLM at all), then falls back to semantic search. The
  neighborhood is assembled through the EXISTING `ContextCompiler` (policy
  enforcement, MMR diversity, budget-fit, citations, proof graph — all
  reused) and answered with the `kg_grounded_answer` canonical prompt.
- **`global_search`** — bounded map-reduce over `:CommunityReport` nodes.
  MAP: ranks reports by embedding-cosine similarity to the query, then a
  bounded (`max_communities`, default 8) per-report LLM call scores a partial
  answer. REDUCE: the top-scored partial answers go back through the SAME
  `ContextCompiler` + `kg_grounded_answer` prompt for final synthesis.
  Auto-builds community reports on first use if none exist yet.
- **`build_community_reports`** — explicitly (re)build `:CommunityReport`
  nodes over the LIVE graph: theme + summary per community (reusing the
  ingest-time summarizer), each one embedded so `global_search` can rank it,
  plus the single level-1 global rollup report.

Supply an active graph engine (resolved server-side); no bespoke setup beyond
having ingested data — `global_search`/`local_search` both degrade cleanly
(never crash) with no LLM/embedding model configured, at reduced answer
quality (deterministic themes, raw-text fallback ranking).

## Invoke
- **MCP:** `load_tools(tools=["graph_engineering"])`, then
  `graph_engineering(action="local_search", query="What does X do?")`.
- **REST twin:** `POST /graph/engineering` with
  `{"action": "global_search", "query": "What are the main themes here?"}`.

## Examples
```
graph_engineering(action="local_search", node_id="entity:acme-corp", depth=1)
graph_engineering(action="local_search", query="who works with Alice")
graph_engineering(action="global_search", query="what are the main risk themes", max_communities=5)
graph_engineering(action="build_community_reports", min_size=8, embed=true)
```

## The 5 canonical prompts

This skill's answer synthesis, and the ingestion/dedup/maintenance paths
elsewhere in the platform, share one packaged prompt library
(`agent_utilities/prompts/kg_*.json`, loaded via
`agent_utilities.prompts.canonical.load_canonical_prompt`):

| Prompt | Wired into |
|---|---|
| `kg_extraction` | `knowledge_graph.extraction.fact_extractor.FACT_EXTRACTION_PROMPT` (ingestion) |
| `kg_normalization` | `knowledge_graph.distillation.deduplicator.KnowledgeDeduplicator.merge_cluster` |
| `kg_graph_query` | `graph_engineering.local_search`'s seed resolution → native Cypher |
| `kg_grounded_answer` | `graph_engineering.local_search`/`global_search`'s reduce step, over a `ContextCompiler` bundle |
| `kg_graph_maintenance` | the `contradictions` action's `ContradictionDetector` findings (`mcp/tools/analysis_tools.py`) — a best-effort LLM recommendation layered on the deterministic detector |

## Honest limitations

- `global_search`'s MAP step is a bounded per-community LLM call (default cap
  8), not an exhaustive scan — this is deliberate cost control (see
  `pipeline/phases/community_reports.py`'s own `_MAX_COMMUNITIES` bound), not
  a bug; raise `max_communities` for a broader (costlier) sweep.
- The `kg_graph_query` canonical prompt only ever produces a structured
  `{entity_name, node_label}` JSON object, executed as a bounded, EXACT-match
  parameterized native-Cypher lookup — it never generates or executes raw
  Cypher text, by design (injection safety).
- A community report written before this skill shipped (or built with
  `embed=false`) has no `embedding` property — `global_search` still finds it
  (falls back to a deterministic largest-community-first order) but cannot
  rank it by relevance to the query until it is rebuilt.

## Delegation

If graph-os is reachable, offload composite multi-step work via
`graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of
hand-running the steps — let the local LLM + Loop engine do it, and resolve
only the exceptions.
