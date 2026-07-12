---
name: kg-uql
skill_type: skill
description: >-
  Runs UQL — the epistemic-graph engine's native cross-modal Unified Query Language —
  a single pipelined text query that composes graph traversal, vector/ANN ranking,
  BM25 text search, relational (DataFusion) filtering, bi-temporal AS-OF/WINDOW,
  OWL reasoning, federation, and epistemic belief/evidence ops over ONE snapshot with
  no impedance mismatch between modalities. Use when a question spans more than one
  modality in one breath — "find docs about X, then rank by similarity to this
  vector, then only the ones live as of last month", "traverse citations then
  diversify the top results", "what supports this claim, discounted by our
  confidence at the time" — or whenever you'd otherwise chain several separate
  Cypher/SQL/vector calls by hand.
license: MIT
tags: [graph-os, engine, uql, query, cross-modal, vector, graph, epistemic]
tier: core
wraps: [engine_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-uql

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_query` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_query"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to a query surface for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.

## What UQL is

UQL is a **pure front-end**: a UQL string parses (`eg_plan::uql::parse`,
`epistemic-graph/crates/eg-plan/src/uql/parser.rs`) to the *exact same* `wire::Plan`
(an ordered list of `Op`s) the structured `UnifiedQuery` API builds by hand — it adds
**no new execution path**. Every stage is a function `RowSet -> RowSet` over one
shared currency (an ordered list of `(id, optional score)` rows), so a `MATCH` (graph
scan) can feed a `TRAVERSE` (graph hop) can feed a `RANK BY` (vector kNN) can feed a
`WHERE` (real DataFusion) can feed a bi-temporal `AS OF`, all inside **one pipeline
over one off-lock snapshot** — no separate round-trips, no result-stitching in
Python. This is the engine's flagship cross-modal query surface; wrap your head
around it via the full grammar + worked examples in
[`references/uql-reference.md`](references/uql-reference.md) before writing anything
non-trivial.

## Invoke

The AU→engine surface is `QueryMixin.uql` (`agent_utilities/knowledge_graph/orchestration/engine_query.py:279`),
reached over the generic `engine_query` MCP/REST passthrough (`tool="engine_query"`,
`action="uql"` — `agent_utilities/mcp/_graphos_action_manifest.py:715`). The generic
dispatcher (`agent_utilities/mcp/tools/engine_tools.py:471` `_dispatch`) calls
`getattr(client.query, "uql")(**params)` where `params` is exactly the decoded
`params_json` object — so its keys must match the engine client's real method
signature: `QueryClient.uql(self, text: str, reorder_filter_selectivity: float |
None = None)` (`epistemic_graph/client.py:3061`). **The keyword is `text`, not
`query`** — a common mistake since the Python docstring examples write
`c.query.uql("MATCH ...")` positionally.

- **MCP:** `load_tools(tools=["engine_query"])`, then:
  ```
  engine_query(action="uql", params_json="{\"text\": \"MATCH (:Doc) WHERE year > 2024 |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[1.0, 0.0, 0.0, 0.0] |> LIMIT 10\"}")
  ```
  Pinned intent-surface form (identical call, explicit tool routing):
  `{"tool": "engine_query", "action": "uql", "params_json": "{\"text\": \"...\"}"}`.
- **REST twin:** `POST /engine/query` with
  `{"action": "uql", "params_json": "{\"text\": \"...\"}"}`.
- **Structured (builder) form**, when you need an `Op` the UQL grammar doesn't
  surface yet (see the reference's "known gaps" — e.g. `TsScan`, `SpatialScan`,
  `TensorScan`, `Cep`, `Probabilistic`, `SparqlBgp`): `action="unified"`,
  `params_json='{"plan": [{"Scan": {"label": "Doc"}}, {"Limit": {"k": 10}}]}'`
  (`QueryClient.unified`, `epistemic_graph/client.py:3024`).
- **Result shape:** `[{"id": str, "score": float | None}, ...]`, in the plan's final
  (post-`RANK`/`RERANK`) order. A UQL **syntax** error surfaces as the transport's
  raised error (a caret-annotated parse message); an **unsupported-in-this-build**
  clause (a feature-gated Op the running server wasn't compiled with) degrades to a
  clear `{"error": "..."}` payload from `_dispatch`'s `TypeError`/exception catch —
  never a silent wrong answer.
- `include_epistemic=True` on the Python-level `QueryMixin.uql` (not yet exposed as
  an `engine_query` param — call `KnowledgeGraph`/`IntelligenceGraphEngine.uql`
  directly, or currency-upgrade the returned ids via `kg-epistemic-answer`'s
  `explain_provenance_by_ids` afterward) attaches confidence/provenance/bitemporal
  columns to the same ids in the same order.

## Examples

Single-modality (graph scan + property filter, real DataFusion):
```
engine_query(action="uql", params_json="{\"text\": \"MATCH (:Doc) WHERE year > 2024 AND lang = 'en'\"}")
```

Graph + vector, the canonical two-modality pipeline:
```
engine_query(action="uql", params_json="{\"text\": \"MATCH (:Doc) |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[0.1, 0.9, 0.0] |> LIMIT 10\"}")
```

Server-side NL→vector rank (no client-side embedding call — needs an embedder bound
on the serving side, `EG_UQL_TEXT_EMBEDDER=hash` for the dependency-free fallback):
```
engine_query(action="uql", params_json="{\"text\": \"MATCH (:Doc) |> RANK BY ~\\\"graph databases\\\" |> LIMIT 5\"}")
```

Bi-temporal point-in-time + diversity rerank (no `MATCH` needed — `AS OF` is a
legal leaf source):
```
engine_query(action="uql", params_json="{\"text\": \"AS OF @1700000000 |> RERANK MMR 0.5 5 |> LIMIT 5\"}")
```

Tri-modal hybrid retrieval — vector + BM25 + graph-proximity fused by reciprocal
rank (needs the `text` build feature):
```
engine_query(action="uql", params_json="{\"text\": \"MATCH (:Doc) |> FUSE [RANK BY ~[1.0, 0.0]] [TEXT \\\"graph databases\\\"] [RERANK NODE_DISTANCE FROM \\\"n1\\\"] |> LIMIT 5\"}")
```

Epistemic — evidence for a claim, discounted by belief-time confidence (needs the
`epistemic` build feature):
```
engine_query(action="uql", params_json="{\"text\": \"MATCH (:Claim) |> EVIDENCE FOR \\\"c1\\\" |> BELIEF AS OF @1700000000 |> LIMIT 10\"}")
```

For the complete grammar (every clause, with `file:line` citations into the parser
and executor), the modality-seams map, ~20 more worked cross-seam patterns from
simple to maximal, and the documented-but-unimplemented operators to treat as
**bug candidates**, read
[`references/uql-reference.md`](references/uql-reference.md).
