---
name: kg-context-compile
skill_type: skill
description: >-
  Assembles a policy-aware, cited, calibrated, token-budgeted LLM context bundle instead
  of a plain relevance-sorted text blob — scores candidates on relevance, MMR diversity,
  evidence quality/freshness (from the epistemic columns), and runs every candidate
  through the live permissioning gate before returning citations + a proof graph +
  a decisions log. Use when assembling context to feed an LLM prompt and you want it
  cited, deduplicated, freshness-aware, and policy-filtered — "build me a context bundle
  for this question", "compile cited context under a token budget".
license: MIT
tags: [graph-os, retrieval, context-compiler, citations, policy, calibration]
tier: core
wraps: [graph_search]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-context-compile

`graph_search(mode="compiled")` (X-7, CONCEPT:AU-KG.retrieval.context-compiler) reuses
the SAME engine ANN/hybrid retriever every other search mode calls, but additionally:

- **MMR-diversifies** the candidate set (greedy max-marginal-relevance over embedding
  cosine, falling back to lexical Jaccard) so near-duplicate hits don't crowd out
  distinct evidence.
- **Scores evidence quality** from the `KnowledgeBatch`-shaped epistemic columns when a
  result carries them (`confidence`/`source_refs`/`evidence_refs`/`proof_ids`/
  `contradiction_ids`/`policy_labels`) — the `"epistemic:contested"` policy label flags
  a disputed claim. Degrades to a neutral prior (never fabricated) when a result doesn't
  carry these columns.
- **Weighs bi-temporal freshness** — recency decay against `event_time`/`valid_from`.
- **Enforces the token budget** via the retrieval budget manager, logging every drop.
- **Runs the live permissioning gate** on every candidate — row-level drop, column-level
  redaction — the same fine-grained `ontology.permissioning.enforce` path the live read
  path uses. No bypass.

Same candidates + same session ⇒ same bundle deterministically, so two runs are
diffable for an audit/benchmark.

## Invoke

- **MCP:** `load_tools(tools=["graph_search"])`, then
  `graph_search(query="...", mode="compiled", top_k=10, token_budget=4000)`.
- **REST twin:** `POST /graph/search` with `{"query": "...", "mode": "compiled", "top_k": 10, "token_budget": 4000}`.

## Example

```jsonc
graph_search(query="what caused the payments-service latency spike",
             mode="compiled", top_k=8, token_budget=3000)
// -> a text bundle assembling the selected ContextItems, each with its per-axis
//    scores, a flat citations list, a proof_graph of supports/contradicts/
//    alternative-to edges, and a decisions log recording every selection/rejection.
```

`as_of` (an ISO-8601 instant) is honored for `mode="compiled"` exactly as for every
other search mode — pass it to compile a bundle as-of a past point in time rather than
now.

## Honest limitations

- `token_budget=0` (the default) uses the compiler's own internal default budget rather
  than an unbounded one — set it explicitly for a hard budget.
- Evidence-quality scoring reads epistemic columns populated by the ENGINE, not by AU —
  a result whose source path never ran through the epistemic-columns-populating write
  path (see `kg-epistemic-answer`'s currency-upgrade note) scores on relevance/diversity/
  freshness/policy alone, with a neutral evidence-quality prior. This is additive
  degradation, not a broken feature.
- Not yet routed through the KV-cache layering path (LMCache) — a compiled bundle is
  recomputed per call, not served from a shared warm cache. If you want to cache a
  compiled bundle's raw blocks yourself, `kg-kvcache` (`graph_kvcache`) is a separate,
  manual content-addressed cache primitive — the two are not wired together today.
- Returns a formatted text bundle (`bundle.as_text()`), not a structured JSON object —
  parse the citations/proof-graph/decisions sections out of the text if you need them
  programmatically; there is no separate structured-output mode for `compiled` today.

## See also

`kg-epistemic-answer` for a single node's own justification/confidence rather than a
whole assembled multi-source bundle; `kg-evidence-cite` for one node's exact multimodal
source loci.
