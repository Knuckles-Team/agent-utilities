---
name: graph-query-and-explanation
skill_type: skill
description: >-
  Answer bounded read-only questions over Graph-OS with grounded evidence. Use
  for natural-language or multi-step data questions, Cypher or UQL reads, local
  or federated semantic search, code navigation, document trees, tables, metrics,
  geospatial lookups, epistemic answers, or cited explanations. For a
  hypothesis-driven multi-source study, mining, learning, or causal evaluation,
  use graph-research-and-analysis.
---

# Graph query and explanation

Turn a question into the smallest safe read plan, execute it, and return an
answer that separates evidence from inference.

## Choose the route

1. Use `graph_ask` for an ordinary natural-language question.
2. Use `ask_data` for a bounded multi-step data question that needs query
   planning, execution, and correction.
3. Use `graph_search` for local discovery and `graph_federated_search` across
   registered external graph references; add `graph_search_synthesis` only when
   the user wants a synthesized answer over several results.
4. Use `graph_query` for an explicit read-only Cypher or UQL query. Use
   `nl_query` when the request needs query planning but no query was supplied.
5. Use `graph_code` or `graph_code_nav` for symbols, callers, definitions, and
   impact. Use `graph_document_tree` for document structure.
6. Use `graph_analyze`, `graph_explain`, `graph_evaluate`, or `graph_epistemic`
   when the answer needs diagnosis, confidence, disagreement, or provenance.
7. Use `graph_table`, `graph_promql`, or `graph_gis` for tabular, metric, or
   spatial reads.

If the task spans several routes or needs independent verification, delegate
through `graph_agents` and require a final evidence synthesis. Keep a
single bounded lookup direct.

## Action reference

| Tool | Actions | Notes |
|---|---|---|
| `graph_ask` / `ask_data` | NL question → generated query (`dialect`=auto\|cypher\|sql\|sparql) + rows | `execute=false` previews the generated query without running it |
| `graph_code` | `code_context` (`target`=how\|usage\|impact), `cross_repo_usages`, `call_graph`, `similar_code`, `routes`, `change_coupling`, `code_evolution`, `blast_radius`, `code_metrics`, `arch_report`, `adr` | query the KG before grepping |
| `graph_code_nav` | `find_definition`, `find_references`, `trace_call_graph`, `impact_of_change`, `connects` (shortest path between two symbols) | start from a `symbol` or exact `node_id`; optional `source_system` scope |
| `graph_context` | `put`/`get`/`list` — a session-scoped `ContextBlob` key/value store, optional `ttl_s` | linked to a `Session` node for id-anchored retrieval |
| `graph_document_tree` | `build`, `structure` (token-cheap text-free table of contents), `content` (fetch cited char/page `ranges`), `retrieve` (tree-walk by relevance, `use_llm=true` tries LLM navigation first) | vectorless PageIndex-style retrieval for one long document — cited `start..end` ranges beat an embedder's recall ceiling; complements `graph_search`, doesn't replace it |
| `graph_analyze` | `inspect`, `enrichment_coverage`, `process_writeback` (push KG intelligence to Camunda/ARIS, `target=camunda\|aris\|both`), `placement_plan` (workload placement), `infra_sweep`, `security_scan` | structural/ops analysis; code/research/eval/Q&A intents route to `graph_code`/`graph_research`/`graph_evaluate`/`graph_explain` instead |
| `graph_explain` | `action=explain`/`context`, `target=<domain>:<intent>` (`code`, `ops` live task-queue, `deploy` is-my-change-live, `entity`/`tickets`/`process`), `target=domains` lists providers | the universal context plane — routes to the right domain provider and returns ONE cited answer |
| `graph_epistemic` | `why` (=`explain_belief`), `status` (=`epistemic_status`), `what_changed`, `resolve_conflict` (argumentation-based resolution over contradicting claims) | purpose-named wrapper over the epistemic layers below — see "Epistemic answers" |
| `graph_search` | `mode`: `hybrid` (default), `hyde`, `deep`, `concept` (look up a `CONCEPT:ID`), `analogy`, `memory`, `discover`, `latent`, `rerank`, `adore`, `hard_negatives`, `chrono_ids`, **`compiled`**; `top_k`, `self_correct`, `as_of`, `target` (named/`all` connections) | `mode="compiled"` is the policy-aware context compiler — see "Compiled context bundles" |
| `graph_search_synthesis` | `synthesize` (evidence subgraph + multi-hop question around an `answer_id`), `diagnose` (solver trajectories / FORT signatures) | |
| `graph_federated_search` | fans a `query` across registered external graph `references`, capped by `top_k` | for ONE specific reference by id instead, use `graph_query(scope="federated")` |
| `graph_table` | `query` (read-only SELECT), `ingest` (mirror a connector into a table), `rows` (insert dicts), `create`, `list`, `drop` | the SQL-table surface of the engine |
| `graph_promql` | `action=instant` (single evaluation at `time`, default now) or `range` (`start`..`end` at `step`) | extra engine kwargs via `params_json`; degrades cleanly with no metrics surface |
| `graph_gis` | `route` (`from`+`to`[+`profile`]), `tile` (`z/x/y`), `nearest` (`lat`+`lon`[+`limit`]), `geo_task` | degrades cleanly with no GIS surface |
| `graph_engineering` (GraphRAG) | `local_search` (one entity + relationship-path neighborhood via the SAME `ContextCompiler`), `global_search` (bounded map-reduce over `:CommunityReport` nodes, `max_communities` default 8), `build_community_reports` | reuses the engine's existing Louvain/label-propagation community detection and the ingest-time report summarizer — no reimplementation; auto-builds reports on first `global_search` use |
| `graph_query` (`scope="sql"`) | read-only SQL over the KG + user tables via the engine's DataFusion surface — the same path the pg-wire listener serves | the query text goes in the `cypher` field regardless of dialect; non-`SELECT` is refused |
| `graph_query` (`scope="sparql"`) | SPARQL 1.1 `SELECT`/`ASK`/`CONSTRUCT`/`DESCRIBE` over the engine's RDF projection of the live graph | RLS-governed exactly like the default Cypher path |
| `graph_query` (`scope="federated"`) | query one registered `ExternalGraphReference` node by `reference_id` | ranking across SEVERAL external graphs at once → `graph_federated_search` instead |
| `engine_query` (`action="uql"`) | UQL — the engine's native cross-modal query language: one pipelined text query composing `MATCH`/`TRAVERSE`/`RANK BY`/`WHERE`/`AS OF`/`RERANK`/`FUSE`/`EVIDENCE FOR`/`BELIEF AS OF` over one snapshot, parsing to the exact same `wire::Plan` the structured `action="unified"` API builds — no second execution path | the keyword arg is **`text`**, not `query`; an unsupported-in-this-build clause degrades to `{"error": ...}`, never a silent wrong answer; full grammar + ~20 worked examples in `references/uql-reference.md` |

### Epistemic answers — confidence, provenance, and "why do we believe this"

A plain `graph_query`/`graph_ask` gives you rows. Four `engine_query` actions layer the
engine's epistemic currency on top, progressively deeper and progressively more
feature-gated (call directly, or through the `graph_epistemic` wrapper above):

1. **`explain_provenance_by_ids`** (CONCEPT:AU-KG.memory.knowledge-currency / Seam 1) — take any
   id list from a prior read and "upgrade" it to
   calibrated, cited, time-versioned rows (`confidence`, `valid_time`, `tx_time`,
   `source_refs`, `policy_labels`, `evidence_spans`). In the default `full` build already.
   `include_epistemic=true` on `graph_query`/`graph_ask` (Cypher dialect) skips this
   two-step dance — each row is already an `EpistemicRow`.
2. **`explain_belief`** — the full `JustificationGraph` rooted at a node: `rule` is one of
   `Asserted`/`DerivedSupport`/`DerivedContradiction`/`BayesianUpdate`. Also in `full`.
   Pass `disclosure_level` (`Full`/`Skeleton`/`ExistenceOnly`) for policy-aware redaction.
3. **`epistemic_status`** (what do we believe, why, under whose authority, what would
   invalidate it) and **`what_changed`** (a whole-graph bitemporal diff between two
   transaction times) — require the **opt-in `epistemic-tms` engine feature**, not in the
   default `full` build. A `{"error": ...}` means "capstone unavailable here," not a crash.
4. **`explain_policy`** — runs a search plan against both the caller's RLS-filtered view and
   the unfiltered one, returning `visible_ids`/`policy_denied_ids` — use when an expected row
   is missing and you need to know whether policy hid it.

**Evidence citation** — `engine_query(action="explain_evidence")` walks the same
support/contradiction/attack topology as `explain_belief`, but returns every
transitively-reachable **located evidence locus** instead (11 `EvidenceSpan` kinds:
`DocumentSpan`, `PageBox`, `CodeSymbol`, `AudioSegment`, `TraceSpan`, …) plus the
`AssetOccurrence`/`Blob` identity chain. Requires the opt-in `evidence-graph` engine
feature; every locus resolves to a real CAS-digest reference, never a fabricated excerpt.

### Compiled context bundles

`graph_search(mode="compiled")` reuses the same engine ANN/hybrid retriever every other
search mode calls, but additionally MMR-diversifies the candidate set, scores evidence
quality from the `KnowledgeBatch`-shaped epistemic columns (degrading to a neutral prior
when absent, never fabricated), weighs bi-temporal freshness, enforces the token budget
(logging every drop — `token_budget=0`, the default, uses the compiler's own internal
budget rather than an unbounded one), and runs the live permissioning gate on every
candidate before returning citations + a proof graph + a decisions log. Same candidates +
session ⇒ same bundle deterministically (diffable for audit/benchmark); `as_of` compiles a
bundle as-of a past point in time. Returns a formatted **text** bundle, not structured
JSON — parse the citations/proof-graph/decisions sections out of it if you need them
programmatically.

## Workflow

### 1. Frame the question

- State the target, scope, time boundary, and requested output.
- Identify whether the user supplied a query or expects natural-language
  planning.
- Treat ambiguous entity names as search terms before using them as identifiers.

### 2. Retrieve narrowly

- Start with the most selective operation and a bounded result count.
- Keep reads tenant- and graph-scoped when that context is available.
- Request only fields needed for the answer.
- Do not turn a read request into a write, ingestion, or workflow execution.

Example direct plan:

```text
graph_search(query="sample service dependency", mode="hybrid", top_k=8)
graph_query(cypher="MATCH (s:Service {id: $id})-[:DEPENDS_ON]->(d) RETURN d.id", params='{"id":"sample-service"}')
```

### 3. Cross-check important claims

- Confirm a semantic match with a structural query when feasible.
- Use `graph_epistemic` or `graph_evaluate` when sources conflict.
- Preserve returned citations, node identifiers, confidence, and time context.
- Label any conclusion not directly returned by a source as an inference.

### 4. Report

Return:

1. the concise answer;
2. the supporting evidence or citations;
3. uncertainty, conflicts, and material gaps;
4. the query or retrieval route used when reproducibility matters.

Use an economy model for bounded retrieval, filtering, and formatting. Escalate
only when evidence conflicts, causal judgment is requested, or synthesis spans
multiple domains.

## Guardrails

- Keep Cypher and UQL read-only.
- Parameterize identifiers and values; never interpolate untrusted text into a
  query.
- Do not claim that an empty result proves absence outside the queried scope.
- Do not expose hidden credentials, tokens, or unrelated tenant data.
- Stop and ask for scope when two interpretations would produce materially
  different answers.
