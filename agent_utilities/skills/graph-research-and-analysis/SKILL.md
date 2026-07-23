---
name: graph-research-and-analysis
skill_type: skill
description: >-
  Run hypothesis-driven, scoped research, graph mining, learning, causal
  analysis, and evaluation in Graph-OS. Use for reproducible multi-source
  studies, comparative analysis, anomaly or association discovery, graph
  learning, causal evaluation, feedback, report persistence, or grounded
  proposals. For one bounded lookup or cited explanation, use
  graph-query-and-explanation.
---

# Graph research and analysis

Build a reproducible evidence set, apply the lightest suitable analysis, and
separate measured results from hypotheses.

## Choose the route

- Use `graph_research` to organize a research question and source set.
- Use `graph_analyze` for structural, comparative, impact, or diagnostic work.
- Use `graph_mine` or `graph_mine_deep` for associations, anomalies, and deeper
  discovery over an existing graph.
- Use `graph_learn` for supported graph-learning tasks.
- Use `graph_ops_causal` when an operational question requires causal evidence.
- Use `graph_feedback` to record corrections or outcome signals.
- Use `research_artifact` to persist an approved report with provenance.

Keep a bounded analysis direct. Delegate a multi-source study through
`graph_agents` when acquisition, independent analyses, critique, and
synthesis can run as separate work items.

## Action reference

| Tool | Actions | Notes |
|---|---|---|
| `graph_research` | `synthesize` (synthesize knowledge from a source), `deep_extract` (entity/relation extraction), `background_research`/`spawn_background` (background jobs — poll via `graph_ingest(action="status")`), `relevance_sweep`, `research_ingest`, `evolve_variants`, `track_citations` | `query` carries the source/topic |
| `graph_evaluate` | `evaluate`/`evaluate_alpha` (score outputs), `evaluate_harness`, `guard_corpus`, `harness_gate` (formal no-regression SHACL gate), `check_constraints`, `specialize` (SAI specialization cycle), `world_model_rollout`, `latent_efficiency_benchmark`, `evolve_model`, `forecast`, `causal`, `invariant` | evaluation, gates, and world-model reasoning |
| `graph_feedback` | `correction_type=`: `outcome` (adjust an entity's reward), `rule` (durable governance/voice/source rule consulted at retrieval), `eval` (add a regression case), `reads_avoided` (close the code_context reads-avoided loop), `action_outcome` (close the loop on any autonomous action so routing prefers what works), `gotcha` (pin a hard-won trap to a file/module so code lookups surface it) | |
| `research_artifact` | persists workflow execution outputs as typed `ExecutionSummary`/`PerformanceAnomaly` nodes — parses `execution_id`/`workflow_name`/timestamps/`status`/`steps_executed`/`raw_logs`, flags anomalies (step duration over threshold), links `HAS_EXECUTION`/`HAS_ANOMALY` | write via `graph_write(action="bulk_ingest")` or raw `add_node`/`add_edge` |
| `graph_ops_causal` | `root_cause` (rank probable causes for a failure `node_id`, favoring true topological-source causes over closer symptoms), `blast_radius` (downstream impact of a change `node_id`), `change_risk` (predict risk from blast radius + `incident_history_json`), `control_evidence` (gather + verify the evidence chain for a governance control), `join` (materialize `links_json` as real edges between EXISTING ids — creates zero new nodes) | joins entities already ingested by the fleet (langfuse-agent, container-manager-mcp, gitlab-api/repository-manager, servicenow-api/atlassian-agent, leanix-agent) and runs the existing causal-reasoning engine (`StructuralCausalModel` + `CausalVerifier` + `SpuriousnessDetector`) — no new traversal algorithm; supply `links_json` for an offline/test-friendly model, or omit it with an active engine + `node_id` to load the neighborhood live; `root_cause`/`blast_radius` accept `as_claim=true` to propose ONE citable, revisable Claim through the SAME governed `graph_claims propose` ClaimFlywheel path, ActionPolicy-gated (a denial adds `claim_denied` without blocking the read-only answer) |
| `graph_mine` / `graph_mine_deep` | `graph_mine` actions: `associate` (frequent-itemset + rules, Apriori/FP-Growth/Eclat, over `transactions` baskets or a graph-derived `source`; `writeback:true` ⇒ `:AssociationRule` nodes), `cluster` (DBSCAN default/hierarchical/GMM/k-medoids over a `features` matrix or a vector `source`; `writeback:true` ⇒ `:Cluster` nodes), `anomaly` (z-score default/isolation-forest/LOF/one-class-SVM over `features`/a 1-D `values` series/a vector `source`; `writeback:true` ⇒ `:Anomaly` nodes); `graph_mine_deep` dispatches the deep-learning family the pure-Rust engine deliberately does not implement — `deep_forecast` (LSTM), `deep_classify` (MLP), `autoencoder_anomaly`, `xgboost`, `embed` — to `agents/data-science-mcp` over MCP and folds results back as typed nodes (`:Forecast`/`:Classification`/`:Anomaly`/`:Embedding`) | ad hoc association-rule/clustering/anomaly/deep-learning discovery, distinct from the GOVERNED mining→claim flywheel below; degrades cleanly (`{"available":false, ...}`) when data-science-mcp is unreachable or its `[training]` extra isn't installed; the raw modality-tier router (`engine_mining`, empty-`action` self-discovery) lives in `graph-engine-and-modalities` |
| `graph_learn` | `fit` (learn a KAN link-predictor model over a graph-derived subgraph — every `node_label`ed node is a vertex, edges among them are positives, non-edges are sampled negatives; `writeback:true` ⇒ `:EdgeFunction` nodes), `predict` (score candidate links or the `top_k` highest-probability missing links with a fitted `model`; `writeback:true` ⇒ `:PredictedEdge` nodes) | interpretable per-feature edge functions (common-neighbors, Jaccard, Adamic-Adar, preferential attachment, PageRank-product, neighbor-cosine, …), not a black-box scorer; the friendlier fixed-action wrapper over the raw `engine_graphlearn` client in `graph-engine-and-modalities` |

### Provenance-aware causal reasoning (do-calculus)

Two `engine_query` actions — pure functions over the request, no graph read; both
require the opt-in `epistemic-causal` engine feature (not in the default `full`
build) — distinct from `graph_ops_causal` above, which reasons over the REAL
ingested ops entity graph:

- **`causal_estimate`** — genuine Pearl do-calculus `P(· | do(X₁=x₁, …))` over a
  caller-supplied linear-Gaussian structural causal model (`variables` in topological
  order, each `{"id", "parents": [[parent_id, weight]], "bias", "noise_var"}`).
  `do_values` fixes named variables via graph surgery — incoming edges to the
  `do`-fixed variable are CUT, not conditioned on, the operationally meaningful
  difference from `observe(X=x)`. Returns one calibrated `{"mean", "variance",
  "interval", "level"}` estimate per variable, in the same order as `variables`. Only
  the **intervention** op is wired; the engine's `observe`/`counterfactual` methods
  exist Rust-side but are crate-internal — no wire `Method` exposes a genuine
  counterfactual ("what WOULD have happened") from outside the engine today.
- **`rank_by_provenance`** — order caller-supplied `candidates` by a weighted blend of
  similarity AND evidence quality (source reliability, corroboration, calibration
  precision, freshness) so a well-sourced/well-corroborated/fresh result isn't
  outranked by a merely-more-similar unsourced one. `weights=
  {"similarity": w1, "evidence_quality": w2}`, default `{0.5, 0.5}`.

### Governed mining → claim flywheel

Two pipelines mine evidence into reviewable KG facts, gated by the SAME
promotion-governance + action-policy checks every autonomous action passes through —
neither exposes a direct `propose`/`accept` call; both only run as a byproduct of
advancing a Loop cycle (`graph_loops`, see `graph-orchestration-and-automation`):

- **Claim flywheel** (C4/C6) — a research Loop cycle's `trace_mining`+
  `insight_validation` stages (default ON) mine Episode/ToolCall/OutcomeEvaluation
  provenance into Claims with a five-state lifecycle (`proposed → validated →
  accepted → deprecated`/`retracted`; **`retracted` is terminal and sticky** — a
  rejected finding is never silently re-proposed from the same content-addressed
  finding id on a later pass). Every transition is an append-only
  `ClaimLifecycleEvent`, never a silent mutation of the Claim's own fields. Advance a
  cycle with `graph_loops(action="run", max_topics=5)` or `drive`, then inspect via
  `graph_query(cypher="MATCH (e:ClaimLifecycleEvent) WHERE e.claim_id = $id RETURN e
  ORDER BY e.at", params='{"id": "<claim_id>"}')`. Cross-reference a resulting claim
  id with `graph-query-and-explanation`'s epistemic-answer/evidence-citation tools
  for its justification tree and source loci. For ad hoc association-rule/clustering/
  anomaly mining OUTSIDE this governance (raw discovery, not claim promotion), use
  `graph_mine`/`graph_mine_deep` directly instead (Action reference above).
- **Placement advisor** (X-5) mirrors the same pattern for infrastructure: mines
  agent-trace co-occurrence (tenant/tool/entity/modality access skew) into typed
  `PlacementProposal`s (`shard_split`/`replica`/`cache_prewarm`/`materialized_join`/
  `embedding_refresh`/`index_change`), persists each as a `Claim` (`status="proposal"`,
  `is_verified=False`), validates and policy-gates it (`apply_placement_change`,
  shipped `approval_required` by default — nothing auto-applies), then runs a
  **measured canary** (apply small, measure the SLO delta against
  `placement_canary_tolerance`, promote or roll back). On promote, the change reaches
  the engine's `PlacementCatalog` admin path (`engine_resharding` — an ADMIN domain
  requiring the `kg:admin` scope, same admin surface, no second placement authority).
  Manual trigger for one governed pass: `graph_loops(action="placement_control",
  placement_scan_limit=200, placement_canary_tolerance=0.10)`. **Nothing runs this on
  a schedule automatically** — wire it into `graph_schedules`
  (`graph-orchestration-and-automation`) for a continuous cadence. A stuck proposal
  awaiting approval is granted via `graph_orchestrate(action="grant_approval", ...)`.

## Workflow

### 1. Define the claim and acceptance test

- State the question, decision it informs, population, time window, and success
  criteria.
- List assumptions that would invalidate the result.
- Decide which outputs are descriptive, predictive, or causal.

### 2. Assemble evidence

- Fix the authorized tenant, graph, audience, source set, and population before
  retrieval or delegation.
- Reuse already-ingested evidence before acquiring more.
- Track source identity, observed time, content hash, and relevance.
- Keep contradictory evidence; do not collapse it into a single claim early.
- Bound the corpus and record why each source was included.
- Request only fields needed for the claim; exclude secrets, direct personal
  identifiers, private endpoints, and unrelated records, and use pseudonyms when
  identity is not part of the analysis.

### 3. Analyze

- Start with descriptive graph structure and baselines.
- Use mining or learning only when the data and task support it.
- Compare against a control, prior result, or held-out sample when possible.
- For causal conclusions, state the intervention, outcome, confounders, and
  identification assumptions.

### 4. Challenge the result

- Run an independent query or alternative method for material claims.
- Check leakage, duplicated evidence, stale data, and selection bias.
- Report uncertainty and negative results.
- Treat mined patterns and predicted links as proposals until reviewed.

### 5. Synthesize and persist

Return the question, method, evidence, findings, uncertainty, and recommended
next action. For read-only work, stop after returning the redacted synthesis; do
not call `research_artifact` or `graph_feedback`. Otherwise, persist with
`research_artifact` only when the report is ready for reuse and authorized for
its audience. Persist neutral source references, digests, and redacted findings
rather than raw source content, endpoints, identities, or trace values. Record
authorized corrections with `graph_feedback`.

Use an economy model for retrieval, extraction, deduplication, and evidence
tables. Use a stronger model for adversarial critique, causal judgment, and the
final synthesis.

## Guardrails

- Do not fabricate measurements, citations, confidence, or source coverage.
- Do not infer causation from correlation alone.
- Preserve tenant, graph, authorization, and audience boundaries across direct
  and delegated work.
- Do not expose credentials, tokens, personal data, private endpoints, or
  unrelated records in prompts, evidence tables, reports, or artifacts.
- Keep compute and result sizes bounded.
- Do not let an analysis job's acceptance stand in for a verified result.
