---
name: graph-research-and-analysis
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
