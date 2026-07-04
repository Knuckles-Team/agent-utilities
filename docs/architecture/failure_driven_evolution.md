# Failure-Driven Evolution (CONCEPT:AU-AHE.harness.failure-evolution)

> The self-evolution loop learns from **failures observed in production telemetry**,
> not only from research papers and unresolved concepts. Failures recorded in
> Langfuse become addressable gaps in the Knowledge Graph, which the golden loop
> turns into regression-gated remediation proposals.

## Why

The research-driven golden loop (`CONCEPT:AU-KG.query.vendor-agnostic-traversal`) assimilated papers, OSS, and
unresolved `Concept` topics — but a running fleet's most valuable signal, its own
failures, never entered the loop. Langfuse already received exported traces
(`CONCEPT:AU-AHE.harness.harness-evolution` Experience Observability), yet the integration was
write-only: nothing read failures back out, the `ExecutionSummary` /
`PerformanceAnomaly` schema sat dormant, and the daemon's `telemetry_ingestion`
sweep referenced a workflow that never existed (it raised every cycle). AU-AHE.harness.failure-evolution
closes that loop.

## The loop

```
Langfuse  ── pull ──▶  cluster ──▶  materialize ──▶  intake ──▶  remediate ──▶  regression-gated merge ──▶  lock regression
(errors,              (recurring     (KG nodes +       (golden     (TeamSpec/      (auto-merge only when      (AHE-3.25: on a
 low scores,           failure        failure_gap       loop)       AgentSpec        the failure is not         verified fix, lock a
 cost/latency)         signatures)    Concept topics)               proposal)        spiking; else hold)        plain-English assertion)
```

1. **Pull** — error observations, low-score traces, and cost/latency anomalies are
   pulled from Langfuse over a configurable window.
2. **Cluster** — records are grouped into recurring **failure signatures**
   (deterministic, LLM-free): `(workflow/observation name, normalized
   status/error class, anomaly type)`. Volatile parts (ids, paths, numbers) are
   normalized out so the *same* failure produces the *same* signature across
   occurrences. A pattern must recur at least `min_occurrences` times (default 2)
   to qualify — single one-offs are noise.
3. **Materialize** — each recurring pattern writes, into the durable KG:
   - a `PerformanceAnomaly` node (`anomaly_type ∈ {ERROR_RATE, LOW_SCORE, TIMEOUT,
     HIGH_COST, HIGH_TOKEN_USAGE}`),
   - an `ExecutionSummary` rollup per failing workflow (`success_rate < 1.0`, so
     `maintainer.trigger_self_improvement` picks it up), and
   - a synthetic **`failure_gap` `Concept`** topic — labelled `Concept` with **no
     `ADDRESSED_BY` edge** and `evidence_trace_ids` linking back to Langfuse — so
     the golden loop's existing `unresolved_topics()` intake surfaces it unchanged.
4. **Remediate** — the failure-ingest path runs one golden-loop cycle that
   addresses the just-materialized gaps **directly** (see *Targeting* below) and
   synthesizes a `TeamSpec`/`AgentSpec` remediation proposal.
5. **Regression-gated merge** — promotion of a failure remediation is gated by a
   regression check bound to the originating failures: it re-queries Langfuse and
   holds the proposal if any signature is actively spiking (current > baseline).
   Each gap is also appended to the durable eval corpus and the failing
   capability's reward is nudged down (`FeedbackService`).
6. **Lock regression (CONCEPT:AU-AHE.evaluation.failure-analysis-loop)** — when the gate *passes* (the fix holds
   against the originally-observed failures), `_lock_regression_cases` promotes one
   **plain-English assertion** case per signature into the durable `EvalCorpus`
   (idempotent) — e.g. *"The response does not reproduce the failure 'X' in workflow
   'Y'."* Thereafter the case is judged by LLM-as-judge (`EvalStrategy.ASSERTION`,
   with an offline lexical fallback) rather than by brittle expected-output matching,
   so the same failure cannot silently recur. This is the "lock-as-regression-test"
   close of the loop (the Opik Test Suite pattern).

## Langfuse integration

The integration is bidirectional and both halves are wired through
`agent_utilities/core/config.py` (the `config` singleton, resolved from the XDG
`config.json` for `graph-os`):

| Direction | Path | Notes |
|---|---|---|
| **Write** (agent-utilities → Langfuse) | `observability/langfuse_exporter.py` + OTEL exporter → `${LANGFUSE_HOST}/api/public/otel` | Exports graph runs as traces. |
| **Read** (Langfuse → AU-AHE.harness.failure-evolution) | `harness/trace_backend.py::LangfuseTraceBackend` | `get_error_observations`, `get_low_score_traces`, `get_cost_latency_anomalies` (+ dataset helpers), wrapping `langfuse_agent.LangfuseApi`. |

**Endpoint note.** Self-hosted Langfuse (validated against 3.182.0) does not expose
the `/api/public/v2/observations` or `/api/public/v2/metrics` routes (404). The
read surface uses the stable `/api/public/observations` and `/api/public/metrics`
endpoints, which take an identical query schema. The metrics query requires a
non-null `toTimestamp`.

**Environment variables** are the official Langfuse SDK names — there is no
deprecated fallback:

- `LANGFUSE_HOST` (host base URL; **not** `LANGFUSE_BASE_URL`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

## Targeting: addressing the right gap

A brand-new `failure_gap` competes for a slot in `unresolved_topics()`'s limited,
arbitrarily-ordered scan over (potentially thousands of) existing concepts — so
generic intake can silently exclude it and the failure would never be remediated.
The failure-ingest path therefore passes its just-materialized gaps **explicitly**
to `GoldenLoopController.run_one_cycle(topics=…)`, bypassing the generic scan so
remediation is deterministic.

## How to run it

- **On demand (MCP):** `graph_orchestrate(action="failure_ingest")` — pulls
  failures, materializes gaps, and runs the regression-gated remediation cycle,
  returning a JSON report (`gap_concepts`, `anomalies`, and a `remediation` block).
- **As a daemon tick:** the consolidated KG daemon registers a `failure_ingest`
  maintenance job when `KG_FAILURE_EVOLUTION` is enabled. This replaced the dead
  `telemetry_ingestion` sweep.

Both share one implementation: `failure_analyzer.run_failure_ingest(engine)`.

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `KG_FAILURE_EVOLUTION` | `False` | Enable the daemon `failure_ingest` tick. Parsed via `to_boolean` (`"True"`/`"False"`). |
| `KG_FAILURE_EVOLUTION_INTERVAL` | `3600` | Daemon tick interval (seconds). |
| `KG_FAILURE_EVOLUTION_WINDOW` | `86400` | How far back to pull telemetry (seconds). |
| `KG_FAILURE_REGRESSION_DATASET` | `False` | Enable the dataset-based regression path (build a Langfuse dataset from the failing traces). |
| `KG_GOLDEN_AUTO_MERGE` | `False` | Master switch for promoting a passing remediation `proposal → active`. Off ⇒ propose-only. |

## Code paths

- `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py` — `FailureAnalyzer`,
  `cluster_failures`, `make_regression_check`, `_lock_regression_cases` (AHE-3.25),
  `run_failure_ingest`.
- `agent_utilities/harness/eval_corpus.py` — `EvalCorpus.add_case(..., assertion=…)`,
  the durable regression-case store the verified fixes lock into.
- `agent_utilities/harness/continuous_evaluation_engine.py` — `TestCase.assertion`,
  `EvalStrategy.ASSERTION`, `EvalRunner._assertion_judge` (AHE-3.25).
- `agent_utilities/harness/trace_backend.py` — the Langfuse failure-read surface.
- `agent_utilities/knowledge_graph/core/engine_tasks.py` — the `_tick_failure_ingest`
  daemon job.
- `agent_utilities/knowledge_graph/research/golden_loop.py` — `run_one_cycle(topics=…)`
  override; regression check threaded into `GovernedAutoMerger`.
- `agent_utilities/mcp/kg_server.py` — `graph_orchestrate(action="failure_ingest")`.
- `agent_utilities/models/schema_definition.py` — `ExecutionSummary`,
  `PerformanceAnomaly`.

## Relationship to other concepts

- Builds on **AHE-3.0** (Experience Observability) and the propose-only golden loop
  (**KG-2.7**).
- Reuses **KG-2.8** feedback primitives (`FeedbackService` `eval`/`outcome`
  corrections) and the **AU-AHE.assimilation.research-auto-merge** `GovernedAutoMerger` (its injectable
  `regression_check`).
- The remediation proposals are the same `TeamSpec`/`AgentSpec` artifacts the
  research-driven synthesis (**AU-KG.enrichment.a2a-capability-extraction**) produces.
- **AHE-3.25** (plain-English regression assertions) closes the loop's final step:
  a *verified* remediation locks a human-readable regression case into the eval
  corpus, the "lock-as-regression-test" guarantee.
