# Pillar 3: Agentic Harness Engineering

## Overview

The **Agentic Harness Engineering** pillar encapsulates the continuous learning, evaluation, and evolutionary refinement of the agent ecosystem. It moves the system from static, pre-programmed behaviors into an adaptive entity that evaluates its own performance, distills lessons, and evolves new models and strategies autonomously.

## Why We Built This (Rationale)

Autonomous agents typically suffer from a "Groundhog Day" effect:
1. **Lack of Continual Learning**: An agent will make the same mistake across 100 sessions because it has no mechanism to convert an execution trace into generalized wisdom.
2. **Evaluation Blind Spots**: Traditional test suites are boolean. LLMs need graded, multi-strategy rubrics to gauge nuance and reasoning degradation.
3. **Catastrophic Forgetting**: As an agent acquires new skills, it often overwrites or corrupts previously established knowledge.

## How It Works (Implementation)

### Trace Distillation & The Experience Node (AHE-3.1 & AU-AHE.harness.self-evolution-narrative)
After a task completes or fails, the orchestrator initiates **Trace Distillation**. By analyzing the gap between failure and successful retry (Cross-Rollout Critique), the system extracts a `Condition -> Action` tactical insight and persists it as an `ExperienceNode`. This allows the agent to intrinsically "remember" how to avoid specific pitfalls in the future.

### EWC Consolidation & Temporal Drift (AU-AHE.harness.evolution-checkpoint)
To prevent catastrophic forgetting when modifying the Knowledge Graph, we implemented a lightweight **Elastic Weight Consolidation (EWC++)**. The system tracks concept drift across node embeddings via coefficient of variation. When drift exceeds a threshold, EWC applies a penalty to preserve the stability of legacy knowledge.

### Heavy Thinking & Horizon-Aware Curriculum (AU-AHE.harness.concept-2 & AHE-3.9)
For complex tasks, **Heavy Thinking Orchestration** spawns multiple parallel thinker agents to explore trajectories before synthesizing a consensus. Simultaneously, the **Horizon-Aware Task Curriculum** uses macro-action composition and subgoal checkpoints to train agents on progressively longer execution horizons without losing focus.

### Agentic-iModels & Interpretability (AU-AHE.harness.self-improvement-overview & AU-AHE.harness.self-improvement-overview)
The **Agent-Interpretable Model Evolver** autonomously evolves scikit-learn compatible models optimized for both predictive accuracy and LLM readability. **LLM-Graded Interpretability Tests** run 200-test protocols to verify the agent can correctly simulate the model's behavior natively.

## Benefits Introduced

- **Self-Healing Knowledge**: The ecosystem autonomously refines its understanding and prevents degradation over time.
- **Explainable Autonomy**: Through the iModels integration, the agents can natively interpret and defend the machine learning models they use.
- **Measurable Evolution**: The Continuous Evaluation Engine (EvalRunner) provides exact Jaccard metrics, cosine semantic tracking, and LLM-as-Judge scores to quantitatively prove the agent is getting smarter.

### Workflow Distillation & Bundle Distribution (ORCH-1.8 × AHE-3.2)
The **Workflow Distillation Hook** closes the evolution feedback loop by automatically promoting successful workflow execution patterns into reusable Workflow+TeamConfig pairs in the Knowledge Graph. When `synthesizer_step` completes a successful execution, an asynchronous background task fires the distillation hook. The hook tracks success counts per canonical workflow pattern (based on agent topology, not task content) and only promotes patterns that exceed the configurable `promotion_threshold` (default: 3 successes). Both the threshold and the `quality_score_minimum` are configurable from `config.json`.

Promoted patterns are persisted as paired `WorkflowDefinition` + `TeamConfigNode` KG entries that package workflows with their proven team compositions.

- **Source Code**: `agent_utilities/workflows/distillation_hook.py`
- **Hot Path**: `synthesizer_step → WorkflowDistillationHook.on_execution_complete()`

## Key Concepts Leveraged
- **AHE-3.1**: Continuous Evaluation Engine
- **AU-AHE.harness.self-evolution-narrative**: Continual Learning & Experience Nodes
- **AU-AHE.harness.evolution-checkpoint**: Continual Learning Engine
- **AU-AHE.harness.concept-2**: Heavy Thinking Orchestration
- **AHE-3.9**: Horizon-Aware Task Curriculum
- **AU-AHE.harness.self-improvement-overview**: Agent-Interpretable Model Evolver

## BrowseComp-Plus Extensions (arXiv:2508.06600)

### Adaptive Reasoning Budget (AHE-3.1)
Continuous 0.0–1.0 float scale for test-time compute scaling. Maps effort level to discrete retrieval parameters (search calls, depth, decomposition). Lightweight heuristic `estimate_query_complexity()` auto-classifies query difficulty.
- **Source**: `agent_utilities/harness/reasoning_effort.py`
- **Hot Path**: `EvaluationEngine.evaluate_and_decompose(reasoning_effort=0.7)`

### Disentangled Evaluation (AHE-3.1)
Separates retriever quality from LLM reasoning quality in evaluation. Returns three independent metric groups: `retriever_metrics` (precision, recall, nDCG, MRR), `reasoning_metrics` (step accuracy, goal achievement), and `citation_metrics` (precision, recall, F1). Enables pinpointing whether failures stem from bad retrieval or bad reasoning.
- **Source**: `agent_utilities/harness/evaluation_engine.py`
- **Hot Path**: `EvaluationEngine.evaluate_disentangled(retrieval_results=..., gold_doc_ids=...)`

### Citation Quality Tracking (AHE-3.1)
Measures citation quality in agent responses. Extracts KG node references (`[KG:id]`), concept IDs (`CONCEPT:X`), external URLs, file paths, and arXiv IDs. Computes precision/recall/F1 against retrieved and gold document sets. Identifies hallucinated citations and uncited evidence.
- **Source**: `agent_utilities/harness/citation_tracker.py`
- **Hot Path**: Lazy-loaded in `EvaluationEngine._lazy_init()` → `CitationTracker.evaluate_citations()`

---

## Evolved Self-Evolution Capabilities (Phase 10 — DSPy-Driven Self-Evolution)

### Physical Knowledge Distillation Engine (AHE-3.9) 🔬
The **Physical Knowledge Distillation Engine** represents a monumental architectural breakthrough in self-evolution. Rather than restricting optimized prompts and tool schemas to dynamic, volatile in-memory Knowledge Graph nodes, the distiller maps semantic components from the graph back into structural, human-readable file system changes. This allows the system to bridge the divide between runtime optimization and permanent code enhancement.
- **Source Code**: `agent_utilities/knowledge_graph/distillation/physical_distiller.py`
- **Hot Path**: `PhysicalDistillationEngine.distill_skill(...)` / `distill_mcp_tool(...)` / `distill_system_prompt(...)`

### Multi-Optimizer Prompt Selection Strategy (AHE-3.10) 🔬
The **Multi-Optimizer Prompt Selection Strategy** ensures that the optimization behavior scales appropriately based on the failure footprint. When optimizing prompt signatures via DSPy, the system dynamically inspects failure cluster scales. For highly localized failures, lightweight bootstrap optimizers (like `BootstrapFewShot`) are used. For widespread systemic regressions, the system employs high-parameter multi-generation optimization (like `MIPROv2`) to perform multi-stage hyperparameter tuning.
- **Source Code**: `agent_utilities/harness/evolve_agent.py`
- **Hot Path**: `EvolveAgent._dspy_optimize_cluster(failure_cluster=...)`

> **What can be evolved, and how DSPy fits the whole substrate** — for the full map of
> the evolvable surface (prompts, sampling profiles, MCP tool descriptions, agent skills,
> KG extraction, routing policies), the optimizer/substrate/metric model, and where a DSPy
> pass hooks into the live loop, see **[The Evolvable Surface](../architecture/evolvable_surface.md)**.
> For per-call inference-parameter evolution (temperature/top_p/…), see
> **[Task-Aware Sampling Profiles](../architecture/sampling_profiles.md)** (AHE-3.38).

### GitOps Commit & Evolution Boundary Traceability (AU-AHE.optimization.gitops-commit-automation) 🔬
Every evolutionary cycle is governed by strict, declarative **GitOps boundaries**. When changes are distilled to the physical file system, a structured, isolated git commit is generated programmatically. This commit is tagged with concept traceability IDs and the source failure cluster ID, linking runtime agent telemetry directly to code version control.
- **Source Code**: `agent_utilities/knowledge_graph/distillation/physical_distiller.py`
- **Hot Path**: `PhysicalDistillationEngine.commit_distilled_changes(...)`

### AHE-3.12 — LongMemEval-S Validation Harness

A FastAPI `/benchmark` surface (`server/routers/benchmark.py`, mounted in `build_agent_app`)
compatible with Quarq's HTTP benchmark runner (`quarqlabs/benchmarks`), used to prove the
memory-first synergy stack (ORCH-1.27 role routing + KG-2.11 bi-temporal + KG-2.12 memory-first
retrieval + KG-2.13 learner) meets or beats Quarq's 98.2% on LongMemEval-S. `POST /benchmark/session`
ingests haystack messages as **episodic** memory into a **frozen, versioned** `EvaluationCorpus`
(reproducible across agent versions — Quarq re-derives FAISS each run); `POST /benchmark/query` runs
the HyDE + self-correcting two-pass pipeline (corpus-scoped), synthesizes via the `generator` role,
and scores via the `judge` role with a deterministic fallback; `GET /benchmark/report/{run_id}`
returns the LongMemEval-style accuracy + per-category breakdown. `scripts/check_longmemeval.py` gates
CI on a frozen-subset floor (default 95%), sharing the exact scoring helpers so gate and live router
never diverge; the full 500-question run is nightly/on-demand. Extends AHE-3.

### AHE-3.1 — In-house training substrate

The harness can fine-tune its own open-weight models end-to-end. The deterministic
reward/data engine (`graph/training_signals.py` + data-science-mcp `training_data.py`)
builds SFT/DPO/GRPO corpora; the gradient trainers (data-science-mcp `trainers/`,
torch/PEFT) consume them; the Rust performance path (epistemic-graph
`datascience/training.rs`) mirrors the loss/optimizer kernels; `eval_hooks` bridge
checkpoints back into the AHE-3.1 reliability suite. Trained checkpoints go live via
the model-registry role seam with no hot-path edit. Full cross-repo design:
[In-House Training Substrate](../architecture/in_house_training_substrate.md).

### AHE-3.0 — Prioritized replay of decisive states

`harness/replay_buffer.PrioritizedReplayBuffer` (inverse-frequency priority,
seed-faithful sampling) is wired into `AgenticEvolutionEngine.run_evolution_cycle`:
each cycle pushes its outcome keyed by base id so rare/decisive bases resurface
preferentially via `sample_replay`. Pairs with MEMO merge-generalize (KG-2.1) for
sample-efficient, weight-free self-evolution (source b4-03). Extends AHE-3.0.

---

## 2026 Reasoning-RL Adaptations (`.specify/specs/reasoning-rl-2026/`)

A comparative analysis of the 2026 reasoning-RL landscape (GRPO, DPO, RLVR, DAPO, Dr.GRPO,
GSPO, DHPO, EP-GRPO, TR-GRPO, DPPO, ARPO, VPO, InSPO, TI-DPO, RAPPO) found that most of the
toolkit is *already covered* by the AHE-3.1 reward spine and the capability reward-EMA router.
The high-leverage gaps are the **agentic adaptations** below, not re-implementing GRPO. See
[`COMPARATIVE_ANALYSIS.md`](../../.specify/specs/reasoning-rl-2026/COMPARATIVE_ANALYSIS.md) and
[`ACTIONABLE_PLAN.md`](../../.specify/specs/reasoning-rl-2026/ACTIONABLE_PLAN.md).

### AU-AHE.reward.this-is-read-back — Agent-Step Policy Optimization (ARPO, arXiv:2507.19849)
For multi-turn tool agents the decisive uncertainty is at *intermediate* tool/decision steps,
not the final answer. ARPO (a) **branches** extra rollouts at high-entropy agent steps and (b)
assigns **advantage at the agent-step granularity**, written back into the capability reward-EMA
so routing learns which intermediate *actions* help — not just which final answers succeed.
- **Source**: `graph/agent_step_po.py` (`step_entropy`, `should_branch`, `write_back_step_credit`);
  per-step credit from `graph/reward_decomposition.py::RewardDecomposer.step_advantages`.
- **Hot Path**: `SubagentLifecyclePolicy.determine_route()` branches to `fan_out` on a
  high-entropy decision step (bounded by `ARPO_MAX_BRANCHES`).

### AU-AHE.harness.width-diverse-best-k — Test-Time Diversity (VPO, arXiv:2605.22817)
Optimizes for a *diverse* candidate set (not a single best) to raise test-time best@k / pass@k.
The diverse fan-out width is **effort-derived** (`ReasoningBudget.diversity_width`) so harder
queries fan out wider; MMR selection trades quality vs. embedding-spread diversity. The
`epistemic-graph` `personalized_pagerank` (seed-diverse propagation) is the optional graph-native
diversity kernel.
- **Source**: `graph/test_time_diversity.py` (`diverse_fan_out_width`, `mean_pairwise_distance`,
  `select_diverse`); `harness/reasoning_effort.py::ReasoningBudget.diversity_width`.

### AU-AHE.harness.preference-corpus-reliability — Preference-Corpus Reliability (RAPPO + TI-DPO + InSPO + DPO)
A first-class, DPO-ready preference corpus consolidated from the eval corpus, distilled episodes,
and human corrections — with **RAPPO** ambiguous-pair filtering (keep-the-best-forget-the-rest),
**TI-DPO** token-importance weights, and **InSPO** reflective conditioning layered on top.
- **Source**: `harness/preference_pairs.py` (`PreferencePair`, `PreferencePairExporter`,
  `reliability_filter`, `attach_token_weights`, `with_reflection`).
- **Hot Path**: `FeedbackService.export_preference_pairs()` — the read-side of the feedback loop.

### AHE-3.1 reward-primitive hardening (Dr.GRPO / DAPO / EP-GRPO / TR-GRPO)
`graph/training_signals.py` gains opt-in primitives that default to the original GRPO behaviour:
`batch_normalized_advantage(length_unbiased=…, mode=…, group_ids=…)` (Dr.GRPO σ-bias removal +
GRPO/REINFORCE++ grouping), `dynamic_sample` (DAPO zero-variance group drop),
`entropy_progress_weights` (EP-GRPO, consumed by `RewardDecomposer.step_advantages`), and
`token_regulation` (TR-GRPO). GSPO/DPPO trainer micro-mechanics are deferred until a policy-gradient
trainer consumes them (Wire-First — no speculative dead code).

### AU-AHE.harness.failure-evolution — Failure-Driven Evolution
The self-evolution loop learns from **failures observed in production telemetry**, not only
from research. Errors, low scores, and cost/latency anomalies are pulled from **Langfuse**,
clustered into recurring **failure signatures**, and materialized into the durable KG as
`PerformanceAnomaly` / `ExecutionSummary` nodes plus synthetic **`failure_gap` `Concept`**
topics (with `evidence_trace_ids` back to Langfuse). The golden loop addresses those gaps
**directly** (an explicit `run_one_cycle(topics=…)` override, so a brand-new gap is never
lost in a limited generic scan) and synthesizes a `TeamSpec`/`AgentSpec` remediation. Merge
of a failure remediation is gated by a **regression check** bound to the originating
failures (held while a signature is spiking; AU-AHE.assimilation.research-auto-merge).
- **Source**: `knowledge_graph/adaptation/failure_analyzer.py` (`FailureAnalyzer`,
  `cluster_failures`, `make_regression_check`, `run_failure_ingest`),
  `harness/trace_backend.py` (Langfuse failure-read surface).
- **Run it**: `graph_orchestrate(action="failure_ingest")` (on demand) or the daemon
  `failure_ingest` tick (opt-in `KG_FAILURE_EVOLUTION`). Replaced the dead
  `telemetry_ingestion` sweep.
- **Langfuse vars** are the official SDK names — `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` (no deprecated `LANGFUSE_BASE_URL` fallback).
- Full detail: [`docs/architecture/failure_driven_evolution.md`](../architecture/failure_driven_evolution.md).

### AU-AHE.optimization.performance-anomaly-consumer — Performance Anomaly Consumer
Closes the loop on the anomalies AU-AHE.harness.failure-evolution persists: a daemon tick
(`knowledge_graph/adaptation/anomaly_consumer.py`) consumes durable
`PerformanceAnomaly` nodes and turns recurring ones into evolution topics for
the golden loop, so a cost/latency/error pattern observed in production becomes
a remediation candidate without a human filing it. Propose-only, like every
evolution ingress.

### AU-AHE.harness.promotion-governance-validator — Promotion Governance Validator
The governed validation gate every promoted proposal must pass before merge
(`knowledge_graph/research/promotion_governance.py`, wired into
`research/auto_merge.py`): promotion is no longer a bare regression check but a
policy surface. The merger's own promotion decision additionally consults the
operational OS-5.24 `ActionPolicy` under the reserved `merge_promotion` kind
before the lifecycle flip — `deny` blocks promotion (fail-closed, audited);
the shipped `approval_required` tier queues the same approval the AHE-3.21
bridge consumes. Together with the recorded regression gate (AU-AHE.harness.failure-evolution), this
is stage two of the safety chain in
[Autonomous Evolution](../guides/autonomous-evolution.md).

### AHE-3.21 — Evolution-to-Branch Publication Bridge
The bridge from "promoted proposal in the KG" to "reviewable change on disk":
**change synthesis** (`knowledge_graph/research/change_synthesis.py`) turns a
promoted proposal into concrete edits, validates them in the RLM sandbox, and a
governed **`ChangePublisher`** seam (`research/change_publisher.py`) publishes
them as a regression-gated **local** git branch — never pushed. Publication
itself is an ActionPolicy-gated action (`publish_proposal` on
`graph_orchestrate`, REST twin `/api/graph/orchestrate/publish-proposal`),
which under the shipped default policy requires human approval (OS-5.24).
Walkthrough: [evolution publication example](../examples/evolution-publication.md).

### AU-ORCH.execution.robust-multi-format-edit — Robust Edit-Application Engine
The harness layer that turns a model's proposed change into a file actually edited.
`harness/edit_engine.py` parses SEARCH/REPLACE blocks or unified diffs and applies them
with a graduated fuzzy-match ladder (exact → leading-whitespace-flexible → drop-blank →
`SequenceMatcher` closest-window), so edits land even when whitespace drifts; failures
return did-you-mean hints, and `apply_with_reflection` re-prompts the model on
malformed/failed edits (with an optional lint/test verify gate as the checker half).
Surfaced as the `apply_edits` tool (`tools/developer_tools.py`); `replace_in_file` is
kept for the trivial exact path. Full design:
[Edit-Application Engine](../architecture/edit_application_engine.md).

### AHE-3.25 — Plain-English Regression Assertions
Closes the "lock-as-regression-test" seam of the self-repair loop. `TestCase.assertion`
+ `EvalStrategy.ASSERTION` let a regression case be a human-readable pass/fail check
judged by LLM-as-judge (with an offline lexical fallback), taking precedence over
expected-output scoring (the Opik Test Suite pattern). When a failure remediation is
*verified* — the AU-AHE.harness.failure-evolution regression gate confirms no spike against the original failing
input — `failure_analyzer._lock_regression_cases` promotes one plain-English assertion
case per failure signature into the durable `EvalCorpus` (idempotent), so the same
failure cannot silently recur. See
[Failure-Driven Evolution](../architecture/failure_driven_evolution.md).

## Key Concepts Leveraged (2026 additions)
- **AU-AHE.reward.this-is-read-back**: Agent-Step Policy Optimization (ARPO)
- **AU-AHE.harness.width-diverse-best-k**: Test-Time Diversity (VPO)
- **AU-AHE.harness.preference-corpus-reliability**: Preference-Corpus Reliability (DPO family)
- **AU-AHE.harness.failure-evolution**: Failure-Driven Evolution (Langfuse failures → remediation proposals)
- **AU-AHE.optimization.performance-anomaly-consumer**: Performance Anomaly Consumer (durable anomalies → evolution topics)
- **AU-AHE.harness.promotion-governance-validator**: Promotion Governance Validator (governed gate on every promotion)
- **AHE-3.21**: Evolution-to-Branch Bridge (change synthesis + RLM-sandbox validation + ActionPolicy-gated local-branch publication)
- **AHE-3.25**: Plain-English regression assertions (LLM-judge `TestCase.assertion`; verified remediations auto-lock a regression case)
- **AU-ORCH.execution.robust-multi-format-edit**: Robust edit-application engine (multi-format fuzzy edits + reflection loop)
- **AU-AHE.harness.empirical-evidence-that-latent**: Latent-native efficiency benchmark (rollout drift KG-2.73b + retrieval type-coherence AU-KG.ontology.optional-populated-from vs round-tripped/flat baselines; `graph_analyze action="latent_efficiency_benchmark"`; distilled from arXiv:2606.09828)
