# Multi-Source Assimilation Program

> **17 shipped concepts** assimilated from **8 arXiv papers + 4 articles**, grouped into
> 6 clusters. Every concept is wired into the *one* epistemic-graph kernel through the
> *one* `LoopController`-style evolution/eval loops, surfaced through exactly two MCP
> tools (`graph_search` modes, `graph_analyze` actions).
> **Surfaces:** `agent_utilities/mcp/tools/query_tools.py` (`graph_search`) ·
> `agent_utilities/mcp/tools/analysis_tools.py` (`graph_analyze`).
> **Engine seams:** `knowledge_graph/orchestration/engine_query.py` (`search_hybrid`) ·
> `harness/agentic_evolution_engine.py` (`run_evolution_cycle`) ·
> `harness/continuous_evaluation_engine.py` (`TraceDistiller.distill`).

## Intro — what was assimilated, and the framing

This program took external research (retrieval gating, temporal semantic IDs, adaptive
stopping, iterative expansion, generative recommendation, decentralized agent memory,
self-play, graph-search code evolution, fast/slow training, eval-set optimization,
forecasting/calibration discipline, contradiction detection, and a night-shift research
swarm) and folded each one into the existing substrate rather than bolting on new
services. Three framing principles held throughout:

1. **One ontology / one graph.** Every new artifact is a typed addition over the single
   Evidence/Capability/Concept knowledge graph; the retrieval concepts annotate the same
   result dicts, and the night-shift swarm reuses the same contradiction primitive.
2. **One loop, two distillers.** Self-evolution concepts are wired into a single
   `run_evolution_cycle`; the enterprise/research-craft concepts are wired into a single
   `TraceDistiller.distill`. No concept ships its own scheduler.
3. **Epistemic-graph kernel, edges deterministic.** Most assimilated mechanisms are pure,
   dependency-injected, stdlib/numpy-only functions (no model, no network) so they run
   in the kernel; LLMs are used only at the edges (the ADORE judge, the graph-search
   coder, the night-shift extractor) and are always injectable for testing.

The five clusters below mirror how the code is organized.

---

## C4 Component view — the assimilation surface

How the two MCP surfaces reach the engine seams and the new modules, grouped by cluster.

<div class="admonition architecture" markdown>
<p class="admonition-title">Two MCP surfaces reach the engine seams and six module clusters</p>

Two MCP/REST surfaces dispatch by mode/action: `graph_search`
(`query_tools.py`) defaults to `search_hybrid()`, and also supports mode
`adore` (`iterative_expansion.py`, KG-2.88) and mode `chrono_ids`
(`temporal_semantic_id.py`). `graph_analyze` (`analysis_tools.py`)
dispatches `recommend` to `generative_recommender.py`, `evolve_code` to
`AgenticEvolutionEngine.run_evolution_cycle()`, `contradictions` to
`contradiction_detector.py`, and `night_shift` to `night_shift.py`.

**Retrieval cluster.** `search_hybrid()` calls `score_gate.py` +
`neural_reranker.py` and `temporal_semantic_id.py`;
`iterative_expansion.py` (ADORE) calls `adaptive_stopping.py`;
`generative_recommender.py` calls `temporal_semantic_id.py`.

**Self-evolution.** `AgenticEvolutionEngine.run_evolution_cycle()` calls
`self_guided_play.py`, `decentralized_memory.py`,
`fast_slow_controller.py`, and `graph_search_evolution.py`.
`decentralized_memory.py` feeds the **memory + bandit** module
`explore_exploit_router.py`; `fast_slow_controller.py` feeds
`substrate_trainer.py`.

**Eval / research-craft.** `TraceDistiller.distill()`
(`continuous_evaluation_engine.py`) calls `forecasting.py`,
`baseline_overfit_gate.py`, `research_log.py`, and
`eval_set_optimizer.py`.

**Night-shift.** `night_shift.py` calls `contradiction_detector.py`.
</div>

---

## Cluster 1 — Retrieval pipeline

All retrieval concepts live under `knowledge_graph/retrieval/`. `ScoreGate` and the
`ChronoID` time-bucket annotation are **default-on** inside `search_hybrid`; `adore`,
`chrono_ids`, and `recommend` are explicit modes/actions on top.

- **CONCEPT:AU-KG.retrieval.unset-dependency-free ScoreGate** — `retrieval/score_gate.py`. `score_gate(...)` fuses a
  bi-encoder `_score` and a cross-encoder `_rerank_score` via per-component
  z-standardization (`fuse_scores`), then keeps everything at/above `keep_z`
  (recall-safe: never below `min_results`, capped at `max_results`). The neural
  cross-encoder lives in `retrieval/neural_reranker.py` (`NeuralCrossEncoderReranker`,
  default model `cross-encoder/ms-marco-MiniLM-L-6-v2`) and is the auto-detected default
  scorer via `reasoning_reranker.py::_auto_scorer` (falls back to
  `LexicalRelevanceScorer` offline). `search_hybrid` calls it with `keep_z=-1.0`.
- **CONCEPT:AU-KG.query.chronoid-fits-residual-quantization ChronoID** — `retrieval/temporal_semantic_id.py`.
  `TemporalSemanticIdEncoder` (`n_codebooks=2`, `codebook_size=64`, `n_time_buckets=16`)
  produces `(time_bucket, *content_codes)`. `time_bucket` is annotated default-on by
  `search_hybrid::_annotate_time_buckets`; full IDs are exposed via `graph_search`
  mode `chrono_ids`.
- **CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval TASR** — `retrieval/adaptive_stopping.py`. `IterativeStopper.update`
  returns a `StopDecision` from three training-free signals in priority order:
  `answer_repeat` → `coverage_saturation` → `max_rounds`. It drives the ADORE loop.
- **CONCEPT:AU-KG.query.adore-concept-expansion ADORE** — `retrieval/iterative_expansion.py`.
  `IterativeQueryExpander.run` loops reformulate → `build_expanded_query` (alpha-repeat)
  → retrieve → graded-relevance judge (0..3) → `IterativeStopper.update`, returning a
  `SearchHistory`. Surfaced via `graph_search` mode `adore`.

<div class="admonition architecture" markdown>
<p class="admonition-title">graph_search dispatches by mode to three retrieval paths</p>

A `graph_search` query dispatches on mode. **Standard/hybrid (default)**:
`search_hybrid()` -> `hybrid_retriever.retrieve_hybrid()` (bi-encoder
`_score` + reranker `_rerank_score`) -> `score_gate(keep_z=-1.0)`
(z-fuses the two scores, trims the weak tail) ->
`_annotate_time_buckets()` (adds `_time_bucket`) -> ranked results
(score, `_time_bucket`).

**`chrono_ids`**: `temporal_semantic_ids()` produces
`(time_bucket, *content_codes)` directly as ranked results.

**`adore`**: `IterativeQueryExpander.run()` (KG-2.88) loops reformulate
-> `build_expanded_query(alpha)` -> retrieve -> judge grade (0..3) ->
`IterativeStopper.update()`, which either stops (on `answer_repeat`,
`coverage_saturation`, or `max_rounds`) and returns
`SearchHistory.final_ranking` as the ranked results, or loops back to
reformulate.
</div>

---

## Cluster 2 — Memory + bandit

Wired into `agentic_evolution_engine.py::run_evolution_cycle` (the memory routing step).

- **CONCEPT:AU-KG.memory.ahe-record-this-base DecentralizedMemory** — `harness/decentralized_memory.py`.
  `DecentralizedMemory` keeps, per agent, an `EXPLOIT` pool (proven trajectories) and an
  `EXPLORE` pool (fresh candidates), plus an append-only `_trace` of `Contribution`s.
  Key methods: `record_trajectory` (→ EXPLOIT), `propose_candidate` (→ EXPLORE),
  `choose_pool` (bandit-routed), `reward`, `promote` (EXPLORE→EXPLOIT),
  `recall` (privacy-scoped to the agent), `router_stats`.
- **CONCEPT:EG-AHE.harness.online-exploit-explore-reference ExploreExploitRouter** — `harness/explore_exploit_router.py`.
  `ExploreExploitRouter(strategy="ucb1"|"thompson")`: `select()` / `update(arm, reward)` /
  `scores()` / `stats()` (with cumulative regret). The free function `ucb1_scores(...)`
  is the canonical UCB1 formula, kept as a parity reference to
  `epistemic_graph.quant.ucb1_scores`.

<div class="admonition architecture" markdown>
<p class="admonition-title">Reward signals the bandit router, which chooses next cycle's pool</p>

`run_evolution_cycle`'s winners feed
`DecentralizedMemory.record_trajectory()` (into the EXPLOIT pool). If the
population collapsed, the cycle rewards EXPLORE (1.0) to encourage
exploration; otherwise it rewards EXPLOIT (1.0) to encourage
exploitation. Either reward feeds `ExploreExploitRouter.update()`
(AHE-3.33), which informs `choose_pool()` for the next cycle (UCB1 or
Thompson `select()`) — feeding back into the next `record_trajectory()`
call. The router also exposes `router_stats()` (counts, means, regret).

Separately, per-agent pools (privacy-scoped recall) hold EXPLOIT (proven
trajectories, fed by `record_trajectory()`) and EXPLORE (fresh
candidates, fed by `propose_candidate()`); EXPLORE candidates move to
EXPLOIT via `promote()`.
</div>

---

## Cluster 3 — Self-evolution

Wired into `agentic_evolution_engine.py`: the main `run_evolution_cycle` (self-play +
fast/slow + trainer) and the `evolve_via_graph_search` alternative (graph-search code
evolution), surfaced via `graph_analyze` action `evolve_code`.

- **CONCEPT:AU-AHE.harness.when-task-is-scope SGS self-play** — `harness/self_guided_play.py`. `SelfGuidedSelfPlay`
  runs Conjecturer → `Guide` (relevance/conciseness/naturalness gate) → Solver each
  round, raising difficulty on solved+accepted and breaking plateaus (`plateau_patience`)
  by perturbing difficulty; returns a `PlayReport` (solve_rate, accept_rate, plateaued).
- **CONCEPT:AU-KG.retrieval.monte-carlo-graph-search MLEvolve graph-search** — `harness/graph_search_evolution.py`.
  `GraphSearchEvolver.run` does Monte-Carlo **graph** search (UCT with a progressive
  `exploration_schedule`), cross-branch `_fuse` nodes (reference edges that don't
  backprop), and a `GlobalCodeMemory` replay (`save`/`retrieve` by label/stage). The
  real RLM LLM coder is the injected `coder_fn(plan, prior_code) -> (plan, code)`.
- **CONCEPT:AU-ORCH.execution.feed-cycle-outcome-fast FastSlowController** — `harness/fast_slow_controller.py`.
  `observe(Trace)` collects production traces; `fast_step()` updates the harness now;
  `slow_step()` finds recurring `task_key`s, computes a GRPO group advantage, and calls
  the `trainer_fn` — emitting `SlowUpdate`s. `swap_model` keeps learning across frontier
  swaps.
- **CONCEPT:AU-ORCH.execution.substrate-training-job-emission SubstrateTrainer** — `harness/substrate_trainer.py`. The fast/slow
  trainer: `build_corpus` turns recurring traces into a GRPO `corpus`, `train` assembles a
  `TrainingJobSpec` (`method="grpo"`, `mean_advantage`, `status` recorded/dispatched/
  skipped) and calls the injected `dispatch_fn` (the DSM substrate); `as_trainer_fn`
  adapts it to the controller.

<div class="admonition architecture" markdown>
<p class="admonition-title">Two evolution paths: the main cycle, and graph-search code evolution</p>

**Main cycle.** `run_evolution_cycle(base_id, task_text)` runs
`tournament_select` + `prune_losers` to produce winners and
population_health, which feed Cluster-2 memory routing
(`DecentralizedMemory` + bandit). That feeds
`SelfGuidedSelfPlay.run()` (Conjecturer -> Guide -> Solver + plateau
breaker), which feeds `FastSlowController.observe(Trace(reward=spread))`.
That in turn drives `fast_step()` (updates the harness now) and
`slow_step()` (finds recurring `task_key`s -> GRPO advantage), which
feeds `SubstrateTrainer.train()` (`build_corpus` -> `TrainingJobSpec`),
which calls `dispatch_fn` into the DSM substrate (recorded/dispatched/
skipped).

**Graph-search code evolution.** `graph_analyze action=evolve_code` calls
`evolve_via_graph_search()`, which runs `GraphSearchEvolver.run()` (UCT
graph search + cross-branch `_fuse` + `GlobalCodeMemory` replay). That
calls the injected RLM `coder_fn(plan, prior)` -> code, and produces the
best `SearchNode` (metric, branch, refs).
</div>

---

## Cluster 4 — Eval / research-craft loop

All four are wired into `continuous_evaluation_engine.py::TraceDistiller.distill`, which
calls `_triage_failures(corpus)` in a fixed order: forecast → baseline-gate → triage →
eval-set-grow. The `ForecastBoard`, `ResearchLog`, and `EvalSet` are constructed once on
the distiller and persist across rounds (compounding IP).

- **CONCEPT:AU-AHE.evaluation.predict-before-resolve-calibration ForecastBoard** — `harness/forecasting.py`. `predict(...)` before a
  round, `resolve(...)` after; `brier_score()` (proper, lower=better), `hit_rate`,
  `calibration_curve`, `surprises`, `summary`.
- **CONCEPT:AU-AHE.assimilation.baseline-overfit-gate baseline/overfit gate** — `harness/baseline_overfit_gate.py`.
  `baseline_gate(candidate, baseline, min_lift)` rejects gains that don't clear the tuned
  baseline; `overfit_smoke_gate(loss_curve)` enforces single-batch loss collapse;
  `PreRunGate` composes both. In `distill` the baseline gate flags regressions vs the
  previous round.
- **CONCEPT:AU-AHE.evaluation.disconfirming-evidence-log FailureTriage + ResearchLog** — `harness/research_log.py`.
  `FailureTriage.from_evidence_corpus` clusters failures into piles; `biggest_pile()`
  surfaces the largest to attack first. `ResearchLog.record(..., supports=...)` makes
  disconfirming evidence first-class (`disconfirming`, `contested`).
- **CONCEPT:AU-ORCH.execution.eval-set-optimization-compounding EvalSetOptimizer** — `rlm/eval_set_optimizer.py`. Each non-passing
  trace becomes an `EvalCase(source="production_failure")` added to the dedup-safe
  `EvalSet`; `optimize_round`/`compounding_loop` keep the suite monotonically growing —
  failures become permanent evals.

<div class="admonition architecture" markdown>
<p class="admonition-title">Distill: forecast, baseline-gate, triage, then grow the eval suite</p>

`TraceDistiller.distill(round_id)` builds an `EvidenceCorpus` (traces ->
classify -> cluster), then runs `_triage_failures(corpus)` through a
fixed pipeline: `forecasts.predict(round holds)` -> `forecasts.resolve
(benchmark_score)` (-> Brier/hit_rate/surprises) -> `baseline_gate(score,
last_round)` (warns on regression) -> `FailureTriage
.from_evidence_corpus()` (`biggest_pile()` first) ->
`ResearchLog.record(supports=…)` (disconfirming evidence first-class) ->
for each non-passing entry, `EvalSet.add(source=production_failure)` ->
a compounding eval suite (an IP asset).
</div>

---

## Cluster 5 — Night-shift swarm

Surfaced via `graph_analyze` action `night_shift`; reuses the AU-KG.research.explicit-node-node-contradiction contradiction
primitive as its Critic and an optional RLM extractor as its Cataloger.

- **CONCEPT:AU-KG.research.explicit-node-node-contradiction ContradictionDetector** — `knowledge_graph/adaptation/contradiction_detector.py`.
  `lexical_similarity` (zero-infra Jaccard+bigram) pre-filters topically related claims;
  `opposes` flags opposing polarity (negation flip / antonym flip / numeric contradiction
  / frame flip). `ContradictionDetector.check` / `scan` return propose-only
  `FrictionFinding`s (never resolves). Also surfaced directly via `graph_analyze` action
  `contradictions`.
- **CONCEPT:AU-KG.research.run-one-autonomous-night NightShiftSwarm** — `knowledge_graph/research/night_shift.py`.
  `NightShiftSwarm.run_shift` runs five stages over a markdown vault:
  `scout` (sources, read-only thereafter) → `catalog` (atomic notes via `extract_fn`,
  each tracing to a source) → `cartograph` (link each atom to ≥`min_links` peers) →
  `critique` (run ContradictionDetector → `[FRICTION]` notes, propose-only) →
  `edit` (cluster threads + write a morning **briefing**). Returns a `ShiftReport`.
  House rules: every atom comes from a source; never delete (retire instead).

<div class="admonition architecture" markdown>
<p class="admonition-title">Night-shift swarm: five stages over a markdown vault</p>

`graph_analyze action=night_shift` runs `scout(items)` (writes
`sources/`, immutable) -> `catalog()` (`extract_fn` -> atomic notes in
`2-atoms/`, each tracing to one source) -> `cartograph()` (links atoms at
or above `min_links` via `lexical_similarity`) -> `critique()`
(`ContradictionDetector.check` -> `[FRICTION]` notes, propose-only) ->
`edit()` (union-find threads in `3-threads/`) -> a briefing (What Came
In, Contradictions To Resolve, Threads That Grew) -> a final
`ShiftReport` (sources, atoms, links, frictions, path).
</div>

---

## Cluster 6 — PauseRec generative recommender

- **CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative PauseRec** — `knowledge_graph/retrieval/generative_recommender.py`.
  `ImplicitReasoningRecommender` adopts PauseRec's *mechanism* at inference time over the
  SIDs from the AU-KG.query.chronoid-fits-residual-quantization `TemporalSemanticIdEncoder` (no backbone training). A query is
  bridged into SID space (`TextSidBridge.project` → shared codebooks), refined for a
  `pause_steps` latent budget (`_latent_refine` blends toward nearby catalog items +
  user history — no decoded rationale), then catalog items are ranked by
  codebook-overlap blended with cosine proximity. Surfaced via `graph_analyze` action
  `recommend`; `explain_budget()` documents that reasoning is implicit.

<div class="admonition architecture" markdown>
<p class="admonition-title">PauseRec: implicit latent refinement, ranked against a fitted catalog</p>

A query embedding (`graph_analyze action=recommend`) passes through
`TextSidBridge.project()` (routes the query through the codebooks into
SID space), producing a target via `reconstruct(query SID)`. That target
is refined through `_latent_refine()` across `pause_steps` (blends toward
nearest items + history SIDs — implicit, no rationale string), then
`_rank()` (0.5·codebook-overlap + 0.5·cosine) produces the top_k
`Recommendation` list (item_id, semantic_id, score).

Separately, `fit_catalog()` turns item embeddings into item content SIDs
+ reconstructed vectors (via `encoder.fit` + `encode_content`), which
`_rank()` ranks candidates against.
</div>

---

## Source → Concept → Surface mapping

| Source (paper / article) | Concept ID | Module | MCP / REST entry point |
|---|---|---|---|
| ScoreGate: Adaptive Chunk Selection via Dual-Score Statistical Fusion (arXiv 2606.x) | AU-KG.retrieval.unset-dependency-free | `knowledge_graph/retrieval/score_gate.py` (+ `neural_reranker.py`, `reasoning_reranker.py::_auto_scorer`) | `graph_search` — default-on inside `search_hybrid` (also `mode="rerank"`); MCP-only |
| Temporal Semantic IDs / ChronoID | AU-KG.query.chronoid-fits-residual-quantization | `knowledge_graph/retrieval/temporal_semantic_id.py` | `graph_search` `mode="chrono_ids"` + default-on `_annotate_time_buckets`; MCP-only |
| TASR: Training-free Adaptive Stopping for Retrieval | AU-KG.retrieval.adaptive-stopping-iterative-retrieval | `knowledge_graph/retrieval/adaptive_stopping.py` | drives ADORE; reached via `graph_search` `mode="adore"`; MCP-only |
| ADORE: iterative query expansion + graded relevance feedback | KG-2.88 | `knowledge_graph/retrieval/iterative_expansion.py` | `graph_search` `mode="adore"`; MCP-only |
| PauseRec: Implicit Reasoning for LLM-based Generative Recommendation (He et al., arXiv:2606.14142) | AU-KG.retrieval.pauserec-implicit-reasoning-generative | `knowledge_graph/retrieval/generative_recommender.py` | `graph_analyze` `action="recommend"`; MCP-only |
| Decentralized agent memory (exploit/explore pools + collaboration trace) | AU-KG.memory.ahe-record-this-base | `harness/decentralized_memory.py` | inside `run_evolution_cycle` (not a direct tool action) |
| Explore/exploit bandit routing (UCB1 / Thompson) | AHE-3.33 | `harness/explore_exploit_router.py` (parity: `epistemic_graph.quant.ucb1_scores`) | inside `DecentralizedMemory` / `run_evolution_cycle` |
| SGS: self-guided self-play (Conjecturer/Solver/Guide) | AU-AHE.harness.when-task-is-scope | `harness/self_guided_play.py` | inside `run_evolution_cycle` |
| MLEvolve: Monte-Carlo graph-search code evolution | AU-KG.retrieval.monte-carlo-graph-search | `harness/graph_search_evolution.py` | `graph_analyze` `action="evolve_code"` → `evolve_via_graph_search` |
| FastSlow: fast harness loop + slow weight loop | AU-ORCH.execution.feed-cycle-outcome-fast | `harness/fast_slow_controller.py` | inside `run_evolution_cycle` |
| SubstrateTrainer: GRPO corpus → DSM training-job spec | AU-ORCH.execution.substrate-training-job-emission | `harness/substrate_trainer.py` | inside `FastSlowController.slow_step` (DSM dispatch) |
| EvalSet optimization (failures → new evals) | AU-ORCH.execution.eval-set-optimization-compounding | `rlm/eval_set_optimizer.py` | inside `TraceDistiller.distill` |
| ForecastBoard: predict-before-resolve calibration | AHE-3.34 | `harness/forecasting.py` | inside `TraceDistiller.distill` |
| Baseline / overfit pre-run gates | AU-AHE.assimilation.baseline-overfit-gate | `harness/baseline_overfit_gate.py` | inside `TraceDistiller.distill` |
| FailureTriage + ResearchLog (disconfirming evidence) | AHE-3.36 | `harness/research_log.py` | inside `TraceDistiller.distill` |
| ContradictionDetector (friction surface) | AU-KG.research.explicit-node-node-contradiction | `knowledge_graph/adaptation/contradiction_detector.py` | `graph_analyze` `action="contradictions"`; MCP-only |
| NightShiftSwarm (vault scout→…→edit) | AU-KG.research.run-one-autonomous-night | `knowledge_graph/research/night_shift.py` | `graph_analyze` `action="night_shift"`; MCP-only |

All entry points are exposed through FastMCP (`query_tools.py` / `analysis_tools.py`).
REST clients use the current `/graph/search` action route and select `mode="hybrid"`
in its typed request.

---

## Deferred

- **FST gradient run (AU-ORCH.execution.feed-cycle-outcome-fast / AU-ORCH.execution.substrate-training-job-emission).** The fast/slow loop builds the GRPO corpus
  and emits a `TrainingJobSpec`, but the actual weight-update gradient run is dispatched to
  the **DSM** substrate on a configured accelerator host. The default `dispatch_fn` records-only
  (`status="recorded" | "skipped_no_substrate"`); the live GPU dispatch is deferred until
  the accelerator substrate is available.
- **Cross-encoder fine-tune (AU-KG.retrieval.unset-dependency-free).** `NeuralCrossEncoderReranker` runs a stock
  distilled model (`cross-encoder/ms-marco-MiniLM-L-6-v2`); fine-tuning the reranker on
  our own graded-relevance traces is deferred.

## Empirical parity + training track (Round 5)

Parity is **measured**, not asserted: `harness/assimilation_benchmark.py` (`CONCEPT:AU-AHE.assimilation.empirical-parity-evidence-assimilation`)
runs each mechanism vs a baseline on a controlled, seeded, CPU-only task and reports the real lift
+ a `claim_reproduced` verdict — surfaced via `graph_analyze action='assimilation_benchmark'`.
**7/7 torch-free mechanisms reproduce their paper's claimed direction** (seed 0).

The PauseRec **training track** — the paper's actual **trainable `<pause>` tokens optimized by
gradient descent** (`CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative`) — was re-homed to **data-science-mcp**
(`data_science_mcp/training/pause_token_trainer.py`, reached over MCP) so agent-utilities core
stays torch-free (see [AGENTS.md](https://github.com/Knuckles-Team/agent-utilities/blob/main/AGENTS.md) "Dependency discipline"). It measured
Recall@3 = 1.00 with vs 0.67 without the trained tokens (torch, CPU); full *scale* reproduction
(paper datasets + GPU training) remains GPU-gated. The **inference-time** deterministic adaptation
(`retrieval/generative_recommender.py`, `bench_pauserec`) is torch-free and stays in core.

<div class="admonition architecture" markdown>
<p class="admonition-title">Torch-free empirical parity in core; the trainable track lives in data-science-mcp</p>

`graph_analyze action=assimilation_benchmark` runs the torch-free
`assimilation_benchmark` suite (`bench_pauserec`/`scoregate`/`tasr`/
`adore`/`decentmem`/`mlevolve`/`sgs`), producing `BenchmarkResult[]` +
`to_markdown` (baseline vs ours + `claim_reproduced`, 7/7). The
PauseRec training track — trainable pause tokens + next-item CE
(`pause_token_trainer.py`) — is re-homed separately to
`data-science-mcp[training]` (torch), outside this torch-free benchmark.
</div>
