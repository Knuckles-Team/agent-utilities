# Multi-Source Comparative Analysis — 8 papers + 4 articles vs agent-utilities / epistemic-graph

**Date:** 2026-06-15
**Scope:** Maximum-feature-extraction comparison of 8 arXiv papers, 4 long-form articles, and 2
cloned codebases (`open-source-libraries/ADORE`, `open-source-libraries/MLEvolve`) against the
`agent-utilities` + `epistemic-graph` ecosystem, with a sequenced plan to **surpass** every source.
**Companion plan:** `~/.claude/plans/soft-honking-canyon.md`. **Branch:** `feat/multi-source-assim`.

## How our ecosystem changes the analysis (synergy lens)

Every gap below is closed *through* four ecosystem-unique substrates rather than as a bolt-on:
- **One ontology-driven KG** — new memory/retrieval/self-play artifacts become OWL classes/links so
  `OntologyReasoningDriver` extrapolates relationships across the whole ecosystem (the standing
  one-ontology principle). New typed links: `EXPLORES_FROM`, `CONTRADICTS`, `REFINES_QUERY`,
  `FUSES_BRANCH`, `GATED_BY_GUIDE`.
- **Single LoopController spine** (`graph_loops`) — ADORE rounds, SGS self-play, eval-set
  optimization, and the night-shift swarm all become *stages of the one loop*, not parallel daemons.
- **epistemic-graph Rust kernel** — bandit/regret + advantage math is mirrored into the kernel with a
  Python↔Rust parity test (the `quant.py` `group_relative_advantage` pattern).
- **Propose-only ProposalQueue governance** — contradictions/friction and harness changes flow
  through the queue; high-stakes sinks never auto-execute.

## Source → theme → arXiv map

| Source | Theme | arXiv |
|---|---|---|
| DecentMem | Decentralized per-agent memory + exploit/explore bandit | 2605.22721 |
| SGS | Self-Guided Self-Play (Conjecturer/Solver/Guide) | 2604.20209 |
| MLEvolve (cloned) | Graph-search code evolution + self-evolving memory | 2606.06473 |
| ScoreGate / ChronoID / TASR / ADORE / (+1) | Retrieval/RAG cluster | 2606.14142 / 14269 / 13905 / 13814 / 14260 (ID↔title resolved at scholarx download) |
| ADORE (cloned) | Iterative query expansion w/ graded relevance feedback | github.com/aminbigdeli/ADORE |
| "Be good at research" | Research-craft → engine features | article |
| Enterprise loop | GEPA + Fast-Slow Training | 2507.19457 / 2605.12484 |
| Second brain "night shift" | Autonomous overnight knowledge swarm | article |

---

## Comparative feature matrix (E=exists, P=partial, M=missing)

| # | Feature (source) | AU/EG today (evidence) | Verdict | Closes via |
|---|---|---|---|---|
| 1 | Decentralized **per-agent private memory** w/ exploit+explore pools (DecentMem) | `harness/evolving_memory.py` `EvolvingMemoryStore` is centralized/shared; `retrieval/capability_index.py` reward-EMA | P→M | KG-2.82 |
| 2 | **Online exploit/explore bandit router**, O(log T) regret (DecentMem) | reward-EMA + `harness/replay_buffer.py` `PrioritizedReplayBuffer`; no UCB/regret | P | AHE-3.33 |
| 3 | Collaboration-aware memory (who-solved-what) (DecentMem) | episodic snapshots; no role/credit trace | P | KG-2.82 |
| 4 | **Self-guided self-play**: Conjecturer/Solver/**Guide** + plateau-break (SGS) | `harness/{agentic_evolution_engine,variant_pool,quality_gates}.py` + `search_synthesis/`; no explicit triad/plateau-breaker | P | AHE-3.37 |
| 5 | **Graph-search code evolution**: reference/cross-branch fusion (MLEvolve) | `harness/program_synthesis.py` = tree/MDL only | P→M | KG-2.89 |
| 6 | Cold-start KB + **dynamic global code memory** (label/stage filter) (MLEvolve) | skill-graph distiller; no labeled code-candidate memory | P | KG-2.89 |
| 7 | **Decoupled planning↔coding** + 3 adaptive coding modes (MLEvolve) | single-pass synth | M | KG-2.89 |
| 8 | **Adaptive chunk selection** via bi+cross-encoder **statistical fusion** (ScoreGate) | `retrieval/autocut.py` knee-cut + `reasoning_reranker.py` lexical fusion | P | KG-2.85 |
| 9 | **Training-free adaptive stopping** for iterative retrieval (TASR) | `retrieval/executable_rag.py` repair loop, no convergence stop | P→M | KG-2.87 |
| 10 | **Iterative query expansion w/ graded relevance feedback** (ADORE) | `retrieval/hyde_planner.py` one-shot multi-query; no feedback rounds | P→M | KG-2.88 |
| 11 | **Temporal signals in semantic IDs** / generative retrieval (ChronoID) | `core/bitemporal.py` + `_recency_boost` query-time only; no semantic IDs/RQ-VAE | M | KG-2.86 |
| 12 | Neural **cross-encoder reranker** (ScoreGate/ADORE) | `LexicalRelevanceScorer` proxy | P | KG-2.85 |
| 13 | **GEPA eval-set optimization** from annotations (GEPA) | `rlm/gepa.py` optimizes prompts; evals static | P | ORCH-1.55 |
| 14 | Compounding **trace→eval→re-optimize** loop (Enterprise loop) | `harness/continuous_evaluation_engine.py` exists; loop not closed as IP | P | ORCH-1.55 |
| 15 | **Fast-Slow Training** controller (FST) | `graph/training_signals.py` (GRPO/DAPO/EP-GRPO) present; no trainer/co-opt loop | P | ORCH-1.56 (trainer deferred) |
| 16 | **Explicit node↔node contradiction / friction surface + report** (Second brain) | `adaptation/failure_analyzer.py`/dedup/SHACL implicit only | P→M | KG-2.83 |
| 17 | **Night-shift swarm** roles + vault + briefing + immutable source (Second brain) | loop engine + cron + breadth-ingest + propose-only; roles implicit, no vault/briefing | P | KG-2.84 |
| 18 | **Predict-before-run forecasting + calibration** (Be-good-at-research) | none | M | AHE-3.34 |
| 19 | **Tuned-baseline gate + single-batch overfit smoke** (Be-good-at-research) | quality gates; no baseline/overfit gate | M | AHE-3.35 |
| 20 | **Failure-transcript triage + disconfirming-evidence log** (Be-good-at-research / Darwin) | `TraceDistiller` clusters; no human-triage queue / belief log | P | AHE-3.36 |

### Where we already meet or exceed the sources (no new work)
- **GEPA core** — genetic-Pareto + held-out generalization already in `rlm/gepa.py` (ORCH-1.13/1.30):
  `split_dataset`, `ParetoCandidatePool`, AgentSpec counterfactual grounding, `select_best_on_heldout`.
- **Bi-temporal retrieval** — `core/bitemporal.py` as-of querying exceeds ChronoID's recency-only view
  (ChronoID adds the *generative-ID* angle, which we lack).
- **Shortcut-resistant task synthesis** — `search_synthesis/` (FORT-Searcher, KG-2.70-72) already
  surpasses naive self-play conjecturing; SGS adds the *Guide gatekeeper + plateau* discipline.
- **RL signal spine** — GRPO/DAPO/Dr.GRPO/EP-GRPO in `graph/training_signals.py` exceeds FST's
  data needs; FST adds the *fast-slow co-optimization controller*.
- **Vector infra** — Nomic-v2 embedder + HNSW + pgvector (`core/embedding_utilities.py`,
  `capability_index.py`, `backends/postgresql_backend.py`).
- **Governance** — propose-only ProposalQueue + auto-merge regression gates exceed the second-brain
  "immutable source" guardrail (we add the markdown-vault ergonomics + friction briefing).

---

## Per-source extracted features (deep dive)

### DecentMem (2605.22721)
Each agent keeps **private memory** split into an **exploitation pool** (reusable past trajectories)
and an **exploration pool** (fresh LLM candidates); a lightweight **online router** reweights the two
from stage-wise judge feedback (modeled as a bandit → O(log T) regret; random-walk-over-strategies
guarantees no permanent local-minimum). Keeps a **collaboration-aware trace** (who solved what) so
decentralization keeps the coordination signal. Results: +~9% (up to 24%) vs centralized, −up to 49%
tokens, ~2.5× faster self-evolution; gains widen as coordination loosens. **We are centralized** →
close via **KG-2.82** (per-agent pools + collab trace over `EvolvingMemoryStore`) + **AHE-3.33**
(UCB/Thompson router, regret-tracked, Rust-parity).

### SGS — Scaling Self-Play with Self-Guidance (2604.20209)
Three roles on one LLM: **Solver**, **Conjecturer** (difficulty-matched question generator),
**Guide** (quality gatekeeper scoring relevance/conciseness/logic-naturalness). The Guide prevents
the learning-plateau failure where the Conjecturer games difficulty with illogical complexity. 7B
surpassed 671B DeepSeek-Prover-V2 on Lean4 after long-cycle SGS. → **AHE-3.37**: explicit triad +
Guide on `quality_gates.py` + plateau-breaker on `population_health()` W1-collapse, fed to the loop.

### MLEvolve (2606.06473, cloned)
**Monte-Carlo Graph Search** (not a tree) with **reference/cross-branch fusion edges**
(`engine/{agent_search,search_node,node_selection}.py`), a piecewise **explore-exploit decay
schedule**, a **cold-start KB** (`engine/coldstart/models_guidance_classified.json`), a **dynamic
global memory** (`agents/memory/global_memory.py`: MemRecord + BM25+FAISS HybridRetriever, label
−1/0/1, stage filter, idempotent save), **decoupled two-stage planning** (free-text plan →
memory-guided JSON, `agents/planner/planner_with_memory.py`), and **3 adaptive coding modes**
(single-pass / stepwise multi-agent / SEARCH-REPLACE diff). 65.3% MLE-bench medal rate. → **KG-2.89**
upgrades `program_synthesis.py` tree→MCGS with all five mechanisms, reusing our embedder + HNSW +
ORCH-1.38 sandbox.

### ADORE (cloned)
5-round max loop per query (`src/agent.py`): **reformulate** (zero-shot pseudo-passages round 1;
feedback-conditioned after) → **search** (alpha-repetition expansion `query_expansion.py`) →
**judge** (UMBRELA 0-3 graded relevance, `tools/judge.py`) → **partition by grade** → repeat until
**quality saturation** (all 3s) or **coverage saturation** (<K new docs for 2 rounds) or max-rounds;
caching + dedup + final-round refinement. → **KG-2.88** reimplements over our retriever+embedder, with
the **TASR** stop (KG-2.87) as the saturation rule.

### ScoreGate / TASR / ChronoID (retrieval papers)
- **ScoreGate** — adaptively control chunk retention by **statistically fusing** bi-encoder +
  cross-encoder scores → **KG-2.85** generalizes `autocut.py`; adds pluggable neural cross-encoder.
- **TASR** — training-free one-line stop for iterative RAG: **halt when the answer repeats** →
  **KG-2.87** `adaptive_stopping.py`, wired into ADORE + executable-RAG repair.
- **ChronoID** — inject **explicit temporal signals into semantic IDs** for generative recommendation
  → **KG-2.86** `temporal_semantic_id.py` (codebook + explicit time-bucket token).

### Enterprise loop — GEPA + FST (Article 3)
The compounding-IP thesis: learning lives in the **harness** (portable across frontier models) and a
small **owned model**; **evals themselves are optimized** from expert annotations; each production
failure becomes a new eval. → **ORCH-1.55** (optimize evals + close the trace→eval→re-opt loop on
`rlm/gepa.py` + `eval_corpus.py`) + **ORCH-1.56** (Fast-Slow controller; weight trainer deferred).

### "Be good at research" (Article 2)
Operationalized into engine features: **predict-before-run** + calibration (AHE-3.34); **tuned-baseline
gate** + single-batch overfit smoke (AHE-3.35); **failure-transcript triage** + Darwin
disconfirming-evidence research log (AHE-3.36).

### Second brain "night shift" (Article 4)
A scheduled overnight swarm with named roles (scout/cataloger/cartographer/critic/editor) over a
local-first markdown vault; immutable source; atomic notes (≥2 links each); explicit **contradiction
[FRICTION] surfacing**; morning briefing; weekly audit. → **KG-2.83** (explicit contradiction/friction
detector + report) + **KG-2.84** (night-shift roles + vault + briefing on the LoopController + cron),
all propose-only.

---

## Sequenced plan (concept allocation)

Next-free at audit: KG-2.82, ORCH-1.55, AHE-3.33, ECO-4.47, OS-5.40 (renumber at merge if a sibling
session claims one). Registered in `docs/concepts.yaml`.

1. **Phase 1 — Retrieval** (one spine, `HybridRetriever`/`engine_query.search_hybrid`): KG-2.85
   ScoreGate, KG-2.87 TASR, KG-2.88 ADORE, KG-2.86 ChronoID. Surfaced as `graph_search` modes/steps
   (MCP + REST nearly free).
2. **Phase 2 — Memory**: KG-2.82 decentralized pools + collab trace; AHE-3.33 bandit router + Rust parity.
3. **Phase 3 — Self-evolution**: AHE-3.37 SGS triad; KG-2.89 MLEvolve MCGS.
4. **Phase 4 — Enterprise loop**: ORCH-1.55 eval-set optimizer; ORCH-1.56 Fast-Slow controller.
5. **Phase 5 — Night-shift swarm**: KG-2.83 contradiction/friction; KG-2.84 swarm + vault + briefing.
6. **Phase 6 — Research craft**: AHE-3.34 forecasting; AHE-3.35 baseline/overfit gate; AHE-3.36 triage+log.
7. **Phase 7 — Ontology + tests + wiring sweep + merge** (local, not pushed).

Each capability ships on **both surfaces** (MCP action + REST twin into the shared `_execute_tool`
core), **default-on / wired into the live path** (Wire-First), and **green pre-commit**.

## Deferred / out of scope (honest)
- Actual **weight trainer** for the FST slow loop (controller + GRPO data spine built; the training
  run needs a GPU — currently blocked by the GB10 power fault).
- Neural cross-encoder **model training** (we wire a pluggable distilled reranker; fine-tuning later).
- Pushing to remotes (merge to main **locally only**, per ecosystem norm).
