# Comparative Analysis — Reasoning RL in 2026 vs. agent-utilities & epistemic-graph

> **Scope.** 14 arXiv papers + 1 OpenReview paper + a survey article spanning the 2026
> reasoning-RL landscape (GRPO, DPO, RLVR, DAPO, Dr.GRPO, GSPO, DHPO, EP-GRPO, TR-GRPO,
> DPPO, ARPO, VPO, InSPO, TI-DPO, RAPPO), compared against existing reward / preference /
> evaluation / routing surface in `agent-utilities` and the scoring/normalization kernels in
> `epistemic-graph`.
>
> **Method.** Code-grounded: every verdict cites the concrete file:function it maps onto.
> Sources are ingested into the KG (`research/papers/`, job `job-0a915501`); the graph-native
> assimilation pass (dedup → gap → synergy → rank, CONCEPT:KG-2.7) will be reconciled against
> this hand-authored map once document enrichment completes. This document is the
> human-readable comparative; the KG assimilation is the automated cross-check.
>
> **Framing.** We are an **agentic harness + KG-driven evolution** system — we are *not*
> primarily training base-model weights with an on-policy GRPO trainer. So the highest-leverage
> ideas are the ones that map onto our **live** mechanisms (reward-weighted routing, the
> evaluation harness, test-time search/fan-out, the distilled preference corpus, the evolution
> loop) — not the token-level trainer micro-mechanics (which only pay off if/when we run an
> actual policy-gradient trainer over AHE-3.1's reward primitives).

## Source map

| # | arXiv / OR | Method | Title (short) |
|---|---|---|---|
| 1 | 2402.03300 | **GRPO** | DeepSeekMath (GRPO origin) |
| 2 | 2305.18290 | **DPO** | Direct Preference Optimization |
| 3 | 2501.03262 | **REINFORCE++** | Critic-free w/ global normalization |
| 4 | 2503.14476 | **DAPO** | Open-source RL system at scale |
| 5 | 2503.20783 | **Dr.GRPO** | Understanding R1-Zero-like training |
| 6 | 2507.18071 | **GSPO** | Group Sequence Policy Optimization |
| 7 | 2601.05607 | **DHPO** | Dynamic Hybrid Policy Optimization |
| 8 | 2605.04960 | **EP-GRPO** | Entropy-Progress Aligned GRPO |
| 9 | 2511.00066 | **TR-GRPO** | Token-Regulated GRPO |
| 10 | 2603.04135 | **DPPO** | Unbiased Dynamic Pruning |
| 11 | 2507.19849 | **ARPO** | Agentic Reinforced Policy Optimization |
| 12 | 2605.22817 | **VPO** | Vector Policy Optimization (test-time diversity) |
| 13 | 2512.23126 | **InSPO** | Intrinsic Self-reflective Preference Optimization |
| 14 | 2505.19653 | **TI-DPO** | Token-Importance Guided DPO |
| 15 | OR LrHfYPFTtg | **RAPPO** | Keep the Best, Forget the Rest (order-aware) |

## Verdict table (anchor → verdict → merge point)

Legend — **covered**: we already implement the core idea; **partial**: a related primitive
exists but the paper's specific mechanism is missing; **net-new**: no current surface.

| Method | agent-utilities anchor | epistemic-graph anchor | Verdict | Leverage |
|---|---|---|---|---|
| **GRPO** | `graph/training_signals.py::batch_normalized_advantage()` — literally `(rᵢ−μ)/σ` | `client.py::FinanceClient.cross_sectional_rank()`, `rolling_zscore()` | **covered** | baseline |
| **REINFORCE++** | same `batch_normalized_advantage()` (global-normalized, critic-free) | `rolling_zscore()` | **covered** | baseline |
| **DPO** | `harness/eval_corpus.py::EvalCorpus`; `training_signals.py::failure_point()` (error-attributed pairs); `knowledge_graph/adaptation/feedback.py::FeedbackService._apply_outcome` | `batch_cosine_similarity()` | **partial** | med |
| **DHPO** | `graph/reward_decomposition.py` — `R_total = R_traj + α·ΣR_step` *is* a token/sequence hybrid | — | **covered (conceptually)** | low |
| **Dr.GRPO** | `batch_normalized_advantage()` (σ-normalize only; no length-unbias) | `rolling_zscore()` | **partial** | med |
| **GSPO** | `reward_decomposition.py` trajectory-level outcome | `pagerank()` | **partial** | low* |
| **DAPO** | `training_signals.py::difficulty_floor_filter()`; `harness/reasoning_effort.py` | — | **partial** | med* |
| **EP-GRPO** | `reward_decomposition.py`, `harness/reasoning_effort.py` (effort/uncertainty) | — | **net-new** | med |
| **TR-GRPO** | `training_signals.py::composite_reward()` (per-component gating) | — | **partial** | med |
| **DPPO** | `graph/routing/strategies/policy.py::SubagentLifecyclePolicy` (fan-out) | `personalized_pagerank()` | **net-new** | low* |
| **ARPO** | `reward_decomposition.py` (step credit); `retrieval/capability_index.py::designate()`/`record_outcome()` (reward-EMA routing); `knowledge_graph/adaptation/trace_distiller.py` (EpisodeToPreferenceRule); `capabilities/adversarial_verifier.py` | — | **partial → strong-fit gap** | **HIGH** |
| **VPO** | `harness/reasoning_effort.py` (test-time compute); `SubagentLifecyclePolicy` (best-of-k fan-out); RLM fan-out (`rlm/`) | `personalized_pagerank()` (seed-diverse) | **partial → gap** | **HIGH** |
| **InSPO** | `capabilities/adversarial_verifier.py` (critic); `trace_distiller.py`; `rlm/` self-reflection | — | **partial** | med |
| **TI-DPO** | `eval_corpus.py`; `composite_reward()` | — | **net-new** | med |
| **RAPPO** | `eval_corpus.py` (pair store); `training_signals.py::difficulty_floor_filter()` (data quality) | `cross_sectional_rank()` | **partial** | med |

\* *Leverage low/med because these are on-policy token-level trainer mechanics (clip/sampling/
sequence-ratio/rollout-pruning) that only pay off when we run an actual GRPO/PPO trainer. They
land as AHE-3.1 reward-primitive hardening for the day we do, not as live-path wins today.*

## Per-method detail

### Already covered (baselines we implement)

- **GRPO (2402.03300) / REINFORCE++ (2501.03262).** The critic-free group/global-relative
  advantage `A = (r − μ)/σ` is exactly `training_signals.batch_normalized_advantage()`
  (CONCEPT:AHE-3.1). On the kernel side, `epistemic-graph` already exposes the same primitive as
  `FinanceClient.cross_sectional_rank()` (within-batch normalization) and `rolling_zscore()`.
  **No gap** — these papers are the reference point our reward math is built on. Merit: keep them
  as the documented baseline and ensure `batch_normalized_advantage` exposes both *group* and
  *global* normalization modes (REINFORCE++'s distinction).
- **DHPO (2601.05607).** "Blend token-level (local credit) + sequence-level (global reward)" is
  what `reward_decomposition.py` already does: `R_total(τ) = R_trajectory(τ) + α·Σ R_step`. Our
  α=0.2 is precisely the hybrid weight DHPO tunes. Merit: low — validate α against DHPO's schedule;
  cite it as prior art in the docstring.

### Partial — a primitive exists, the specific mechanism doesn't

- **DPO (2305.18290).** We construct *implicit* preference signal — `failure_point()` gives the
  first-divergence index for error-attributed (chosen=corrected / rejected=original) pairs, and
  `FeedbackService._apply_outcome` turns a human correction into a scalar reward — but there is no
  explicit DPO loss / reference-anchored chosen-vs-rejected training over `EvalCorpus`. **Merge
  point:** formalize a `PreferencePair` export from `eval_corpus.py` + `trace_distiller`'s
  `EpisodeToPreferenceRule` (successful vs failed episode → pair). This is the substrate the whole
  DPO family (InSPO/TI-DPO/RAPPO) layers onto.
- **Dr.GRPO (2503.20783).** Identifies GRPO's length bias (dividing the advantage by response
  length over-rewards short answers / penalizes long traces). Our `batch_normalized_advantage`
  normalizes by σ but does not apply the length-unbiased correction. **Merge point:** add a
  `length_unbiased=True` option (fixed max/completion length) to `batch_normalized_advantage` —
  the single most surgical, clearly-correct trainer-primitive upgrade.
- **DAPO (2503.14476).** Decoupled clip-higher/clip-lower + dynamic sampling (drop zero-variance
  groups that yield no gradient) + overlong filtering. We have data filtering
  (`difficulty_floor_filter`) and test-time budget (`reasoning_effort`) but not the clip/dynamic-
  sampling mechanics. **Merge point:** a `dynamic_sample()` helper in `training_signals.py` that
  drops zero-variance reward groups (pairs naturally with EP-GRPO's zero-variance-collapse target).
- **GSPO (2507.18071).** Sequence-level importance ratio (stable for MoE). We optimize at the
  trajectory-outcome level conceptually but don't compute token/sequence importance ratios (we're
  not running on-policy token RL). **Merge point:** informs how `reward_decomposition` aggregates;
  low live-path leverage now.
- **TR-GRPO (2511.00066).** Weight tokens by estimated contribution to the final reward.
  `composite_reward()` already gates *components* conditionally — extend the same idea to
  per-token regulation. Pairs with **TI-DPO** (token-importance on the preference side).
- **RAPPO (OR LrHfYPFTtg).** Order-aware preference; "keep the best, forget the rest"; filter
  *ambiguous* pairs that hurt generalization. We filter training data by *difficulty*
  (`difficulty_floor_filter`) but not by preference-pair *reliability/ambiguity*. **Merge point:**
  a `reliability_filter` over the `PreferencePair` export — directly improves DPO-family corpus
  quality at near-zero cost.
- **InSPO (2512.23126).** Self-reflection inside preference optimization (condition the policy on
  an alternative response); plug-and-play for DPO-family. We have an adversarial critic
  (`adversarial_verifier.run_adversarial_pass`) and reflection in `rlm/`, but not the
  condition-on-alternative preference enhancement. **Merge point:** layer onto the `PreferencePair`
  export as an optional reflective-conditioning step.

### Net-new (no current surface) — highest novelty

- **ARPO (2507.19849) — TOP RECOMMENDATION.** Agent-step-level policy optimization for multi-turn
  tool agents: branch/rollout at *high-entropy* tool-call steps and assign advantage at the
  agent-step (not just final answer). This is the **best fit for our architecture** because we are
  agentic: it maps directly onto `reward_decomposition.py` (step-level credit), the reward-EMA
  routing in `capability_index.designate()`/`record_outcome()`, and `SubagentLifecyclePolicy`
  fan-out. **Merge point / proposal:** a new AHE-3.x concept "Agent-Step Policy Optimization" —
  entropy-gated branching at tool/decision steps + step-advantage write-back into the capability
  reward-EMA. Live-path: the router already records outcomes; ARPO upgrades *when* and *at what
  granularity* it branches and credits.
- **VPO (2605.22817) — SECOND.** Train for *diverse* solution sets under reward vectors to improve
  test-time best@k / pass@k. We scale test-time compute (`reasoning_effort`) and fan out
  (`SubagentLifecyclePolicy`, `rlm/` parallel sub-calls) but optimize for a single best, not for
  diversity. **Merge point / proposal:** an AHE-3.x "Test-Time Diversity" concept — reward-vector-
  driven diverse sampling in the subagent fan-out; `epistemic-graph::personalized_pagerank()`
  (seed-diverse propagation) is a natural kernel for diversity scoring.
- **EP-GRPO (2605.04960).** Track entropy change across reasoning steps ("progress") to reweight
  step/token advantages; explicitly targets zero-variance collapse and wrong-polarity credit.
  **Merge point:** an entropy-progress signal feeding `reward_decomposition`'s step weights;
  composes with `reasoning_effort` (which already reasons about uncertainty/complexity).
- **TI-DPO (2505.19653).** Token-importance weights + triplet loss for fine-grained preference.
  Net-new on the preference side; pairs with TR-GRPO. **Merge point:** token-importance weighting
  on the `PreferencePair` export.
- **DPPO (2603.04135).** Dynamic pruning of redundant rollouts + importance-sampling correction
  (keeps the gradient unbiased). Efficiency play; only relevant if we sample many rollouts/
  subagents. **Merge point:** unbiased pruning of low-value subagent rollouts in
  `SubagentLifecyclePolicy`.

## Synthesis — recommended concept extensions

Ranked by leverage **for our agentic/KG architecture** (not for a base-model trainer):

1. **AHE-3.x — Agent-Step Policy Optimization (ARPO).** Entropy-gated branching at tool/decision
   steps + agent-step advantage write-back into capability reward-EMA. Touches
   `reward_decomposition.py`, `capability_index.py`, `routing/strategies/policy.py`,
   `trace_distiller.py`. **Highest fit — live-path on the router we already run.**
2. **AHE-3.x — Test-Time Diversity (VPO).** Reward-vector-driven diverse subagent fan-out for
   best@k/pass@k. Touches `harness/reasoning_effort.py`, `SubagentLifecyclePolicy`, `rlm/` fan-out;
   `epistemic-graph::personalized_pagerank` as the diversity kernel.
3. **AHE-3.x — Preference-Corpus Reliability (RAPPO + TI-DPO + InSPO + DPO).** A first-class
   `PreferencePair` export from `eval_corpus.py`/`trace_distiller.py` with (a) RAPPO ambiguous-pair
   filtering, (b) TI-DPO token-importance weights, (c) InSPO reflective conditioning. Turns our
   distilled traces into a clean DPO-ready corpus. **Cheap, broadly enabling.**
4. **AHE-3.1 hardening — reward-primitive upgrades (Dr.GRPO + DAPO + EP-GRPO + TR-GRPO).** Add to
   `training_signals.py`: `length_unbiased` normalization (Dr.GRPO), `dynamic_sample()` zero-
   variance drop (DAPO), entropy-progress step reweighting (EP-GRPO), per-token regulation
   (TR-GRPO). Banked for when we run an actual policy-gradient trainer.
5. **epistemic-graph kernel layer.** Promote the GRPO/REINFORCE++ normalization (already present as
   `cross_sectional_rank`/`rolling_zscore`) to a named **group-relative advantage** kernel, plus a
   length-unbiased variant — so the reward math lives natively in the Rust engine, off the Python
   hot path (consistent with [[epistemic-graph-transport]]: batch over the wire).

### What is explicitly NOT worth chasing now
GSPO sequence-ratio MoE stability, DAPO clip mechanics, DPPO rollout pruning as *trainer*
features — they presume an on-policy token-level GRPO trainer we do not run. Captured as AHE-3.1
reward-primitive notes (item 4) rather than standalone builds, to avoid speculative dead code
(per the repo's Wire-First mandate).

## KG reconciliation (pending)
Once `job-0a915501` ingest + `enrichment/` extraction complete, run
`graph_orchestrate(action="assimilate", task="synthesize")` and compare its
`SATISFIED_BY` edges + ranked `open_features` against this table. Expected: the "covered"
rows (GRPO/REINFORCE++/DHPO) auto-satisfy against AHE-3.1/`training_signals`/`reward_decomposition`
concepts; ARPO/VPO/preference-reliability surface as top-ranked open gaps. Discrepancies are a
signal to refine either this map or the concept embeddings.
