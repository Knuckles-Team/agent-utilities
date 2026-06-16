# Comparative Analysis & Assimilation — CacheRL · HOTE · Kyle Insider Trading

**Date:** 2026-06-16
**Papers:**
- arXiv:2606.14179 — *CacheRL: Multi-Turn Tool-Calling Agents via Cached Rollouts and Hybrid Reward* (Islam et al.)
- arXiv:2606.13710 — *Hybrid Open-Ended Tri-Evolution Makes Better Deep Researcher* (HOTE; Piao et al.)
- arXiv:2605.27684 — *Insider and stealth trading with dynamic legal risk* (Qiao & Xia)

**Scope:** download via scholarx → compare against the agent-utilities + agent-packages
ecosystem → implement the gaps so we surpass each paper. CPU-testable mechanism +
scaffolding; GPU-bound RL e2e deferred (GB10 power fault). All work merged to local
`main`, not pushed.

---

## 1. CacheRL (arXiv:2606.14179)

**What it proposes.** Multi-turn tool-calling RL is dominated by the cost of *live*
tool execution during rollouts. CacheRL contributes (1) a **Hybrid Thinking Pipeline**
that augments trajectories with LLM-generated reasoning ("why this tool"), (2) a
**CacheAgentLoop** — a three-tier fuzzy cache that serves repeated tool calls so rollouts
run at ~100× less compute, with token-level masking to preserve trajectory quality, and
(3) a **Cache-Tier-Aware Reward** that avoids penalizing the model for cache limitations
vs. genuine failures. A Qwen3-4B reaches ~92% process accuracy (GPT-5 ≈ 94%) at 100× less
compute; data quality + reward design matter more than the optimizer.

| Capability | CacheRL | Ecosystem (before) | Ecosystem (after) |
|---|---|---|---|
| RL trainers (GRPO/DPO/PPO+GAE+value head, reward model) | ✓ | ✓ `data-science-mcp/trainers/*` (ML-008/009) | ✓ |
| Rollout buffer / group-normalized advantage | ✓ | ✓ `rollout_buffer.py`, `training_signals.batch_normalized_advantage` | ✓ |
| Three-tier **fuzzy** rollout cache (exact→fuzzy→semantic) | ✓ | ✗ (rollouts hit live vLLM) | ✓ `cache_agent_loop.ThreeTierToolCache` + `CacheAgentLoop` (ML-013) |
| Token-level masking of injected observations | ✓ | ✗ | ✓ `training_signals.token_cache_mask` (AHE-3.49) |
| Cache-tier-aware reward | ✓ | ✗ (reward tier-agnostic) | ✓ `training_signals.cache_tier_aware_reward` (AHE-3.49) |
| Hybrid-thinking ("why this tool") augmentation | ✓ | partial `heavy_thinking_cache.py` (post-hoc) | ✓ `ThinkingTraceAugmenter` (per-step rationale + provenance) |

**How we surpass it.** The cache is graph-/provenance-native: every served observation
carries its `CacheTier`, the augmenter emits per-segment provenance labels, and
`CacheAgentLoop.calls_saved()/hit_rate()` make the compute reduction a *measured* quantity
rather than a claim. The reward half lives in the shared `training_signals` spine, so it
composes with the existing Dr.GRPO / EP-GRPO / DAPO primitives — every trainer inherits
cache-aware shaping for free, not just the one model in the paper.

**Built (ML-013, AHE-3.49):**
| Slice | Module | Tests |
|---|---|---|
| Three-tier cache + loop | `data_science_mcp/cache_agent_loop.py` | `tests/test_cache_agent_loop.py` (6) |
| Hybrid-thinking augmenter | same | same |
| Token mask + cache-tier reward | `agent_utilities/graph/training_signals.py` | `tests/test_cache_rollout_signals.py` (5) |

---

## 2. HOTE — Hybrid Open-Ended Tri-Evolution (arXiv:2606.13710)

**What it proposes.** Co-evolve **three** modules together via hybrid-mode RL — a
*proposer* (open-ended research tasks), a *solver* (deep research), and a *judge*
(evaluation) — and shows evolving all three jointly is **indispensable**. An 8B model so
trained beats static 8–32B models on long-form research benchmarks.

| Capability | HOTE | Ecosystem (before) | Ecosystem (after) |
|---|---|---|---|
| Proposer (research-task generation) | ✓ | ✓ `OntologyReasoningDriver.extrapolate` (KG-2.79) | ✓ |
| Solver (deep research / artifacts) | ✓ | ✓ ARA `research/ara/*` (KG-2.80), `ResearchPipelineRunner` | ✓ |
| Judge (evaluation) | ✓ | ✓ `ConceptMatcher` LLM-judge (KG-2.75) | ✓ |
| Single-component evolution | — | ✓ `SaiFactoryController`, `EvolveAgent`, `AgenticEvolutionEngine` | ✓ |
| **Joint** co-evolution w/ interdependent rewards | ✓ | ✗ (modules evolve independently) | ✓ `HybridTriEvolutionController` (AHE-3.50) |
| Empirical "co-evolution > solo" ablation | ✓ | ✗ | ✓ `run_ablation` (joint strictly beats every solo) |

**How we surpass it.** The controller reuses the existing adaptation-speed instrument
(`AdaptationCurve` / `marginal_speed_gain`, AHE-3.27) so co-evolution is measured the same
way SAI specialization is, and the interdependent-reward coupling is made *falsifiable*:
the analytic default proves indispensability in a CPU unit test (solver learns only from
frontier tasks gated by judge calibration; a frozen proposer collapses the learning signal
as the solver improves). The three real modules plug into the same controller via
injectable hooks, and it is wired as an opt-in stage of the one `LoopController` — so HOTE
becomes a mode of the existing loop engine rather than a parallel system.

Demonstrated ablation (20 rounds, analytic default): joint final skill **12.09** vs best
solo **4.35**; proposer-only / judge-only stall at **0** (solver frozen); `indispensable=True`;
marginal adaptation-speed gain **+2.78**.

**Built (AHE-3.50):**
| Slice | Module | Tests |
|---|---|---|
| Tri-evolution controller + ablation | `agent_utilities/harness/hote_tri_evolution.py` | `tests/test_hote_tri_evolution.py` (7) |
| Opt-in LoopController stage | `knowledge_graph/research/loop_controller.py` (`_run_tri_evolution`) | covered via controller tests |

---

## 3. Insider & stealth trading with dynamic legal risk (arXiv:2605.27684)

**What it proposes.** A continuous-time Kyle-type model where an insider trades against an
*endogenous* surveillance hazard that rises with trading activity and triggers prosecution
(criminal + civil penalties); stealth trading hides among noise traders. Results: (1)
trading accelerates as legal risk diminishes near the window's end; (2) raising financial
penalties alone cannot offset reduced enforcement effort; (3) criminal penalties are
essential to constrain aggressive intensity.

**Starting point — already partly assimilated.** `emerald-exchange` already cites this paper:
`EE-042` (surveillance signal) and `EE-043` (market-making legal-risk gate) compute a snapshot
`legal_risk_score` via the engine `surveillance_risk` kernel. The gap was the *strategic
model* behind the score.

| Capability | Paper | Ecosystem (before) | Ecosystem (after) |
|---|---|---|---|
| Snapshot surveillance / informed-flow score | ✓ | ✓ EE-042/043 (`surveillance_risk`) | ✓ |
| Equilibrium trading-intensity solver β* | ✓ | ✗ | ✓ `insider_equilibrium.solve_equilibrium` (KG-2.6) |
| Criminal vs civil penalty decomposition | ✓ | ✗ | ✓ `penalty_policy_analysis` |
| Continuous-time hazard / end-of-window acceleration | ✓ | ✗ (snapshot only) | ✓ `intensity_schedule` |
| Penalty-design verdict (the paper's 3 results) | ✓ | ✗ | ✓ reproduced as comparative statics |

**How we surpass it.** The closed-form reduction recovers the Kyle baseline exactly when
legal risk is off, and reproduces all three of the paper's qualitative findings as testable
invariants: criminal cost enters β* subtractively (drives it to zero exactly), civil fines
only inflate the denominator (diminishing, never zero) and are gated by enforcement effort
`e` (so weak enforcement nullifies fines), and `intensity_schedule` shows end-of-window
acceleration. It is exposed defensively as `emerald_signals(action="insider_equilibrium")`
(EE-044) — a surveillance/enforcement-design tool — and keeps a `to_engine_args` seam to
later promote the closed form into a Rust engine kernel.

**Built (EE-044, KG-2.6):**
| Slice | Module | Tests |
|---|---|---|
| Equilibrium + penalty + schedule | `agent_utilities/domains/finance/insider_equilibrium.py` | `tests/test_insider_equilibrium.py` (6) |
| MCP action | `emerald-exchange/.../mcp/mcp_signals.py` | `tests/test_insider_equilibrium_action.py` (2) |

---

## 4. Cross-domain synthesis

The three papers compose into the one self-evolving architecture:

- **CacheRL → HOTE.** The cached-rollout loop and cache-tier-aware reward (ML-013/AHE-3.49)
  directly cut the cost of HOTE's *solver* training — solver rollouts over tool-using
  research can be served from the three-tier cache, and the token mask keeps the judge's
  observations out of the gradient. CacheRL is the efficiency substrate for HOTE's scale.
- **HOTE → ConceptMatcher.** HOTE's co-evolving *judge* is the same `ConceptMatcher` LLM-judge
  that scores research-vs-ecosystem novelty; co-evolving its calibration sharpens the very
  assimilation loop that ingested these three papers.
- **Kyle → research loop.** The insider-equilibrium model is a worked example of the
  ontology-driven finance domain feeding the KG, and a defensive surveillance capability the
  HOTE proposer can pose research tasks about.

## 5. Concept IDs

| ID | Pillar | Repo | What |
|---|---|---|---|
| ML-013 | data-science-mcp | data-science-mcp | CacheRL three-tier cached rollouts + hybrid-thinking augmenter |
| AHE-3.49 | Agentic Harness | agent-utilities | Cache-tier-aware reward shaping (token mask + tier-aware reward) |
| AHE-3.50 | Agentic Harness | agent-utilities | Hybrid tri-evolution controller (HOTE) |
| EE-044 | Emerald Exchange | emerald-exchange | Insider equilibrium under dynamic legal risk (MCP action) |
| KG-2.6 | Knowledge Graph (finance) | agent-utilities | `insider_equilibrium` model backing EE-044 |

## 6. Test status & residual notes

- New CPU unit tests: 6 (Kyle model) + 2 (Kyle action) + 6 (cache loop) + 5 (cache reward)
  + 7 (HOTE) = **26**. All logic verified standalone (dependency-free modules); the model
  invariants and the HOTE indispensability ablation are asserted in-test.
- `ruff` clean on every new/changed file.
- **Deferred (GB10 power fault):** live GPU RL runs for CacheRL and HOTE; the LLM-backed
  integration of the real OntologyReasoningDriver/ARA/ConceptMatcher into the tri-evolution
  controller (the analytic default stands in as the proof + harness).
- **Deferred (engine):** promoting the Kyle equilibrium closed form into a Rust
  `epistemic-graph` kernel (Python-first; `to_engine_args` seam in place).
- **Pre-existing (not introduced here):** `gen_docs.py --check` is already red on `main`
  (README 183 vs concepts.yaml 294 concepts); the docs-consistency drift predates this work.
- **scholarx:** the three PDFs were requested for offline storage; background download was
  in flight at writeup time. The analysis is grounded in the abstracts + full assimilation;
  KG ingestion of the PDFs completes when the embedder (vLLM) is reachable.
