# Comparative Analysis: GEPA paper + predict-rlm + AppWorld RLM-GEPA → agent-utilities RLM

**Sources:** GEPA paper (Agrawal et al., ICLR 2026 Oral, `2507.19457v2`); `Trampoline-AI/predict-rlm@edaddfe`
(pinned); the "RLM-GEPA on AppWorld" summary (mismanaged-genius hypothesis, skill-as-SOP, TGC/SGC).
**Target:** `agent-utilities` (`agent_utilities/rlm`: ORCH-1.12 Predict-RLM, ORCH-1.13 GEPA) + `epistemic-graph`.
**Mode:** Lightweight. Pipeline: pin → explore→ledger → verify → score → scaffold-SDD → wiring-audit.
**Artifacts:** `reports/rlm-ca/{ledger,verified,scored,wiring}.json`; SDD stubs under `.specify/`.

## Integrity

9 candidate innovations curated from ~20 explored. **8/9 verified against the predict-rlm source;**
the 9th (graph-native synergy) verified against our own `gepa.py`. 0 refuted, 0 unverified.

## Key finding: our GEPA/RLM is already strong

The capability map confirms agent-utilities **already implements** the GEPA paper's core: reflective
prompt mutation (`gepa.py:122`), a textual feedback function μ_f (`:144`), **Pareto-per-instance
selection** (`ParetoCandidatePool`, `:42-100`), **system-aware crossover** (`:180`), ancestry, minibatch
eval, and three sandboxes (local/container/WASM, `repl.py:301-441`). So this is **not** a rebuild — it's
closing the *generalization, role-specialization, telemetry, and resilience* gaps that make the
optimized skills actually transfer (the AppWorld summary's whole point: +7.2 pp SGC that held off-split).

## Verified innovations → SDD features

| # | SDD feature | Concept | Bundles (verified rows) | Wiring |
|---|---|---|---|---|
| **A** | **Generalizing GEPA** | extends ORCH-1.13 | gepa-heldout-pareto-split, agent-spec-grounding, patch-merge-selection | ⚠️ needs RLM-GEPA entry point |
| **B** | Role-Specialized RLM Optimization | extends ORCH-1.27 / ORCH-1.12 | rlm-role-specialized-optimization | ⚠️ via entry point |
| **C** | Composable Skills + Generic Adapter | extends ORCH-1.12 | composable-skill-system, generic-environment-adapter | ✅ `specialist.py` (1), `repl.py` (2) |
| **D** | RLM Resilience + Telemetry | extends ORCH-1.12 | structured-runtrace-telemetry, recoverable-vs-fatal-timeout | ✅ `repl.py` (2) |
| **E** | Graph-Native Optimization State | extends ORCH-1.13 (+KG-2.7) | epistemic-graph-backed-gepa-state | ⚠️ via entry point |

**Highest leverage (score order):** agent-spec-grounding, composable-skill-system, recoverable-vs-fatal-timeout.
**Build order (deps respected):** C/D foundations → A (heldout-split → agent-spec/patch-merge) → B → E.

## Wiring finding (pre-implementation, from `check_wiring.py`)
- ✅ `hook.py` (1 hop), `specialist.py` (1), `repl.py` (2) are on live call paths.
- ⚠️ **`gepa.py` and `predict_rlm.py` are unreachable from any entry point** — they're library classes
  invoked manually (or in tests). **Prerequisite (P0):** add an RLM-GEPA entry point — a
  `graph_orchestrate(action="rlm_optimize" / "rlm_run")` MCP action (and/or a CLI subcommand) that
  reaches `PredictRLM` and `GEPAOptimizer` — so features A/B/E wire ≤3 hops instead of being dead code.

## Synergy thesis — how each makes our RLM superior

- **A — Generalizing GEPA (the headline).** Our GEPA optimizes and selects on the same minibatches
  (overfit risk). The paper's `D_train → D_feedback + D_pareto` held-out split + predict-rlm's `AgentSpec`
  grounding (use-cases + counterfactual axis) + patch-merge grafts turn it into an optimizer that
  produces **transferable** skills — exactly the AppWorld result (skill optimized on a cheap proxy lifts
  a strong executor off-split).
- **B — Role-specialized optimization.** Wire the executor / proposer / sub-LM split onto the
  **ORCH-1.27 role registry we just shipped** (planner/generator/learner/judge → add rlm-executor /
  rlm-proposer / rlm-sublm). This is the AppWorld cost trick (cheap `gpt-5.4-mini` proxy executor,
  strong proposer) made portable across any provider pool.
- **C — Composable skills + generic adapter ("less harness").** Upgrade skill-as-SOP from raw source
  mounting to composable `Skill` units (instructions+packages+modules+tools, conflict-checked merge),
  and add a minimal generic environment adapter (list/describe/call/SUBMIT) that preserves an external
  evaluator — so the policy lives entirely in the optimizable skill.
- **D — Resilience + telemetry.** Structured `RunTrace` + `FailureClass` give the proposer classified,
  high-signal feedback (better than raw text); recoverable-tool-timeout vs `SandboxFatalError` stops
  the RLM burning iterations on a dead sandbox.
- **E — Graph-native optimization state (our unique edge).** Persist the Pareto frontier, candidate
  ancestry, and RunTraces into the durable epistemic-graph (extending the existing
  `OptimizationTrajectoryNode`/`EvaluatorFeedbackNode` writes) → **resumable, cross-session GEPA** that
  accumulates learnings — something neither predict-rlm nor the paper's reference impl does.

## Risks
- **P0 entry point is the gate** — without it, A/B/E are Wire-First violations. Build it first.
- **Held-out split changes GEPA's eval contract** — additive (`D_pareto` reserved); keep the no-split
  path as default-off to preserve current behavior.
- **Role separation** rides ORCH-1.27 — additive role keys, no change to existing roles.

## Next step
9 features scaffolded as DSTDD stubs under `.specify/design|specs/<id>/`. Promote per the build order,
**starting with the P0 RLM-GEPA entry point**, then A (generalization) as the highest-leverage cluster.
