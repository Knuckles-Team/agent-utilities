# Generalizing GEPA (CONCEPT:ORCH-1.30)

## Overview

Makes GEPA-optimized skills **transfer** off the optimization split — the AppWorld RLM-GEPA result
(the +7.2 pp SGC that held on held-out test). Adds the GEPA paper's held-out feedback/Pareto split,
predict-rlm's `AgentSpec` anti-overfit grounding, and held-out candidate selection. Extends **ORCH-1.13**.

## How it works

- **Held-out split** (`split_dataset`) — `D_train → D_feedback` (propose on) + `D_pareto` (held-out
  select on). With `dev_fraction > 0`, `GEPAOptimizer.optimize` proposes/evaluates on the feedback set
  and **selects the final candidate by held-out score** (`select_best_on_heldout` over
  `_score_candidate_on`), so a candidate that merely memorized the minibatch does not win.
- **`AgentSpec` grounding** — `use_cases` + `runtime_grounding` + `scoring_rule` + `counterfactual_axis`
  are prepended (`as_prompt()`) to the reflective-mutation prompt, steering the proposer toward a
  general standard-operating-procedure rather than rules that overfit the examples.
- **Patch-merge selection** — `select_best_on_heldout` picks the winning instruction graft on the
  held-out set (ties → earlier/simpler generation).

## Key files / API

| Piece | Location |
|---|---|
| Generalization core | `rlm/gepa.py` (`AgentSpec`, `split_dataset`, `select_best_on_heldout`, `GEPAOptimizer.optimize(dev_fraction=...)`, `_score_candidate_on`) |

## Wiring (≤3 hops)
`graph_orchestrate(action="rlm_optimize")` → `optimize_rlm_skill` → `GEPAOptimizer.optimize` (≤3 hops).

## Research provenance
GEPA paper (Agrawal et al., ICLR 2026 — `D_feedback`/`D_pareto` split, Algorithm 1); predict-rlm `src/rlm_gepa/schema.py` (`AgentSpec`) — verified.
