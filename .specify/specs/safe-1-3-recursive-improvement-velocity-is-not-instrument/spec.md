# Spec: Recursive-Improvement Velocity Ledger (AU-OS.audit.recursive-improvement-velocity-tracker)

> Status: **proposed**. Closes part of the AGI→ASI distance in "From AGI to ASI" §7.4(a/f)/§7.1(g):
> recursive self-improvement is "poorly understood"; the paper demands we identify each improvement
> mechanism, **measure** its current effect, and **"measure and track the research productivity
> (improvements) of AI Scientist systems."** AU already *runs* the loop — it just never scores it.
>
> **Wire-First.** Extend the cycle node AU already persists — `golden_loop.py`
> `GoldenLoopController._finalize_metrics` (it writes an `evolution_cycle:` `orchestration_cycle`
> node with `duration_ms`/`open_gaps`/`error_count`/`stage_ms`) — and surface a derived
> productivity series on the existing AU-ECO.mcp.usage-cost-observability-surface observability router
> (`gateway/usage_api.py`, mounted at `/api/observability`). Two existing hops, no new loop.

## Pre-Flight Checklist
- [x] Extension target identified: `_finalize_metrics` already emits the per-cycle ledger node;
      it lacks the productivity-bearing fields (proposals, merged?, eval-delta, compute/tokens) and
      no one computes a rate. AU-ECO.mcp.usage-cost-observability-surface `usage_api.py` already exposes `/api/observability/*`.
- [x] New CONCEPT:AU-OS.audit.recursive-improvement-velocity-tracker justified: this is the *measurement* axis (research-productivity curve +
      derivative), distinct from the loop itself (AU-AHE.harness.failure-evolution), the regression gate (AHE-3.1), and cost
      accounting (ECO-4.40) — it is the missing scorer that joins them.
- [x] Wire-First confirmed: enrich the existing node write in `_finalize_metrics`; read it back over
      the existing AU-ECO.mcp.usage-cost-observability-surface router. No new daemon, no new loop, pure wiring over existing nodes.
- [x] Success metric defined: after N≥2 cycles, `/api/observability/improvement-velocity` returns a
      `research_productivity = capability_delta / compute_input` series with a sign on its derivative
      ("accelerating | constant | decelerating"), and a negative derivative raises a tracked signal.

## User Stories

### US-1 — Every cycle leaves a productivity-bearing ledger entry
**As** the recursive-improvement loop, **I want** each cycle to record what it actually produced and
what it cost, **so that** the loop's *returns* — not just its duration — are queryable.
- **AC1**: `_finalize_metrics` enriches the `evolution_cycle:` `orchestration_cycle` node with
  `proposals` (count from the publish/synthesize stages), `merged` (bool/count from the
  AU-AHE.harness.failure-evolution `GovernedAutoMerger` path), `eval_delta` (the AHE-3.1 regression-gate score delta, or
  `null` when the gate didn't run), and `compute_input` (tokens/cost from ECO-4.40 usage, falling
  back to `duration_ms` when no token meter is wired). Existing fields are unchanged (additive).
- **AC2**: All new fields are JSON-safe (scalars or `json.dumps`'d), persist best-effort exactly like
  the current node write, and a missing upstream signal yields `null`/`0` — never aborts the cycle.

### US-2 — A measured research-productivity curve with a friction signal
**As** an operator (and the system itself), **I want** the rolling
`research_productivity = capability_delta / compute_input` and the sign of its derivative, **so that**
I can tell whether self-improvement is accelerating or hitting §7.4's "research gets harder" friction.
- **AC3**: A pure helper `improvement_velocity(cycles) -> ImprovementVelocity` (capability-delta per
  unit compute, rolling window, derivative sign ∈ {`accelerating`,`constant`,`decelerating`}) is
  LLM-free and deterministic over a list of ledger dicts.
- **AC4**: `GET /api/observability/improvement-velocity` on the AU-ECO.mcp.usage-cost-observability-surface router queries the
  `orchestration_cycle` nodes, applies the helper, and returns the series + current productivity +
  derivative sign; with fewer than 2 cycles it returns `{status:"insufficient", cycles:n}`.
- **AC5**: A negative derivative surfaces a tracked `research_friction` flag in that response
  (the §7.4 "diminishing-returns" signal) — observability only, it gates nothing.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_safe_1_3_improvement_velocity.py`
  (`@pytest.mark.concept(id="AU-OS.audit.recursive-improvement-velocity-tracker")`), ≤60s, no live engine/LLM: a fake engine captures the
  enriched node write (US-1) and the pure helper + a `TestClient` route assert the curve, derivative
  sign, friction flag, and the `<2`-cycle guard (US-2).
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run so `docs/concepts.yaml`
  carries AU-OS.audit.recursive-improvement-velocity-tracker; `scripts/check_concepts.py` passes.
- Per-concept doc authored under `docs/` (cite "From AGI to ASI" §7.4 in the module docstring; name
  from purpose — *improvement velocity*, never the paper/section).
