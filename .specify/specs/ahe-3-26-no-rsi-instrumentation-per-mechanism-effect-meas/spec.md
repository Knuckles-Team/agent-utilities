# Spec: Recursive-Improvement Velocity Ledger (AU-AHE.sdd.recursive-improvement-instrumentation-aggregating)

> Status: **proposed.** **Wire-First:** EXTENDS `knowledge_graph/research/golden_loop.py`
> (`_finalize_metrics` already persists `orchestration_cycle`/`EvolutionCycle` nodes via
> `engine.add_node`) on the write side and `gateway/fleet.py` (`mount_fleet_routes`,
> the `/api/fleet/*` plane) on the read side — reuse both, do not rebuild. Read sources are
> already-persisted `EvolutionCycle` + `ActionExecution` audit nodes
> (`observability/audit_logger.py`, `change_publisher.py:_audit_publish`).
> *(Shares one component with AU-OS.audit.recursive-improvement-velocity-tracker — implement the single ledger once; this spec is the
> AU-AHE.sdd.recursive-improvement-instrumentation-aggregating view: per-mechanism effect measurement + early scaling-curve fit.)*

## Pre-Flight Checklist
- [x] Extension target identified: `golden_loop.py:_finalize_metrics` (write) + `gateway/fleet.py`
      `mount_fleet_routes` (read); read corpus = persisted `EvolutionCycle`/`ActionExecution` nodes.
- [x] New CONCEPT:AU-AHE.sdd.recursive-improvement-instrumentation-aggregating justified: today **zero consumers read the cycle nodes back** — no
      capability-delta-per-merge, no acceptance rate, no compute-per-improvement, no per-mechanism
      breakdown. The ledger is the missing read/aggregate seam, not new persistence.
- [x] Wire-First confirmed: persisted cycle/audit nodes → `RsiLedger.series()` → new
      `GET /api/fleet/rsi` handler mounted by the existing `mount_fleet_routes`, ≤ 3 hops.
- [x] Success metric defined: `series()` returns a per-mechanism row vector for every persisted
      cycle and `slope()` flags a negative capability-delta derivative ("research gets harder").

## User Stories

### US-1 — Per-mechanism improvement ledger over already-persisted nodes
**As** the evolution loop, **I want** each cycle's audit/publication nodes aggregated into one
queryable series, **so that** AU can answer "is the loop improving us, how fast, and which
mechanism contributes most" instead of being blind to its own dynamics.
- **AC1**: `_finalize_metrics` additionally records, on the `EvolutionCycle` node, a
  `mechanism` tag (one of `prose_spec` / `prompt_optimization` / `code_edit`) and a
  `capability_delta_vector` (post-vs-pre scores; empty/`None` when no ratchet ran) — opt-in fields,
  prior node shape unchanged when absent.
- **AC2**: `RsiLedger.series(...)` reads persisted `EvolutionCycle` + `ActionExecution` nodes and
  yields one row per cycle `{cycle_id, mechanism, proposals, merges, capability_delta_vector,
  compute_cost, wall_clock}`; pure aggregation, no node writes.
- **AC3**: `RsiLedger.acceptance_rate(mechanism=...)` and `.by_mechanism()` derive merges/proposals
  and mean capability-delta per mechanism from the same rows.

### US-2 — Early recursive-improvement curve + negative-derivative alert
**As** a forecaster (agenda 4a/4f), **I want** an early-curve fit per mechanism, **so that** AU
gets a first empirical recursive-improvement datapoint and is warned when velocity turns down.
- **AC4**: `RsiLedger.slope(metric="capability_delta", mechanism=...)` fits a simple early-curve
  (least-squares trend over ordered cycles) and returns slope + confidence; flags
  `degrading=True` when the derivative is negative beyond tolerance.
- **AC5**: pure-math helpers (`series`, `acceptance_rate`, `slope`) are LLM-free and deterministic
  given the node rows, so CI exercises them without a live engine.

### US-3 — Queryable fleet surface
- **AC6**: a `GET /api/fleet/rsi` handler (Starlette, mirroring `fleet_trace`/`fleet_touched`) is
  registered via the existing `mount_fleet_routes`, returns the per-mechanism series + slope JSON,
  and is tenant/identity-scoped like its siblings.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_ahe_3_26_rsi_ledger.py` (`@pytest.mark.concept(id="AU-AHE.sdd.recursive-improvement-instrumentation-aggregating")`),
  ≤60s, no live engine/LLM (fake persisted nodes + pure aggregation/fit + route via TestClient).
- `pre-commit` green; `docs/concepts.yaml` regenerated (`scripts/build_concepts_yaml.py`,
  `scripts/check_concepts.py`); per-concept doc authored (`docs/architecture/` note).
