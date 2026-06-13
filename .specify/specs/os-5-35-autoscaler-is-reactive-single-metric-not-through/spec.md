# Spec: Throughput-per-Dollar Heterogeneous Autoscaling (OS-5.35)

> Status: **proposed**. Closes part of the AGI→ASI "digital-worker-collective" lever
> ("From AGI to ASI" §5.1): co-scaling compute/cost in compute-optimal regimes means
> placing work on the *cheapest/fastest available* compute, not adding replicas on
> whatever node a single hot metric points at.
>
> **Wire-First** — extend the *existing* OS-5.29 autoscaler, do not rebuild it.
> Primary extension: `agent_utilities/orchestration/fleet_autoscaler.py`
> (`compute_desired_replicas` / `FleetAutoscaler._evaluate_service`). It already owns
> the leader-tick → signal → policy-gate → actuate → deploy-watch path; we widen its
> *objective* from single-signal target-tracking to a cost/throughput utility and add a
> placement-tier choice. Reuses `pricing/catalog.py` (ECO-4.40 `PricingCatalog.cost_for`)
> for $-per-token and `orchestration/scaling_signals.py` (OS-5.29 signal seam) for load.

## Pre-Flight Checklist
- [x] Extension target identified: `fleet_autoscaler.py` (`compute_desired_replicas`,
  `_evaluate_service`) + `fleet_reconciler.ScalingSpec` (the `scaling:` block) — verified
  to do single-signal `desired = ceil(current * value_per_replica / target)` today.
- [x] New CONCEPT:OS-5.35 justified: a distinct *objective* axis (throughput-per-USD
  across heterogeneous node classes) layered onto OS-5.29's *mechanism* (target-tracking
  on one signal), not a replacement — OS-5.29 stays the default when no cost/node metadata
  is declared.
- [x] Wire-First confirmed: 1 hop — `autoscale_fleet(engine)` → `FleetAutoscaler.evaluate`
  → `_evaluate_service` already chooses replica count; we extend that same method to also
  choose the placement tier and rank by utility. The deployment-planner's compute-weighted
  placement (today only in universal-skills) is folded in here as the affinity step
  `agent_dispatch.py` documents as "future work".
- [x] Success metric defined: for a service with a `cost:` block, the autoscaler picks the
  `(replicas, tier)` that **maximizes served-throughput-per-USD under the SLA bound**; on a
  fixture where a GPU tier is N× faster but M× pricier, it prefers GPU iff `N/M > 1` at the
  observed load, else cheap-CPU — and OS-5.29 behavior is byte-for-byte unchanged when no
  `cost:` block is present.

## User Stories

### US-1 — Cost/throughput objective, not single-signal target-tracking
**As** the fleet, **I want** the autoscaler to choose replica count by throughput-per-dollar
under an SLA, **so that** the digital-worker collective runs compute-optimally, not "add a
replica when one metric is hot".
- **AC1**: `ScalingSpec` gains an optional `cost` block (node-class capability metadata:
  per-class `usd_per_hour`, `throughput_per_replica`, optional `model_tier` priced via
  ECO-4.40 `PricingCatalog.cost_for`); absent ⇒ the spec is the plain OS-5.29 spec and the
  legacy path runs unchanged.
- **AC2**: a pure `throughput_per_usd(load, replicas, tier_caps, pricing)` helper in
  `fleet_autoscaler.py` returns the served-throughput-per-USD utility for a candidate
  `(replicas, tier)` and is LLM-free / deterministic.
- **AC3**: `compute_desired_replicas` (or a `compute_optimal_plan` wrapper) returns the
  `(replicas, tier)` maximizing AC2's utility subject to `load_served ≥ SLA` and the
  existing `[min,max]` clamp + step-cap + cooldown; with no `cost` block it returns exactly
  today's number (regression-locked).

### US-2 — Heterogeneous placement tier folded into the live loop
**As** an operator with GB10-GPU and CPU hosts, **I want** the chosen tier (cheap-CPU vs GPU)
emitted on the scale action, **so that** compute-weighted placement runs in the live OS-5.29
loop, closing the `agent_dispatch.py` "affinity-aware placement … is future work" gap.
- **AC4**: `_evaluate_service` emits the chosen `tier` in the `scale_service` `ActionRequest`
  params and reason, gated by the same OS-5.24 ActionPolicy and recorded in the
  `AutoscaleEvaluation` audit node — no new actuation path, no new gate.
- **AC5**: when a GPU tier is faster-but-pricier, the verdict flips with load: cheap-CPU at
  low load, GPU once `throughput_per_usd(GPU) > throughput_per_usd(CPU)` at the observed
  signal — asserted on a fixture, no live hardware required (coefficients are spec inputs).

## Non-Functional Requirements
- `tests/unit/orchestration/test_os_5_35_cost_throughput_autoscaling.py`
  (`@pytest.mark.concept(id="OS-5.35")`), ≤60s, no live engine/Prometheus/hardware: pure
  utility math + a `FleetAutoscaler` with injected signal provider + fake `ScalingSpec.cost`.
- Includes the **OS-5.29 regression lock**: identical inputs with no `cost` block ⇒ identical
  `compute_desired_replicas` output as today.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` regenerated (OS-5.35 in
  `docs/concepts.yaml`); `scripts/check_concepts.py` passes; per-concept doc authored under
  `docs/architecture/fleet_autonomy.md` (the OS-5.29 home) and CHANGELOG + OS-5 pillar count
  updated. Configuration discipline: no new env flag — the `cost:` block lives in the existing
  registry/override, auto-detected node-class metadata is preferred over a knob.
