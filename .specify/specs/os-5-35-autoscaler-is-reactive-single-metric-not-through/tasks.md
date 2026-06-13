# Tasks: Throughput-per-Dollar Heterogeneous Autoscaling (OS-5.35)

Wire-first: extend the live OS-5.29 modules before adding anything new. Order matters —
each step keeps the legacy single-signal path byte-for-byte unchanged when no `cost:` block
is declared.

## T1 — Extend the registry spec  [code] (US-1)
- [ ] Add an optional `cost` block to `ScalingSpec` in
  `agent_utilities/orchestration/fleet_reconciler.py` (per-node-class `usd_per_hour`,
  `throughput_per_replica`, optional `model_tier`, `sla_min_throughput`); extend
  `parse_scaling_spec` to validate-or-drop it (typo ⇒ plain OS-5.29 spec, never surprise
  cost scaling). No block ⇒ `cost is None`.

## T2 — Pure utility helper  [code] (US-1 AC2)
- [ ] Add `throughput_per_usd(load, replicas, tier_caps, pricing)` to
  `agent_utilities/orchestration/fleet_autoscaler.py`, pricing `model_tier` via
  `agent_utilities/pricing/catalog.py` `get_pricing_catalog().cost_for(...)` (ECO-4.40).
  Deterministic, LLM-free, no I/O.

## T3 — Cost/throughput objective in the live computation  [code] (US-1 AC3)
- [ ] Extend `compute_desired_replicas` (add a `compute_optimal_plan` wrapper returning
  `(replicas, tier)`) in `fleet_autoscaler.py` to maximize T2's utility under the SLA bound,
  reusing the existing `[min,max]` clamp + `scale_up_step`/`scale_down_step` cap. When
  `spec.cost is None`, delegate to today's formula unchanged (regression-locked).

## T4 — Emit tier in the live loop  [code] (US-2 AC4)
- [ ] In `FleetAutoscaler._evaluate_service`, call the planner, put the chosen `tier` in the
  `scale_service` `ActionRequest.params` + `reason`, and surface it in `ServiceEvaluation` /
  the `AutoscaleEvaluation` audit row. Reuse the existing OS-5.24 policy gate, FleetActuator
  seam and OS-5.27 deploy-watch — no new actuation/gate.

## T5 — Close the dispatch "future work" note  [docs/code] (US-2)
- [ ] Update the `agent_utilities/orchestration/agent_dispatch.py` header note: affinity-aware
  / compute-weighted placement now lives in the OS-5.35 autoscaler tier choice (was deferred).

## T6 — Tests  [test] (NFR)
- [ ] `tests/unit/orchestration/test_os_5_35_cost_throughput_autoscaling.py`
  (`@pytest.mark.concept(id="OS-5.35")`): T2 utility math; AC5 GPU-vs-CPU flip with load;
  AC3 SLA-bounded plan; **OS-5.29 regression lock** (no `cost` block ⇒ identical output);
  injected signal provider + fake spec, no live engine/hardware.

## T7 — Artifacts  [docs] (NFR)
- [ ] `CONCEPT:OS-5.35` marker on the new code; `python scripts/build_concepts_yaml.py`;
  `python scripts/check_concepts.py`; OS-5.35 section in `docs/architecture/fleet_autonomy.md`;
  CHANGELOG entry; OS-5 pillar count in README/AGENTS regenerated; `pre-commit run --all-files`.
