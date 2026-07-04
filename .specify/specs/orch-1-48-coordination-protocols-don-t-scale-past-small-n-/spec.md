# Spec: Hierarchical Federated Coordination (AU-ORCH.planning.repo-map-skeleton)

> Status: **proposed**. **Wire-First:** EXTENDS `graph/coordination.py`
> (`ProtocolRegistry.apply_protocol`, ORCH-1.3) and is invoked from
> `graph/parallel_engine.py:276` (the single `apply_protocol` call over the full
> agent set). Reuses the MASS neighborhood already built for ORCH-1.32
> (`graph/social_system.py` `observable_messages` / `degree_centrality`). No new
> protocol primitive is built — small-N protocols are *composed* over graph clusters.

## Pre-Flight Checklist
- [x] Extension target identified: `ProtocolRegistry.apply_protocol`
  (`coordination.py`) is the existing global-aggregation seam; `parallel_engine.py:276`
  is its only live caller.
- [x] New CONCEPT:AU-ORCH.planning.repo-map-skeleton justified: a *federated* coordination strategy is a
  distinct capability from the small-N `BUILTIN_PROTOCOLS` (pair_consensus min 2,
  team_voting min 3) — it bounds interaction density as N grows; not a config knob.
- [x] Wire-First confirmed: `parallel_engine.py:276` routes to the federated path
  when the agent set exceeds the protocol's `max_participants`, recursing
  cluster-local `apply_protocol` over MASS neighborhoods (≤ 3 hops).
- [x] Success metric defined: for N ≫ small-N, global aggregation work drops from
  O(N) to O(N / cluster_size) per level and convergence has no single global
  failure point; small-N results remain bit-identical to the current path.

## User Stories

### US-1 — Federation only kicks in above small-N
**As** the parallel engine coordinating a large agent population, **I want** the
coordination step to cluster agents and aggregate hierarchically, **so that** it
does not become an O(N) global-convergence bottleneck or single failure point.
- **AC1**: `FederatedCoordinationStrategy.apply(participants, protocol, graph)`
  partitions `participants` into MASS-neighborhood clusters via
  `social_system.observable_messages` / `degree_centrality`; each cluster runs the
  **existing** `apply_protocol` locally.
- **AC2**: each cluster elects a representative (highest `degree_centrality`); only
  representatives are aggregated upward, recursing the tree until a cluster fits
  within the protocol's `max_participants`.
- **AC3**: upward roll-up reuses the existing named log-pool aggregation operators
  from `coordination.py` (no new aggregation math).

### US-2 — Small-N is unchanged
**As** an operator running ≤ `max_participants` agents, **I want** byte-identical
behavior, **so that** federation is purely additive (Wire-First, No-Legacy default-on).
- **AC4**: when `len(participants) <= protocol.max_participants` the live path calls
  the original `apply_protocol` directly — federation is a no-op, result unchanged.
- **AC5**: `parallel_engine.py:276` selects the federated path automatically by
  population size (no env flag; auto-sized cluster fan-out from `max_participants`).

### US-3 — Bounded interaction density
- **AC6**: per-level cluster size is bounded by the protocol's `max_participants`;
  total cross-agent messages scale ~O(N·log N), not O(N²), recorded for the
  AU-ORCH.execution.robust-multi-format-edit scaling-law harness to consume.

## Non-Functional Requirements
- `tests/unit/graph/test_orch_1_48_federated_coordination.py`
  (`@pytest.mark.concept(id="AU-ORCH.planning.repo-map-skeleton")`), ≤60s, no live engine/LLM (synthetic
  `SocialSystem` graph + `BUILTIN_PROTOCOLS`), asserting AC1–AC6 incl. the
  small-N identity case (AC4).
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` regenerated
  (AU-ORCH.planning.repo-map-skeleton) and `scripts/check_concepts.py` passes.
- Per-concept doc authored (note in `docs/architecture/multi_agent_social_system.md`).
