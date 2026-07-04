# Tasks: Hierarchical Federated Coordination (AU-ORCH.planning.repo-map-skeleton)

## T1 — Federated strategy (US-1,3)  [code]
- [ ] In `agent_utilities/graph/coordination.py` add `FederatedCoordinationStrategy`:
  partition participants into MASS clusters, run the existing `apply_protocol`
  per cluster, elect representatives by `degree_centrality`, recurse upward, and
  roll up via the existing named log-pool operators. `CONCEPT:AU-ORCH.planning.repo-map-skeleton` tag.

## T2 — Neighborhood access (US-1)  [code]
- [ ] Reuse `agent_utilities/graph/social_system.py` `observable_messages` /
  `degree_centrality` for clustering + representative election (gives the unused
  `observable_messages` a live caller — closes the ORCH-1.32 dead-code note).

## T3 — Live wiring (US-2)  [code]
- [ ] At `agent_utilities/graph/parallel_engine.py:276`, route the single
  `apply_protocol` call to the federated strategy when
  `len(participants) > protocol.max_participants`; else call the original path
  unchanged (auto-sized by `max_participants`, no env flag).

## T4 — Tests (NFR)  [test]
- [ ] `tests/unit/graph/test_orch_1_48_federated_coordination.py` — AC1–AC6,
  including the small-N byte-identity case (AC4) and the message-count bound (AC6).

## T5 — Artifacts (NFR)  [docs]
- [ ] `scripts/build_concepts_yaml.py` regen (AU-ORCH.planning.repo-map-skeleton); `scripts/check_concepts.py`;
  CHANGELOG; README ORCH-1 count; note in
  `docs/architecture/multi_agent_social_system.md`.
