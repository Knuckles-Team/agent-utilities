# Tasks: Recursive-Improvement Velocity Ledger (AHE-3.26)

> Wire-First, ordered. Shares the single ledger with SAFE-1.3 — build once.

## T1 — Tag the cycle node at the write seam (US-1)  [code]
- [ ] `knowledge_graph/research/golden_loop.py:_finalize_metrics` — add `mechanism` +
      `capability_delta_vector` fields to the `orchestration_cycle`/`EvolutionCycle` `add_node`
      payload (opt-in; node shape unchanged when absent).

## T2 — RsiLedger aggregator (US-1,2)  [code]
- [ ] New `knowledge_graph/research/rsi_ledger.py` (`RsiLedger`): reads persisted `EvolutionCycle`
      + `ActionExecution` audit nodes (`observability/audit_logger.py`,
      `change_publisher.py:_audit_publish`); pure `series()`, `by_mechanism()`,
      `acceptance_rate()`, `slope()` (early-curve fit + negative-derivative flag). No writes.

## T3 — Fleet read surface (US-3)  [code]
- [ ] `gateway/fleet.py` — add a `fleet_rsi` Starlette handler (mirror `fleet_trace`/`fleet_touched`,
      tenant/identity-scoped) and register `GET /api/fleet/rsi` inside `mount_fleet_routes`.

## T4 — Tests (NFR)  [test]
- [ ] `tests/unit/knowledge_graph/test_ahe_3_26_rsi_ledger.py` — AC1–AC6 with fake persisted nodes
      (`@pytest.mark.concept(id="AHE-3.26")`), incl. a live-path test through `mount_fleet_routes`.

## T5 — Artifacts (NFR)  [docs]
- [ ] CONCEPT:AHE-3.26 marker on `rsi_ledger.py`; `docs/concepts.yaml` regen
      (`scripts/build_concepts_yaml.py`) + `scripts/check_concepts.py`; CHANGELOG; README AHE-3 count;
      `docs/architecture/` per-concept note; `pre-commit run --all-files` green.
