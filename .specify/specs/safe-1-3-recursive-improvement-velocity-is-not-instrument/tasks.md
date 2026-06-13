# Tasks: Recursive-Improvement Velocity Ledger (SAFE-1.3)

Wire-first: enrich the cycle node AU already writes, then read it back over the existing ECO-4.41
router. Do not add a new loop, daemon, or store.

1. **Enrich the existing ledger node.** In
   `agent_utilities/knowledge_graph/research/golden_loop.py`, extend
   `GoldenLoopController._finalize_metrics` so the `evolution_cycle:` `orchestration_cycle` node also
   carries `proposals`, `merged`, `eval_delta`, and `compute_input`. Pull `proposals`/`merged` from
   the report's publish/synthesize/auto-merge stages (the AHE-3.18 `GovernedAutoMerger` path),
   `eval_delta` from the AHE-3.1 regression-gate result already threaded through `regression_check`,
   and `compute_input` from ECO-4.40 usage (fall back to `duration_ms`). Keep the write best-effort
   and additive — existing fields untouched. (US-1 / AC1, AC2)

2. **Add the pure velocity helper.** Add `improvement_velocity(cycles)` next to the loop (a new
   `agent_utilities/knowledge_graph/research/improvement_velocity.py`, tagged
   `CONCEPT:SAFE-1.3`, cite "From AGI to ASI" §7.4 in the docstring): compute
   `research_productivity = capability_delta / compute_input` per cycle, a rolling value, and the
   derivative sign ∈ {`accelerating`,`constant`,`decelerating`} + a `research_friction` bool. Pure,
   LLM-free, deterministic over a list of ledger dicts. (US-2 / AC3, AC5)

3. **Surface it on the existing observability router.** In `agent_utilities/gateway/usage_api.py`
   (ECO-4.41, `/api/observability`) add `GET /improvement-velocity`: query the `orchestration_cycle`
   nodes, feed them to the helper, return series + current productivity + derivative sign + friction
   flag; `<2` cycles → `{status:"insufficient", cycles:n}`. (US-2 / AC4)

4. **Test the live path.** Add `tests/unit/knowledge_graph/test_safe_1_3_improvement_velocity.py`
   (`@pytest.mark.concept(id="SAFE-1.3")`): a fake engine asserts `_finalize_metrics` writes the new
   fields; unit-test the pure helper (curve, three derivative signs, friction); a `TestClient`
   asserts the route and the `<2`-cycle guard. (NFR)

5. **Register + document.** Run `scripts/build_concepts_yaml.py`; confirm `scripts/check_concepts.py`
   passes. Author the per-concept doc under `docs/`. Drive `pre-commit run --all-files` fully green
   (including pre-existing). (NFR)
