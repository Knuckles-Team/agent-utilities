# Tasks: First-Class Reasoner / Paradigm Abstraction (KG-2.65)

Wire-first, ordered. Cite the real files; reuse the three engines, do not rewrite them.

1. **Define the seam.** Add `Reasoner` (`typing.Protocol`: `name`, `capability_tags`,
   `reason(context, goal) -> ReasonerResult`) and `ReasonerResult` (answer/action + paradigm-agnostic
   `trace`) in a new `agent_utilities/reasoning/reasoner.py`. Keep the shape minimal.

2. **Mirror the selector.** Add `ReasonerRegistry` with `register()` and
   `pick_for_task(required_tags=..., topology=...)` in
   `agent_utilities/reasoning/reasoner_registry.py`, copying the tag-filter / exact-match / default
   fallback algorithm from `agent_utilities/models/model_registry.py` (`ModelRegistry.pick_for_task`,
   ~L195). Do **not** import the model registry — mirror the idiom, separate domain.

3. **Register the three existing paradigms as thin adapters** (delegate, never duplicate):
   - `PredictRLM` → `agent_utilities/rlm/predict_rlm.py` (ORCH-1.12), tag `tool-use`.
   - `GEPAOptimizer` → `agent_utilities/rlm/gepa.py` (AU-ORCH.optimization.optimize-skill-prompt-gepa), tag `evolutionary-search`.
   - `OWLBridge` → `agent_utilities/knowledge_graph/core/owl_bridge.py` (AU-KG.domains.legal-automation), tag
     `deductive-symbolic`.
   Adapters live under `agent_utilities/reasoning/adapters/`; built-ins register at import
   (per the ontology registry convention — real built-ins, never an empty shell).

4. **Wire the live path.** Have an orchestrator/router seam (e.g. `agent_utilities/graph/routing/` or
   `core/execution/protocol.py ExecutionEngine` caller) resolve a paradigm via
   `ReasonerRegistry.pick_for_task(...).reason(...)` with a default that reproduces current behaviour.
   Grep that the hot path actually invokes it (Wire-First); run `scripts/check_wiring.py`.

5. **Tests.** `tests/unit/knowledge_graph/test_kg_2_65_reasoner_registry.py`
   (`@pytest.mark.concept(id="KG-2.65")`): unit-test the selection algorithm + a registration-only
   "fourth paradigm" case; add a `*_live_path` test asserting the selected reasoner's `reason(...)`
   actually ran on the seam.

6. **Concept + docs.** Add `CONCEPT:AU-KG.compute.code-intelligence-tools` markers in the new modules; regenerate
   `docs/concepts.yaml` via `scripts/build_concepts_yaml.py`; run `scripts/check_concepts.py`; author the
   per-concept doc. `pre-commit run --all-files` green.
