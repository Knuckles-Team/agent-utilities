# Tasks: Action-Conditioned World Model (KG-2.64)

Wire-First, ordered. Cite the real files; reuse the Markov/SCM kernels — do not rebuild them.

## T1 — WorldModel abstraction + backend Protocol (US-1)  [code]
- [ ] Add `world_model.py` under `knowledge_graph/core/`: `WorldModel`, `Prediction`, `Trajectory`,
  `WorldModelBackend` Protocol; `step`/`rollout` are read-only on the graph.

## T2 — Symbolic forward-sim backend (US-2)  [code]
- [ ] `SymbolicForwardSimBackend` reusing `formal_reasoning_core.py`
  (`MarkovTransitionModel.predict_next_states`/`forecast_from_state`, `CausalScm.do_intervention`,
  `CounterfactualGenerator`) + KG-2.23 OWL closure + historical `OutcomeEvaluationNode` transitions.
  Register it as the default; leave a named `parametric` backend slot (registered, unimplemented).

## T3 — Facade wiring (US-1)  [code]
- [ ] `knowledge_graph/facade.py`: add `KnowledgeGraph.world_model()` accessor binding the backend to the
  live store / `owl_bridge` / retrieval (mirror the `ontology` property pattern). No kernel imports leak
  into the execution plane.

## T4 — Graph-native rollouts (US-3)  [code]
- [ ] `WorldModelRollout` (+ per-step prediction) node models in `models/knowledge_graph.py`;
  `rollout` persists and links them to origin state + actions for replay.

## T5 — Tests (NFR)  [test]
- [ ] `tests/unit/knowledge_graph/test_kg_2_64_world_model.py` — AC1–AC6, `@pytest.mark.concept(id="KG-2.64")`.

## T6 — Artifacts (NFR)  [docs]
- [ ] `docs/concepts.yaml` regen (KG-2.64) via `scripts/build_concepts_yaml.py`; `scripts/check_concepts.py`;
  CHANGELOG; README/AGENTS KG-2 count; per-concept doc under `docs/pillars/2_epistemic_knowledge_graph/`.
