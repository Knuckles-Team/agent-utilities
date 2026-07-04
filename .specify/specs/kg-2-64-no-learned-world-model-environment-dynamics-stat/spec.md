# Spec: Action-Conditioned World Model (KG-2.64)

> Status: **proposed**. Wire-First: EXTENDS `knowledge_graph/core/formal_reasoning_core.py`
> (reuse `MarkovTransitionModel.predict_next_states`/`multi_step_transition`/`forecast_from_state`,
> `CausalScm.do_intervention`, `CounterfactualGenerator`, KG-2.6) reached through the live
> `knowledge_graph/facade.py` (`KnowledgeGraph`) — wrap the existing graph-as-state, do NOT
> rebuild a predictive kernel.

## Pre-Flight Checklist
- [x] Extension target identified: `formal_reasoning_core.py` Markov/SCM kernels + `KnowledgeGraph`
  facade; rollouts persisted as KG nodes alongside `OutcomeEvaluationNode` transitions.
- [x] New CONCEPT:AU-KG.enrichment.atomic-triple-extraction justified — the descriptive KG (KG-2.1/2.6/2.23) has no agent-facing
  `state × action → next_state + predicted_reward` seam; this adds the *predictive/imaginative* axis.
- [x] Wire-First confirmed: `KnowledgeGraph.world_model().rollout(...)` → existing Markov/SCM kernels
  + AU-KG.domains.legal-automation OWL closure + historical `OutcomeEvaluationNode` transitions, ≤ 3 hops from the facade.
- [x] Success metric: a planner can score ≥ 2 candidate actions by simulated next-state/reward
  **without** mutating the live graph; symbolic backend forecasts match the Markov kernel on a held-out
  transition set (parity), and every rollout is replayable from its persisted KG node.

## User Stories

### US-1 — First-class WorldModel abstraction over the graph
**As** a planner/agent, **I want** a `WorldModel` whose `step(state, action)` returns a predicted
next-state delta + reward, **so that** I can imagine futures instead of only retrieving the past.
- **AC1**: `WorldModel` exposes `step(state, action) -> Prediction(next_state_delta, predicted_reward, confidence)`
  and `rollout(state, actions) -> Trajectory`, reached only via `KnowledgeGraph.world_model()` (facade-bound,
  never importing the kernel modules from the execution plane).
- **AC2**: a pluggable `WorldModelBackend` Protocol with a concrete `SymbolicForwardSimBackend` registered
  by default and a `parametric` slot left open (registered, not implemented — Wire-First / No-Legacy clean).
- **AC3**: `step` is **read-only** on the live graph — it computes predicted node/edge deltas without writing
  them; calling `rollout` never mutates persisted state.

### US-2 — Symbolic forward-simulation backend (reuse kernels)
**As** the default backend, **I want** to predict deltas from existing machinery, **so that** no new
predictive model is built.
- **AC4**: `SymbolicForwardSimBackend` derives next-state from (a) AU-KG.domains.legal-automation OWL-RL closure over the
  action's asserted effects, and (b) `MarkovTransitionModel.predict_next_states` /
  `forecast_from_state` fitted from historical `OutcomeEvaluationNode` transitions; `predicted_reward`
  reuses the recorded outcome distribution.
- **AC5**: counterfactual "what-if-action" branches use `CausalScm.do_intervention` /
  `CounterfactualGenerator` rather than a new interventional path.

### US-3 — Graph-native, replayable rollouts
- **AC6**: each `rollout` persists a `WorldModelRollout` node (+ per-step prediction nodes) linked to the
  origin state and the actions, so a trajectory is queryable and replayable from the KG.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_64_world_model.py` tagged `@pytest.mark.concept(id="KG-2.64")`,
  ≤60s, no live engine/LLM (in-memory facade + synthetic transition history); asserts read-only `step`,
  symbolic↔Markov parity, and rollout-node replay.
- `pre-commit run --all-files` green; `docs/concepts.yaml` regenerated (`scripts/build_concepts_yaml.py`,
  `scripts/check_concepts.py`); per-concept doc authored under `docs/pillars/2_epistemic_knowledge_graph/`.
- Symbolic backend ships with a live consumer; the parametric backend stays specified-not-implemented
  until a learned-dynamics consumer exists (no speculative code).
