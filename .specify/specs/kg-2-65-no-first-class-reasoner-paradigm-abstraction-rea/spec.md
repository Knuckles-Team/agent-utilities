# Spec: First-Class Reasoner / Paradigm Abstraction (KG-2.65)

> Status: **proposed**. **Wire-First:** EXTEND the existing registry-select pattern —
> add a `Reasoner` protocol + `ReasonerRegistry` mirroring
> `agent_utilities/models/model_registry.py` (`ModelRegistry.pick_for_task()`),
> and **register existing engines as concrete `Reasoner`s without rewriting them**:
> `agent_utilities/rlm/predict_rlm.py` (`PredictRLM`, ORCH-1.12), `agent_utilities/rlm/gepa.py`
> (`GEPAOptimizer`, ORCH-1.13), and `agent_utilities/knowledge_graph/core/owl_bridge.py`
> (`OWLBridge`, KG-2.23). Reuse — do not rebuild — these three engines and the
> `pick_for_task` capability-tag selection idiom.

## Pre-Flight Checklist
- [x] Extension target identified: `ModelRegistry.pick_for_task()` (capability-tag select) is mirrored
      for paradigms; the three engines above are wrapped, never re-implemented.
- [x] New `CONCEPT:KG-2.65` justified: AU is pluggable at the **model** layer (ORCH-1.27) and the
      **symbolic-backend** layer (KG-2.23) but has **no unifying reasoning-paradigm seam** — a new
      inference paradigm today needs bespoke modules, not a registration.
- [x] Wire-First confirmed: registry is invoked on a live path — an orchestrator/router seam calls
      `ReasonerRegistry.pick_for_task(...).reason(context, goal)`, asserted by a live-path test.
- [x] Success metric defined: a new paradigm is added by **registering one `Reasoner`** (≤1 new module,
      zero edits to the registry or the three existing engines); selection resolves by capability tag.

## User Stories

### US-1 — One reasoning seam, three registered paradigms
**As** the orchestrator, **I want** a single `Reasoner` protocol that the LLM-tool (Predict-RLM),
genetic-pareto (GEPA), and OWL-inference paradigms all satisfy, **so that** I invoke a paradigm
through one uniform call instead of three bespoke entry points.
- **AC1**: `Reasoner` is a `typing.Protocol` with `name`, `capability_tags: list[str]`, and
  `reason(context, goal) -> ReasonerResult` where `ReasonerResult` carries `answer`/`action` **and** a
  structured `trace` (paradigm-agnostic shape).
- **AC2**: three thin adapters register the existing engines unchanged — `PredictRLM`,
  `GEPAOptimizer`, `OWLBridge` — each tagged (e.g. `tool-use` / `evolutionary-search` /
  `deductive-symbolic`); the adapters delegate, never duplicate engine logic.
- **AC3**: registering a fourth paradigm is a new adapter module + one `register()` call — **no edit**
  to `reasoner_registry.py` or to the three existing engines (verified by the test importing only the
  new module).

### US-2 — Capability/topology-based selection
**As** a caller, **I want** `ReasonerRegistry.pick_for_task(required_tags=..., topology=...)` mirroring
`ModelRegistry.pick_for_task`, **so that** the right paradigm is chosen by task shape, not hardcoded.
- **AC4**: `pick_for_task` filters by `required_tags` (AND semantics), prefers an exact capability
  match, and falls back to a registry default — never raising on an unknown tag.
- **AC5**: a live orchestrator/router path calls the registry and invokes the selected reasoner's
  `reason(...)`; a `*_live_path` test asserts the selected paradigm actually ran (side effect observed),
  not merely that the registry returned an object.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_65_reasoner_registry.py` (`@pytest.mark.concept(id="KG-2.65")`),
  ≤60s, no live engine/LLM (adapters exercised with stub reasoners + the real selection algorithm).
- Adapters are **opt-in default-on** at the seam (the registry default reproduces today's behaviour);
  No Legacy — no `try_new_then_old` branch, the registry is the single dispatch path once wired.
- `pre-commit` green; `docs/concepts.yaml` regenerated (`scripts/build_concepts_yaml.py`) and
  `scripts/check_concepts.py` passing; per-concept doc authored for KG-2.65.
