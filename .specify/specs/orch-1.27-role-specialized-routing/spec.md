# Spec: Role-Specialized Model Routing (ORCH-1.27)

> References design: `.specify/design/orch-1.27-role-specialized-routing/design.md`

## Pre-Flight Checklist

- [x] Design document exists and KG-nearest-concepts table completed.
- [x] Extension target identified (ORCH-1.2, similarity 0.83 ≥ 0.70).
- [x] New CONCEPT:AU-ORCH.routing.conductor-per-step-model justified in design as augmentation (sub-concept of ORCH-1.2).
- [x] Wire-First confirmed: ≤2 hops from `graph_orchestrate` / `graph_configure`.

## User Stories

### US-1 — Resolve a model by functional role
**As** the orchestrator spawning a pipeline stage,
**I want** `registry.pick_for_role("planner")` to return a concrete `ModelDefinition`,
**so that** planner/generator/learner/judge stages each use an appropriate model tier
without hardcoding model IDs.
- **AC1**: `pick_for_role` exists for roles `planner|generator|learner|judge` and returns a valid `ModelDefinition` for any non-empty registry.
- **AC2**: Default map: planner→light+tags[plan,json]; generator→heavy+tags[synthesis]; learner→heavy+tags[extraction]; judge→reasoning. Unknown role falls back to `medium`.
- **AC3**: When the preferred tier/tags are absent, resolution degrades via the existing `pick_for_task` fallback (never raises unless registry is empty).

### US-2 — Override the role map via config / MCP
**As** an operator,
**I want** to set a custom role→(tier,tags) map in config or via `graph_configure`,
**so that** I can pin the planner to a cheap local model and the judge to a frontier model.
- **AC4**: `ModelRegistry.role_routing` (optional field) round-trips through JSON/YAML (`load_from_file`, `to_api_payload`).
- **AC5**: `AgentConfig` exposes a default role map override; an explicit per-call override arg wins over both registry and defaults.

### US-3 — Factory + MCP integration
**As** a developer,
**I want** `create_model(role="generator")` and a `graph_configure` action to set the map,
**so that** the role layer is reachable from a live entry point (Wire-First).
- **AC6**: `core/model_factory.create_model` accepts an optional `role` that resolves through the active registry.
- **AC7**: `graph_configure` accepts a `role_routing` payload that updates the active registry's map (covered by an integration test).

## Non-Functional Requirements

- Tests in `tests/unit/` (`@pytest.mark.concept(id="ORCH-1.27")`), ≤60s, no network.
- `pre-commit run --all-files` green (ruff/mypy/bandit + guardrail gates).
- Post-modification artifacts: docs pillar page, AGENTS.md (if tool surface changes), CHANGELOG, README feature line, `docs/concepts.yaml` regen, this `.specify/` set, tests.
