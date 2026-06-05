# Tasks: Role-Specialized Model Routing (ORCH-1.27)

> RED → GREEN → REFACTOR. References spec.md / design.md.

## T1 — Registry role layer (US-1, US-2)  [code]
- [ ] Add `ModelRole = Literal["planner","generator","learner","judge"]` to `models/model_registry.py`.
- [ ] Add `RoleSpec(BaseModel)` (`tier: ModelTier`, `tags: list[str]`) and module default `_DEFAULT_ROLE_ROUTING`.
- [ ] Add optional `role_routing: dict[str, RoleSpec]` field on `ModelRegistry` (round-trips JSON).
- [ ] Add `pick_for_role(role, *, override=None) -> ModelDefinition` delegating to `pick_for_task`.

## T2 — Config default map (US-2)  [code]
- [ ] Add `role_routing` override to `core/config.py` (`AgentConfig`), defaulting to the built-in map.

## T3 — Factory integration (US-3)  [code]
- [ ] `core/model_factory.create_model(..., role: ModelRole | None = None)` resolves via active registry when `role` given.

## T4 — MCP wiring (US-3)  [code]
- [ ] `graph_configure` gains a `role_routing` action that updates the active registry map.

## T5 — Tests (NFR)  [test]
- [ ] `tests/unit/test_orch_1_27_role_routing.py`: AC1–AC5 (resolution, defaults, fallback, JSON round-trip, override precedence).
- [ ] Integration: `create_model(role=...)` returns a model; `graph_configure` updates the map.

## T6 — Artifacts (NFR)  [docs]
- [ ] CHANGELOG entry; README feature line; pillar doc note; `scripts/build_concepts_yaml.py` picks up ORCH-1.27; `concepts.yaml` entry.
