"""CONCEPT:AU-ORCH.routing.conductor-per-step-model — Role-Specialized Model Routing.

Verifies that functional roles (planner/generator/learner/judge) resolve to concrete
models via the existing registry tier-fallback, that the role map round-trips through
JSON, and that override precedence (call override > registry > defaults) holds.
"""

from __future__ import annotations

import pytest

from agent_utilities.models.model_registry import (
    _DEFAULT_ROLE_ROUTING,
    ModelDefinition,
    ModelRegistry,
    RoleSpec,
)


def _model(mid: str, tier: str, tags: list[str]) -> ModelDefinition:
    return ModelDefinition(
        id=mid,
        name=mid,
        provider="openai",
        model_id=mid,
        tier=tier,
        tags=tags,  # type: ignore[arg-type]
    )


def _full_pool() -> ModelRegistry:
    return ModelRegistry(
        models=[
            _model("fast", "light", ["plan", "json"]),
            _model("mid", "medium", []),
            _model("big", "heavy", ["synthesis", "extraction"]),
            _model("think", "reasoning", []),
        ]
    )


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_default_role_bindings_resolve_expected_models():
    reg = _full_pool()
    assert reg.pick_for_role("planner").id == "fast"  # light + plan/json
    assert reg.pick_for_role("generator").id == "big"  # heavy + synthesis
    assert reg.pick_for_role("learner").id == "big"  # heavy + extraction
    assert reg.pick_for_role("judge").id == "think"  # reasoning


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_unknown_role_falls_back_to_medium():
    reg = _full_pool()
    spec = reg.resolve_role("nonexistent-role")
    assert spec.tier == "medium"
    assert reg.pick_for_role("nonexistent-role").id == "mid"


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_sparse_pool_degrades_via_tier_fallback():
    # Only a single medium model exists; every role must still resolve, never raise.
    reg = ModelRegistry(models=[_model("only", "medium", [])])
    for role in ("planner", "generator", "learner", "judge"):
        assert reg.pick_for_role(role).id == "only"


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_empty_registry_raises():
    with pytest.raises(ValueError):
        ModelRegistry().pick_for_role("planner")


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_registry_role_override_wins_over_default():
    reg = _full_pool()
    # Pin the planner to the reasoning tier instead of the default light tier.
    reg.role_routing["planner"] = RoleSpec(tier="reasoning", tags=[])
    assert reg.pick_for_role("planner").id == "think"


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_call_override_wins_over_registry_and_default():
    reg = _full_pool()
    reg.role_routing["planner"] = RoleSpec(tier="reasoning")
    picked = reg.pick_for_role("planner", override=RoleSpec(tier="medium"))
    assert picked.id == "mid"


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_role_routing_round_trips_through_json():
    reg = _full_pool()
    reg.role_routing["planner"] = RoleSpec(tier="heavy", tags=["synthesis"])
    dumped = reg.model_dump_json()
    restored = ModelRegistry.model_validate_json(dumped)
    assert restored.role_routing["planner"].tier == "heavy"
    assert restored.role_routing["planner"].tags == ["synthesis"]
    assert restored.pick_for_role("planner").id == "big"


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_default_map_covers_all_quarq_roles():
    # The four Quarq roles plus the RLM-GEPA roles (executor/sub-LM/proposer).
    assert {"planner", "generator", "learner", "judge"} <= set(_DEFAULT_ROLE_ROUTING)
    assert {"rlm-executor", "rlm-sublm", "rlm-proposer"} <= set(_DEFAULT_ROLE_ROUTING)


@pytest.mark.concept(id="AU-ORCH.routing.conductor-per-step-model")
def test_create_model_accepts_role_kwarg():
    # Under AGENT_UTILITIES_TESTING the factory returns a TestModel; the point is that
    # the new `role` kwarg is accepted and resolution never crashes the factory.
    from agent_utilities.core.model_factory import create_model

    assert create_model(role="planner") is not None
