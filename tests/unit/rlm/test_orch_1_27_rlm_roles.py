"""CONCEPT:ORCH-1.27 (RLM extension) — RLM-GEPA role bindings + resolution."""

from __future__ import annotations

import pytest

from agent_utilities.models.model_registry import (
    _DEFAULT_ROLE_ROUTING,
    ModelDefinition,
    ModelRegistry,
)
from agent_utilities.rlm.roles import RLM_ROLES, rlm_role_model


def _model(mid, tier, tags):
    return ModelDefinition(
        id=mid, name=mid, provider="openai", model_id=mid, tier=tier, tags=tags
    )


@pytest.mark.concept(id="ORCH-1.27")
def test_rlm_roles_registered_in_default_map():
    for role in RLM_ROLES:
        assert role in _DEFAULT_ROLE_ROUTING, role
    # Executor/sub-LM are cheap (light); proposer is strong (reasoning).
    assert _DEFAULT_ROLE_ROUTING["rlm-executor"].tier == "light"
    assert _DEFAULT_ROLE_ROUTING["rlm-sublm"].tier == "light"
    assert _DEFAULT_ROLE_ROUTING["rlm-proposer"].tier == "reasoning"


@pytest.mark.concept(id="ORCH-1.27")
def test_pick_for_role_resolves_rlm_roles_on_a_pool():
    reg = ModelRegistry(
        models=[
            _model("cheap", "light", ["code"]),
            _model("mid", "medium", []),
            _model("strong", "reasoning", ["synthesis"]),
        ]
    )
    assert reg.pick_for_role("rlm-executor").id == "cheap"
    assert reg.pick_for_role("rlm-proposer").id == "strong"


@pytest.mark.concept(id="ORCH-1.27")
def test_rlm_role_model_falls_back_without_raising():
    # No registry file configured under test → returns the fallback, never raises.
    assert rlm_role_model("rlm-proposer", fallback="openai:gpt-4o-mini") is not None
