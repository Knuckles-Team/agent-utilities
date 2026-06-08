"""Plan 03 Step 4 — unified planning facade tests.

Asserts:
  (a) ``from agent_utilities.graph.planning import Planner`` works.
  (b) ``Planner().decompose(goal, ctx)`` returns a plan for a simple goal
      (LLM mocked via ``pydantic_ai.models.test.TestModel``).
  (c) re-exported classes are the SAME objects as in the original modules
      (identity check — single source of API).
  (d) the original import paths still resolve.
"""

from __future__ import annotations

import pytest

from agent_utilities.graph.planning import Planner
from agent_utilities.models.graph import ExecutionStep, GraphPlan


# --- (a) facade import resolves ---------------------------------------------
def test_planner_facade_imports():
    from agent_utilities.graph.planning import Plan
    from agent_utilities.graph.planning import Planner as P

    assert P is Planner
    # ``Plan`` is the canonical plan model alias.
    assert Plan is GraphPlan
    assert hasattr(Planner, "decompose")
    assert hasattr(Planner, "refine")


# --- helpers ----------------------------------------------------------------
def _test_model():
    """A pydantic-ai TestModel that returns a valid 3-step GraphPlan."""
    from pydantic_ai.models.test import TestModel

    plan = GraphPlan(
        steps=[
            ExecutionStep(id="step_1", description="gather context"),
            ExecutionStep(id="step_2", description="implement"),
            ExecutionStep(id="step_3", description="verify"),
        ],
        metadata={"reasoning": "test plan"},
    )
    # custom_output_args drives the structured output_type (GraphPlan).
    return TestModel(custom_output_args=plan.model_dump())


# --- (b) decompose returns a plan -------------------------------------------
@pytest.mark.asyncio
async def test_decompose_with_injected_model():
    planner = Planner(model=_test_model())
    plan = await planner.decompose("Build a CSV export feature")
    assert isinstance(plan, GraphPlan)
    assert len(plan.steps) == 3
    assert plan.steps[0].id == "step_1"


@pytest.mark.asyncio
async def test_decompose_model_from_ctx():
    class _Deps:
        agent_model = _test_model()

    class _Ctx:
        deps = _Deps()

    planner = Planner()
    plan = await planner.decompose("Refactor the auth module", _Ctx())
    assert isinstance(plan, GraphPlan)
    assert len(plan.steps) == 3


@pytest.mark.asyncio
async def test_decompose_fallback_without_model():
    # No model anywhere -> deterministic single-step fallback plan (real output).
    planner = Planner()
    plan = await planner.decompose("Some goal")
    assert isinstance(plan, GraphPlan)
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "Some goal"
    assert plan.metadata.get("reasoning") == "no model available"


# --- refine -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_refine_single_shot():
    planner = Planner(model=_test_model())
    original = GraphPlan(
        steps=[ExecutionStep(id="bad", description="wrong approach")],
        metadata={"goal": "Build a CSV export feature"},
    )
    refined = await planner.refine(original, "Use streaming, not in-memory")
    assert isinstance(refined, GraphPlan)
    assert len(refined.steps) == 3


@pytest.mark.asyncio
async def test_refine_fallback_without_model_returns_original():
    planner = Planner()
    original = GraphPlan(steps=[ExecutionStep(id="x", description="y")])
    refined = await planner.refine(original, "feedback")
    assert refined is original


# --- (c) identity proof: single source of API -------------------------------
def test_reexport_identity_htn():
    from agent_utilities.graph import hierarchical_planner as orig
    from agent_utilities.graph.planning import (
        LATSPlanner,
        architect_step,
        fetch_epistemic_context,
        memory_selection_step,
        planner_step,
        researcher_step,
    )

    assert LATSPlanner is orig.LATSPlanner
    assert planner_step is orig.planner_step
    assert researcher_step is orig.researcher_step
    assert architect_step is orig.architect_step
    assert memory_selection_step is orig.memory_selection_step
    assert fetch_epistemic_context is orig.fetch_epistemic_context


def test_reexport_identity_curriculum():
    from agent_utilities.graph import horizon_curriculum as orig
    from agent_utilities.graph.planning import (
        CurriculumStage,
        HorizonCurriculum,
        MacroAction,
        SubgoalCheckpoint,
    )

    assert HorizonCurriculum is orig.HorizonCurriculum
    assert CurriculumStage is orig.CurriculumStage
    assert MacroAction is orig.MacroAction
    assert SubgoalCheckpoint is orig.SubgoalCheckpoint


def test_reexport_identity_reward():
    from agent_utilities.graph import reward_decomposition as orig
    from agent_utilities.graph.planning import (
        RewardDecomposer,
        StepReward,
        TrajectoryReward,
    )

    assert RewardDecomposer is orig.RewardDecomposer
    assert StepReward is orig.StepReward
    assert TrajectoryReward is orig.TrajectoryReward


def test_reexport_identity_manifest():
    from agent_utilities.graph import manifest_generators as orig
    from agent_utilities.graph.planning import (
        manifest_for_enterprise,
        manifest_from_planner,
        manifest_from_teamconfig,
    )

    assert manifest_from_planner is orig.manifest_from_planner
    assert manifest_from_teamconfig is orig.manifest_from_teamconfig
    assert manifest_for_enterprise is orig.manifest_for_enterprise


# --- (d) original import paths still resolve --------------------------------
def test_original_import_paths_resolve():
    # And the router that depends on these still imports.
    import agent_utilities.graph._router_impl  # noqa: F401
    from agent_utilities.graph.hierarchical_planner import (  # noqa: F401
        LATSPlanner,
        planner_step,
        researcher_step,
    )
    from agent_utilities.graph.horizon_curriculum import (  # noqa: F401
        HorizonCurriculum,
    )
    from agent_utilities.graph.manifest_generators import (  # noqa: F401
        manifest_from_planner,
    )
    from agent_utilities.graph.reward_decomposition import (  # noqa: F401
        RewardDecomposer,
    )
