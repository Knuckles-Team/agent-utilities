"""Unified planning package (Plan 03 Step 4).

This package consolidates the previously standalone planning family into a
single import surface and a single :class:`Planner` facade.

Strangled / re-exported implementation modules (each original remains the live
implementation behind this package — see the per-module STRANGLER notes):

- :mod:`agent_utilities.graph.planning.htn`        ← ``graph/hierarchical_planner.py``
- :mod:`agent_utilities.graph.planning.curriculum` ← ``graph/horizon_curriculum.py``
- :mod:`agent_utilities.graph.planning.reward`     ← ``graph/reward_decomposition.py``
- :mod:`agent_utilities.graph.planning.manifest`   ← ``graph/manifest_generators.py``

The NEW unified entrypoint is :class:`Planner`, which provides a single
HTN-style planning API::

    planner = Planner(model=...)
    plan = await planner.decompose(goal, ctx)
    plan = await planner.refine(plan, feedback)

All re-exported classes are the *same* objects as in the original modules,
guaranteeing a single source of the planning API.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.models.graph import ExecutionStep, GraphPlan

# --- Re-export the strangled planning surface (single source of API) ---------
from .curriculum import (
    CurriculumStage,
    HorizonCurriculum,
    HorizonStageConfig,
    MacroAction,
    PromotionPolicy,
    SubgoalCheckpoint,
)
from .htn import (
    AggregationStrategy,
    ConvergenceMonitor,
    EvolutionaryAggregator,
    GroupFitness,
    LATSPlanner,
    RecursionDepthExceeded,
    RecursiveContext,
    architect_step,
    execute_recursive_graph,
    fetch_epistemic_context,
    memory_selection_step,
    planner_step,
    researcher_step,
)
from .manifest import (
    manifest_for_enterprise,
    manifest_from_department,
    manifest_from_planner,
    manifest_from_preset,
    manifest_from_teamconfig,
    manifest_from_workflow,
)
from .reward import (
    DecomposedRewardRecord,
    RewardDecomposer,
    StepOutcome,
    StepReward,
    TrajectoryOutcome,
    TrajectoryReward,
)

logger = logging.getLogger(__name__)

# ``Plan`` is an alias for the canonical plan model so the facade signature
# (-> Plan) reads naturally while remaining a single object identity.
Plan = GraphPlan


_DECOMPOSE_SYSTEM_PROMPT = (
    "You are an HTN (Hierarchical Task Network) planner. Decompose the user's "
    "high-level goal into a concrete, ordered list of executable steps. Each "
    "step must have a stable 'id' and a clear 'description'. Steps that can run "
    "concurrently should set 'parallel=True'. Keep steps coarse-grained and "
    "delegate fine refinement to the executor."
)

_REFINE_SYSTEM_PROMPT = (
    "You are an HTN (Hierarchical Task Network) re-planner. You are given a "
    "previous plan and verification feedback explaining why it was insufficient. "
    "Produce a CORRECTED plan that addresses the feedback. You MUST change the "
    "approach rather than repeat the failed strategy."
)


class Planner:
    """Unified planning facade — the single planning entrypoint (Plan 03 Step 4).

    Consolidates the HTN decomposition / refinement logic (and exposes the
    curriculum, reward-decomposition, and manifest-generation surfaces via the
    package re-exports).

    The facade is intentionally thin and self-contained: ``decompose`` and
    ``refine`` build a ``pydantic_ai.Agent`` whose ``output_type`` is
    :class:`GraphPlan`. The model can be injected directly (``model=...``), or
    derived from the ``ctx`` passed to :meth:`decompose` (``ctx.deps.agent_model``
    or ``ctx.agent_model``), which lets callers reuse the graph's configured
    model and lets tests inject ``pydantic_ai.models.test.TestModel``.
    """

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    # -- model / context resolution -------------------------------------------
    @staticmethod
    def _resolve_model(model: Any | None, ctx: Any | None) -> Any | None:
        if model is not None:
            return model
        if ctx is None:
            return None
        deps = getattr(ctx, "deps", None)
        for holder in (deps, ctx):
            if holder is None:
                continue
            candidate = getattr(holder, "agent_model", None)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _resolve_deps(ctx: Any | None) -> Any | None:
        if ctx is None:
            return None
        deps = getattr(ctx, "deps", None)
        return deps if deps is not None else ctx

    @staticmethod
    def _goal_text(goal: Any) -> str:
        if isinstance(goal, str):
            return goal
        for attr in ("query", "goal", "description", "objective"):
            value = getattr(goal, attr, None)
            if isinstance(value, str) and value:
                return value
        return str(goal)

    def _build_agent(self, model: Any, system_prompt: str) -> Any:
        from pydantic_ai import Agent

        return Agent(
            model=model,
            output_type=GraphPlan,
            system_prompt=system_prompt,
        )

    @staticmethod
    def _fallback_plan(goal_text: str, reason: str) -> GraphPlan:
        """Deterministic single-step plan used when no model is available."""
        return GraphPlan(
            steps=[ExecutionStep(id="execute", description=goal_text)],
            metadata={"reasoning": reason, "planner": "Planner.fallback"},
        )

    # -- public facade API ----------------------------------------------------
    async def decompose(self, goal: Any, ctx: Any | None = None) -> Plan:
        """Decompose a high-level ``goal`` into an executable :class:`GraphPlan`.

        Args:
            goal: The objective. Either a plain string or any object exposing a
                ``query``/``goal``/``description``/``objective`` attribute.
            ctx: Optional planning context. If it (or its ``.deps``) exposes
                ``agent_model``, that model is used. Otherwise the model passed
                to the constructor is used.

        Returns:
            A :class:`GraphPlan` (aliased as ``Plan``).
        """
        goal_text = self._goal_text(goal)
        model = self._resolve_model(self.model, ctx)
        if model is None:
            logger.warning(
                "Planner.decompose: no model available; emitting fallback plan."
            )
            return self._fallback_plan(goal_text, "no model available")

        agent = self._build_agent(model, _DECOMPOSE_SYSTEM_PROMPT)
        deps = self._resolve_deps(ctx)
        prompt = f"Decompose this goal into an execution plan:\n\n{goal_text}"
        try:
            if deps is not None:
                res = await agent.run(prompt, deps=deps)
            else:
                res = await agent.run(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Planner.decompose failed ({exc}); emitting fallback.")
            return self._fallback_plan(goal_text, f"decompose error: {exc}")

        plan = getattr(res, "output", None)
        if not isinstance(plan, GraphPlan):
            return self._fallback_plan(goal_text, "non-plan model output")
        if not plan.steps:
            plan.steps = [ExecutionStep(id="execute", description=goal_text)]
        plan.metadata.setdefault("planner", "Planner.decompose")
        return plan

    async def refine(self, plan: Plan, feedback: Any, ctx: Any | None = None) -> Plan:
        """Refine an existing ``plan`` using verification ``feedback``.

        When a graph ``ctx`` carrying ``deps`` is supplied, refinement routes
        through :class:`LATSPlanner` (Language Agent Tree Search) — the same
        deep re-planning path used by ``planner_step``. Otherwise a single-shot
        re-planning agent is used.

        Args:
            plan: The previous :class:`GraphPlan`.
            feedback: Verification feedback (string or object with ``feedback``/
                ``message``/``error``).
            ctx: Optional planning context (model / deps source).

        Returns:
            A corrected :class:`GraphPlan`.
        """
        feedback_text = self._feedback_text(feedback)
        model = self._resolve_model(self.model, ctx)
        deps = self._resolve_deps(ctx)
        goal_text = self._plan_goal_text(plan)

        if model is None:
            logger.warning(
                "Planner.refine: no model available; returning original plan."
            )
            plan.metadata.setdefault("refine", "no model available")
            return plan

        previous = "\n".join(
            f"- {s.id}: {self._goal_text(s.description) if s.description else ''}"
            for s in plan.steps
        )
        query = (
            f"Goal: {goal_text}\n\n"
            f"### PREVIOUS PLAN\n{previous}\n\n"
            f"### VERIFICATION FEEDBACK (address this)\n{feedback_text}"
        )

        # Deep re-planning via LATS when full graph deps are available.
        if deps is not None and getattr(deps, "agent_model", None) is not None:
            try:
                lats = LATSPlanner(
                    context=(
                        f"### PREVIOUS PLAN\n{previous}\n\n"
                        f"### VERIFICATION FEEDBACK\n{feedback_text}"
                    ),
                    deps=deps,
                    model=model,
                )
                refined: Any = await lats.search(goal_text or query)
                if isinstance(refined, GraphPlan) and refined.steps:
                    refined.metadata.setdefault("planner", "Planner.refine.lats")
                    return refined
            except Exception as exc:
                logger.warning(
                    f"Planner.refine: LATS path failed ({exc}); falling back to "
                    "single-shot re-planning."
                )

        agent = self._build_agent(model, _REFINE_SYSTEM_PROMPT)
        try:
            if deps is not None:
                res = await agent.run(query, deps=deps)
            else:
                res = await agent.run(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Planner.refine failed ({exc}); returning original plan.")
            plan.metadata.setdefault("refine", f"error: {exc}")
            return plan

        refined = getattr(res, "output", None)
        if not isinstance(refined, GraphPlan):
            plan.metadata.setdefault("refine", "non-plan model output")
            return plan
        if not refined.steps:
            refined.steps = list(plan.steps)
        refined.metadata.setdefault("planner", "Planner.refine")
        return refined

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _feedback_text(feedback: Any) -> str:
        if isinstance(feedback, str):
            return feedback
        for attr in ("feedback", "message", "error", "reason"):
            value = getattr(feedback, attr, None)
            if isinstance(value, str) and value:
                return value
        return str(feedback)

    @classmethod
    def _plan_goal_text(cls, plan: Plan) -> str:
        goal = plan.metadata.get("goal") or plan.metadata.get("query")
        if isinstance(goal, str) and goal:
            return goal
        if plan.steps:
            first = plan.steps[0]
            return cls._goal_text(first.description) or first.id
        return ""


__all__ = [
    # Unified facade
    "Planner",
    "Plan",
    # Canonical plan models
    "GraphPlan",
    "ExecutionStep",
    # HTN surface (re-exported)
    "AggregationStrategy",
    "ConvergenceMonitor",
    "EvolutionaryAggregator",
    "GroupFitness",
    "LATSPlanner",
    "RecursionDepthExceeded",
    "RecursiveContext",
    "architect_step",
    "execute_recursive_graph",
    "fetch_epistemic_context",
    "memory_selection_step",
    "planner_step",
    "researcher_step",
    # Curriculum surface
    "CurriculumStage",
    "HorizonCurriculum",
    "HorizonStageConfig",
    "MacroAction",
    "PromotionPolicy",
    "SubgoalCheckpoint",
    # Reward surface
    "DecomposedRewardRecord",
    "RewardDecomposer",
    "StepOutcome",
    "StepReward",
    "TrajectoryOutcome",
    "TrajectoryReward",
    # Manifest surface
    "manifest_for_enterprise",
    "manifest_from_department",
    "manifest_from_planner",
    "manifest_from_preset",
    "manifest_from_teamconfig",
    "manifest_from_workflow",
]
