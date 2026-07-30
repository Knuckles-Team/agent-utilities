#!/usr/bin/python
from __future__ import annotations

"""Loop / tool / token / cost / time budgets + classified termination.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

Every topology run in this package carries a :class:`Budgets` declaration and
enforces it through :class:`BudgetTracker`, which wraps the EXISTING
time/token/cost governor
(:class:`agent_utilities.graph.reactive.budget.BudgetGuard`) instead of
re-implementing wall-clock/token/cost accounting, and adds the two axes that
governor does not cover: loop-count and tool-call-count.

Truthfulness (``AGENTS.md``): on any budget exhaustion, :class:`BudgetExhausted`
carries a :class:`TerminationProof` with ``degraded=True``. Every ``run_*``
entry point in this package catches it, returns the state accumulated so far
(never discarded), and reports the proof as-is — a budget-halted run is never
reported as a clean success.

Seam / deferred dependency: a parallel lane is separately building a
general-purpose budget/repair layer for the whole harness. This module is
deliberately the narrow, swappable seam that layer can replace — see
``reports/deferred/lane-4.3.md`` (D-4.3-1) for the recorded dependency.
"""

from dataclasses import dataclass
from enum import StrEnum

from ..reactive.budget import BudgetGuard, BudgetTrippedException


class TerminationReason(StrEnum):
    """Classified termination outcome — never left implicit."""

    GOAL_REACHED = "goal_reached"
    CONVERGED = "converged"  # search exhausted with a stable best candidate
    LOOP_BUDGET_EXHAUSTED = "loop_budget_exhausted"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    GROUNDING_FAILURE = "grounding_failure"  # repeated tool failure / doom loop
    MAX_DEPTH_REACHED = "max_depth_reached"


@dataclass
class TerminationProof:
    """The truthful, classified outcome of a topology run.

    ``success`` is the topology's own judgment that it reached a usable
    answer; ``degraded`` is set whenever the run was halted early by a budget
    rather than concluding naturally — callers MUST gate "clean success"
    reporting on ``success and not degraded``, never on ``success`` alone.
    """

    reason: TerminationReason
    success: bool
    degraded: bool
    detail: str = ""

    def as_report(self) -> dict[str, object]:
        """Render a truthful outcome dict — a degraded run is never framed as
        a clean success, matching the AGENTS.md truthfulness contract."""
        return {
            "reason": self.reason.value,
            "success": bool(self.success) and not self.degraded,
            "degraded": self.degraded,
            "detail": self.detail,
        }


@dataclass
class Budgets:
    """Declared resource ceilings a topology run must not exceed."""

    loop_budget: int
    tool_budget: int = 0
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    time_budget_s: float | None = None


class BudgetExhausted(Exception):
    """Raised the moment any axis of a :class:`Budgets` is exceeded."""

    def __init__(self, proof: TerminationProof) -> None:
        super().__init__(proof.detail or proof.reason.value)
        self.proof = proof


_GUARD_REASON = {
    "time": TerminationReason.TIME_BUDGET_EXHAUSTED,
    "tokens": TerminationReason.TOKEN_BUDGET_EXHAUSTED,
    "cost": TerminationReason.COST_BUDGET_EXHAUSTED,
}


class BudgetTracker:
    """Enforces a :class:`Budgets` declaration across a single topology run."""

    def __init__(
        self,
        budgets: Budgets,
        *,
        run_id: str = "",
        guard: BudgetGuard | None = None,
    ) -> None:
        self.budgets = budgets
        self._loops = 0
        self._tool_calls = 0
        self._run_id = run_id or f"reasoning-{id(self)}"
        self._guard = guard or BudgetGuard(
            max_time_seconds=budgets.time_budget_s,
            max_tokens=budgets.token_budget,
            max_cost_usd=budgets.cost_budget_usd,
        )

    @property
    def loops_used(self) -> int:
        return self._loops

    @property
    def tool_calls_used(self) -> int:
        return self._tool_calls

    def tick_loop(self) -> None:
        """Count one loop iteration; raise once ``loop_budget`` is exceeded."""
        self._loops += 1
        if self._loops > self.budgets.loop_budget:
            raise BudgetExhausted(
                TerminationProof(
                    reason=TerminationReason.LOOP_BUDGET_EXHAUSTED,
                    success=False,
                    degraded=True,
                    detail=f"{self._loops} loops used (budget {self.budgets.loop_budget})",
                )
            )
        self.check_time_and_cost()

    def tick_tool_call(self) -> None:
        """Count one tool call; raise once ``tool_budget`` is exceeded (0 = no tools allowed)."""
        self._tool_calls += 1
        if self.budgets.tool_budget and self._tool_calls > self.budgets.tool_budget:
            raise BudgetExhausted(
                TerminationProof(
                    reason=TerminationReason.TOOL_BUDGET_EXHAUSTED,
                    success=False,
                    degraded=True,
                    detail=(
                        f"{self._tool_calls} tool calls used "
                        f"(budget {self.budgets.tool_budget})"
                    ),
                )
            )
        self.check_time_and_cost()

    def check_time_and_cost(self) -> None:
        """Delegate wall-clock/token/cost enforcement to the shared ``BudgetGuard``."""
        try:
            self._guard.check_limits(self._run_id)
        except BudgetTrippedException as exc:
            reason = _GUARD_REASON.get(
                exc.limit_type, TerminationReason.TIME_BUDGET_EXHAUSTED
            )
            raise BudgetExhausted(
                TerminationProof(
                    reason=reason, success=False, degraded=True, detail=str(exc)
                )
            ) from exc
