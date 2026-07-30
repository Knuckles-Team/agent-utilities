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

Token and cost accounting is CALLER-REPORTED, not inferred. This package never
sees a model response object, so it cannot know real token counts; inventing an
estimate and enforcing a ceiling against it would be a fabricated measurement.
Instead, :meth:`BudgetTracker.record_usage` is the seam: a caller that wants
``token_budget``/``cost_budget_usd`` enforced constructs the tracker itself,
passes it into the ``run_*`` entry point via ``tracker=``, and calls
``record_usage(...)`` from inside its own step/generate/simulate callable with
the real counts its model client reported. Without that, ``token_budget`` and
``cost_budget_usd`` are honestly inert — nothing in this package can fill them
in, and ``Budgets.declared_axes()`` says so.

Seam / deferred dependency: a parallel lane is separately building a
general-purpose budget/repair layer for the whole harness. This module is
deliberately the narrow, swappable seam that layer can replace — see
``reports/deferred/waves1-5-gate.md`` (D-W15-1) for the recorded dependency.
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
    """Declared resource ceilings a topology run must not exceed.

    ``tool_budget=0`` means NO tool calls are allowed (the ceiling is zero), not
    "unlimited" — :meth:`BudgetTracker.tick_tool_call` trips on the first call.
    Use ``tool_budget=None`` for an unbounded tool axis.
    """

    loop_budget: int
    tool_budget: int | None = 0
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    time_budget_s: float | None = None

    def declared_axes(self) -> dict[str, bool]:
        """Which axes this declaration actually constrains.

        Truthfulness seam: ``token``/``cost`` are only enforceable when the
        caller reports usage through :meth:`BudgetTracker.record_usage` (this
        package cannot measure tokens itself), so a consumer rendering a budget
        report can distinguish "declared and enforced" from "declared but the
        caller never reported anything to enforce it against".
        """
        return {
            "loop": True,
            "tool": self.tool_budget is not None,
            "token": self.token_budget is not None,
            "cost": self.cost_budget_usd is not None,
            "time": self.time_budget_s is not None,
        }


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
        """Count one tool call; raise once ``tool_budget`` is exceeded.

        ``tool_budget=0`` is a REAL ceiling of zero (the first tool call trips);
        ``tool_budget=None`` leaves the tool axis unbounded. The previous
        ``if self.budgets.tool_budget and ...`` guard made 0 mean "unlimited",
        silently contradicting its own docstring.
        """
        self._tool_calls += 1
        if (
            self.budgets.tool_budget is not None
            and self._tool_calls > self.budgets.tool_budget
        ):
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

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        thoughts_tokens: int = 0,
        tool_use_tokens: int = 0,
        model: str = "",
        agent_name: str = "reasoning-topology",
    ) -> None:
        """Report REAL, caller-measured token usage for this run, then re-check.

        Without this, ``token_budget``/``cost_budget_usd`` can never trip: the
        shared :class:`~agent_utilities.graph.reactive.budget.BudgetGuard` sums
        usage from the token tracker keyed by this run's id, and nothing in this
        package can measure a model call it never sees. This is the seam that
        makes those two axes real — call it from inside the ``step_fn`` /
        ``generate_fn`` / ``simulate_fn`` you pass to a ``run_*`` entry point,
        with the counts your model client actually reported (never an estimate;
        a fabricated token count enforced as a ceiling is worse than no ceiling).

        Cost is derived by the same guard from these token counts and its own
        configured per-token rates — it is not a second, independently
        fabricated number.

        Raises :class:`BudgetExhausted` immediately if this usage crosses the
        token or cost ceiling, so the enclosing ``run_*`` returns the state
        accumulated so far with ``degraded=True`` rather than continuing to spend.
        """
        from ...observability.token_tracker import TokenUsageRecord

        self._guard.token_tracker.record(
            TokenUsageRecord(
                session_id=self._run_id,
                agent_name=agent_name,
                model_name=model,
                prompt_tokens=int(prompt_tokens),
                response_tokens=int(response_tokens),
                thoughts_tokens=int(thoughts_tokens),
                tool_use_tokens=int(tool_use_tokens),
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
