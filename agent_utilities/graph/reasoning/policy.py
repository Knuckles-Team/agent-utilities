#!/usr/bin/python
from __future__ import annotations

"""Cost-aware topology escalation policy.

CONCEPT:AU-ORCH.routing.topology-escalation-policy

Default policy: start with the cheapest ADEQUATE topology and escalate only on
MEASURED uncertainty or failure — never a hardcoded "always use the most
expensive topology" choice. :class:`EscalationPolicy` is a pure, testable
decision function driven by live :class:`~.benchmark.TopologyStats` and/or a
run's own reported confidence; it holds no state itself beyond the ladder and
thresholds, so it composes cleanly with whatever loop drives repeated
escalation attempts.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .benchmark import TopologyStats

#: Escalation rung order from cheapest to most expensive. ``react`` is
#: intentionally NOT on this ladder — it is selected by task shape (a task
#: that needs tools), not by an escalation rung; see :meth:`EscalationPolicy.choose`.
DEFAULT_LADDER: tuple[str, ...] = (
    "cot",
    "self_consistent_cot",
    "tot_bfs",
    "got",
    "rap",
)


@dataclass(frozen=True)
class EscalationDecision:
    """Which topology to run next, and why."""

    topology: str
    rung: int
    reason: str  # "initial" | "needs_tools" | "low_confidence" | "prior_failure" | "adequate"


class EscalationPolicy:
    """Chooses the cheapest adequate topology, escalating on measured signal."""

    def __init__(
        self,
        ladder: Sequence[str] = DEFAULT_LADDER,
        *,
        confidence_floor: float = 0.6,
        reliability_floor: float = 0.6,
    ) -> None:
        if not ladder:
            raise ValueError("ladder must be non-empty")
        self.ladder = tuple(ladder)
        self.confidence_floor = confidence_floor
        self.reliability_floor = reliability_floor

    def choose(
        self,
        *,
        needs_tools: bool = False,
        prior_confidence: float | None = None,
        prior_stats: TopologyStats | None = None,
        current_rung: int = 0,
    ) -> EscalationDecision:
        """Decide the next topology to run.

        ``needs_tools`` routes straight to ReAct regardless of rung (an
        orthogonal axis: the task shape, not measured uncertainty).
        Otherwise, the FIRST call (no prior signal at all) starts at rung 0 —
        the cheapest topology. A later call escalates one rung when
        ``prior_confidence`` is below ``confidence_floor`` or
        ``prior_stats.reliability`` is below ``reliability_floor``; it never
        de-escalates and never exceeds the top of the ladder.
        """
        if needs_tools:
            return EscalationDecision(
                topology="react", rung=current_rung, reason="needs_tools"
            )

        current_rung = max(0, min(current_rung, len(self.ladder) - 1))
        if prior_confidence is None and prior_stats is None:
            return EscalationDecision(topology=self.ladder[0], rung=0, reason="initial")

        escalate = False
        reason = "adequate"
        if prior_confidence is not None and prior_confidence < self.confidence_floor:
            escalate, reason = True, "low_confidence"
        if prior_stats is not None and prior_stats.reliability < self.reliability_floor:
            escalate, reason = True, "prior_failure"

        rung = current_rung
        if escalate and rung < len(self.ladder) - 1:
            rung += 1
        return EscalationDecision(topology=self.ladder[rung], rung=rung, reason=reason)
