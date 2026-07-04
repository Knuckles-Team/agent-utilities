"""Value-aware test-time-compute governor (CONCEPT:AU-OS.scaling.value-test-time-compute).

Stops an iterative test-time search once a result satisfices or the marginal gain per
attempt drops below a floor, so compute is spent where the return is highest. Consumed
by the reasoning router's adaptive mode.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.compute_governor import ComputeGovernor
from agent_utilities.knowledge_graph.core.reasoner import (
    ReasonerRouter,
    ReasoningResult,
    ReasoningTask,
)

pytestmark = pytest.mark.concept("AU-OS.scaling.value-test-time-compute")


class TestGovernor:
    def test_empty_always_continues(self):
        assert ComputeGovernor().should_continue([]) is True

    def test_stops_on_satisfice(self):
        assert ComputeGovernor(satisfice=0.95).should_continue([0.97]) is False

    def test_stops_on_diminishing_returns(self):
        # second attempt added no gain over the first ⇒ stop
        assert (
            ComputeGovernor(min_marginal_gain=0.05).should_continue([0.5, 0.5]) is False
        )

    def test_continues_on_meaningful_gain(self):
        assert (
            ComputeGovernor(min_marginal_gain=0.05).should_continue([0.5, 0.7]) is True
        )

    def test_hard_cap(self):
        assert ComputeGovernor(max_attempts=2).should_continue([0.1, 0.2]) is False


class _Para:
    def __init__(self, name, score):
        self.name = name
        self.capability_tags = ("shared",)
        self._score = score
        self.calls = 0

    def reason(self, task):
        self.calls += 1
        return ReasoningResult(answer=self.name, reasoner=self.name, score=self._score)


class TestRouterAdaptive:
    def test_stops_after_satisficing_paradigm(self):
        router = ReasonerRouter(reward_weight=0.5)
        winner = _Para("winner", 1.0)
        loser = _Para("loser", 0.4)
        router.register(winner)
        router.register(loser)
        # bias routing so the winner is tried first
        router._index.record_outcome("winner", reward=1.0)
        result = router.reason_adaptive(
            ReasoningTask(goal="x", tags=("shared",)), ComputeGovernor(satisfice=0.95)
        )
        assert result.reasoner == "winner" and result.trace["attempts"] == 1
        assert loser.calls == 0  # never tried — compute saved

    def test_tries_more_when_unsatisfied(self):
        router = ReasonerRouter()
        router.register(_Para("a", 0.5))
        router.register(_Para("b", 0.5))
        result = router.reason_adaptive(
            ReasoningTask(goal="x", tags=("shared",)),
            ComputeGovernor(satisfice=0.95, min_marginal_gain=0.05, max_attempts=4),
        )
        # both score 0.5: tries a (0.5), then b adds no gain ⇒ stops at 2
        assert result.trace["attempts"] == 2
        assert result.trace["governor"]["best"] == 0.5
