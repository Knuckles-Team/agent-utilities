"""Pluggable reasoning paradigms + outcome-learning router (CONCEPT:AU-KG.compute.first-class-reasoner-paradigm).

The router selects a paradigm per task (capability tags blended with a learned
reward EMA over the existing CapabilityIndex) and feeds the scored result back, so
routing self-improves. Built-in paradigms wire KG-2.69 (induction) and KG-2.67
(planning) plus deductive + generative behind one seam.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.reasoner import (
    GenerativeReasoner,
    ReasonerRouter,
    ReasoningResult,
    ReasoningTask,
    get_reasoner_router,
    reason,
)
from agent_utilities.knowledge_graph.core.world_model import WorldModel

pytestmark = pytest.mark.concept("AU-KG.compute.first-class-reasoner-paradigm")


# ── built-in paradigms behind the router ─────────────────────────────


class TestParadigmsViaRouter:
    def test_induction_routes_to_program_synthesis(self):
        task = ReasoningTask(
            goal="learn the mapping",
            tags=("induction",),
            payload={
                "primitives": {"double": lambda x: x * 2, "inc": lambda x: x + 1},
                "examples": [(1, 2), (3, 6)],
            },
        )
        result = reason(task)
        assert result.reasoner == "program_synthesis"
        assert result.answer.ops == ("double",) and result.score == 1.0

    def test_deduction_forward_chains(self):
        task = ReasoningTask(
            goal="derive C",
            tags=("deduction",),
            payload={
                "facts": ["A"],
                "rules": [(("A",), "B"), (("B",), "C")],
                "goal_fact": "C",
            },
        )
        result = get_reasoner_router().reason(task)
        assert result.reasoner == "deductive"
        assert "C" in result.answer and result.score == 1.0

    def test_planning_routes_to_world_model(self):
        wm = WorldModel()
        wm.observe("a", "go", "b")
        wm.observe("b", "go", "goal")
        task = ReasoningTask(
            goal="reach goal",
            tags=("planning",),
            payload={
                "world_model": wm,
                "start": "a",
                "policy": lambda _s: "go",
                "horizon": 2,
                "goal_state": "goal",
            },
        )
        result = get_reasoner_router().reason(task)
        assert result.reasoner == "world_model" and result.score == 1.0

    def test_generative_uses_injected_fn(self):
        router = ReasonerRouter()
        router.register(
            GenerativeReasoner(
                llm_fn=lambda p: "Paris",
                verifier=lambda ans, task: 1.0 if ans == "Paris" else 0.0,
            )
        )
        result = router.reason(
            ReasoningTask(goal="capital of France?", tags=("generative",))
        )
        assert result.reasoner == "generative" and result.score == 1.0


# ── the learning loop ────────────────────────────────────────────────


class _Stub:
    def __init__(self, name, score):
        self.name = name
        self.capability_tags = ("shared",)
        self._score = score

    def reason(self, task):
        return ReasoningResult(answer=self.name, reasoner=self.name, score=self._score)


class TestLearningRouter:
    def test_router_converges_to_higher_reward_paradigm(self):
        router = ReasonerRouter(reward_weight=0.5)
        router.register(_Stub("good", 1.0))
        router.register(_Stub("bad", 0.0))
        task = ReasoningTask(goal="x", tags=("shared",))
        # Both start neutral (0.5). Run several rounds; the router records each
        # outcome and should settle on the consistently-higher-scoring paradigm.
        picks = [router.reason(task).reasoner for _ in range(8)]
        assert picks[-1] == "good"
        assert router.reward("good") > router.reward("bad")

    def test_outcome_feedback_updates_reward(self):
        router = ReasonerRouter()
        router.register(_Stub("good", 1.0))
        before = router.reward("good")
        router.reason(ReasoningTask(goal="x", tags=("shared",)))
        assert router.reward("good") > before  # the score was fed back

    def test_unknown_tags_fall_back_not_crash(self):
        router = ReasonerRouter()
        router.register(_Stub("only", 1.0))
        result = router.reason(ReasoningTask(goal="x", tags=("nonexistent",)))
        assert result is not None and result.reasoner == "only"

    def test_empty_router_returns_none(self):
        assert ReasonerRouter().reason(ReasoningTask(goal="x")) is None
