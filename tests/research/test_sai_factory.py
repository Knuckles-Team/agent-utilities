#!/usr/bin/python
"""Tests for the SAI factory closed-loop controller (AHE-3.29).

CPU-only: a toy verifier (counts a target token) + toy generators stand in for the
real kernel verifier and LLM generator, exercising the scaffolding arm, the
injected weight arm, marginal-speed arm attribution, and the promotion ratchet.
"""

from __future__ import annotations

from agent_utilities.harness.sai_task import SpecializationTask, VerifierResult
from agent_utilities.knowledge_graph.research.sai_factory import (
    Harvested,
    SaiFactoryController,
    Specialist,
)


class _CountVerifier:
    """Reward = min(count('good')/5, 1.0); passes when at least one is present."""

    def verify(self, candidate: str) -> VerifierResult:
        n = (candidate or "").split().count("good")
        reward = min(n / 5.0, 1.0)
        return VerifierResult(reward=reward, passed=n > 0, detail={"count": n})


def _gen_from(quality: dict[str, int]):
    """A generator that emits ``quality[scaffold]`` 'good' tokens for a scaffold."""

    def gen(scaffold: str) -> str:
        return " ".join(["good"] * quality.get(scaffold, 0))

    return gen


def _task() -> SpecializationTask:
    return SpecializationTask(
        task_id="toy",
        prompt_corpus=["poor", "rich"],
        verifier=_CountVerifier(),
        target_tau=0.6,
    )


def test_scaffold_arm_selects_best_variant_and_promotes():
    task = _task()
    ctrl = SaiFactoryController(task, _gen_from({"poor": 1, "rich": 4}), scaffolds=["poor", "rich"])
    result = ctrl.run(rounds=1)
    # "rich" → 4/5 = 0.8 beats "poor" → 0.2
    assert result.specialist.scaffold == "rich"
    assert result.specialist.reward == 0.8
    assert result.rounds[0].arm == "scaffold"
    assert result.rounds[0].promoted is True


def test_curve_reaches_target_and_metrics_populated():
    task = _task()
    ctrl = SaiFactoryController(task, _gen_from({"poor": 1, "rich": 4}), scaffolds=["poor", "rich"])
    result = ctrl.run(rounds=2)
    assert result.curve.reached(task.target_tau) is True
    m = result.metrics()
    assert m["reached"] is True
    assert m["final_specialist_reward"] == 0.8
    assert m["promotions"] >= 1
    assert m["sample_complexity"] is not None


def test_weight_arm_overtakes_scaffolding_and_is_attributed():
    task = _task()

    def weight_arm(_task, harvested: list[Harvested]):
        # The trainer "learns" to emit 5 good tokens regardless of scaffold → reward 1.0
        assert harvested, "weight arm should receive harvested winners"
        return ("adapter-1", _gen_from({"poor": 5, "rich": 5}))

    ctrl = SaiFactoryController(
        task,
        _gen_from({"poor": 1, "rich": 4}),
        scaffolds=["poor", "rich"],
        weight_arm=weight_arm,
    )
    result = ctrl.run(rounds=1)
    assert result.specialist.reward == 1.0
    assert result.specialist.adapter_id == "adapter-1"
    assert result.rounds[0].arm == "weights"
    assert result.rounds[0].weight_gain != 0.0


def test_ratchet_blocks_a_regressing_challenger():
    task = _task()
    ctrl = SaiFactoryController(task, _gen_from({"poor": 1, "rich": 2}), scaffolds=["poor", "rich"])
    # Force a high incumbent the scaffold arm (max 0.4) cannot match.
    ctrl.specialist = Specialist(scaffold="rich", generator=ctrl.specialist.generator, reward=0.9)
    rr = ctrl.run_round(0)
    assert rr.promoted is False
    assert rr.arm == "none"
    assert ctrl.specialist.reward == 0.9  # incumbent preserved


def test_weight_arm_absent_is_pure_scaffolding():
    task = _task()
    ctrl = SaiFactoryController(task, _gen_from({"poor": 1, "rich": 4}), scaffolds=["poor", "rich"])
    result = ctrl.run(rounds=2)
    assert all(r.arm in ("scaffold", "none") for r in result.rounds)
    assert all(r.weight_gain == 0.0 for r in result.rounds)
