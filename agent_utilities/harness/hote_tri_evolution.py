#!/usr/bin/python
from __future__ import annotations

"""Hybrid Open-Ended Tri-Evolution — CONCEPT:AHE-3.50.

Distils **Hybrid Open-Ended Tri-Evolution Makes Better Deep Researcher** (HOTE,
arXiv:2606.13710). The ecosystem already owns the three deep-research modules as
*separate* pieces — a proposer (``OntologyReasoningDriver.extrapolate`` →
research topics), a solver (``ResearchPipelineRunner`` + ARA artifacts), and a
judge (``ConceptMatcher`` LLM-judge) — and evolves them *one at a time*
(``SaiFactoryController``, ``EvolveAgent``). HOTE's contribution, which we lacked,
is co-evolving all three *together* with **interdependent rewards**, and the claim
that doing so is *indispensable*: freeze any one module and the others stall.

This controller is that joint loop. Its default dynamics are an analytic, fully
deterministic coupling that makes the indispensability claim falsifiable in a CPU
unit test (``run_ablation``), while every module is also injectable so the real
OntologyReasoningDriver / ARA / ConceptMatcher can drive it in production:

* **Solver** improves only from *frontier* tasks (max learning signal at an
  intermediate success rate) and only as fast as the *judge* is calibrated — its
  reward is the judge's score.
* **Proposer** is rewarded for keeping the solver near a productive-struggle band,
  so it must track the *rising* solver skill — a frozen proposer makes tasks
  trivial as the solver improves, collapsing the learning signal.
* **Judge** is rewarded for calibration against a verifier; a miscalibrated judge
  feeds the solver a biased reward and slows it.

Reuses the ``AdaptationCurve`` adaptation-speed instrument (CONCEPT:AHE-3.27) and
``marginal_speed_gain`` so co-evolution is measured the same way SAI specialization
is. Pure Python — no model, no network.

Concept: hote-tri-evolution
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.adaptation_speed import (
    AdaptationCurve,
    marginal_speed_gain,
)

__all__ = [
    "TriModulePolicies",
    "RoundRecord",
    "TriEvolutionResult",
    "HybridTriEvolutionController",
]


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass
class TriModulePolicies:
    """The three co-evolving policies (one evolvable scalar each)."""

    proposer_difficulty: float = 0.0  # absolute task difficulty the proposer sets
    solver_skill: float = 0.0  # solver capability (the quantity we ultimately grow)
    judge_strictness: float = 0.5  # 0.5 = perfectly calibrated; deviation = bias

    def copy(self) -> TriModulePolicies:
        return TriModulePolicies(
            self.proposer_difficulty, self.solver_skill, self.judge_strictness
        )


@dataclass
class RoundRecord:
    """Per-round telemetry for one co-evolution step."""

    round: int
    success_prob: float
    informativeness: float
    reward_proposer: float
    reward_solver: float
    reward_judge: float
    solver_skill: float


@dataclass
class TriEvolutionResult:
    """Outcome of an evolution run under one mode."""

    mode: str
    final_skill: float
    rewards: dict[str, float]  # final per-module reward
    curve_metrics: dict[str, Any]
    curve: AdaptationCurve | None = None
    records: list[RoundRecord] = field(default_factory=list)


# Module names that can be (co-)evolved; "joint" evolves all three.
_MODULES = ("proposer", "solver", "judge")


@dataclass
class HybridTriEvolutionController:
    """Co-evolve proposer/solver/judge with interdependent rewards (HOTE).

    Injectable hooks let the real research modules drive the loop; the analytic
    defaults below encode the paper's coupling so the dynamics are testable:

    * ``success_fn(difficulty, skill) -> p`` — probability the solver solves a task
      of the given difficulty (default: ``logistic(skill − difficulty)``).
    * ``judge_fn(true_quality, strictness) -> score`` — the judge's (possibly
      biased) quality estimate (default: ``true_quality + (strictness − 0.5)``,
      clamped — calibrated iff ``strictness == 0.5``).

    ``band`` is the target success rate (productive-struggle frontier). ``lr`` is the
    solver learning rate; ``step`` is the proposer/judge adaptation step.
    """

    success_fn: Callable[[float, float], float] | None = None
    judge_fn: Callable[[float, float], float] | None = None
    band: float = 0.6
    lr: float = 1.0
    step: float = 0.5

    def _success(self, difficulty: float, skill: float) -> float:
        if self.success_fn is not None:
            return min(max(self.success_fn(difficulty, skill), 0.0), 1.0)
        return _logistic(skill - difficulty)

    def _judge(self, true_quality: float, strictness: float) -> float:
        if self.judge_fn is not None:
            return min(max(self.judge_fn(true_quality, strictness), 0.0), 1.0)
        return min(max(true_quality + (strictness - 0.5), 0.0), 1.0)

    def _round(self, pol: TriModulePolicies, i: int) -> RoundRecord:
        """Evaluate one round at the current policies (no mutation)."""
        p = self._success(pol.proposer_difficulty, pol.solver_skill)
        true_quality = p
        judge_score = self._judge(true_quality, pol.judge_strictness)
        # Frontier learning signal: maximal at p = 0.5, zero at p ∈ {0, 1}.
        informativeness = 4.0 * p * (1.0 - p)
        # A calibrated judge yields a trustworthy reward; bias erodes it.
        judge_reliability = 1.0 - abs(judge_score - true_quality)
        reward_solver = judge_score
        reward_proposer = 1.0 - abs(p - self.band)
        reward_judge = judge_reliability
        return RoundRecord(
            round=i,
            success_prob=round(p, 6),
            informativeness=round(informativeness, 6),
            reward_proposer=round(reward_proposer, 6),
            reward_solver=round(reward_solver, 6),
            reward_judge=round(reward_judge, 6),
            solver_skill=round(pol.solver_skill, 6),
        )

    def _advance(self, pol: TriModulePolicies, rec: RoundRecord, evolve: set[str]) -> None:
        """Mutate the policies in ``evolve`` using this round's interdependent signals."""
        if "judge" in evolve:
            # Calibrate toward 0.5 (the unbiased point).
            pol.judge_strictness += self.step * (0.5 - pol.judge_strictness)
        if "proposer" in evolve:
            # Aim difficulty so success lands on the frontier band: tracks rising skill.
            target = pol.solver_skill - _logit(self.band)
            pol.proposer_difficulty += self.step * (target - pol.proposer_difficulty)
        if "solver" in evolve:
            # Skill grows with task informativeness, gated by judge reliability —
            # so a frozen proposer (trivial tasks) or miscalibrated judge stalls it.
            judge_reliability = rec.reward_judge
            pol.solver_skill += self.lr * rec.informativeness * judge_reliability

    def evolve(
        self,
        rounds: int = 20,
        *,
        mode: str = "joint",
        policies: TriModulePolicies | None = None,
    ) -> TriEvolutionResult:
        """Run ``rounds`` of co-evolution under ``mode``.

        ``mode`` ∈ ``{"joint", "proposer", "solver", "judge"}``: "joint" evolves all
        three modules together (HOTE); the others evolve only the named module and
        freeze the rest (the ablations). Returns the final solver skill, the final
        per-module rewards, and the ``AdaptationCurve`` metrics over the skill curve.
        """
        if mode == "joint":
            evolve = set(_MODULES)
        elif mode in _MODULES:
            evolve = {mode}
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        pol = (policies or TriModulePolicies()).copy()
        curve = AdaptationCurve()
        records: list[RoundRecord] = []
        last = self._round(pol, 0)
        for i in range(1, rounds + 1):
            rec = self._round(pol, i)
            records.append(rec)
            # Adaptation-speed instrument over the system objective (solver skill).
            curve.record(t_wall=float(i), n_samples=i, reward=rec.solver_skill)
            self._advance(pol, rec, evolve)
            last = rec

        return TriEvolutionResult(
            mode=mode,
            final_skill=round(pol.solver_skill, 6),
            rewards={
                "proposer": last.reward_proposer,
                "solver": last.reward_solver,
                "judge": last.reward_judge,
            },
            curve_metrics=curve.metrics(tau=last.solver_skill or 0.0),
            curve=curve,
            records=records,
        )

    def run_ablation(
        self, rounds: int = 20, *, policies: TriModulePolicies | None = None
    ) -> dict[str, Any]:
        """A/B harness: joint vs each solo ablation (the paper's central claim).

        Returns each mode's final solver skill, the marginal speed gain of joint
        over the best solo ablation, and ``indispensable`` — True when joint
        co-evolution strictly beats *every* solo ablation (i.e. evolving all three
        together is necessary, not merely sufficient).
        """
        base = (policies or TriModulePolicies()).copy()
        results = {
            m: self.evolve(rounds, mode=m, policies=base) for m in ("joint", *_MODULES)
        }
        finals = {m: r.final_skill for m, r in results.items()}
        joint = finals["joint"]
        # Pick the strongest solo ablation by learning-AUC as the comparison baseline.
        best_solo_mode = max(
            _MODULES, key=lambda m: results[m].curve_metrics.get("learning_auc", 0.0)
        )
        joint_curve = results["joint"].curve or AdaptationCurve()
        best_solo_curve = results[best_solo_mode].curve or AdaptationCurve()
        gain = marginal_speed_gain(best_solo_curve, joint_curve, tau=joint)
        return {
            "final_skill": finals,
            "best_solo": max(finals[m] for m in _MODULES),
            "joint": joint,
            "indispensable": all(joint > finals[m] for m in _MODULES),
            "marginal_speed_gain": round(gain, 6),
            "rewards": {m: results[m].rewards for m in ("joint", *_MODULES)},
        }
