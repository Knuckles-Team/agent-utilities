#!/usr/bin/python
from __future__ import annotations

"""SAI factory — the closed specialization loop that produces superhuman specialists.

CONCEPT:AU-AHE.harness.sai-controller — the controller that operationalizes Superhuman Adaptable
Intelligence (arXiv:2602.23643): given an important task and a machine-verifiable
reward, it produces a *certified-better specialist* by running two improvement
arms against the same verifier and keeping whichever advances capability —

* the **scaffolding arm** (AHE-3.2): search a set of prompt/scaffold variants,
  keep the one whose generated candidate earns the highest verified reward; and
* the **weight arm** (OS-5.34 harvest → AHE-3.25 distillation): harvest the
  verified-winning candidates into training data and fine-tune a specialist
  generator (injected callable, so the heavy trainer + GPU stay out of this
  module and out of CI).

The loop is *measured and steered by adaptation speed* (AHE-3.27): every
evaluation is appended to an :class:`~agent_utilities.harness.adaptation_speed.AdaptationCurve`,
and each round is attributed to the arm that bought the larger marginal
adaptation-speed gain — directly answering the paper's "better scaffolding vs.
weight updates: which helped" question. Promotion is a **monotone per-task
ratchet** (the AHE-3.24 idea applied to task reward instead of pytest pass-rates):
the incumbent specialist is replaced only when a challenger's verified reward on a
fresh evaluation is ≥ incumbent − tolerance; otherwise the round rolls back.

What was missing before this: AU had the scaffolding evolver, the trainer
(ML-001..007), the verifier suite, and the harvest seam — but *no factory between
them*. This is that factory. The heavy arms (real generator, real trainer,
real harvest) are constructor-injected callables; the default toys make the whole
loop importable and unit-testable on CPU, and the live engine tick (Wire-First)
passes the production arms.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.adaptation_speed import (
    AdaptationCurve,
    marginal_speed_gain,
)
from agent_utilities.harness.sai_task import SpecializationTask, VerifierResult

logger = logging.getLogger(__name__)

#: A generator turns a scaffold/prompt into a candidate output for the task.
GenerateFn = Callable[[str], str]

#: A weight arm consumes harvested (scaffold, candidate, reward) winners and
#: returns (adapter_id, new_generator) — or ``None`` if it declined/failed to
#: train (e.g. too few winners, no GPU). It is the seam to OS-5.34 + AHE-3.25.
WeightArm = Callable[
    [SpecializationTask, "list[Harvested]"], "tuple[str, GenerateFn] | None"
]


@dataclass(frozen=True)
class Harvested:
    """One verified-winning candidate, the unit the weight arm distills."""

    scaffold: str
    candidate: str
    reward: float


@dataclass
class Specialist:
    """The current best specialization artifact for a task."""

    scaffold: str
    generator: GenerateFn
    reward: float
    adapter_id: str | None = None

    def generate(self) -> str:
        return self.generator(self.scaffold)


@dataclass
class RoundResult:
    """Outcome of one factory cycle."""

    round_index: int
    arm: str  # "scaffold" | "weights" | "none"
    challenger_reward: float
    incumbent_reward: float
    promoted: bool
    scaffold_gain: float
    weight_gain: float


@dataclass
class FactoryResult:
    """The product of a full factory run."""

    task_id: str
    specialist: Specialist
    curve: AdaptationCurve
    rounds: list[RoundResult] = field(default_factory=list)

    def metrics(self) -> dict[str, Any]:
        m = self.curve.metrics(self._tau)
        m["promotions"] = sum(1 for r in self.rounds if r.promoted)
        m["arm_attribution"] = {
            arm: sum(1 for r in self.rounds if r.arm == arm)
            for arm in ("scaffold", "weights", "none")
        }
        m["final_specialist_reward"] = round(self.specialist.reward, 6)
        return m

    _tau: float = 0.0


class SaiFactoryController:
    """Run the closed scaffolding+weights specialization loop for one task."""

    def __init__(
        self,
        task: SpecializationTask,
        generate_fn: GenerateFn,
        *,
        scaffolds: Sequence[str] | None = None,
        weight_arm: WeightArm | None = None,
        tolerance: float = 0.0,
        harvest_min_reward: float = 0.0,
    ) -> None:
        self.task = task
        self.scaffolds: list[str] = list(scaffolds or task.prompt_corpus or [""])
        if not self.scaffolds:
            self.scaffolds = [""]
        self.weight_arm = weight_arm
        self.tolerance = float(tolerance)
        self.harvest_min_reward = float(harvest_min_reward)
        self.curve = AdaptationCurve(task_id=task.task_id)
        self._harvest: list[Harvested] = []
        self._n_samples = 0
        self._t = 0.0
        # Seed the incumbent from the initial generator on the first scaffold.
        self.specialist = Specialist(
            scaffold=self.scaffolds[0], generator=generate_fn, reward=float("-inf")
        )

    # ── evaluation ───────────────────────────────────────────────────────
    def _evaluate(
        self, scaffold: str, generator: GenerateFn
    ) -> tuple[float, VerifierResult, str]:
        """Generate a candidate for ``scaffold`` and verify it, advancing the curve."""
        candidate = generator(scaffold)
        result = self.task.score(candidate)
        self._n_samples += 1
        self._t += 1.0
        self.curve.record(
            t_wall=self._t, n_samples=self._n_samples, reward=result.reward
        )
        if result.passed and result.reward >= self.harvest_min_reward:
            self._harvest.append(
                Harvested(scaffold=scaffold, candidate=candidate, reward=result.reward)
            )
        return result.reward, result, candidate

    # ── arms ─────────────────────────────────────────────────────────────
    def _scaffold_arm(self, generator: GenerateFn) -> tuple[str, float]:
        """Search scaffold variants with ``generator``; return the best (scaffold, reward)."""
        best_scaffold = self.scaffolds[0]
        best_reward = float("-inf")
        for scaffold in self.scaffolds:
            reward, _res, _cand = self._evaluate(scaffold, generator)
            if reward > best_reward:
                best_reward, best_scaffold = reward, scaffold
        return best_scaffold, best_reward

    def _weight_arm(self) -> tuple[str, GenerateFn, str, float] | None:
        """Distill harvested winners into a specialist generator; evaluate it.

        Returns ``(adapter_id, new_generator, best_scaffold, best_reward)`` or
        ``None`` when the weight arm is absent or declines.
        """
        if self.weight_arm is None or not self._harvest:
            return None
        try:
            trained = self.weight_arm(self.task, list(self._harvest))
        except Exception as exc:  # noqa: BLE001 — a failed train is a no-op round, not a crash
            logger.warning("[AHE-3.29] weight arm failed: %s", exc)
            return None
        if trained is None:
            return None
        adapter_id, new_generator = trained
        best_scaffold, best_reward = self._scaffold_arm(new_generator)
        return adapter_id, new_generator, best_scaffold, best_reward

    # ── loop ─────────────────────────────────────────────────────────────
    def run_round(self, index: int) -> RoundResult:
        """One factory cycle: run both arms, attribute by marginal speed, ratchet."""
        before = AdaptationCurve(
            task_id=self.task.task_id, points=list(self.curve.points)
        )

        # Scaffolding arm.
        scaffold_best, scaffold_reward = self._scaffold_arm(self.specialist.generator)
        after_scaffold = AdaptationCurve(
            task_id=self.task.task_id, points=list(self.curve.points)
        )
        scaffold_gain = marginal_speed_gain(
            before, after_scaffold, self.task.target_tau
        )

        challenger = Specialist(
            scaffold=scaffold_best,
            generator=self.specialist.generator,
            reward=scaffold_reward,
        )
        arm = "scaffold"
        weight_gain = 0.0

        # Weight arm (optional) — harvest → distil → re-evaluate.
        trained = self._weight_arm()
        if trained is not None:
            adapter_id, new_generator, w_scaffold, w_reward = trained
            after_weights = AdaptationCurve(
                task_id=self.task.task_id, points=list(self.curve.points)
            )
            weight_gain = marginal_speed_gain(
                after_scaffold, after_weights, self.task.target_tau
            )
            if w_reward > challenger.reward:
                challenger = Specialist(
                    scaffold=w_scaffold,
                    generator=new_generator,
                    reward=w_reward,
                    adapter_id=adapter_id,
                )
                arm = "weights"

        # Promotion ratchet (AHE-3.24 idea on task reward): replace the incumbent
        # only if the challenger does not regress beyond tolerance.
        incumbent_reward = self.specialist.reward
        promoted = challenger.reward >= incumbent_reward - self.tolerance
        if promoted:
            self.specialist = challenger
        else:
            arm = "none"

        return RoundResult(
            round_index=index,
            arm=arm,
            challenger_reward=challenger.reward,
            incumbent_reward=(
                0.0 if incumbent_reward == float("-inf") else incumbent_reward
            ),
            promoted=promoted,
            scaffold_gain=scaffold_gain,
            weight_gain=weight_gain,
        )

    def run(self, rounds: int = 3) -> FactoryResult:
        """Run ``rounds`` factory cycles and return the produced specialist + curve."""
        results: list[RoundResult] = []
        for i in range(rounds):
            results.append(self.run_round(i))
        result = FactoryResult(
            task_id=self.task.task_id,
            specialist=self.specialist,
            curve=self.curve,
            rounds=results,
        )
        result._tau = self.task.target_tau
        logger.info(
            "[AHE-3.29] factory done task=%s reward=%.4f promotions=%d",
            self.task.task_id,
            self.specialist.reward,
            sum(1 for r in results if r.promoted),
        )
        return result
