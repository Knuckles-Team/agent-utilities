#!/usr/bin/python
from __future__ import annotations

"""Superhuman-certification gate for SAI specialists (CONCEPT:SAFE-1.6).

The SAI paper's promotion criterion is *exceed humans at the task* — but its own
remark is that comparing to a human point-estimate "will not produce useful signal
to quantitatively distinguish superhuman AIs". AU's :class:`CapabilityRatchet`
(AHE-3.24) only ratchets against a *self* baseline; nothing certified beating a
*human*. This gate closes that: a specialist is certified SUPERHUMAN only when the
**bootstrap lower confidence bound** of its verified-reward distribution exceeds a
recorded ``human_baseline`` by a margin — a statistical claim, not a lucky single
run.

It composes the existing frontier signals rather than reinventing them:
``saturation_detector`` (SAFE-1.1) flags a task whose human-relative signal has
collapsed to the ceiling (certification there is meaningless — escalate to a
frontier/relative scorer), and ``population_spread`` (SAFE-1.4) reports the
reward-distribution spread so a degenerate (collapsed) sample is visible. The SAI
factory consults this gate before labelling a specialist *certified*, and the
adaptation benchmark (SAFE-1.7) reports its verdict per task.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CertificationResult:
    """Outcome of a superhuman-certification check."""

    certified: bool
    mean_reward: float
    ci_lower: float
    ci_upper: float
    human_baseline: float | None
    margin: float
    reward_spread: float
    saturated: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "mean_reward": round(self.mean_reward, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "human_baseline": self.human_baseline,
            "margin": self.margin,
            "reward_spread": round(self.reward_spread, 6),
            "saturated": self.saturated,
            "reason": self.reason,
        }


class SuperhumanCertifier:
    """Certify a specialist superhuman via a bootstrap-CI human-baseline comparison."""

    def __init__(
        self,
        *,
        confidence: float = 0.95,
        margin: float = 0.0,
        n_boot: int = 1000,
        seed: int = 0,
        saturation_ceiling: float = 0.98,
    ) -> None:
        self.confidence = float(confidence)
        self.margin = float(margin)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.saturation_ceiling = float(saturation_ceiling)

    def _bootstrap_ci(self, rewards: list[float]) -> tuple[float, float, float]:
        """Deterministic bootstrap CI of the mean reward (seeded; reproducible)."""
        from agent_utilities.numeric import xp as np

        arr = np.asarray(rewards, dtype=np.float64)
        mean = float(arr.mean())
        if arr.size == 1:
            return mean, mean, mean
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, arr.size, size=(self.n_boot, arr.size))
        boot_means = arr[idx].mean(axis=1)
        alpha = 1.0 - self.confidence
        lo = float(np.quantile(boot_means, alpha / 2.0))
        hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
        return mean, lo, hi

    def certify(
        self,
        rewards: list[float],
        human_baseline: float | None,
        *,
        pass_rate_history: list[float] | None = None,
    ) -> CertificationResult:
        """Certify SUPERHUMAN iff the reward CI-lower-bound clears the human baseline."""
        from agent_utilities.graph.population_drift import population_spread
        from agent_utilities.harness.frontier_scorers import saturation_detector

        if not rewards:
            return CertificationResult(
                certified=False,
                mean_reward=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                human_baseline=human_baseline,
                margin=self.margin,
                reward_spread=0.0,
                saturated=False,
                reason="no reward samples",
            )

        mean, lo, hi = self._bootstrap_ci(rewards)
        spread = population_spread(list(rewards))
        sat = saturation_detector(
            pass_rate_history or [], ceiling=self.saturation_ceiling
        )
        saturated = bool(sat.get("saturated"))

        if human_baseline is None:
            reason = "no human baseline recorded — superhuman is unprovable (self-improvement only)"
            certified = False
        elif saturated:
            reason = "task saturated at ceiling — human comparison uninformative; escalate to a frontier scorer"
            certified = False
        elif lo > human_baseline + self.margin:
            reason = f"CI lower bound {lo:.4f} > human {human_baseline:.4f} + margin {self.margin}"
            certified = True
        else:
            reason = f"CI lower bound {lo:.4f} does not clear human {human_baseline:.4f} + margin {self.margin}"
            certified = False

        return CertificationResult(
            certified=certified,
            mean_reward=mean,
            ci_lower=lo,
            ci_upper=hi,
            human_baseline=human_baseline,
            margin=self.margin,
            reward_spread=spread,
            saturated=saturated,
            reason=reason,
            detail={"saturation": sat, "n": len(rewards)},
        )
