#!/usr/bin/python
from __future__ import annotations

"""Adaptation-speed metric — the SAI primary measure of specialization.

CONCEPT:AU-AHE.harness.per-task-adaptation-speed — per-task adaptation-speed instrumentation (time-to-target +
sample-complexity + learning-AUC) that measures *how fast* a specialization run
reaches a target capability, rather than only its final quality.

The SAI thesis ("AI Must Embrace Specialization via Superhuman Adaptable
Intelligence", arXiv:2602.23643) names the primary metric of an adaptable agent
as *"the speed and efficiency with which new skills are acquired under realistic
resource constraints"* — explicitly **not** a fixed-competency checklist. AU
already measures terminal quality everywhere (``reliability_scorers``) and
cross-cycle repo cadence (``ImprovementVelocity``/SAFE-1.3), but **nothing
measures per-task time-to-target / sample-complexity** — the learning *curve* of
a single specialization. This module is that missing measurement: the object the
SAI factory controller (AHE-3.29) optimizes and the certifier (SAFE-1.1) gates.

A run appends ``(t_wall, n_samples, verified_reward)`` points as the specialist
improves (each point = one evaluated iteration / training checkpoint); the curve
then yields:

* :meth:`time_to_target`  — wall-seconds to first reach a reward target ``tau``
* :meth:`sample_complexity` — examples/rollouts consumed to first reach ``tau``
* :meth:`learning_auc`     — normalized area under the best-so-far reward vs
  samples (higher ⇒ faster riser), a single comparable scalar.

Rewards are transformed to **best-so-far** before any threshold test, so a noisy
verifier (a kernel that regresses on one candidate) never makes an already-met
target "un-meet".
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CurvePoint:
    """One evaluated iteration of a specialization run."""

    t_wall: float  # seconds since run start
    n_samples: int  # cumulative examples/rollouts consumed to reach this point
    reward: float  # verified reward at this point (the raw, un-smoothed score)


@dataclass
class AdaptationCurve:
    """The reward-vs-(time, samples) trajectory of a single specialization run.

    Append points in the order they are produced; the metric accessors derive
    everything from the accumulated points and are safe to call at any time
    (they return ``None`` for an unreached target rather than raising).
    """

    task_id: str = ""
    points: list[CurvePoint] = field(default_factory=list)

    def record(self, t_wall: float, n_samples: int, reward: float) -> CurvePoint:
        """Append one evaluated iteration and return the stored point.

        Points must be appended in non-decreasing ``(t_wall, n_samples)`` order
        (the natural order of a run); this is asserted defensively because an
        out-of-order append would silently corrupt time-to-target.
        """
        if self.points:
            last = self.points[-1]
            if t_wall < last.t_wall or n_samples < last.n_samples:
                raise ValueError(
                    "adaptation curve points must be non-decreasing in "
                    f"(t_wall, n_samples); got ({t_wall}, {n_samples}) after "
                    f"({last.t_wall}, {last.n_samples})"
                )
        point = CurvePoint(
            t_wall=float(t_wall), n_samples=int(n_samples), reward=float(reward)
        )
        self.points.append(point)
        return point

    # -- best-so-far transform -------------------------------------------------

    def _best_so_far(self) -> list[tuple[float, int, float]]:
        """Points with reward replaced by the running maximum reward."""
        out: list[tuple[float, int, float]] = []
        best = float("-inf")
        for p in self.points:
            best = max(best, p.reward)
            out.append((p.t_wall, p.n_samples, best))
        return out

    # -- SAI metrics -----------------------------------------------------------

    def reached(self, tau: float) -> bool:
        """Whether the run ever attained a verified reward ≥ ``tau``."""
        return any(p.reward >= tau for p in self.points)

    def time_to_target(self, tau: float) -> float | None:
        """Wall-seconds to the first point whose best-so-far reward ≥ ``tau``.

        ``None`` if the target was never reached.
        """
        for t_wall, _n, best in self._best_so_far():
            if best >= tau:
                return t_wall
        return None

    def sample_complexity(self, tau: float) -> int | None:
        """Cumulative samples consumed to first reach best-so-far reward ≥ ``tau``.

        ``None`` if the target was never reached.
        """
        for _t, n_samples, best in self._best_so_far():
            if best >= tau:
                return n_samples
        return None

    def peak_reward(self) -> float:
        return max((p.reward for p in self.points), default=0.0)

    def final_reward(self) -> float:
        return self.points[-1].reward if self.points else 0.0

    def learning_auc(self) -> float:
        """Normalized area under the best-so-far reward vs samples curve.

        Trapezoidal integral of best-so-far reward over the sample axis, divided
        by the sample span, giving a span-independent scalar in the reward's
        units: a run that reaches high reward with few samples scores higher than
        one that reaches the same reward slowly. Returns the single observed
        reward when only one point exists, and ``0.0`` when empty.
        """
        bsf = self._best_so_far()
        if not bsf:
            return 0.0
        if len(bsf) == 1:
            return bsf[0][2]
        first_n = bsf[0][1]
        last_n = bsf[-1][1]
        span = last_n - first_n
        if span <= 0:
            # No sample progression — fall back to the mean best-so-far reward.
            return sum(b[2] for b in bsf) / len(bsf)
        area = 0.0
        for (n0, r0), (n1, r1) in zip(
            [(b[1], b[2]) for b in bsf[:-1]],
            [(b[1], b[2]) for b in bsf[1:]],
            strict=False,
        ):
            area += (r0 + r1) / 2.0 * (n1 - n0)
        return area / span

    def metrics(self, tau: float) -> dict[str, Any]:
        """A compact, serializable summary at a given target ``tau``."""
        return {
            "task_id": self.task_id,
            "tau": tau,
            "iterations": len(self.points),
            "reached": self.reached(tau),
            "time_to_target_s": self.time_to_target(tau),
            "sample_complexity": self.sample_complexity(tau),
            "learning_auc": round(self.learning_auc(), 6),
            "peak_reward": round(self.peak_reward(), 6),
            "final_reward": round(self.final_reward(), 6),
        }


def marginal_speed_gain(
    before: AdaptationCurve, after: AdaptationCurve, tau: float
) -> float:
    """Improvement in learning-AUC from ``before`` → ``after`` at target ``tau``.

    The SAI factory controller (AHE-3.29) uses this to pick, per round, the arm
    (scaffolding vs. weight-training) that bought the larger adaptation-speed gain.
    ``tau`` is accepted for call-site symmetry with the threshold metrics; AUC is
    target-independent, so it is reserved for future target-relative scoring.
    """
    del tau  # AUC is target-independent today; kept for a stable controller API.
    return after.learning_auc() - before.learning_auc()
