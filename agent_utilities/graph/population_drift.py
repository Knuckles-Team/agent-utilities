#!/usr/bin/python
from __future__ import annotations

"""Population-drift monitor for evolutionary / swarm loops.

CONCEPT:AU-AHE.harness.evolutionary-aggregation — Evolutionary Aggregation Engine (diversity-collapse detection)

A distributional collapse detector distilled from the MASS social-swarm research
(`.specify/specs/research-evolution-20260606/` plan b2-01): instead of a single
scalar diversity heuristic, track the *score distribution* of a population across
generations using the 1-D Wasserstein-1 distance. Two signals:

* **spread** — within-population dispersion (population stdev). Collapse = spread
  shrinking toward 0 (everyone converging to the same score), which is the central
  failure mode of verifier-free evolution (b6-67) and mean-reward tournaments
  (b4-03).
* **drift** — W1 between consecutive generations' score distributions. Healthy
  exploration keeps drift > 0; drift → 0 with low spread confirms collapse.

Pure Python (``bisect``/``statistics``) — no numpy, no model, no network.

Concept: population-drift
"""

import statistics
from bisect import bisect_right
from dataclasses import dataclass, field


def wasserstein1(a: list[float], b: list[float]) -> float:
    """1-D Wasserstein-1 (earth-mover) distance between two empirical samples.

    Computed as the integral of ``|F_a(x) − F_b(x)|`` over the support — the
    standard closed form for the 1-D case. Returns 0.0 if either sample is empty.
    """
    if not a or not b:
        return 0.0
    a_s, b_s = sorted(a), sorted(b)
    na, nb = len(a_s), len(b_s)
    points = sorted(set(a_s) | set(b_s))
    total = 0.0
    for i in range(len(points) - 1):
        x, nxt = points[i], points[i + 1]
        fa = bisect_right(a_s, x) / na
        fb = bisect_right(b_s, x) / nb
        total += abs(fa - fb) * (nxt - x)
    return total


def population_spread(scores: list[float]) -> float:
    """Within-population dispersion (population standard deviation)."""
    if len(scores) < 2:
        return 0.0
    return statistics.pstdev(scores)


@dataclass
class DriftReading:
    """One drift-monitor reading."""

    spread: float
    drift: float | None  # W1 vs previous generation (None on the first reading)
    collapsed: bool
    low_streak: int
    population: int


@dataclass
class PopulationDriftMonitor:
    """Stateful collapse detector over successive population score distributions.

    Args:
        collapse_threshold: Spread at/below which a generation counts as "low".
        patience: Consecutive low-spread generations before declaring collapse.
    """

    collapse_threshold: float = 0.05
    patience: int = 2
    _prev: list[float] = field(default_factory=list)
    _low_streak: int = 0

    def update(self, scores: list[float]) -> DriftReading:
        """Record a generation's scores and return the drift reading."""
        spread = population_spread(scores)
        drift = wasserstein1(scores, self._prev) if self._prev else None

        if spread <= self.collapse_threshold:
            self._low_streak += 1
        else:
            self._low_streak = 0
        collapsed = self._low_streak >= self.patience

        self._prev = list(scores)
        return DriftReading(
            spread=round(spread, 6),
            drift=None if drift is None else round(drift, 6),
            collapsed=collapsed,
            low_streak=self._low_streak,
            population=len(scores),
        )

    def reset(self) -> None:
        """Reset for a new evolutionary loop."""
        self._prev = []
        self._low_streak = 0
