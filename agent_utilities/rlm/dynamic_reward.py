#!/usr/bin/python
from __future__ import annotations

"""Dynamic multi-reward weighting (DW-GRPO) for the GEPA optimizer.

CONCEPT:ORCH-1.30 — Selection on Unseen Data (anti-seesaw reward weighting)

Distilled from Deep GraphRAG's DW-GRPO (`.specify/specs/research-evolution-20260606/`
plan b2-04): when an evolutionary optimizer maximises several rewards at once it
tends to ride the *easiest* one and let the others stagnate (the "seesaw"). This
weighter tracks each objective's recent improvement **slope** across generations
and shifts weight toward the *lagging* objectives, so optimization pressure keeps
moving where progress has stalled.

Pure Python, deterministic — the training-free core of DW-GRPO (the GRPO-trained
integrator is the GPU-gated Wave-C/D part; the slope-weighting itself needs no
training). Consumed by :class:`~agent_utilities.rlm.gepa.ParetoCandidatePool`.

Concept: dynamic-reward
"""

import math


class DynamicRewardWeighter:
    """Slope-tracking reward weighter (up-weights lagging objectives).

    Args:
        objectives: Objective names (e.g. ``["accuracy", "efficiency"]``).
        window: Number of recent generations the slope is measured over.
        temperature: Softmax temperature for the weight distribution.
    """

    def __init__(
        self, objectives: list[str], *, window: int = 4, temperature: float = 1.0
    ) -> None:
        self.objectives = list(objectives)
        self.window = max(2, window)
        self.temperature = max(1e-6, temperature)
        self._history: list[dict[str, float]] = []

    def observe(self, best_by_objective: dict[str, float]) -> None:
        """Record one generation's best score per objective."""
        self._history.append(
            {obj: float(best_by_objective.get(obj, 0.0)) for obj in self.objectives}
        )

    @property
    def ready(self) -> bool:
        """True once there is enough history to differentiate slopes."""
        return len(self._history) >= 2

    def slopes(self) -> dict[str, float]:
        """Per-objective improvement over the recent window (last − first)."""
        if not self.ready:
            return {obj: 0.0 for obj in self.objectives}
        window = self._history[-self.window :]
        first, last = window[0], window[-1]
        span = max(1, len(window) - 1)
        return {
            obj: (last.get(obj, 0.0) - first.get(obj, 0.0)) / span
            for obj in self.objectives
        }

    def weights(self) -> dict[str, float]:
        """Normalised weights that up-weight low-slope (lagging) objectives.

        Uniform until there is slope history. Otherwise ``softmax(−slope/scale)``
        so faster-improving objectives get *less* weight and stalled ones get more.
        """
        n = len(self.objectives)
        if n == 0:
            return {}
        if not self.ready:
            return {obj: 1.0 / n for obj in self.objectives}

        slopes = self.slopes()
        scale = max(abs(s) for s in slopes.values()) + 1e-9
        logits = {
            obj: -slopes[obj] / (scale * self.temperature) for obj in self.objectives
        }
        m = max(logits.values())
        exps = {obj: math.exp(logits[obj] - m) for obj in self.objectives}
        total = sum(exps.values()) or 1.0
        return {obj: exps[obj] / total for obj in self.objectives}

    def scalarize(self, scores: dict[str, float]) -> float:
        """Weighted sum of a candidate's per-objective scores."""
        w = self.weights()
        return sum(
            w.get(obj, 0.0) * float(scores.get(obj, 0.0)) for obj in self.objectives
        )
