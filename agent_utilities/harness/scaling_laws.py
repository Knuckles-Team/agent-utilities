#!/usr/bin/python
from __future__ import annotations

"""Multi-agent scaling-law measurement.

CONCEPT:AU-OS.scaling.multi-agent-scaling-law — a multi-agent scaling-law harness that sweeps collective size over a fixed task and fits capability ~ N^alpha so the platform can measure whether adding agents helps super- or sub-linearly instead of assuming it does

The paper (§5.4/§7.5) asks the pivotal open question: does collective capability emerge
*linearly or superlinearly* with the size, density and speed of organised collaboration —
the **Multi-Agent Scaling Laws** — and does a homogeneous LLM collective yield synergy at
all? AU runs real collectives (ORCH-1.8 ParallelEngine, ORCH-1.32 social system) and
detects collapse (AHE-3.2), but only produced per-run *health* snapshots, never
*capability-per-N*. This harness closes that: hold a task fixed, sweep collective size,
and fit a power law over (N, collective_quality) to recover the scaling exponent and a
regime verdict (superlinear / linear / sublinear / flat). It is backend-agnostic — given
any ``collective_eval(n) -> score`` it fits the law — so the production hook wraps the
ParallelEngine over a fixed suite while tests use a synthetic collective.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScalingLaw:
    """A fitted ``capability ~ N^alpha`` law over a collective-size sweep."""

    alpha: float  # scaling exponent
    intercept: float  # log-space intercept
    r_squared: float
    regime: str  # superlinear | linear | sublinear | flat
    points: list[tuple[int, float]] = field(default_factory=list)

    def predict(self, n: int) -> float:
        """Predicted collective quality at size ``n`` from the fitted law."""
        return math.exp(self.intercept) * (max(1, n) ** self.alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": round(self.alpha, 4),
            "r_squared": round(self.r_squared, 4),
            "regime": self.regime,
            "points": [[n, round(s, 4)] for n, s in self.points],
        }


def _regime(alpha: float) -> str:
    if alpha >= 1.15:
        return "superlinear"
    if alpha >= 0.85:
        return "linear"
    if alpha > 0.15:
        return "sublinear"
    return "flat"


def fit_scaling_law(points: list[tuple[int, float]]) -> ScalingLaw | None:
    """Least-squares fit of ``log(quality) = alpha·log(N) + c`` (≥2 distinct N)."""
    pts = [(int(n), float(q)) for n, q in points if n > 0 and q > 0]
    if len({n for n, _ in pts}) < 2:
        return None
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(q) for _, q in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    if sxx == 0:
        return None
    alpha = sxy / sxx
    intercept = my - alpha * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum(
        (y - (alpha * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=False)
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return ScalingLaw(alpha, intercept, r2, _regime(alpha), pts)


class MultiAgentScalingHarness:
    """Sweep a collective's size over a fixed task and fit its scaling law."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine

    def measure(
        self, collective_eval: Callable[[int], float], sizes: list[int]
    ) -> ScalingLaw | None:
        """Run ``collective_eval(n)`` for each ``n`` and fit the power law.

        ``collective_eval`` runs the collective of ``n`` agents on the *fixed* task
        suite and returns a collective-quality score in any positive unit. Returns
        ``None`` when fewer than two distinct sizes produced a positive score.
        """
        points: list[tuple[int, float]] = []
        for n in sorted({int(s) for s in sizes if int(s) > 0}):
            try:
                score = float(collective_eval(n))
            except Exception as exc:  # noqa: BLE001 — a failed size is simply skipped
                logger.warning("[SAFE-1.2] collective_eval(%s) failed: %s", n, exc)
                continue
            if score > 0:
                points.append((n, score))
        law = fit_scaling_law(points)
        if law is not None:
            self._persist(law)
        return law

    def _persist(self, law: ScalingLaw) -> None:
        if self.engine is None:
            return
        import json
        import time
        import uuid

        try:
            self.engine.add_node(
                f"scaling_law:{uuid.uuid4().hex[:12]}",
                "ScalingLawMeasurement",
                properties={
                    "alpha": law.alpha,
                    "regime": law.regime,
                    "r_squared": law.r_squared,
                    "metrics_json": json.dumps(law.to_dict()),
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("[SAFE-1.2] could not persist scaling law: %s", exc)
