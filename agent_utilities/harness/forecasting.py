#!/usr/bin/python
from __future__ import annotations

"""Predict-before-run forecasting register + calibration scoreboard.

CONCEPT:AHE-3.34 — operationalizes the research-craft habit of *forecasting every
experiment's result before running it, then scoring calibration* — turning taste
into a trainable, measured loop.

Source: the "Be good at research" craft — predict-before-run + calibration
discipline (the Hamming/Karpathy forecasting habit). Before an experiment is run,
the researcher writes down a prediction *and* a confidence; after the run, the
forecast is resolved against the observed result. Over many forecasts this yields
proper scores (Brier), a hit-rate, and a calibration curve that diagnoses
over/under-confidence — so the act of "staring at the outputs" of the worst
divergences (the surprises) becomes a structured, repeatable feedback loop rather
than an unmeasured intuition.

Where ``reliability_scorers.BrierSkillScorer`` (AHE-3.1) scores a *single*
probabilistic forecast against one realised outcome, this module is the
*longitudinal register*: it accumulates many predict-before-run forecasts across
experiments and produces the calibration scoreboard over the whole set.

``predicted`` and ``actual`` are treated as probabilities / normalized metrics in
``[0, 1]`` for the proper-score and calibration accessors; the threshold accessors
(:meth:`ForecastBoard.hit_rate`, :meth:`ForecastBoard.surprises`) work on the raw
absolute deviation, so point metrics outside ``[0, 1]`` still get a sensible
hit/surprise verdict. All accessors are deterministic and derive everything from
the accumulated forecasts.
"""

from dataclasses import dataclass
from typing import Any

__all__ = ["Forecast", "ForecastBoard"]


def _clamp01(value: float) -> float:
    """Clamp a float into the closed unit interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))


@dataclass
class Forecast:
    """A single predict-before-run record and its later resolution.

    A forecast is created (with :attr:`resolved` ``False``) the moment a
    prediction is committed — *before* the experiment runs. It is resolved later
    by filling in :attr:`actual` and flipping :attr:`resolved`, so an unresolved
    forecast can never be retroactively edited to match its result.
    """

    experiment_id: str
    hypothesis: str
    predicted: float  # predicted metric/probability in [0,1] (or any float for point metrics)
    confidence: float  # subjective confidence in [0,1]
    actual: float | None = None
    resolved: bool = False

    def deviation(self) -> float | None:
        """Absolute distance ``|predicted - actual|`` once resolved, else ``None``."""
        if not self.resolved or self.actual is None:
            return None
        return abs(self.predicted - self.actual)


class ForecastBoard:
    """Predict-before-run register + calibration scoreboard (CONCEPT:AHE-3.34).

    Forecasts are keyed by ``experiment_id``; predicting an id that already
    exists overwrites its prior (still-open) forecast, which models revising a
    prediction before the run. Resolution is order-independent and the scoreboard
    accessors only ever consider resolved forecasts.
    """

    def __init__(self) -> None:
        # Insertion-ordered so calibration/surprise output is deterministic.
        self._forecasts: dict[str, Forecast] = {}

    # -- recording -------------------------------------------------------------

    def predict(
        self,
        experiment_id: str,
        hypothesis: str,
        predicted: float,
        confidence: float = 0.5,
    ) -> Forecast:
        """Record a forecast *before* the run and return it.

        ``confidence`` is clamped to ``[0, 1]``; ``predicted`` is stored as given
        (point metrics may legitimately fall outside the unit interval). Re-using
        an ``experiment_id`` replaces the earlier open forecast for that id.
        """
        forecast = Forecast(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            predicted=float(predicted),
            confidence=_clamp01(confidence),
        )
        self._forecasts[experiment_id] = forecast
        return forecast

    def resolve(self, experiment_id: str, actual: float) -> Forecast | None:
        """Fill in the observed result and mark the forecast resolved.

        Returns the updated :class:`Forecast`, or ``None`` if no forecast was
        ever registered under ``experiment_id``.
        """
        forecast = self._forecasts.get(experiment_id)
        if forecast is None:
            return None
        forecast.actual = float(actual)
        forecast.resolved = True
        return forecast

    # -- views -----------------------------------------------------------------

    @property
    def forecasts(self) -> list[Forecast]:
        """All forecasts in insertion order (resolved and open)."""
        return list(self._forecasts.values())

    def _resolved(self) -> list[Forecast]:
        return [
            f for f in self._forecasts.values() if f.resolved and f.actual is not None
        ]

    # -- proper score ----------------------------------------------------------

    def brier_score(self) -> float | None:
        """Mean squared error ``(predicted - actual)^2`` over resolved forecasts.

        Treats ``predicted``/``actual`` as probabilities (clamped to ``[0, 1]``),
        so this is a proper score in ``[0, 1]`` where lower is better. ``None``
        when nothing is resolved.
        """
        resolved = self._resolved()
        if not resolved:
            return None
        total = 0.0
        for f in resolved:
            p = _clamp01(f.predicted)
            a = _clamp01(f.actual if f.actual is not None else 0.0)
            total += (p - a) ** 2
        return total / len(resolved)

    # -- threshold accuracy ----------------------------------------------------

    def hit_rate(self, tolerance: float = 0.1) -> float | None:
        """Fraction of resolved forecasts with ``|predicted - actual| <= tolerance``.

        Operates on the raw (un-clamped) deviation so point metrics outside
        ``[0, 1]`` still get a fair hit verdict. ``None`` when nothing is
        resolved.
        """
        resolved = self._resolved()
        if not resolved:
            return None
        hits = sum(
            1
            for f in resolved
            if abs(f.predicted - float(f.actual or 0.0)) <= tolerance
        )
        return hits / len(resolved)

    # -- calibration -----------------------------------------------------------

    def calibration_curve(
        self, bins: int = 5, tolerance: float = 0.1
    ) -> list[tuple[float, float, int]]:
        """Per-confidence-bin calibration table for overconfidence diagnosis.

        Buckets resolved forecasts by stated ``confidence`` into ``bins`` equal
        slices of ``[0, 1]``, then within each non-empty bin reports
        ``(mean_confidence, empirical_hit_rate, count)``. A bin whose mean
        confidence far exceeds its empirical hit-rate exposes overconfidence (and
        the inverse exposes under-confidence). The hit-rate uses the same
        ``tolerance`` semantics as :meth:`hit_rate`.

        Returns one tuple per non-empty bin, ordered from lowest to highest
        confidence band.
        """
        if bins < 1:
            raise ValueError(f"bins must be >= 1; got {bins}")
        resolved = self._resolved()
        if not resolved:
            return []

        buckets: list[list[Forecast]] = [[] for _ in range(bins)]
        for f in resolved:
            c = _clamp01(f.confidence)
            # Map confidence to a bin index; the top edge (1.0) lands in the last bin.
            idx = min(bins - 1, int(c * bins))
            buckets[idx].append(f)

        curve: list[tuple[float, float, int]] = []
        for bucket in buckets:
            if not bucket:
                continue
            mean_conf = sum(_clamp01(f.confidence) for f in bucket) / len(bucket)
            hits = sum(
                1
                for f in bucket
                if abs(f.predicted - float(f.actual or 0.0)) <= tolerance
            )
            curve.append((mean_conf, hits / len(bucket), len(bucket)))
        return curve

    # -- diagnostics -----------------------------------------------------------

    def surprises(self, tolerance: float = 0.25) -> list[Forecast]:
        """Resolved forecasts whose result diverged most from the prediction.

        Returns the resolved forecasts with ``|predicted - actual| > tolerance``,
        worst (largest deviation) first — the outputs worth staring at. Ties keep
        insertion order for determinism.
        """
        diverged = [
            f
            for f in self._resolved()
            if abs(f.predicted - float(f.actual or 0.0)) > tolerance
        ]
        # ``sorted`` is stable, so equal-deviation forecasts keep insertion order.
        return sorted(
            diverged,
            key=lambda f: abs(f.predicted - float(f.actual or 0.0)),
            reverse=True,
        )

    # -- summary ---------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compact, serializable scoreboard over all recorded forecasts."""
        resolved = self._resolved()
        mean_conf = (
            sum(_clamp01(f.confidence) for f in self._forecasts.values())
            / len(self._forecasts)
            if self._forecasts
            else None
        )
        brier = self.brier_score()
        hit_rate = self.hit_rate()
        return {
            "total": len(self._forecasts),
            "resolved": len(resolved),
            "brier": None if brier is None else round(brier, 6),
            "hit_rate": None if hit_rate is None else round(hit_rate, 6),
            "mean_confidence": None if mean_conf is None else round(mean_conf, 6),
            "n_surprises": len(self.surprises()),
        }
