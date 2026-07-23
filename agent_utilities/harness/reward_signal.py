#!/usr/bin/python
from __future__ import annotations

"""Unified reward contract (CONCEPT:AU-AHE.reward.unified-reward-signal).

Every reward-producing function in the codebase today independently converges on
the SAME informal contract: a ``float`` clamped to ``[0, 1]``, ``0.5`` treated as
neutral/unknown by at least two of them:

- ``SkillSignalProvider.score()`` -> ``(success, score, fail_reason)``, ``score``
  in ``[0, 1]`` (``skill_evolution.py``).
- ``blend_reward(internal_reward, langfuse_reward, weight)`` — weighted average,
  both operands and the result clamped ``[0, 1]`` (``langfuse_signal.py``).
- ``graded_score(expected, actual)`` -> ``[0, 1]`` similarity
  (``program_optimization.py``).
- ``eg-program``'s ``TrainingExample.score: f64``, validated ``(0.0..=1.0)`` at
  the Rust layer (``eg-program/src/contracts.rs``).
- ``CapabilityIndex.record_outcome`` -> EMA in ``[0, 1]``, ``0.5`` = neutral prior
  (``capability_index.py``).
- ``ClaimFlywheel.record_outcome`` blends an internal reward with
  ``langfuse_signal.fetch_reward`` (also ``[0, 1]``).

This module NAMES that already-universal convention — it does not change any
existing numeric behavior. :class:`RewardSignal` adds the one thing the bare
floats above lose: provenance (which source produced this number, and how much
to trust it), so a promotion decision can show its work. :func:`blend`
generalizes the existing 2-operand, weight-only ``langfuse_signal.blend_reward``
to N operands with confidence — ``blend_reward(a, b, weight=w)`` is exactly
``blend(RewardSignal(a, "internal"), RewardSignal(b, "langfuse_score"),
weights=[1 - w, w]).value`` for a non-``None`` ``b``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["RewardSignal", "RewardSource", "blend"]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """One provenance-carrying reward observation, ``value`` in ``[0, 1]``.

    ``0.5`` is the conventional neutral/unknown prior (matching
    ``CapabilityIndex``'s EMA seed and ``blend_reward``'s degrade-to-internal
    default). ``source`` names WHICH producer computed ``value`` — one of
    ``"langfuse_score"`` | ``"internal_corpus"`` | ``"capability_ema"`` |
    ``"claim_flywheel"`` | ``"eval_corpus"`` | ``"failure_trace"`` in the shipped
    producers, but any short adapter-chosen string is valid (this is a naming
    convention, not a closed enum — a new signal source names itself).
    ``confidence`` is how much to trust ``value`` (``0`` = ignore entirely) and
    drives :func:`blend`'s effective weighting; it does not change ``value``
    itself. ``provenance_ref`` is an opaque reference (trace id, eval-case id,
    ``RunTrace`` id, ...) a reviewer can follow back to the underlying evidence.
    ``reason`` mirrors ``SkillTrajectory.fail_reason`` — a short human-readable
    pass/fail explanation.
    """

    value: float
    source: str
    confidence: float = 1.0
    provenance_ref: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        # frozen dataclass -> bypass __setattr__ via object.__setattr__, mirroring
        # every existing producer's own clamp-at-the-boundary discipline (never a
        # silent out-of-range reward reaching a promotion decision).
        object.__setattr__(self, "value", _clamp01(self.value))
        object.__setattr__(self, "confidence", _clamp01(self.confidence))


@runtime_checkable
class RewardSource(Protocol):
    """One reward producer, generalizing ``SkillSignalProvider.score``'s shape.

    ``LangfuseSignalProvider.score()`` already computes exactly this — the
    refactor to implement this protocol is mechanical: wrap the existing
    ``(success, score, fail_reason)`` return in ``RewardSignal(value=score,
    source="langfuse_score", provenance_ref=trace_id, reason=fail_reason)``. No
    existing producer's math changes; this is a naming + provenance exercise.
    """

    def reward_for(self, artifact_ref: str, example_ref: str) -> RewardSignal: ...


def blend(
    *signals: RewardSignal, weights: Sequence[float] | None = None
) -> RewardSignal:
    """Confidence- and weight-aware blend of N :class:`RewardSignal`\\ s into one.

    Generalizes ``langfuse_signal.blend_reward`` (2-operand, weight-only) to N
    operands with confidence: each signal's effective weight is its explicit
    ``weights[i]`` (defaulting to an even split) times its own ``confidence`` —
    a ``confidence=0`` signal contributes nothing, matching ``blend_reward``'s
    "no Langfuse signal yet -> return the internal reward unchanged" behavior in
    the 2-operand, ``confidence=(1.0, 0.0)`` case. Zero signals or all-zero
    effective weight returns the neutral ``0.5`` prior (never raises, never
    divides by zero). ``source``/``provenance_ref``/``reason`` on the returned
    signal name the blend itself, not any one input.
    """
    if not signals:
        return RewardSignal(value=0.5, source="blend:empty")
    if weights is None:
        weights = [1.0 / len(signals)] * len(signals)
    if len(weights) != len(signals):
        raise ValueError("blend: weights must be the same length as signals")

    effective = [
        max(0.0, w) * s.confidence for s, w in zip(signals, weights, strict=True)
    ]
    total = sum(effective)
    if total <= 0.0:
        # No signal is trusted enough to contribute — degrade to the neutral
        # prior rather than fabricating a confident number from nothing.
        return RewardSignal(value=0.5, source="blend:no_confidence")

    blended_value = (
        sum(s.value * e for s, e in zip(signals, effective, strict=True)) / total
    )
    sources = "+".join(dict.fromkeys(s.source for s in signals))
    refs = ",".join(s.provenance_ref for s in signals if s.provenance_ref)
    return RewardSignal(
        value=blended_value,
        source=f"blend:{sources}",
        confidence=min(1.0, total),
        provenance_ref=refs,
    )
