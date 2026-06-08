#!/usr/bin/python
from __future__ import annotations

"""Deterministic reward & dataset-prep signals — the training spine.

CONCEPT:AHE-3.1 — Reward / dataset primitives

Pure, model-free functions that turn execution traces into the reward and
dataset signals every training-gated paper consumes — distilled from the
research plans (`.specify/specs/research-evolution-20260606/`):

* :func:`batch_normalized_advantage` — group-normalized advantage ``(r−μ)/σ``,
  the GRPO/PPO core (b6-04 ATLAS, b7-03 SDAR, b6-01 MedCausalX-II).
* :func:`failure_point` — first-divergence step index for error-attributed
  preference-pair construction (b6-01 MedCausalX).
* :func:`composite_reward` — weighted sum of reward components with *conditional*
  terms (a component gated off contributes 0), e.g. ATLAS's func-reward granted
  only when the token was used AND correct (b6-04, b6-01).
* :func:`difficulty_floor_filter` — drop trajectories below a difficulty floor
  (min tool-calls/steps), the data-quality engine (b3-02 OpenSeeker).

"Build once" so that when the Wave-C trainers land, each training paper is "add
the optimizer". These are wired today into :class:`RewardDecomposer` (advantage +
failure-point surface through ``get_distillation_insights``); the dataset-prep
helpers feed the Wave-C dataset builders.

Concept: training-signals
"""

import statistics
from typing import Any


def batch_normalized_advantage(
    rewards: list[float], *, eps: float = 1e-8
) -> list[float]:
    """Group-normalized advantage ``A_i = (r_i − μ) / σ`` (GRPO core).

    Returns zeros for a degenerate group (≤1 sample or ~zero variance), so a
    collapsed batch produces no spurious gradient signal.
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * n
    return [round((r - mean) / std, 6) for r in rewards]


def failure_point(step_failed: list[bool]) -> int | None:
    """First-divergence index: index of the first failed step, else ``None``.

    ``step_failed[i]`` is True when step ``i`` diverged (e.g. outcome INCORRECT).
    This is the anchor for error-attributed preference pairs (winner = corrected
    trajectory from ``t_fail`` onward).
    """
    for i, failed in enumerate(step_failed):
        if failed:
            return i
    return None


def composite_reward(
    components: dict[str, float],
    weights: dict[str, float],
    *,
    conditions: dict[str, bool] | None = None,
) -> float:
    """Weighted sum of reward components with optional per-component gating.

    A component whose ``conditions[name]`` is ``False`` contributes 0 (e.g. a
    functional-token reward granted only if the token was used AND the answer is
    correct). Components default to "on" when unconditioned.
    """
    conditions = conditions or {}
    total = 0.0
    for name, value in components.items():
        if conditions.get(name, True):
            total += weights.get(name, 0.0) * value
    return round(total, 6)


def difficulty_floor_filter(
    items: list[Any],
    *,
    min_count: int = 1,
    count_key: str = "step_count",
) -> list[Any]:
    """Keep only trajectories at/above a difficulty floor (min steps/tool-calls).

    ``items`` may be dicts (read ``count_key``) or objects (read the attribute).
    The data-quality filter that makes a synthesized SFT corpus hard enough to be
    worth training on.
    """
    out: list[Any] = []
    for it in items:
        count = (
            it.get(count_key, 0) if isinstance(it, dict) else getattr(it, count_key, 0)
        )
        if int(count or 0) >= min_count:
            out.append(it)
    return out
