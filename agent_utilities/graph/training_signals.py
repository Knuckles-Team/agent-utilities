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

2026 reasoning-RL extensions (`.specify/specs/reasoning-rl-2026/`):

* :func:`batch_normalized_advantage` gains ``length_unbiased`` (Dr.GRPO, drop the
  difficulty-biased ``/σ``) and ``mode``/``group_ids`` (GRPO per-prompt grouping vs
  REINFORCE++ global normalization).
* :func:`dynamic_sample` — DAPO zero-variance group dropping.
* :func:`entropy_progress_weights` — EP-GRPO entropy-progress step reweighting.
* :func:`token_regulation` — TR-GRPO token-importance regulation.

These are opt-in reward *primitives*: ``length_unbiased`` and ``mode`` default to the
original GRPO behaviour, and the new helpers ship with consumers (see
:class:`RewardDecomposer` for entropy-progress) rather than as speculative dead code.

"Build once" so that when the Wave-C trainers land, each training paper is "add
the optimizer". These are wired today into :class:`RewardDecomposer` (advantage +
failure-point surface through ``get_distillation_insights``); the dataset-prep
helpers feed the Wave-C dataset builders.

Concept: training-signals
"""

import statistics
from collections import defaultdict
from typing import Any


def _normalize_advantage(
    rewards: list[float], eps: float, length_unbiased: bool
) -> list[float]:
    """Normalize one group of rewards to advantages. See public wrapper."""
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if length_unbiased:
        # Dr.GRPO (arXiv:2503.20783): omit the ``/σ`` term. Dividing by the
        # per-group std injects a question-difficulty bias (easy questions with
        # small σ get inflated advantages) that, coupled with token-length loss
        # normalization, over-rewards short responses. The centered advantage
        # ``(r − μ)`` is the length-/difficulty-unbiased signal.
        return [round(c, 6) for c in centered]
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * n
    return [round(c / std, 6) for c in centered]


def batch_normalized_advantage(
    rewards: list[float],
    *,
    eps: float = 1e-8,
    length_unbiased: bool = False,
    mode: str = "group",
    group_ids: list[Any] | None = None,
) -> list[float]:
    """Group-normalized advantage ``A_i = (r_i − μ) / σ`` (GRPO core).

    Returns zeros for a degenerate group (≤1 sample or ~zero variance), so a
    collapsed batch produces no spurious gradient signal.

    - ``mode="group"`` with ``group_ids`` normalizes *within* each group id (GRPO's
      per-prompt grouping); without ``group_ids`` the whole list is one group.
    - ``mode="global"`` normalizes across the entire list regardless of ``group_ids``
      (REINFORCE++ global normalization, arXiv:2501.03262).
    - ``length_unbiased=True`` applies the Dr.GRPO correction (arXiv:2503.20783):
      the centered ``(r − μ)`` advantage without the difficulty-biased ``/σ`` term.

    Defaults reproduce the original GRPO behaviour exactly (group, σ-normalized).
    """
    n = len(rewards)
    if n == 0:
        return []
    if mode == "group" and group_ids is not None:
        if len(group_ids) != n:
            raise ValueError("group_ids must be the same length as rewards")
        out = [0.0] * n
        buckets: dict[Any, list[int]] = defaultdict(list)
        for i, g in enumerate(group_ids):
            buckets[g].append(i)
        for idxs in buckets.values():
            adv = _normalize_advantage([rewards[i] for i in idxs], eps, length_unbiased)
            for i, a in zip(idxs, adv, strict=True):
                out[i] = a
        return out
    return _normalize_advantage(rewards, eps, length_unbiased)


def dynamic_sample(
    groups: list[list[float]], *, eps: float = 1e-8
) -> tuple[list[list[float]], int]:
    """DAPO (arXiv:2503.14476) dynamic sampling: drop zero-variance reward groups.

    A group whose rewards are (near-)identical yields a ~zero advantage and thus no
    gradient — keeping it wastes a rollout slot. Returns ``(kept_groups,
    dropped_count)``. Pairs with EP-GRPO's zero-variance-collapse target. Callers
    should ``log`` the dropped count (no silent truncation).
    """
    kept: list[list[float]] = []
    dropped = 0
    for g in groups:
        if len(g) >= 2 and statistics.pstdev(g) >= eps:
            kept.append(g)
        else:
            dropped += 1
    return kept, dropped


def entropy_progress_weights(
    entropies: list[float], *, floor: float = 0.0
) -> list[float]:
    """EP-GRPO (arXiv:2605.04960): per-step weights from entropy *progress*.

    A reasoning step that *reduces* entropy relative to the previous step is making
    progress toward a solution and earns a higher weight; non-progress steps decay to
    ``floor``. Weights are normalized so the largest progress step is ``1.0``. When no
    step reduces entropy (no clear progress signal) the weights are uniform ``1.0`` so
    the caller falls back to unweighted credit. Targets GRPO's wrong-polarity and
    zero-variance-collapse credit-assignment failures.
    """
    n = len(entropies)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    deltas = [0.0] + [entropies[i - 1] - entropies[i] for i in range(1, n)]
    max_d = max(deltas)
    if max_d <= 0:
        return [1.0] * n
    return [
        round(max(floor, d / max_d), 6) if d > 0 else round(floor, 6) for d in deltas
    ]


def token_regulation(
    token_rewards: list[float], importances: list[float], *, normalize: bool = True
) -> list[float]:
    """TR-GRPO (arXiv:2511.00066): regulate per-token rewards by token importance.

    Scales each token's reward by its estimated contribution to the final outcome, so
    noisy/unhelpful tokens contribute less while important reasoning/action tokens keep
    their signal. With ``normalize`` the importances are scaled to a max magnitude of 1.
    """
    if len(token_rewards) != len(importances):
        raise ValueError("token_rewards and importances must be the same length")
    if normalize and importances:
        m = max((abs(x) for x in importances), default=0.0) or 1.0
        importances = [x / m for x in importances]
    return [round(r * w, 6) for r, w in zip(token_rewards, importances, strict=True)]


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
