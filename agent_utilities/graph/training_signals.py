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
from collections.abc import Iterable
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


# ── Cached-rollout reward shaping (CONCEPT:AHE-3.49) ──────────────────────────
# The reward-side half of CacheRL (arXiv:2606.14179): when multi-turn tool-calling
# rollouts are served from a fuzzy cache (see data-science-mcp ``cache_agent_loop``),
# two corrections keep the gradient honest — (1) token-level masking so injected
# tool observations earn no gradient (only the model's own thoughts/actions do),
# and (2) cache-tier-aware reward so a failure caused by a stale/approximate cache
# tier is not charged to the model as if it were a genuine performance failure.
# Pure, model-free; opt-in consumers (the trainers) only.

# Default reliability per cache tier: an EXACT hit or a LIVE execution is fully
# trustworthy (a failure under them is genuinely the model's); FUZZY/SEMANTIC hits
# may have returned an approximate observation, so a failure under them is partly
# the cache's fault and is discounted.
CACHE_TIER_RELIABILITY: dict[str, float] = {
    "exact": 1.0,
    "live": 1.0,
    "fuzzy": 0.85,
    "semantic": 0.7,
}

# Segment sources whose tokens the model is trained to produce.
_TRAINABLE_SOURCES = frozenset({"model", "action"})


def token_cache_mask(
    sources: list[str],
    *,
    trainable: Iterable[str] = _TRAINABLE_SOURCES,
) -> list[float]:
    """Per-token loss mask: ``1.0`` for model-generated tokens, ``0.0`` for injected.

    ``sources`` is the per-token (or per-segment) provenance label produced by the
    hybrid-thinking augmenter — ``"model"`` (reasoning), ``"action"`` (the tool
    call the model emits), or ``"observation"`` (a tool result, cached or live).
    Only ``trainable`` sources contribute to the RL/SFT loss; masking observations
    is what lets cached rollouts preserve trajectory quality (CacheRL token-level
    masking). Length-preserving so it can multiply a per-token loss directly.
    """
    keep = set(trainable)
    return [1.0 if s in keep else 0.0 for s in sources]


def cache_tier_aware_reward(
    outcome_reward: float,
    tiers_used: list[str],
    *,
    tier_reliability: dict[str, float] | None = None,
) -> float:
    """Discount a *negative* reward by the weakest cache tier the trajectory used.

    CacheRL's cache-tier-aware reward: a rollout that relied on a low-reliability
    cache tier (``fuzzy``/``semantic``) and then *failed* should not be penalized as
    harshly as a rollout that failed on real (``live``/``exact``) observations — the
    failure may be the cache's fault. Successes (``reward ≥ 0``) pass through
    unchanged (we never inflate a win for using the cache). The penalty is scaled
    by the minimum tier reliability across ``tiers_used`` (the weakest link).
    """
    if outcome_reward >= 0 or not tiers_used:
        return round(outcome_reward, 6)
    table = tier_reliability or CACHE_TIER_RELIABILITY
    reliability = min(table.get(t, 1.0) for t in tiers_used)
    return round(outcome_reward * reliability, 6)


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


# ── Realized search-difficulty signatures (CONCEPT:AHE-3.30) ──────────────────
# Observable trajectory diagnostics for deep-search tasks, distilling the
# shortcut-aware difficulty framework of FORT-Searcher (arXiv:2606.12087, §2.4,
# eqs 15/16/18): apparent trajectory length alone does NOT prove a task forced
# real search — the answer may have surfaced early or been named from prior
# knowledge. These three signatures separate genuinely search-heavy data from
# shortcut-prone data, and feed task acceptance (a synthesized task whose
# trajectories show low cost / early hit / high prior-binding is too easy and is
# re-synthesized harder). Pure, model-free; opt-in consumers only.
#
# A *trajectory* is ``{"steps": [...], "answer_aliases": [...]}`` where each step
# is a dict; a step counts as a retrieval when ``step["is_retrieval"]`` is truthy
# or ``step["kind"]`` is one of the retrieval kinds. ``observation`` holds the
# tool result text; ``model_text`` holds the model's visible generated text.

_RETRIEVAL_KINDS = frozenset({"search", "retrieval", "tool", "read", "browse"})


def _is_retrieval(step: dict[str, Any]) -> bool:
    if step.get("is_retrieval"):
        return True
    return str(step.get("kind", "")).lower() in _RETRIEVAL_KINDS


def _alias_in(text: Any, aliases: list[str]) -> bool:
    if not text:
        return False
    low = str(text).lower()
    return any(a and a.lower() in low for a in aliases)


def _hit_times(
    steps: list[dict[str, Any]], aliases: list[str]
) -> tuple[int | None, int | None]:
    """1-based positions of the first tool-observation hit and first model-text hit.

    ``T_tool`` is the first step whose tool ``observation`` contains the gold
    answer (or an alias); ``T_model`` is the first step whose ``model_text``
    mentions it. Positions are on the shared step timeline so they are directly
    comparable (FORT eqs 16/18). ``None`` means the answer never surfaced there.
    """
    t_tool: int | None = None
    t_model: int | None = None
    for i, step in enumerate(steps, start=1):
        if (
            t_tool is None
            and _is_retrieval(step)
            and _alias_in(step.get("observation"), aliases)
        ):
            t_tool = i
        if t_model is None and _alias_in(step.get("model_text"), aliases):
            t_model = i
        if t_tool is not None and t_model is not None:
            break
    return t_tool, t_model


def _retrieval_count(steps: list[dict[str, Any]]) -> int:
    return sum(1 for s in steps if _is_retrieval(s))


def solving_cost(trajectories: list[dict[str, Any]]) -> float:
    """Realized solving cost ``Ω̂`` — mean retrieval calls per trajectory (FORT eq 15).

    Counts only retrieval steps (the cost of interest), not model-only turns.
    A high ``Ω̂`` means the solver issued many searches — but on its own does not
    prove the answer was hard to find (see :func:`mean_answer_hit_time`).
    """
    if not trajectories:
        return 0.0
    return round(
        sum(_retrieval_count(t.get("steps", [])) for t in trajectories)
        / len(trajectories),
        6,
    )


def answer_hit_time(trajectory: dict[str, Any]) -> int:
    """Answer hit time ``T_hit = min(T_tool, T_model)`` for one trajectory (FORT eq 16).

    The earliest step at which the gold answer becomes available — either in a
    tool observation or in the model's own text. A *late* hit time is the
    observable signature of suppressed cheap identifying routes. Returns
    ``len(steps) + 1`` (a beyond-the-end sentinel) when the answer never surfaces.
    """
    steps = trajectory.get("steps", [])
    aliases = list(trajectory.get("answer_aliases", []))
    t_tool, t_model = _hit_times(steps, aliases)
    cands = [t for t in (t_tool, t_model) if t is not None]
    return min(cands) if cands else len(steps) + 1


def mean_answer_hit_time(trajectories: list[dict[str, Any]]) -> float:
    """Dataset mean answer hit time ``T̄_hit`` (FORT eq 17). Higher = harder."""
    if not trajectories:
        return 0.0
    return round(sum(answer_hit_time(t) for t in trajectories) / len(trajectories), 6)


def prior_shortcut_rate(trajectories: list[dict[str, Any]]) -> float:
    """Prior-shortcut rate ``p̂_prior`` (FORT eq 18) — fraction with ``T_model < T_tool``.

    A trajectory is prior-bound when the model names the answer *before* any
    retrieved observation anchors it — a conservative proxy for solver-side
    prior-knowledge binding. Lower is better training data.
    """
    if not trajectories:
        return 0.0
    bound = 0
    for t in trajectories:
        t_tool, t_model = _hit_times(
            t.get("steps", []), list(t.get("answer_aliases", []))
        )
        if t_model is not None and (t_tool is None or t_model < t_tool):
            bound += 1
    return round(bound / len(trajectories), 6)


def realized_difficulty(
    trajectories: list[dict[str, Any]],
    *,
    min_cost: float = 3.0,
    min_hit_time: float = 2.0,
    max_prior_rate: float = 0.2,
) -> dict[str, Any]:
    """Bundle the three FORT signatures + a search-heavy verdict for task gating.

    ``search_heavy`` is True when the trajectories show enough realized cost, a
    late-enough mean hit time, and a low-enough prior-shortcut rate — i.e. the
    task genuinely forces multi-turn evidence acquisition. A task that fails this
    gate is too easy and should be re-synthesized with more hops / stricter
    shortcut control. Thresholds are conservative defaults; callers tune per
    benchmark.
    """
    cost = solving_cost(trajectories)
    hit = mean_answer_hit_time(trajectories)
    prior = prior_shortcut_rate(trajectories)
    return {
        "solving_cost": cost,
        "answer_hit_time": hit,
        "prior_shortcut_rate": prior,
        "search_heavy": bool(
            cost >= min_cost and hit >= min_hit_time and prior <= max_prior_rate
        ),
    }
