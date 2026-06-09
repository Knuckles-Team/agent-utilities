#!/usr/bin/python
from __future__ import annotations

"""Agent-Step Policy Optimization (ARPO).

CONCEPT:AHE-3.15 — ARPO (arXiv:2507.19849)

For multi-turn tool agents the decisive uncertainty is at *intermediate* tool/
decision steps, not the final answer. ARPO (a) branches extra rollouts at
high-entropy agent steps and (b) assigns advantage at the agent-step granularity.

This module is the live-path glue for both, reusing what we already run:

* :func:`step_entropy` — normalized entropy of a candidate-score distribution at a
  decision/tool boundary (how unsure the agent is which action to take).
* :func:`should_branch` — the entropy gate (bounded by ``ARPO_MAX_BRANCHES``), read
  by ``SubagentLifecyclePolicy`` to branch at moderate complexity when uncertainty is
  high.
* :func:`write_back_step_credit` — pushes per-step advantages (from
  ``RewardDecomposer.step_advantages``) into the capability reward-EMA via
  ``record_outcome``, so routing learns which *intermediate actions* help — not just
  which final answers succeed.

Source: ``.specify/specs/reasoning-rl-2026/spec-arpo-agent-step-po.md``.
"""

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

# Defaults are env-tunable; branching defaults ON at a sane threshold (Wire-First:
# the live router activates it without a flag).
DEFAULT_BRANCH_ENTROPY = float(os.getenv("ARPO_BRANCH_ENTROPY", "0.6"))
DEFAULT_MAX_BRANCHES = int(os.getenv("ARPO_MAX_BRANCHES", "3"))


def step_entropy(scores: list[float]) -> float:
    """Normalized Shannon entropy ``∈ [0, 1]`` of a candidate-score distribution.

    The scores are the agent's preferences over the candidate actions/tools at one
    decision step (e.g. router similarity scores). A softmax turns them into a
    distribution; entropy near 1 means the agent is uncertain which action to take —
    exactly where an extra rollout (branch) pays off. ``≤1`` candidate ⇒ ``0.0``.
    """
    vals = [float(s) for s in scores if s is not None]
    if len(vals) <= 1:
        return 0.0
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return round(h / math.log(len(probs)), 6)  # normalize by max entropy log(k)


def should_branch(
    entropy: float,
    *,
    threshold: float = DEFAULT_BRANCH_ENTROPY,
    branch_count: int = 0,
    max_branches: int = DEFAULT_MAX_BRANCHES,
) -> bool:
    """Entropy gate for ARPO branching, bounded so it can't wedge the worker pool."""
    return entropy >= threshold and branch_count < max_branches


def write_back_step_credit(
    capability_index: Any,
    step_agent_ids: list[str | None],
    step_advantages: list[float],
    *,
    alpha: float = 0.3,
) -> int:
    """ARPO credit assignment: write each agent-step's advantage into the capability
    reward-EMA, so routing learns which intermediate actions improve the trajectory.

    Advantages (roughly ``[-x, x]``) are squashed to a reward in ``[0, 1]`` via a
    sigmoid before ``record_outcome``. Returns the number of steps credited.
    """
    if capability_index is None or not hasattr(capability_index, "record_outcome"):
        return 0
    written = 0
    for aid, adv in zip(step_agent_ids, step_advantages, strict=False):
        if not aid:
            continue
        reward = 1.0 / (1.0 + math.exp(-float(adv)))  # sigmoid → (0, 1)
        capability_index.record_outcome(aid, reward=reward, alpha=alpha)
        written += 1
    if written:
        logger.debug("[AHE-3.15] ARPO step-credit written for %d agent-steps", written)
    return written
