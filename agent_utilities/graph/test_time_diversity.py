#!/usr/bin/python
from __future__ import annotations

"""Test-Time Diversity (VPO).

CONCEPT:AU-AHE.harness.width-diverse-best-k — VPO (arXiv:2605.22817)

We already scale test-time compute (``harness/reasoning_effort``) and fan out
subagents (``SubagentLifecyclePolicy``, ``rlm/`` parallel sub-calls), but we sample
toward a *single* best answer. VPO shows that optimizing for a *diverse* candidate set
raises test-time best@k / pass@k. This module supplies:

* :func:`diverse_fan_out_width` — the effort-derived width of the diverse fan-out
  (reads ``ReasoningBudget.diversity_width`` on the live budget path; no new agent knob).
* :func:`mean_pairwise_distance` — a diversity metric over a candidate set.
* :func:`select_diverse` — MMR-style best-of-k selection trading quality vs diversity.

The optional graph-native diversity kernel is ``epistemic-graph``'s
``personalized_pagerank`` (seed-diverse propagation); the embedding-spread selection
here is the dependency-free default.

Source: ``.specify/specs/reasoning-rl-2026/spec-vpo-test-time-diversity.md``.
"""

import logging
from typing import Any

from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_pairwise_distance(embeddings: list[Any]) -> float:
    """Mean pairwise cosine *distance* (``1 − sim``) over a candidate set.

    Higher = more diverse. ``< 2`` candidates ⇒ ``0.0``.
    """
    vecs = [np.asarray(e, dtype=np.float32) for e in embeddings]
    n = len(vecs)
    if n < 2:
        return 0.0
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - _cos(vecs[i], vecs[j])
            cnt += 1
    return round(total / cnt, 6) if cnt else 0.0


def select_diverse(
    scores: list[float],
    embeddings: list[Any],
    k: int,
    *,
    quality_weight: float = 0.5,
) -> list[int]:
    """VPO best-of-k: greedily select ``k`` candidate indices maximizing
    ``quality_weight·quality + (1−quality_weight)·diversity`` (MMR).

    Seeds with the highest-quality candidate, then repeatedly adds the candidate that
    best balances its own quality against its distance to the already-selected set —
    so the returned set is high-quality *and* spread out. Returns selected indices.
    """
    n = len(scores)
    if n == 0:
        return []
    if len(embeddings) != n:
        raise ValueError("scores and embeddings must align")
    k = max(1, min(k, n))
    vecs = [np.asarray(e, dtype=np.float32) for e in embeddings]
    smin, smax = min(scores), max(scores)
    rng = (smax - smin) or 1.0
    q = [(s - smin) / rng for s in scores]

    selected: list[int] = [max(range(n), key=lambda i: q[i])]
    remaining = set(range(n)) - set(selected)
    while remaining and len(selected) < k:

        def _mmr(i: int) -> float:
            div = min(1.0 - _cos(vecs[i], vecs[j]) for j in selected)
            return quality_weight * q[i] + (1.0 - quality_weight) * div

        nxt = max(remaining, key=_mmr)
        selected.append(nxt)
        remaining.discard(nxt)
    return selected


def diverse_fan_out_width(effort: float) -> int:
    """Effort-derived width of the diverse fan-out (the live VPO knob).

    Reads ``ReasoningBudget.diversity_width`` so harder queries get a wider, more
    diverse candidate search without any new agent-facing parameter.
    """
    from agent_utilities.harness.reasoning_effort import get_budget

    return get_budget(effort).diversity_width
