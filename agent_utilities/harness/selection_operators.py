#!/usr/bin/python
from __future__ import annotations

"""Unified selection / aggregation operators.

CONCEPT:ORCH-1.30 — Selection on Unseen Data

One registry of verifier-free, diversity-preserving, uncertainty-aware operators
for "pick the best of N", consumed wherever the stack currently hard-codes a
mean-reward argmax (`VariantPool` tournament, `EvolutionaryAggregator`,
TeamConfig promotion, `CoordinationLayer` aggregation). Distilled from three
research plans (`.specify/specs/research-evolution-20260606/`) that each asked
for a different selection primitive:

* :func:`bradley_terry_scores` — global ranking from pairwise judgments (b6-67;
  the verifier-free pairwise-judge selector — previously absent in the codebase).
* :func:`conservative_rating` — uncertainty-aware online rating with an LCB
  (μ−κσ) score, so a lucky high-mean candidate doesn't beat a reliable one
  (b4-03, TrueSkill-LCB spirit).
* :func:`contribution_weighted_vote` — votes weighted by contribution H
  (w = 1 + β·H), crediting collaborators whose findings entered the record
  (b5-02).

Plus the scalar :func:`select_top_k` (score / LCB) used by the existing
tournament path. Pure Python — no model, no network, no heavy deps.

Concept: selection-operators
"""

import math
from typing import Any

# ---------------------------------------------------------------------------
# Pairwise → global ranking
# ---------------------------------------------------------------------------


def bradley_terry_scores(
    items: list[str],
    comparisons: list[tuple[str, str]],
    *,
    iters: int = 200,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Bradley–Terry strengths from pairwise (winner, loser) outcomes.

    Solves the BT model by the standard minorization–maximization iteration
    (no SciPy): ``p_i ← W_i / Σ_j n_ij/(p_i+p_j)``. Robust at small sample
    counts, internalises opponent strength, and yields a stable global ranking
    from noisy pairwise judgments. Returns scores normalised to sum to 1.

    Args:
        items: All candidate ids.
        comparisons: ``(winner_id, loser_id)`` pairs (ties may be split by the caller).
    """
    ids = list(dict.fromkeys(items))
    if not ids:
        return {}
    idx = {x: i for i, x in enumerate(ids)}
    n = len(ids)
    wins = [0.0] * n
    games = [[0.0] * n for _ in range(n)]
    for w, ll in comparisons:
        if w not in idx or ll not in idx:
            continue
        wins[idx[w]] += 1.0
        games[idx[w]][idx[ll]] += 1.0
        games[idx[ll]][idx[w]] += 1.0

    p = [1.0] * n
    for _ in range(iters):
        new_p = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                nij = games[i][j]
                if nij:
                    denom += nij / (p[i] + p[j])
            new_p[i] = (wins[i] / denom) if denom > 0 else 0.0
        s = sum(new_p)
        if s <= 0:
            break
        new_p = [x * n / s for x in new_p]  # keep scale stable
        if max(abs(new_p[i] - p[i]) for i in range(n)) < tol:
            p = new_p
            break
        p = new_p

    total = sum(p) or 1.0
    return {ids[i]: p[i] / total for i in range(n)}


def conservative_rating(
    items: list[str],
    comparisons: list[tuple[str, str]],
    *,
    kappa: float = 1.0,
    k: float = 0.1,
    init_sigma: float = 1.0,
    sigma_decay: float = 0.97,
    min_sigma: float = 0.05,
) -> dict[str, float]:
    """Uncertainty-aware conservative score (TrueSkill-LCB spirit).

    Runs an online logistic rating over the (winner, loser) stream, tracking a
    mean μ and a shrinking uncertainty σ per item, and returns the lower
    confidence bound ``μ − κσ`` — so selection prefers candidates that are both
    strong *and* well-evidenced, curbing the "promote the lucky one" failure of
    mean-reward selection.
    """
    ids = list(dict.fromkeys(items))
    mu = {x: 0.0 for x in ids}
    sigma = {x: init_sigma for x in ids}
    for w, ll in comparisons:
        if w not in mu or ll not in mu:
            continue
        expected_w = 1.0 / (1.0 + math.exp(-(mu[w] - mu[ll])))
        mu[w] += k * (1.0 - expected_w)
        mu[ll] -= k * (1.0 - expected_w)
        sigma[w] = max(min_sigma, sigma[w] * sigma_decay)
        sigma[ll] = max(min_sigma, sigma[ll] * sigma_decay)
    return {x: mu[x] - kappa * sigma[x] for x in ids}


def contribution_weighted_vote(
    ballots: list[tuple[str, float]], *, beta: float = 1.0
) -> dict[str, float]:
    """Tally votes weighted by contribution: ``w_i = 1 + β·H_i``.

    Args:
        ballots: ``(choice_id, contribution_H)`` per voter.
        beta: Contribution weight.

    Returns:
        choice_id → weighted tally (higher is better).
    """
    tally: dict[str, float] = {}
    for choice, h in ballots:
        tally[choice] = tally.get(choice, 0.0) + (1.0 + beta * float(h))
    return tally


# ---------------------------------------------------------------------------
# Scalar candidate selection (used by the live tournament path)
# ---------------------------------------------------------------------------


def select_top_k(
    candidates: list[dict[str, Any]],
    k: int,
    *,
    method: str = "score",
    score_key: str = "fitness",
    std_key: str = "fitness_std",
    kappa: float = 1.0,
) -> list[dict[str, Any]]:
    """Rank scalar candidate dicts and return the top ``k``.

    Methods:
        * ``score`` — descending by ``score_key`` (plain mean-reward).
        * ``lcb`` — descending by ``score_key − κ·std_key`` (conservative; a
          candidate with no recorded variance degrades gracefully to ``score``).
    """
    if method == "score":
        ranked = sorted(
            candidates, key=lambda c: float(c.get(score_key, 0.0) or 0.0), reverse=True
        )
    elif method == "lcb":
        ranked = sorted(
            candidates,
            key=lambda c: (
                float(c.get(score_key, 0.0) or 0.0)
                - kappa * float(c.get(std_key, 0.0) or 0.0)
            ),
            reverse=True,
        )
    else:
        raise ValueError(f"unknown scalar selection method: {method!r}")
    return ranked[:k]


def rank_from_comparisons(
    items: list[str],
    comparisons: list[tuple[str, str]],
    *,
    method: str = "bradley_terry",
    kappa: float = 1.0,
) -> list[tuple[str, float]]:
    """Dispatch a pairwise method and return ``(id, score)`` sorted descending."""
    if method == "bradley_terry":
        scores = bradley_terry_scores(items, comparisons)
    elif method == "conservative_rating":
        scores = conservative_rating(items, comparisons, kappa=kappa)
    else:
        raise ValueError(f"unknown pairwise selection method: {method!r}")
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
