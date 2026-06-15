#!/usr/bin/python
from __future__ import annotations

"""Dual-score statistical fusion gate for retrieval result sets.

CONCEPT:KG-2.85 — Adaptive Chunk Selection (dual-score fusion)

Implements ScoreGate from "ScoreGate: Adaptive Chunk Selection for
Retrieval-Augmented Generation via Dual-Score Statistical Fusion"
(arXiv 2606.x). Instead of forcing the caller to pick a fixed ``top_k``, the
gate decides *how many* retrieved chunks to keep by statistically fusing two
complementary relevance signals — a fast bi-encoder (vector) score and a slower,
sharper cross-encoder (reranker) score — and retaining the chunks that stand
above the fused mean.

This generalizes :func:`autocut` (a single-score knee detector): where autocut
trims one ranked list at its largest relative drop, ScoreGate first puts the two
scores on a common, scale-free footing (per-component z-standardization), fuses
them into one ``_fused_score``, and then gates on a z-threshold. The two signals
are reconciled rather than chosen between, so a chunk that is strong on either
encoder survives while one that is weak on both is dropped.

The gate is conservative and recall-safe, mirroring autocut's philosophy:
- It never returns fewer than ``min_results`` (small sets pass through intact, so
  a thin retrieval never collapses to a single hit).
- A flat distribution (no spread on either signal) standardizes to all-zeros and
  every item lands at the mean, so the gate keeps everything up to ``max_results``.
- Missing cross-encoder scores fall back to the bi-encoder score, so the gate
  degrades gracefully to single-signal behavior when reranking has not run.

It is a pure, deterministic function of its inputs (stdlib + ``statistics``
only) — no model, no network, no environment reads.
"""

from statistics import fmean, pstdev
from typing import Any

_EPS = 1e-9


def _zscores(values: list[float]) -> list[float]:
    """Standardize ``values`` to z-scores; flat sets standardize to all-zeros.

    A zero or near-zero population stdev carries no discriminating information,
    so that component is treated as uniformly average (every z == 0.0) rather
    than amplifying floating-point noise.
    """
    if not values:
        return []
    mean = fmean(values)
    spread = pstdev(values)
    if spread <= _EPS:
        return [0.0 for _ in values]
    return [(v - mean) / spread for v in values]


def fuse_scores(
    scored: list[dict[str, Any]],
    *,
    bi_key: str = "_score",
    cross_key: str = "_rerank_score",
    fused_key: str = "_fused_score",
) -> list[dict[str, Any]]:
    """Annotate each item with a fused z-score and return them sorted descending.

    The bi-encoder and cross-encoder scores are standardized independently across
    the set (so their differing scales never let one dominate), then averaged into
    a single ``fused_key``. When an item has no ``cross_key`` it falls back to its
    ``bi_key`` value before standardization.

    Args:
        scored: Result dicts carrying numeric ``bi_key`` (and optionally ``cross_key``).
        bi_key: Key holding each item's bi-encoder (vector) score.
        cross_key: Key holding each item's cross-encoder (reranker) score.
        fused_key: Key under which the fused z-score is written.

    Returns:
        A new list (the same dict objects, mutated with ``fused_key``) sorted by
        fused score descending.
    """
    bis = [float(item.get(bi_key, 0.0)) for item in scored]
    crosses = [float(item.get(cross_key, item.get(bi_key, 0.0))) for item in scored]
    z_bi = _zscores(bis)
    z_cross = _zscores(crosses)

    for item, zb, zc in zip(scored, z_bi, z_cross, strict=True):
        item[fused_key] = 0.5 * zb + 0.5 * zc

    return sorted(scored, key=lambda item: float(item[fused_key]), reverse=True)


def score_gate(
    scored: list[dict[str, Any]],
    *,
    bi_key: str = "_score",
    cross_key: str = "_rerank_score",
    fused_key: str = "_fused_score",
    min_results: int = 3,
    max_results: int | None = None,
    keep_z: float = 0.0,
) -> list[dict[str, Any]]:
    """Keep the chunks whose fused z-score clears ``keep_z`` (default = the mean).

    Each item's bi-encoder and cross-encoder scores are fused into a scale-free
    ``fused_key`` (see :func:`fuse_scores`), then the set is gated: every item at
    or above ``keep_z`` standard deviations is retained, the weak tail below it is
    dropped. The cut is recall-safe — it never falls below ``min_results`` — and
    capped at ``max_results`` when supplied.

    Args:
        scored: Result dicts carrying numeric ``bi_key`` (and optionally ``cross_key``).
        bi_key: Key holding each item's bi-encoder (vector) score.
        cross_key: Key holding each item's cross-encoder (reranker) score; falls
            back to ``bi_key`` per item when absent.
        fused_key: Key under which the fused z-score is written on every item.
        min_results: Never return fewer than this many results; sets at or below
            this size are returned (sorted, annotated) intact.
        max_results: Optional hard cap on the number of results returned.
        keep_z: Fused z-score threshold to retain an item. ``0.0`` keeps items at
            or above the mean; raise it to keep only the standout chunks.

    Returns:
        The fused-sorted prefix passing the gate, floored at ``min_results`` and
        capped at ``max_results``.
    """
    ordered = fuse_scores(
        scored, bi_key=bi_key, cross_key=cross_key, fused_key=fused_key
    )

    if len(ordered) <= min_results:
        return ordered

    kept = [item for item in ordered if float(item[fused_key]) >= keep_z]
    if len(kept) < min_results:
        kept = ordered[:min_results]

    if max_results is not None:
        kept = kept[:max_results]
    return kept


__all__ = ["score_gate", "fuse_scores"]
