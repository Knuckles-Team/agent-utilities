#!/usr/bin/python
from __future__ import annotations

"""Score-discontinuity autocut for retrieval result sets.

CONCEPT:AU-KG.retrieval.pack-retrieval-signals — Pack-Driven Retrieval Signals

Mirrors gbrain's ``autocut.ts``: instead of forcing the caller to pick a ``top_k``,
trim the ranked result list at the largest *relative* score drop ("knee"). This
returns the natural cluster of high-relevance results and drops the long tail of
weak matches that only dilute the context window.

The cut is conservative and recall-safe:
- It never trims a set smaller than ``min_results`` (so small result sets are
  returned intact — a single weak query never collapses to one hit).
- It only cuts when the largest relative drop meets ``threshold``; on a flat score
  distribution (no clear knee) the full set is returned.
"""


from typing import Any


def autocut(
    scored: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
    min_results: int = 5,
    score_key: str = "_score",
) -> list[dict[str, Any]]:
    """Trim ``scored`` at the largest relative score drop ≥ ``threshold``.

    Args:
        scored: Result dicts carrying a numeric ``score_key``. Not required to be
            pre-sorted — the function sorts a copy descending by score.
        threshold: Minimum relative drop ``(prev - cur) / prev`` that triggers a cut.
        min_results: Never return fewer than this many results; sets at or below
            this size are returned unchanged.
        score_key: Key holding each result's score.

    Returns:
        The (sorted) prefix up to the knee, or the full sorted list when no drop
        meets ``threshold`` or the set is too small to cut.
    """
    if len(scored) <= min_results:
        return scored

    ordered = sorted(scored, key=lambda n: float(n.get(score_key, 0.0)), reverse=True)

    best_cut = len(ordered)
    best_drop = 0.0
    # Only consider cut points that still leave >= min_results items.
    for i in range(min_results, len(ordered)):
        prev = float(ordered[i - 1].get(score_key, 0.0))
        cur = float(ordered[i].get(score_key, 0.0))
        if prev <= 0.0:
            continue
        drop = (prev - cur) / prev
        if drop > best_drop:
            best_drop = drop
            best_cut = i

    if best_drop >= threshold:
        return ordered[:best_cut]
    return ordered


__all__ = ["autocut"]
