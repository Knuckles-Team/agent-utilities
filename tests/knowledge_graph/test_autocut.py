from __future__ import annotations

"""Tests for score-discontinuity autocut.

CONCEPT:EG-KG.compute.rust-native-training-loss — Pack-Driven Retrieval Signals (autocut)
"""


from agent_utilities.knowledge_graph.retrieval.autocut import autocut


def _scored(scores):
    return [{"_score": s, "id": f"n{i}"} for i, s in enumerate(scores)]


def test_cuts_at_knee():
    # Clear knee between 0.85 and 0.20 → keep the top cluster of 4.
    res = autocut(
        _scored([0.9, 0.88, 0.85, 0.84, 0.2, 0.1, 0.05]), threshold=0.5, min_results=4
    )
    assert [round(n["_score"], 2) for n in res] == [0.9, 0.88, 0.85, 0.84]


def test_never_trims_small_sets():
    # 3 items with min_results=5 → returned unchanged even with a big drop.
    src = _scored([0.9, 0.1, 0.05])
    assert autocut(src, threshold=0.5, min_results=5) == src


def test_flat_distribution_returns_all():
    res = autocut(
        _scored([0.8, 0.79, 0.78, 0.77, 0.76, 0.75]), threshold=0.5, min_results=2
    )
    assert len(res) == 6


def test_sorts_before_cutting():
    # Unsorted input is sorted descending before the knee is found.
    res = autocut(
        _scored([0.1, 0.9, 0.85, 0.88, 0.05, 0.2]), threshold=0.5, min_results=3
    )
    scores = [n["_score"] for n in res]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 0.9


def test_threshold_gating():
    # A modest 30% drop should not trigger a cut when threshold is 0.5.
    res = autocut(
        _scored([1.0, 0.7, 0.5, 0.35, 0.25, 0.2]), threshold=0.5, min_results=2
    )
    assert len(res) == 6
