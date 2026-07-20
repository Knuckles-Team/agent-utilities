#!/usr/bin/python
from __future__ import annotations

"""Tests for the dual-score statistical fusion gate (CONCEPT:AU-KG.retrieval.unset-dependency-free)."""

from agent_utilities.knowledge_graph.retrieval.score_gate import (
    fuse_scores,
    score_gate,
)


def _bimodal() -> list[dict[str, float]]:
    """A clear strong cluster and a clearly weak tail on both encoders."""
    return [
        {"_score": 0.95, "_rerank_score": 0.97},
        {"_score": 0.93, "_rerank_score": 0.94},
        {"_score": 0.90, "_rerank_score": 0.91},
        {"_score": 0.88, "_rerank_score": 0.89},
        {"_score": 0.12, "_rerank_score": 0.10},
        {"_score": 0.09, "_rerank_score": 0.08},
        {"_score": 0.05, "_rerank_score": 0.04},
        {"_score": 0.02, "_rerank_score": 0.03},
    ]


def test_bimodal_cuts_weak_tail() -> None:
    kept = score_gate(_bimodal(), min_results=3)
    # The four weak-tail chunks fall below the fused mean and are dropped.
    assert len(kept) == 4
    assert all(item["_score"] >= 0.88 for item in kept)


def test_flat_distribution_returns_at_least_min_results() -> None:
    flat = [{"_score": 0.5, "_rerank_score": 0.5} for _ in range(10)]
    kept = score_gate(flat, min_results=3)
    # Flat -> all z-scores 0.0 -> everything is at the mean, nothing cut.
    assert len(kept) == 10
    assert all(item["_fused_score"] == 0.0 for item in kept)


def test_flat_distribution_min_results_floor() -> None:
    # Even if the gate would somehow cut, the floor guarantees min_results.
    flat = [{"_score": 0.5} for _ in range(2)]
    kept = score_gate(flat, min_results=3)
    assert len(kept) == 2  # small set returned intact


def test_min_results_floor_on_skewed_set() -> None:
    # One dominant chunk; everything else is far below the mean and would be cut,
    # but the floor keeps the top min_results.
    skewed = [{"_score": 1.0, "_rerank_score": 1.0}]
    skewed += [{"_score": 0.01, "_rerank_score": 0.01} for _ in range(9)]
    kept = score_gate(skewed, min_results=3)
    assert len(kept) == 3
    assert kept[0]["_score"] == 1.0


def test_max_results_cap_respected() -> None:
    kept = score_gate(_bimodal(), min_results=3, max_results=2)
    assert len(kept) == 2
    assert all(item["_score"] >= 0.93 for item in kept)


def test_cross_key_missing_falls_back_to_bi() -> None:
    # No cross scores anywhere: fused == standardized bi, gate still works.
    bi_only = [
        {"_score": 0.95},
        {"_score": 0.93},
        {"_score": 0.90},
        {"_score": 0.88},
        {"_score": 0.05},
        {"_score": 0.02},
    ]
    kept = score_gate(bi_only, min_results=3)
    assert len(kept) == 4
    assert all(item["_score"] >= 0.88 for item in kept)


def test_fused_key_annotated_on_every_returned_item() -> None:
    kept = score_gate(_bimodal(), min_results=3)
    assert all("_fused_score" in item for item in kept)
    assert all(isinstance(item["_fused_score"], float) for item in kept)


def test_fuse_scores_annotates_and_sorts() -> None:
    ordered = fuse_scores(_bimodal())
    assert all("_fused_score" in item for item in ordered)
    fused = [item["_fused_score"] for item in ordered]
    assert fused == sorted(fused, reverse=True)


def test_determinism() -> None:
    data1 = _bimodal()
    data2 = _bimodal()
    out1 = score_gate(data1, min_results=3)
    out2 = score_gate(data2, min_results=3)
    assert [item["_score"] for item in out1] == [item["_score"] for item in out2]
    assert [item["_fused_score"] for item in out1] == [
        item["_fused_score"] for item in out2
    ]


def test_higher_keep_z_keeps_fewer() -> None:
    relaxed = score_gate(_bimodal(), min_results=3, keep_z=0.0)
    strict = score_gate(_bimodal(), min_results=3, keep_z=1.0)
    assert len(strict) <= len(relaxed)
