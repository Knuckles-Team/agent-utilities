#!/usr/bin/python
from __future__ import annotations

"""Unit tests for the implicit-reasoning generative recommender (CONCEPT:KG-2.93).

Deterministic, no LLM/network: a small synthetic catalog of clustered item
embeddings is encoded into semantic IDs via a real (seeded)
``TemporalSemanticIdEncoder``, and the PauseRec-adapted recommender is exercised
end to end.
"""

import numpy as np
import pytest

from agent_utilities.knowledge_graph.retrieval.generative_recommender import (
    ImplicitReasoningRecommender,
    Recommendation,
    TextSidBridge,
)
from agent_utilities.knowledge_graph.retrieval.temporal_semantic_id import (
    TemporalSemanticIdEncoder,
)

_DIM = 16
_N_PER_CLUSTER = 12
_SEED = 7


def _clustered_catalog() -> tuple[
    list[tuple[str, list[float]]], dict[str, int], np.ndarray
]:
    """Build three well-separated clusters of unit-ish embeddings.

    Returns the ``(item_id, embedding)`` catalog, an ``item_id -> cluster``
    map, and the three cluster centers (for crafting near-cluster queries).
    """
    rng = np.random.default_rng(_SEED)
    centers = np.zeros((3, _DIM), dtype=np.float64)
    centers[0, 0] = 5.0
    centers[1, 5] = 5.0
    centers[2, 10] = 5.0

    catalog: list[tuple[str, list[float]]] = []
    cluster_of: dict[str, int] = {}
    for c in range(3):
        for j in range(_N_PER_CLUSTER):
            vec = centers[c] + 0.1 * rng.standard_normal(_DIM)
            item_id = f"c{c}_i{j}"
            catalog.append((item_id, vec.tolist()))
            cluster_of[item_id] = c
    return catalog, cluster_of, centers


def _make_encoder() -> TemporalSemanticIdEncoder:
    return TemporalSemanticIdEncoder(
        n_codebooks=3, codebook_size=8, n_time_buckets=8, seed=_SEED
    )


def _fitted_recommender(
    pause_steps: int = 2,
) -> tuple[ImplicitReasoningRecommender, dict[str, int], np.ndarray]:
    catalog, cluster_of, centers = _clustered_catalog()
    rec = ImplicitReasoningRecommender(_make_encoder(), pause_steps=pause_steps)
    rec.fit_catalog(catalog)
    return rec, cluster_of, centers


def test_fit_catalog_assigns_sids_to_all_items() -> None:
    rec, cluster_of, _ = _fitted_recommender()
    assert rec.catalog_size == len(cluster_of)
    assert len(rec._item_sids) == rec.catalog_size
    for sid in rec._item_sids:
        assert len(sid) == 3  # n_codebooks
        assert all(0 <= code < 8 for code in sid)  # codebook_size


def test_fit_catalog_rejects_empty() -> None:
    rec = ImplicitReasoningRecommender(_make_encoder())
    with pytest.raises(ValueError):
        rec.fit_catalog([])


def test_temporal_event_times_require_now_epoch() -> None:
    catalog, _, _ = _clustered_catalog()
    rec = ImplicitReasoningRecommender(_make_encoder())
    with pytest.raises(ValueError):
        rec.fit_catalog(catalog, event_times={catalog[0][0]: 0.0})


def test_fit_catalog_with_temporal_sids() -> None:
    catalog, _, _ = _clustered_catalog()
    rec = ImplicitReasoningRecommender(_make_encoder())
    event_times = {item_id: float(i) for i, (item_id, _) in enumerate(catalog)}
    rec.fit_catalog(catalog, event_times=event_times, now_epoch=1_000_000.0)
    assert rec.catalog_size == len(catalog)
    # Content SIDs (time token dropped) keep content-code length.
    assert all(len(sid) == 3 for sid in rec._item_sids)


def test_bridge_project_returns_in_range_code_tuple() -> None:
    rec, _, centers = _fitted_recommender()
    bridge = TextSidBridge(rec._encoder)
    sid = bridge.project(centers[0].tolist())
    assert isinstance(sid, tuple)
    assert len(sid) == 3
    assert all(0 <= code < 8 for code in sid)


def test_recommend_returns_top_k_ranked_recommendations() -> None:
    rec, _, centers = _fitted_recommender()
    out = rec.recommend(centers[1].tolist(), top_k=5)
    assert len(out) == 5
    assert all(isinstance(r, Recommendation) for r in out)
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)  # best-first
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_query_near_cluster_recommends_that_cluster() -> None:
    rec, cluster_of, centers = _fitted_recommender()
    for target_cluster in range(3):
        out = rec.recommend(centers[target_cluster].tolist(), top_k=5)
        clusters = [cluster_of[r.item_id] for r in out]
        # The dominant recommended cluster must be the queried cluster.
        majority = max(set(clusters), key=clusters.count)
        assert majority == target_cluster, (target_cluster, clusters)


def test_history_biases_recommendations() -> None:
    rec, cluster_of, centers = _fitted_recommender()
    # Query sits between cluster 0 and cluster 2; history points at cluster 2.
    ambiguous = (0.5 * centers[0] + 0.5 * centers[2]).tolist()
    cluster2_item = next(i for i, c in cluster_of.items() if c == 2)
    hist_sid = rec._encoder.encode_content(dict(_clustered_catalog()[0])[cluster2_item])

    no_hist = rec.recommend(ambiguous, top_k=6)
    with_hist = rec.recommend(ambiguous, top_k=6, history_sids=[hist_sid])

    frac2_no = sum(cluster_of[r.item_id] == 2 for r in no_hist) / len(no_hist)
    frac2_hist = sum(cluster_of[r.item_id] == 2 for r in with_hist) / len(with_hist)
    assert frac2_hist >= frac2_no
    # History must change the ranking outcome.
    assert [r.item_id for r in with_hist] != [r.item_id for r in no_hist]


def test_pause_steps_zero_vs_positive_differ() -> None:
    catalog, _, centers = _clustered_catalog()
    ambiguous = (0.6 * centers[0] + 0.4 * centers[1]).tolist()

    rec0 = ImplicitReasoningRecommender(_make_encoder(), pause_steps=0)
    rec0.fit_catalog(catalog)
    rec3 = ImplicitReasoningRecommender(_make_encoder(), pause_steps=3)
    rec3.fit_catalog(catalog)

    out0 = rec0.recommend(ambiguous, top_k=8)
    out3 = rec3.recommend(ambiguous, top_k=8)
    assert len(out0) == 8 and len(out3) == 8  # both budgets produce results
    # More pause steps sharpens / changes the ranking.
    assert [r.item_id for r in out0] != [r.item_id for r in out3]


def test_explain_budget_reports_implicit_no_rationale() -> None:
    rec, _, _ = _fitted_recommender(pause_steps=4)
    info = rec.explain_budget()
    assert info["pause_steps"] == 4
    assert info["implicit"] is True
    assert info["rationale"] is None
    assert info["decodes_rationale"] is False
    assert "2606.14142" in info["paper"]
    assert info["concept"] == "KG-2.93"


def test_determinism() -> None:
    _, _, centers = _clustered_catalog()
    query = centers[2].tolist()

    rec_a, _, _ = _fitted_recommender()
    rec_b, _, _ = _fitted_recommender()
    out_a = rec_a.recommend(query, top_k=5)
    out_b = rec_b.recommend(query, top_k=5)
    assert [(r.item_id, r.semantic_id, r.score) for r in out_a] == [
        (r.item_id, r.semantic_id, r.score) for r in out_b
    ]


def test_recommend_before_fit_raises() -> None:
    rec = ImplicitReasoningRecommender(_make_encoder())
    with pytest.raises(RuntimeError):
        rec.recommend([0.0] * _DIM)


def test_invalid_pause_steps_rejected() -> None:
    with pytest.raises(ValueError):
        ImplicitReasoningRecommender(_make_encoder(), pause_steps=-1)
