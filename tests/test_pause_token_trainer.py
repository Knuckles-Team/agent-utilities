#!/usr/bin/python
"""Tests for the trainable ``<pause>``-token recommender (CONCEPT:KG-2.93, PauseRec).

Validates the *real* PauseRec mechanism: pause tokens that are trained by gradient
descent on the next-item objective, and that they HELP (the paper's core claim),
reproduced at tiny / CPU scale. The whole module is skipped when torch is absent
so torch-free CI still passes.

Toy task (built so the pause tokens genuinely matter): item ids are laid out so
that the *target* item is a NON-LINEAR (parity / XOR) interaction of two history
items' latent factors. Concretely each history pair ``(a, b)`` maps to a target
chosen by ``(parity(a) XOR parity(b))`` together with a content bucket -- a rule a
plain linear readout over the mean history cannot separate, but the latent
pause-token computation can. So the with-pause model should reach a strictly
higher Recall@k than the same model with its pause rows zeroed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from agent_utilities.knowledge_graph.retrieval.pause_token_trainer import (  # noqa: E402
    PauseTokenRecommender,
    TrainResult,
    is_available,
)

N_ITEMS = 12
DIM = 6  # tiny: ~228 trainable params total
# Recall@1 is the discriminating cutoff: the linear (no-pause) path can rank the
# right family near the top but cannot pin the exact XOR target first, while the
# pause path can -- so the with/without gap shows up cleanly at k=1.
K = 1


def _xor_corpus() -> list[list[int]]:
    """Build sequences whose next item is a non-linear (XOR/parity) function of history.

    For every ordered pair of "factor" items ``(a, b)`` drawn from two disjoint
    groups, the target is selected by the XOR of their parities (plus a fixed
    content offset), so the next item depends on a non-linear interaction of the
    history -- the regime where pause-token latent computation pays off. The
    corpus is fully deterministic (no randomness).
    """
    group_a = [0, 1, 2, 3]
    group_b = [4, 5, 6, 7]
    # Four distinct target items, one per XOR outcome bucket pair.
    targets = {(0, 0): 8, (0, 1): 9, (1, 0): 10, (1, 1): 11}
    seqs: list[list[int]] = []
    for a in group_a:
        for b in group_b:
            key = (a % 2, b % 2)
            seqs.append([a, b, targets[key]])
    return seqs


def test_is_available_true_when_torch_present() -> None:
    """torch was import-skipped above, so the trainable mechanism is available."""
    assert is_available() is True


def test_fit_learns_and_pause_helps() -> None:
    """fit() reduces loss and the pause tokens improve Recall@k (PauseRec's claim)."""
    rec = PauseTokenRecommender(n_items=N_ITEMS, dim=DIM, n_pause_tokens=4, seed=0)
    result = rec.fit(_xor_corpus(), epochs=300, lr=0.05, k=K)

    assert isinstance(result, TrainResult)
    # It actually learns.
    assert result.final_loss < result.initial_loss
    # Budget reflected back.
    assert result.n_pause_tokens == 4
    # PauseRec core claim: the trained pause tokens help (zeroing them hurts recall).
    assert result.recall_at_k_with_pause >= result.recall_at_k_without_pause
    # On this non-linear task the help is real, not a tie: with-pause solves the
    # XOR target perfectly while the ablated linear path misses a clear margin.
    assert result.recall_at_k_with_pause == 1.0
    assert result.recall_at_k_without_pause < 0.9
    assert result.recall_at_k_with_pause > result.recall_at_k_without_pause


def test_recommend_returns_k_distinct_valid_ids() -> None:
    """recommend() yields k distinct, in-range item ids."""
    rec = PauseTokenRecommender(n_items=N_ITEMS, dim=DIM, n_pause_tokens=4, seed=0)
    rec.fit(_xor_corpus(), epochs=80, lr=0.05, k=K)

    top = 4
    recs = rec.recommend([0, 4], k=top)
    assert len(recs) == top
    assert len(set(recs)) == top  # distinct
    assert all(0 <= i < N_ITEMS for i in recs)  # valid


def test_determinism_same_seed_same_loss() -> None:
    """Two trainers with the same seed produce identical final loss."""
    corpus = _xor_corpus()
    r1 = PauseTokenRecommender(n_items=N_ITEMS, dim=DIM, n_pause_tokens=4, seed=7)
    r2 = PauseTokenRecommender(n_items=N_ITEMS, dim=DIM, n_pause_tokens=4, seed=7)
    out1 = r1.fit(corpus, epochs=120, lr=0.05, k=K)
    out2 = r2.fit(corpus, epochs=120, lr=0.05, k=K)
    assert out1.final_loss == pytest.approx(out2.final_loss, rel=0, abs=1e-9)
    assert out1.recall_at_k_with_pause == out2.recall_at_k_with_pause
