"""Plan 08 Synergy 5: reward write-back closes the learning loop.

Repeated successful use of an entity for a query class must raise its rank for
that class on the next identical query.
"""

from __future__ import annotations

import numpy as np

from agent_utilities.knowledge_graph.retrieval.capability_index import CapabilityIndex


def _idx() -> CapabilityIndex:
    idx = CapabilityIndex(dim=3, prefer_backend="numpy")
    # Two entities with the SAME capability and nearly-identical embeddings, so
    # similarity alone barely separates them — reward must break the tie.
    idx.add("tool_a", [1.0, 0.0, 0.02], capabilities={"search"})
    idx.add("tool_b", [1.0, 0.0, 0.0], capabilities={"search"})
    return idx


def test_record_outcome_updates_ema_bounded():
    idx = _idx()
    assert idx.reward_of("tool_a") == 0.5  # neutral prior
    r1 = idx.record_outcome("tool_a", success=True)
    assert 0.5 < r1 <= 1.0
    # Many successes converge toward but never exceed 1.0.
    for _ in range(50):
        idx.record_outcome("tool_a", success=True)
    assert idx.reward_of("tool_a") <= 1.0
    assert idx.reward_of("tool_a") > 0.9


def test_success_raises_rank_for_query_class():
    idx = _idx()
    query = [1.0, 0.0, 0.0]  # equally close to both, tool_b marginally closer

    before = idx.designate(query, required_caps=["search"], k=2)
    # Reinforce tool_a repeatedly for this query class.
    for _ in range(10):
        idx.record_outcome("tool_a", success=True)
        idx.record_outcome("tool_b", success=False)

    after = idx.designate(query, required_caps=["search"], k=2)
    assert after[0].id == "tool_a", f"reward should lift tool_a to rank 1: {[d.id for d in after]}"
    # Provenance surfaces the learned reward.
    assert after[0].provenance["reward"] > 0.5
    # The pre-reinforcement ranking should not already favour tool_a by reward.
    assert before[0].provenance["reward"] == 0.5


def test_reward_weight_zero_disables_boost():
    idx = _idx()
    for _ in range(10):
        idx.record_outcome("tool_a", success=True)
    # With the boost off, ranking is pure similarity (tool_b is closer).
    out = idx.designate([1.0, 0.0, 0.0], required_caps=["search"], k=2, reward_weight=0.0)
    assert out[0].id == "tool_b"


def test_decay_moves_toward_neutral():
    idx = _idx()
    for _ in range(20):
        idx.record_outcome("tool_a", success=True)
    high = idx.reward_of("tool_a")
    idx.decay_rewards(factor=0.5)
    assert 0.5 < idx.reward_of("tool_a") < high


def test_reward_persists_through_save_load(tmp_path):
    idx = _idx()
    for _ in range(5):
        idx.record_outcome("tool_a", success=True)
    expected = idx.reward_of("tool_a")
    idx.save(tmp_path / "idx")
    reloaded = CapabilityIndex.load(tmp_path / "idx")
    assert abs(reloaded.reward_of("tool_a") - expected) < 1e-9
