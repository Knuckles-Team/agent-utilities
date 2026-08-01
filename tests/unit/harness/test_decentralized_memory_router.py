"""Bandit router integrated into per-agent decentralized memory (KG-2.82/AHE-3.33)."""

from agent_utilities.harness.decentralized_memory import (
    DecentralizedMemory,
    MemoryPool,
)


def test_choose_pool_and_reward_shift_balance():
    mem = DecentralizedMemory()
    # Each agent starts with an untried bandit; both arms get explored first.
    first = mem.choose_pool("agentA")
    assert isinstance(first, MemoryPool)
    # Reward exploitation strongly, exploration poorly, over many rounds.
    for _ in range(50):
        mem.reward("agentA", MemoryPool.EXPLOIT, 1.0)
        mem.reward("agentA", MemoryPool.EXPLORE, 0.0)
    stats = mem.router_stats("agentA")
    assert stats["arms"]["exploit"]["mean"] > stats["arms"]["explore"]["mean"]
    assert mem.choose_pool("agentA") is MemoryPool.EXPLOIT


def test_routers_are_per_agent():
    mem = DecentralizedMemory()
    mem.reward("a", MemoryPool.EXPLOIT, 1.0)
    # agent 'b' has its own independent router (no plays yet).
    assert mem.router_stats("b")["total_plays"] == 0
    assert mem.router_stats("a")["total_plays"] == 1


def test_recall_still_returns_only_own_records_with_router():
    mem = DecentralizedMemory()
    mem.record_trajectory("a", "alpha beta gamma")
    mem.propose_candidate("a", "delta epsilon")
    mem.record_trajectory("b", "private to b")
    # Bias the router toward explore, then recall (both pools still searched).
    for _ in range(20):
        mem.reward("a", MemoryPool.EXPLORE, 1.0)
    out = mem.recall("a", "alpha", top_k=5)
    contents = {r.content for r in out}
    assert "private to b" not in contents  # privacy preserved under routing
    assert contents  # found a's own records
