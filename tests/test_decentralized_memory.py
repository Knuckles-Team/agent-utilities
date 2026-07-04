#!/usr/bin/python
"""Unit tests for decentralized-but-collaborative agent memory (CONCEPT:AU-KG.memory.ahe-record-this-base).

Covers DecentMem (arXiv:2605.22721) behaviour: per-agent privacy, exploit/explore
pool separation, promotion explore->exploit, and the collaboration-aware trace.
Fully deterministic — no LLM, no network (engine=None, embedder=None).
"""

from __future__ import annotations

from agent_utilities.harness.decentralized_memory import (
    Contribution,
    DecentralizedMemory,
    MemoryPool,
)
from agent_utilities.harness.evolving_memory import MemoryBank, MemoryRecord


def test_record_trajectory_lands_in_exploit_pool() -> None:
    mem = DecentralizedMemory()
    rec = mem.record_trajectory("alice", "retry failed http calls with backoff")
    assert isinstance(rec, MemoryRecord)
    assert rec.bank is MemoryBank.SKILL
    assert rec.metadata["agent_id"] == "alice"
    assert rec.metadata["pool"] == MemoryPool.EXPLOIT.value
    assert mem.pool_size("alice", MemoryPool.EXPLOIT) == 1
    assert mem.pool_size("alice", MemoryPool.EXPLORE) == 0


def test_propose_candidate_lands_in_explore_pool() -> None:
    mem = DecentralizedMemory()
    rec = mem.propose_candidate("alice", "maybe cache the embedder per cycle")
    assert rec.bank is MemoryBank.INSIGHT
    assert rec.metadata["pool"] == MemoryPool.EXPLORE.value
    assert mem.pool_size("alice", MemoryPool.EXPLORE) == 1
    assert mem.pool_size("alice", MemoryPool.EXPLOIT) == 0


def test_per_agent_isolation_recall() -> None:
    mem = DecentralizedMemory()
    mem.record_trajectory("alice", "alice secret strategy about widgets")
    mem.record_trajectory("bob", "bob secret strategy about gadgets")

    alice_hits = mem.recall("alice", "secret strategy")
    bob_hits = mem.recall("bob", "secret strategy")

    assert alice_hits and bob_hits
    for rec in alice_hits:
        assert rec.metadata["agent_id"] == "alice"
        assert "alice" in rec.content
    for rec in bob_hits:
        assert rec.metadata["agent_id"] == "bob"
        assert "bob" in rec.content

    # Alice can never see bob's record and vice versa.
    alice_contents = {r.content for r in alice_hits}
    assert "bob secret strategy about gadgets" not in alice_contents


def test_recall_unknown_agent_is_empty() -> None:
    mem = DecentralizedMemory()
    mem.record_trajectory("alice", "something")
    assert mem.recall("nobody", "something") == []


def test_recall_scoped_to_single_pool() -> None:
    mem = DecentralizedMemory()
    mem.record_trajectory("alice", "proven approach to parsing logs")
    mem.propose_candidate("alice", "candidate approach to parsing logs")

    exploit_only = mem.recall("alice", "parsing logs", pool=MemoryPool.EXPLOIT)
    explore_only = mem.recall("alice", "parsing logs", pool=MemoryPool.EXPLORE)
    both = mem.recall("alice", "parsing logs")

    assert all(r.metadata["pool"] == "exploit" for r in exploit_only)
    assert all(r.metadata["pool"] == "explore" for r in explore_only)
    assert len(both) == 2


def test_promote_moves_explore_to_exploit() -> None:
    mem = DecentralizedMemory()
    cand = mem.propose_candidate("alice", "promote me once proven")
    assert mem.pool_size("alice", MemoryPool.EXPLORE) == 1
    assert mem.pool_size("alice", MemoryPool.EXPLOIT) == 0

    promoted = mem.promote("alice", cand.id)

    assert promoted is not None
    assert promoted.content == "promote me once proven"
    assert promoted.metadata["pool"] == MemoryPool.EXPLOIT.value
    assert promoted.metadata["promoted_from"] == cand.id

    # Exploit grew, explore record retired (no longer active).
    assert mem.pool_size("alice", MemoryPool.EXPLOIT) == 1
    assert mem.pool_size("alice", MemoryPool.EXPLORE) == 0

    explore_store = mem._stores[("alice", MemoryPool.EXPLORE)]
    assert explore_store.get(cand.id).status == "retired"


def test_promote_missing_returns_none() -> None:
    mem = DecentralizedMemory()
    assert mem.promote("alice", "mem:insight:doesnotexist") is None
    mem.propose_candidate("alice", "real one")
    assert mem.promote("alice", "mem:insight:stillnope") is None


def test_promote_twice_is_idempotent_noop_second_time() -> None:
    mem = DecentralizedMemory()
    cand = mem.propose_candidate("alice", "promote once")
    first = mem.promote("alice", cand.id)
    second = mem.promote("alice", cand.id)
    assert first is not None
    assert second is None  # already retired in explore pool
    assert mem.pool_size("alice", MemoryPool.EXPLOIT) == 1


def test_collaboration_trace_records_and_filters() -> None:
    mem = DecentralizedMemory()
    c1 = mem.record_contribution("alice", "task-1", "planner", "drew up the plan")
    mem.record_contribution("bob", "task-1", "coder", "implemented step 2")
    mem.record_contribution("alice", "task-2", "reviewer", "reviewed the diff")

    assert isinstance(c1, Contribution)
    assert c1.agent_id == "alice"

    all_entries = mem.collaboration_trace()
    assert len(all_entries) == 3
    # Insertion order preserved.
    assert [c.task_id for c in all_entries] == ["task-1", "task-1", "task-2"]

    task1 = mem.collaboration_trace("task-1")
    assert len(task1) == 2
    assert {c.agent_id for c in task1} == {"alice", "bob"}
    assert {c.role for c in task1} == {"planner", "coder"}

    assert mem.collaboration_trace("missing") == []


def test_agents_lists_known_agents() -> None:
    mem = DecentralizedMemory()
    assert mem.agents() == []
    mem.record_trajectory("alice", "x")
    mem.propose_candidate("bob", "y")
    mem.record_trajectory("alice", "z")  # alice again, both pools touched
    assert set(mem.agents()) == {"alice", "bob"}


def test_total_size_across_agents_and_pools() -> None:
    mem = DecentralizedMemory()
    assert mem.total_size() == 0
    mem.record_trajectory("alice", "a1")
    mem.record_trajectory("alice", "a2")
    mem.propose_candidate("alice", "a3 candidate")
    mem.record_trajectory("bob", "b1")
    assert mem.total_size() == 4
    assert mem.pool_size("alice", MemoryPool.EXPLOIT) == 2
    assert mem.pool_size("alice", MemoryPool.EXPLORE) == 1
    assert mem.pool_size("bob", MemoryPool.EXPLOIT) == 1


def test_lazy_store_creation_no_phantom_agents() -> None:
    mem = DecentralizedMemory()
    # Reading sizes for an untouched agent must not create stores or list them.
    assert mem.pool_size("ghost", MemoryPool.EXPLOIT) == 0
    assert mem.agents() == []
