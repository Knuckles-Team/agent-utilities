#!/usr/bin/python
"""Tests for the prioritized replay buffer (MEMO b4-03 F4).

CONCEPT:AHE-3.0
"""

import pytest

from agent_utilities.harness.replay_buffer import PrioritizedReplayBuffer

pytestmark = pytest.mark.concept("AHE-3.0")


def test_add_and_len():
    buf = PrioritizedReplayBuffer()
    a = buf.add({"k": "a"}, key="a")
    buf.add({"k": "b"}, key="b")
    assert len(buf) == 2
    assert a.key == "a"


def test_inverse_frequency_priority():
    buf = PrioritizedReplayBuffer(alpha=1.0)
    ra = buf.add({"k": "a"}, key="a")
    rb = buf.add({"k": "b"}, key="b")
    assert buf.priority(ra) == pytest.approx(buf.priority(rb))  # both unseen
    buf._key_visits["a"] = 3  # 'a' visited more → lower priority
    assert buf.priority(ra) < buf.priority(rb)


def test_sample_is_seed_faithful():
    def fresh():
        b = PrioritizedReplayBuffer()
        for k in ("a", "b", "c", "d"):
            b.add({"k": k}, key=k)
        return b

    s1 = fresh().sample(3, seed=7)
    s2 = fresh().sample(3, seed=7)
    assert s1 == s2  # deterministic for a fixed seed
    assert len(s1) == 3


def test_sample_without_replacement_within_call():
    buf = PrioritizedReplayBuffer()
    for k in ("a", "b", "c"):
        buf.add({"k": k}, key=k)
    out = buf.sample(3, seed=1)
    keys = [o["k"] for o in out]
    assert sorted(keys) == ["a", "b", "c"]  # all distinct, no repeats


def test_sample_increments_visits_and_shifts_priority():
    buf = PrioritizedReplayBuffer(alpha=1.0)
    ra = buf.add({"k": "a"}, key="a")
    buf.add({"k": "b"}, key="b")
    buf.sample(1, seed=0)  # picks one, increments its key's visits
    # at least one key now has a visit recorded
    assert sum(buf._key_visits.values()) == 1
    # the sampled key's priority dropped below the unseen one's
    assert buf.priority(ra) <= 1.0


def test_capacity_evicts_oldest_on_tie():
    buf = PrioritizedReplayBuffer(capacity=2, alpha=1.0)
    for k in ("a", "b", "c"):  # equal priority (all unseen) → keep newest 2
        buf.add({"k": k}, key=k)
    assert len(buf) == 2
    kept = {it.key for it in buf._items}
    assert kept == {"b", "c"}  # oldest ('a') evicted


def test_empty_sample():
    buf = PrioritizedReplayBuffer()
    assert buf.sample(3, seed=0) == []
    assert buf.stats()["size"] == 0
