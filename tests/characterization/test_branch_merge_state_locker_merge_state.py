"""Characterization tests for BranchMergeStateLocker.merge_state (CCN 27),
agent_utilities/harness/distributed_state_manager.py.

tests/harness/test_stateful_harness.py already covers the fast-forward path,
the default three-way recursive merge, and a custom resolver's happy path.
This file targets branches not covered there: no-branch-state short circuit,
resolver exceptions, the three sub-cases of the non-recursive default merge
(new key, unchanged-on-branch-side "pass", unchanged-on-base-side overwrite,
and true conflict overwrite), and the fast-forward update_state failure path.

Pins OBSERVED behaviour before a complexity-reduction refactor. Must stay
byte-identical across the refactor commit.
"""

from __future__ import annotations

from agent_utilities.harness.distributed_state_manager import BranchMergeStateLocker


def test_merge_with_no_branch_state_returns_false():
    locker = BranchMergeStateLocker(use_redis=False)
    assert locker.merge_state("nope", "no-such-branch") is False


def test_resolver_exception_is_swallowed_and_merge_fails():
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k1"
    locker.update_state(base_key, {"a": 1}, expected_version=0)
    locker.fork_state(base_key, "b1")
    # Concurrent base change so we hit the resolver path, not fast-forward.
    locker.update_state(base_key, {"a": 2}, expected_version=1)

    def boom(_base, _branch):
        raise RuntimeError("resolver exploded")

    assert locker.merge_state(base_key, "b1", resolver=boom) is False
    # Base state must be untouched (merge failed before update_state).
    assert locker.get_state(base_key)["data"] == {"a": 2}


def test_default_merge_new_branch_key_is_added():
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k2"
    locker.update_state(base_key, {"x": 1}, expected_version=0)
    locker.fork_state(base_key, "b2")
    locker.update_state(base_key, {"x": 1, "y": 2}, expected_version=1)  # concurrent
    locker.update_branch_state(base_key, "b2", {"x": 1, "new_from_branch": "z"})

    assert locker.merge_state(base_key, "b2") is True
    merged = locker.get_state(base_key)["data"]
    assert merged["new_from_branch"] == "z"
    assert merged["y"] == 2


def test_default_merge_branch_value_wins_even_when_branch_did_not_change_it():
    # OBSERVED (not the apparent intent): `merged_data = dict(base_data)`, so for
    # the FIRST (and only) time a key k is processed, `merged_data[k] ==
    # base_data.get(k)` is trivially always true. That makes the first `elif`
    # unconditionally take `merged_data[k] = v` -- the branch's value -- for
    # every scalar key present in both dicts. The `elif v == base_data.get(k):
    # pass` arm and the final `else: merged_data[k] = v` arm are therefore
    # DEAD CODE: unreachable given the loop's own structure. Pinning the
    # actual behaviour (branch always wins for a shared scalar key), not the
    # apparent 3-way-merge intent. See BUGS FOUND in the lane report.
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k3"
    locker.update_state(base_key, {"shared": "orig"}, expected_version=0)
    locker.fork_state(base_key, "b3")
    locker.update_state(base_key, {"shared": "concurrent-change"}, expected_version=1)
    # Branch never modified "shared" -- still "orig". Under the apparent
    # 3-way-merge intent this SHOULD keep "concurrent-change" (base's
    # unilateral change survives an untouched branch key). It does not.
    locker.update_branch_state(base_key, "b3", {"shared": "orig"})

    assert locker.merge_state(base_key, "b3") is True
    merged = locker.get_state(base_key)["data"]
    assert merged["shared"] == "orig"  # branch's value, not base's concurrent change


def test_default_merge_true_conflict_also_prefers_branch_value():
    # Same dead-code shape: both base and branch changed the same key to
    # DIFFERENT values from the original -- the branch value still wins,
    # via the same always-true first `elif`, not the final `else` arm one
    # might expect to be reached for a "true" conflict.
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k4"
    locker.update_state(base_key, {"shared": "orig"}, expected_version=0)
    locker.fork_state(base_key, "b4")
    locker.update_state(base_key, {"shared": "base-changed"}, expected_version=1)
    locker.update_branch_state(base_key, "b4", {"shared": "branch-changed"})

    assert locker.merge_state(base_key, "b4") is True
    merged = locker.get_state(base_key)["data"]
    assert merged["shared"] == "branch-changed"


def test_fast_forward_update_state_race_failure_returns_false(monkeypatch):
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k5"
    locker.update_state(base_key, {"a": 1}, expected_version=0)
    locker.fork_state(base_key, "b5")  # base_version == forked_base_version -> FF path

    # Simulate a concurrent writer winning the race right before update_state.
    monkeypatch.setattr(locker, "update_state", lambda *a, **kw: False)
    assert locker.merge_state(base_key, "b5") is False


def test_conflict_path_update_state_failure_returns_false(monkeypatch):
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k6"
    locker.update_state(base_key, {"a": 1}, expected_version=0)
    locker.fork_state(base_key, "b6")
    locker.update_state(base_key, {"a": 2}, expected_version=1)  # force conflict path
    locker.update_branch_state(base_key, "b6", {"a": 3})

    monkeypatch.setattr(locker, "update_state", lambda *a, **kw: False)
    assert locker.merge_state(base_key, "b6") is False


def test_successful_merge_deletes_the_branch_regardless_of_path():
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "k7"
    locker.update_state(base_key, {"a": 1}, expected_version=0)
    locker.fork_state(base_key, "b7")
    assert locker.merge_state(base_key, "b7") is True  # fast-forward
    assert locker.get_branch_state(base_key, "b7") is None
