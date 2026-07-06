"""The OPTIONAL zero-copy KV-fork rung on :class:`CrossModalForkFanout`.

CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout + CONCEPT:EG-KG.memory.zero-copy-snapshot-fork.
Proves the two things the rung must guarantee:

* **Default-off / opt-in** — with no ``kv_page_keys`` the fan-out is byte-for-byte the
  existing forkserver copy path: the KV backend is NEVER touched and the result's KV
  fields stay empty.
* **Opt-in plumbing** — with ``kv_page_keys`` supplied the fan-out snapshots the pages
  ONCE, forks one copy-on-write branch per cohort branch, exposes each branch's KV
  branch id to its snippet as ``kv_branch_id``, and lands the fork ids + ``/kv/fork/stats``
  on the result. Degrades to the copy path (never fails the cohort) when the backend is
  unavailable.

A fake in-process KV backend stands in for the engine ``/kv`` fork surface, so these run
without a live engine (the driver's own live roundtrip is covered in
``tests/unit/kvcache/test_kv_fork.py``).
"""

from __future__ import annotations

import multiprocessing

import pytest

from agent_utilities.runtime.crossmodal_fork import CrossModalForkFanout

pytestmark = pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="forkserver start method unavailable on this platform",
)


@pytest.fixture
def clean_registry():
    from agent_utilities.runtime.warm_registry import WarmParentRegistry

    WarmParentRegistry._instance = None  # noqa: SLF001 — test isolation
    yield
    WarmParentRegistry.drain_active()
    WarmParentRegistry._instance = None  # noqa: SLF001


@pytest.fixture
def sandbox():
    from agent_utilities.rlm.sandboxes.forkserver_backend import ForkServerSandbox

    return ForkServerSandbox(preload=())


def _retriever(_query: str) -> list[dict]:
    return [{"id": "n1", "score": 0.9}, {"id": "n2", "score": 0.5}]


class _FakeKvBackend:
    """In-process stand-in for :class:`EpistemicGraphKVBackend`'s fork rung, with call spies."""

    def __init__(self, *, snapshot_id: int | None = 42) -> None:
        self._snapshot_id = snapshot_id
        self.snapshot_calls: list[list[str]] = []
        self.fork_calls: list[int] = []
        self._next_branch = 100
        self.fork_stats_calls = 0

    def snapshot(self, keys):
        self.snapshot_calls.append(list(keys))
        return self._snapshot_id

    def fork(self, snapshot_id):
        self.fork_calls.append(snapshot_id)
        bid = self._next_branch
        self._next_branch += 1
        return bid

    def fork_stats(self):
        self.fork_stats_calls += 1
        # shared_bytes flat regardless of branch count == the zero-copy proof.
        return {
            "branches": len(self.fork_calls),
            "shared_bytes": 4096,
            "shared_pages": 1,
            "overlay_bytes": 0,
        }


# ── default-off: existing behavior is byte-for-byte unchanged ──────────────────
async def test_kv_rung_default_off_does_not_touch_backend(clean_registry, sandbox):
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["FINAL_VAR('out', len(candidates))"] * 3)

    # No kv_page_keys ⇒ the rung stays dormant; the copy path ran exactly as before.
    assert kv.snapshot_calls == []
    assert kv.fork_calls == []
    assert res.kv_snapshot_id is None
    assert res.kv_branch_ids == []
    assert res.kv_fork_stats == {}
    assert res.kv_fork_shared is False
    assert all(b.ok for b in res.branches)
    assert all(b.output == 2 for b in res.branches)


# ── opt-in: snapshot once, fork per branch, surface ids + stats ────────────────
async def test_kv_rung_opt_in_snapshots_once_and_forks_per_branch(
    clean_registry, sandbox
):
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    # Each branch reports the KV branch id it was handed — proving the plumbing reaches
    # the branch namespace as `kv_branch_id`.
    res = await fanout.fan_out(
        "q",
        ["FINAL_VAR('out', kv_branch_id)"] * 3,
        kv_page_keys=["page-a", "page-b"],
    )

    # Snapshot pinned the caller's pages exactly ONCE for the whole cohort.
    assert kv.snapshot_calls == [["page-a", "page-b"]]
    # One CoW branch forked per cohort branch, all off the single snapshot.
    assert kv.fork_calls == [42, 42, 42]

    assert res.kv_snapshot_id == 42
    assert res.kv_branch_ids == [100, 101, 102]
    assert res.kv_fork_stats["shared_bytes"] == 4096
    assert res.kv_fork_shared is True  # forked + shared (Arc'd) bytes > 0

    # Every branch saw its OWN kv_branch_id (no cross-branch leakage).
    got = {b.index: b.output for b in res.branches}
    assert got == {0: 100, 1: 101, 2: 102}


# ── degradation: an unavailable/failed backend falls back to the copy path ─────
async def test_kv_rung_degrades_to_copy_path_when_snapshot_fails(
    clean_registry, sandbox
):
    kv = _FakeKvBackend(snapshot_id=None)  # engine snapshot unavailable
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out(
        "q",
        ["FINAL_VAR('out', len(candidates))"] * 2,
        kv_page_keys=["page-a"],
    )

    assert kv.snapshot_calls == [["page-a"]]
    assert kv.fork_calls == []  # never forked — snapshot failed
    assert res.kv_snapshot_id is None
    assert res.kv_branch_ids == []
    assert res.kv_fork_shared is False
    # The cohort still ran on the copy path (kv_branch_id simply never bound).
    assert all(b.ok for b in res.branches), [b.error for b in res.branches]
    assert all(b.output == 2 for b in res.branches)
