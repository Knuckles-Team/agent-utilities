"""The DEFAULT-ON zero-copy KV-fork rung on :class:`CrossModalForkFanout`.

CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout + CONCEPT:EG-KG.memory.zero-copy-snapshot-fork.
Proves what the rung must guarantee:

* **Default-on, auto-derived** — with no ``kv_page_keys`` and more than one branch, a
  backend that advertises ``supports_fork()`` gets pages derived from the candidate set
  automatically (content-hashed + stored), and the fan-out snapshots them ONCE, forks one
  copy-on-write branch per cohort branch, exposes each branch's KV branch id to its
  snippet as ``kv_branch_id``, and lands the fork ids + ``/kv/fork/stats`` on the result.
* **Explicit override** — ``kv_page_keys`` supplied by the caller wins over auto-derivation.
* **Explicit opt-out** — ``kv_page_keys=[]`` disables the rung entirely, no auto-derivation
  attempted.
* **No sharing benefit, no attempt** — a single-branch cohort never touches the backend
  (nothing to share).
* **Transparent fallback** — an unavailable backend, one that doesn't advertise fork
  support, or one that can't derive any pages degrades to the copy path; the cohort never
  fails either way.

A fake in-process KV backend stands in for the engine ``/kv`` fork surface, so these run
without a live engine (the driver's own live roundtrip is covered in
``tests/unit/kvcache/test_kv_fork.py``).
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.sandboxes.base import (
    ForkableSandbox,
    ParentHandle,
    SandboxCapabilities,
    SandboxEnv,
    SandboxResult,
    WarmSpec,
)
from agent_utilities.runtime.crossmodal_fork import CrossModalForkFanout


class _TestForkableSandbox(ForkableSandbox):
    name = "test-firecracker"
    capabilities = SandboxCapabilities(
        host_callbacks=True,
        third_party_libs=True,
        classes=True,
        full_stdlib=True,
        network=False,
        isolated=True,
        preference_rank=25,
        warm_fork=True,
    )

    def is_available(self) -> bool:
        return True

    def warm_spec(self) -> WarmSpec:
        return WarmSpec(backend=self.name)

    async def warm(self, spec: WarmSpec) -> ParentHandle:
        return ParentHandle(backend=self.name, spec=spec, ref={"snapshot": "test"})

    async def run_forked(
        self, parent: ParentHandle, code: str, env: SandboxEnv
    ) -> SandboxResult:
        del parent
        if code == "length":
            output = len(env.vars["candidates"])
        elif code == "kv_branch_id":
            output = env.vars["kv_branch_id"]
        else:  # pragma: no cover - test misuse
            return SandboxResult({}, "", f"unknown test operation: {code}")
        env.helpers["FINAL_VAR"]("out", output)
        return SandboxResult(updated_vars={}, stdout="")


@pytest.fixture
def clean_registry():
    from agent_utilities.runtime.warm_registry import WarmParentRegistry

    WarmParentRegistry._instance = None  # noqa: SLF001 — test isolation
    yield
    WarmParentRegistry.drain_active()
    WarmParentRegistry._instance = None  # noqa: SLF001


@pytest.fixture
def sandbox():
    return _TestForkableSandbox()


def _retriever(_query: str) -> list[dict]:
    return [{"id": "n1", "score": 0.9}, {"id": "n2", "score": 0.5}]


class _FakeKvBackend:
    """In-process stand-in for :class:`EpistemicGraphKVBackend`'s fork rung, with call spies."""

    def __init__(
        self, *, snapshot_id: int | None = 42, fork_supported: bool = True
    ) -> None:
        self._snapshot_id = snapshot_id
        self._fork_supported = fork_supported
        self.snapshot_calls: list[list[str]] = []
        self.fork_calls: list[int] = []
        self.put_calls: list[str] = []
        self._next_branch = 100
        self.fork_stats_calls = 0
        self.drop_branch_calls: list[int] = []
        self.release_snapshot_calls: list[int] = []

    def supports_fork(self) -> bool:
        return self._fork_supported

    def put(self, key: str, value: bytes) -> bool:
        self.put_calls.append(key)
        return True

    def snapshot(self, keys):
        self.snapshot_calls.append(list(keys))
        return self._snapshot_id

    def fork(self, snapshot_id):
        self.fork_calls.append(snapshot_id)
        bid = self._next_branch
        self._next_branch += 1
        return bid

    def drop_branch(self, branch_id):
        self.drop_branch_calls.append(branch_id)
        return "released"

    def release_snapshot(self, snapshot_id):
        self.release_snapshot_calls.append(snapshot_id)
        return "released"

    def fork_stats(self):
        self.fork_stats_calls += 1
        # shared_bytes flat regardless of branch count == the zero-copy proof.
        return {
            "branches": len(self.fork_calls),
            "shared_bytes": 4096,
            "shared_pages": 1,
            "overlay_bytes": 0,
        }


class _MinimalFakeKvBackend:
    """A KV backend double with NO ``supports_fork``/``put`` (e.g. a stale test double or a
    minimal remote implementation) — proves the rung degrades gracefully rather than
    assuming every backend exposes the newer capability-probe/derive surface."""

    def __init__(self) -> None:
        self.snapshot_calls: list[list[str]] = []

    def snapshot(self, keys):  # pragma: no cover - never reached (put fails first)
        self.snapshot_calls.append(list(keys))
        return None


# ── default-on: auto-derives pages from the candidate set and forks ────────────
async def test_kv_rung_default_on_auto_derives_and_forks(clean_registry, sandbox):
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    # No kv_page_keys, >1 branch, backend supports fork ⇒ the rung engages automatically.
    res = await fanout.fan_out("q", ["kv_branch_id"] * 3)

    # One content-hash key derived + stored per candidate (2 candidates from _retriever).
    assert len(kv.put_calls) == 2
    assert kv.snapshot_calls == [kv.put_calls]
    assert kv.fork_calls == [42, 42, 42]

    assert res.kv_snapshot_id == 42
    assert res.kv_branch_ids == [100, 101, 102]
    assert res.kv_fork_shared is True

    got = {b.index: b.output for b in res.branches}
    assert got == {0: 100, 1: 101, 2: 102}


# ── single branch: nothing to share, the rung is never attempted ──────────────
async def test_kv_rung_skips_auto_derivation_for_a_single_branch(
    clean_registry, sandbox
):
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["length"])

    assert kv.put_calls == []
    assert kv.snapshot_calls == []
    assert res.kv_snapshot_id is None


# ── explicit opt-out: kv_page_keys=[] disables the rung entirely ──────────────
async def test_kv_rung_explicit_empty_list_opts_out(clean_registry, sandbox):
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["length"] * 3, kv_page_keys=[])

    assert kv.put_calls == []
    assert kv.snapshot_calls == []
    assert res.kv_snapshot_id is None
    assert all(b.ok for b in res.branches)


# ── unsupported backend: capability probe says no, auto-derivation skipped ────
async def test_kv_rung_falls_back_when_backend_does_not_support_fork(
    clean_registry, sandbox
):
    kv = _FakeKvBackend(fork_supported=False)
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["length"] * 3)

    assert kv.put_calls == []  # never even tried to derive pages
    assert kv.snapshot_calls == []
    assert res.kv_snapshot_id is None
    assert all(b.ok for b in res.branches)
    assert all(b.output == 2 for b in res.branches)


# ── minimal backend with no capability surface: degrades, never crashes ───────
async def test_kv_rung_degrades_gracefully_for_a_minimal_backend(
    clean_registry, sandbox
):
    kv = _MinimalFakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["length"] * 3)

    assert kv.snapshot_calls == []  # derivation failed (no .put) before any snapshot
    assert res.kv_snapshot_id is None
    assert all(b.ok for b in res.branches)


# ── explicit override: caller-supplied kv_page_keys wins over auto-derivation ──
async def test_kv_rung_explicit_keys_win_over_auto_derivation(clean_registry, sandbox):
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    # Each branch reports the KV branch id it was handed — proving the plumbing reaches
    # the branch namespace as `kv_branch_id`.
    res = await fanout.fan_out(
        "q",
        ["kv_branch_id"] * 3,
        kv_page_keys=["page-a", "page-b"],
    )

    # No auto-derivation — the caller's own keys were used verbatim.
    assert kv.put_calls == []
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
        ["length"] * 2,
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


# ── D-KVR-2: cohort completion releases every forked branch + the snapshot ─────
async def test_fan_out_releases_kv_fork_resources_on_cohort_completion(
    clean_registry, sandbox
):
    """A fan-out that engaged the KV-fork rung must drop every branch (in fork
    order) THEN release the parent snapshot once the cohort finishes — closing
    the fork/snapshot leak this rung would otherwise create on every call
    (D-KVR-2: the engine's DELETE routes only close the leak if a caller
    actually issues them)."""
    kv = _FakeKvBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["kv_branch_id"] * 3)

    assert res.kv_snapshot_id == 42
    assert res.kv_branch_ids == [100, 101, 102]
    # Branches dropped first (parent snapshot release would 409 otherwise),
    # in the same order they were forked.
    assert kv.drop_branch_calls == [100, 101, 102]
    assert kv.release_snapshot_calls == [42]


async def test_fan_out_skips_kv_release_when_rung_never_engaged(
    clean_registry, sandbox
):
    """No snapshot/branches were forked (rung off) ⇒ no release calls at all."""
    kv = _FakeKvBackend(fork_supported=False)
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    res = await fanout.fan_out("q", ["length"] * 3)

    assert res.kv_snapshot_id is None
    assert kv.drop_branch_calls == []
    assert kv.release_snapshot_calls == []


async def test_fan_out_release_relies_on_the_connectors_own_no_raise_contract(
    clean_registry, sandbox
):
    """`_release_kv_fork_resources` does not wrap its calls in try/except — it
    relies on `EpistemicGraphKVBackend.release_snapshot`/`drop_branch` already
    honoring the connector's graceful-degradation contract (every transport/
    protocol error maps to `"error"`, never a raise). This proves that contract
    is load-bearing: a backend that violates it (raises instead of degrading)
    is NOT silently swallowed — the exception propagates from the `finally`
    block rather than being caught-and-hidden, and cleanup was still attempted
    (the branch appears in `drop_branch_calls`) before it propagated."""

    class _FlakyReleaseBackend(_FakeKvBackend):
        def drop_branch(self, branch_id):
            self.drop_branch_calls.append(branch_id)
            raise RuntimeError("engine unreachable during cleanup")

    kv = _FlakyReleaseBackend()
    fanout = CrossModalForkFanout(retriever=_retriever, sandbox=sandbox, kv_backend=kv)

    with pytest.raises(RuntimeError, match="engine unreachable during cleanup"):
        await fanout.fan_out("q", ["kv_branch_id"] * 3)
    assert kv.drop_branch_calls == [100]


def test_kv_backend_is_resolved_once_per_process_not_per_fanout(monkeypatch):
    """The lazily-built KV backend must NOT be rebuilt per fanout instance.

    Regression (waves 1-5 gate): the backend was cached on ``self``, but
    ``CrossModalForkFanout()`` is constructed fresh on every ``graph_fork`` MCP
    call, and ``EpistemicGraphKVBackend.from_env()`` owns a pooled ``httpx.Client``
    (``max_connections=32``) that nothing ever closes. Once this rung became
    default-on for every >1-branch cohort, that leaked a brand-new keep-alive
    connection pool per call. It also defeated ``supports_fork``'s own probe cache,
    whose docstring promises at most one round trip.
    """
    import agent_utilities.runtime.crossmodal_fork as cmf

    monkeypatch.setattr(cmf, "_SHARED_KV_BACKEND", None, raising=False)
    monkeypatch.setattr(cmf, "_SHARED_KV_BACKEND_RESOLVED", False, raising=False)

    builds: list[int] = []

    class _Backend:
        pass

    class _Module:
        @staticmethod
        def from_env():
            builds.append(1)
            return _Backend()

    import sys
    import types

    fake = types.ModuleType("agent_utilities.kvcache")
    fake.EpistemicGraphKVBackend = _Module
    monkeypatch.setitem(sys.modules, "agent_utilities.kvcache", fake)

    first = cmf.CrossModalForkFanout()._resolve_kv_backend()
    second = cmf.CrossModalForkFanout()._resolve_kv_backend()
    third = cmf.CrossModalForkFanout()._resolve_kv_backend()

    assert builds == [1], f"backend rebuilt per fanout instance: {len(builds)} builds"
    assert first is second is third


def test_unreachable_engine_is_resolved_once_and_stays_the_copy_path(monkeypatch):
    """A failed resolve must also be remembered — not retried on every call."""
    import agent_utilities.runtime.crossmodal_fork as cmf

    monkeypatch.setattr(cmf, "_SHARED_KV_BACKEND", None, raising=False)
    monkeypatch.setattr(cmf, "_SHARED_KV_BACKEND_RESOLVED", False, raising=False)

    attempts: list[int] = []

    class _Module:
        @staticmethod
        def from_env():
            attempts.append(1)
            raise RuntimeError("no engine")

    import sys
    import types

    fake = types.ModuleType("agent_utilities.kvcache")
    fake.EpistemicGraphKVBackend = _Module
    monkeypatch.setitem(sys.modules, "agent_utilities.kvcache", fake)

    assert cmf.CrossModalForkFanout()._resolve_kv_backend() is None
    assert cmf.CrossModalForkFanout()._resolve_kv_backend() is None
    assert attempts == [1]


def test_injected_backend_still_wins_over_the_process_wide_one(monkeypatch):
    import agent_utilities.runtime.crossmodal_fork as cmf

    sentinel = object()
    monkeypatch.setattr(cmf, "_SHARED_KV_BACKEND", object(), raising=False)
    monkeypatch.setattr(cmf, "_SHARED_KV_BACKEND_RESOLVED", True, raising=False)

    fanout = cmf.CrossModalForkFanout(kv_backend=sentinel)
    assert fanout._resolve_kv_backend() is sentinel
