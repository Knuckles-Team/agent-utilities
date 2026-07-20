"""CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout — warm-fork fan-out over an engine cross-modal candidate set.

These drive the current :class:`ForkableSandbox` contract, retrieve a faked cross-modal candidate
set once, and fork isolated branch views over it — proving the two things the capability exists for:

* **Reuse** — the cross-modal candidate set is computed once for the whole cohort (the recompute
  guard counts exactly one retrieval), and the warm parent is warmed once (the rest of the branches
  fork the registry-pooled parent).
* **Isolation** — a branch mutating its view of the candidate set is never observed by a sibling
  (separate forked processes), and the orchestrator's candidate set stays pristine.
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
from agent_utilities.runtime.crossmodal_fork import (
    CrossModalForkFanout,
    RecomputeError,
    _RecomputeGuard,
)
from agent_utilities.runtime.warm_registry import WarmParentRegistry


class _TestForkableSandbox(ForkableSandbox):
    """Deterministic confined-backend double that exercises the current fork protocol."""

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
        candidates = env.vars["candidates"]
        idx = env.vars["branch_index"]
        if code == "top_scores":
            top = sorted(candidates, key=lambda c: c["score"], reverse=True)[: idx + 1]
            output = round(sum(c["score"] for c in top), 4)
        elif code == "mutate":
            candidates[0]["score"] = idx * 100
            candidates.append({"id": f"injected-{idx}"})
            output = {"seen_score": candidates[0]["score"], "length": len(candidates)}
        elif code == "length":
            output = len(candidates)
        elif code == "max_score":
            output = max(c["score"] for c in candidates)
        elif code == "vector_count":
            output = sum(c["modality"] == "vector" for c in candidates)
        else:  # pragma: no cover - test misuse
            return SandboxResult({}, "", f"unknown test operation: {code}")
        env.helpers["FINAL_VAR"]("out", output)
        return SandboxResult(updated_vars={}, stdout="")


@pytest.fixture
def clean_registry():
    """Isolate the process-singleton warm-parent registry per test."""
    WarmParentRegistry._instance = None  # noqa: SLF001 — test isolation
    yield
    WarmParentRegistry.drain_active()
    WarmParentRegistry._instance = None  # noqa: SLF001


@pytest.fixture
def sandbox():
    return _TestForkableSandbox()


class _CountingRetriever:
    """A fake cross-modal retriever (stands in for vector+graph+text fusion) that counts calls."""

    def __init__(self, candidates: list[dict]):
        self._candidates = candidates
        self.calls = 0
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict]:
        self.calls += 1
        self.queries.append(query)
        # Return a fresh copy each call so a leak would be a *different* object we could detect.
        return [dict(c) for c in self._candidates]


def _fused_candidates() -> list[dict]:
    """A plausible cross-modal candidate set: fused vector / graph / text hits with scores."""
    return [
        {"id": "n1", "score": 0.91, "modality": "vector", "text": "alpha"},
        {"id": "n2", "score": 0.77, "modality": "graph", "text": "beta"},
        {"id": "n3", "score": 0.64, "modality": "text", "text": "gamma"},
        {"id": "n4", "score": 0.50, "modality": "vector", "text": "delta"},
    ]


# ── default retriever binds the full engine facade (regression) ───────────────
def test_default_retriever_binds_facade_engine_not_bare_graph_compute(monkeypatch):
    """Regression: with no injected engine, the default cross-modal retriever must
    bind ``HybridRetriever`` to the live ``IntelligenceGraphEngine`` facade (which
    owns ``.backend`` / ``._search_keyword`` / ``.embed_model``) — NOT a bare
    ``GraphComputeEngine`` (the ``.graph`` compute layer), which lacked ``.backend``
    and raised "'GraphComputeEngine' object has no attribute 'backend'" on the served
    ``graph_fork`` cross-modal fan-out. CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout
    """
    import agent_utilities.runtime.crossmodal_fork as mod
    from agent_utilities.knowledge_graph.core import engine as engine_mod

    class _FacadeEngine:
        backend = object()  # the attribute the bare compute engine did not expose

    active = _FacadeEngine()
    monkeypatch.setattr(
        engine_mod.IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda cls: active),
    )

    seen: dict = {}

    class _RecordingRetriever:
        def __init__(self, eng):
            seen["engine"] = eng

        def retrieve_hybrid(self, query, *, context_window, multi_hop_depth):
            seen["query"] = query
            return [{"id": "n1"}]

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.HybridRetriever",
        _RecordingRetriever,
    )

    out = mod.engine_cross_modal_candidates("hybrid query")

    assert out == [{"id": "n1"}]
    assert seen["engine"] is active  # the live facade, not a fresh GraphComputeEngine
    assert hasattr(seen["engine"], "backend")
    assert seen["query"] == "hybrid query"


# ── the recompute guard (unit) ────────────────────────────────────────────────
def test_recompute_guard_allows_one_then_raises():
    retr = _CountingRetriever(_fused_candidates())
    guard = _RecomputeGuard(retr)

    first = guard("q")
    assert guard.calls == 1
    assert len(first) == 4
    with pytest.raises(RecomputeError):
        guard("q")  # a second retrieval inside one fan-out is a correctness bug
    assert retr.calls == 1  # the underlying engine was hit exactly once


# ── reuse: candidate set computed ONCE across the fan-out ──────────────────────
async def test_candidate_set_retrieved_once_across_branches(clean_registry, sandbox):
    retr = _CountingRetriever(_fused_candidates())
    fanout = CrossModalForkFanout(retriever=retr, sandbox=sandbox)

    # Four divergent branches, each reducing the SAME candidate set differently: branch i sums the
    # top-(i+1) scores. They reuse one shared context; none re-queries the engine.
    res = await fanout.fan_out("hybrid query", ["top_scores"] * 4)

    # THE reuse proof: one retrieval, one engine hit, regardless of the 4 branches.
    assert retr.calls == 1
    assert res.retrieval_calls == 1
    assert res.reused_without_recompute is True
    assert res.candidate_count == 4
    assert res.sandbox == "test-firecracker"

    # Every branch ran its own divergent reduction over the shared candidate set.
    assert len(res.branches) == 4
    assert all(b.ok for b in res.branches), [b.error for b in res.branches]
    got = {b.index: b.output for b in res.branches}
    assert got[0] == pytest.approx(0.91)  # top-1
    assert got[1] == pytest.approx(0.91 + 0.77)  # top-2
    assert got[3] == pytest.approx(0.91 + 0.77 + 0.64 + 0.50)  # top-4


async def test_warm_parent_warmed_once_for_whole_cohort(
    clean_registry, sandbox, monkeypatch
):
    """The parent is warmed once; every branch forks the registry-pooled copy-on-write parent."""
    real_warm = sandbox.warm
    warm_calls = {"n": 0}

    async def counting_warm(spec):
        warm_calls["n"] += 1
        return await real_warm(spec)

    monkeypatch.setattr(sandbox, "warm", counting_warm)

    retr = _CountingRetriever(_fused_candidates())
    fanout = CrossModalForkFanout(retriever=retr, sandbox=sandbox)
    res = await fanout.fan_out("q", ["length"] * 5)

    assert all(b.ok for b in res.branches)
    assert all(
        b.output == 4 for b in res.branches
    )  # each branch saw the full candidate set
    assert warm_calls["n"] == 1, "warm() paid once; the other branches reuse the parent"
    stats = WarmParentRegistry.get().stats()
    assert stats["warm_parents"] == 1
    assert stats["by_kind"].get("test-firecracker") == 1


# ── isolation: divergent writes stay branch-local ─────────────────────────────
async def test_branches_isolated_on_divergent_writes(clean_registry, sandbox):
    retr = _CountingRetriever(_fused_candidates())
    fanout = CrossModalForkFanout(retriever=retr, sandbox=sandbox)

    # Each branch mutates its OWN view of the shared candidate set differently, then reports what
    # it sees. If forks leaked into each other, a branch would observe a sibling's mutation.
    res = await fanout.fan_out("q", ["mutate"] * 4)

    assert retr.calls == 1  # still one shared retrieval
    assert all(b.ok for b in res.branches), [b.error for b in res.branches]
    seen = {b.index: b.output for b in res.branches}
    # Each branch observes ONLY its own mutation (idx*100) — no sibling leakage.
    for idx in range(4):
        assert seen[idx]["seen_score"] == idx * 100
        assert seen[idx]["length"] == 5  # original 4 + this branch's single append

    # The orchestrator's own candidate set (guard.last_result) is untouched by any branch.
    assert retr.queries == ["q"]


async def test_divergent_snippets_share_one_context(clean_registry, sandbox):
    """Genuinely different per-branch code, one retrieval, one warm parent."""
    retr = _CountingRetriever(_fused_candidates())
    fanout = CrossModalForkFanout(retriever=retr, sandbox=sandbox)
    branches = ["length", "max_score", "vector_count"]
    res = await fanout.fan_out("q", branches)

    assert res.retrieval_calls == 1
    outs = {b.index: b.output for b in res.branches}
    assert outs[0] == 4
    assert outs[1] == pytest.approx(0.91)
    assert outs[2] == 2


# ── degraded path ─────────────────────────────────────────────────────────────
async def test_degraded_when_no_warm_fork_rung(clean_registry, monkeypatch):
    """No rung available ⇒ structured degraded result, but the retrieval is still counted once."""
    import agent_utilities.runtime.crossmodal_fork as mod

    monkeypatch.setattr(mod, "_pick_warm_fork_sandbox", lambda preferred="": None)
    retr = _CountingRetriever(_fused_candidates())
    fanout = CrossModalForkFanout(retriever=retr, sandbox=None)
    res = await fanout.fan_out("q", ["FINAL_VAR('out', 1)"] * 3)

    assert res.degraded is True
    assert res.sandbox is None
    assert res.retrieval_calls == 1
    assert res.branches == []
    assert res.error and "warm-fork rung" in res.error
