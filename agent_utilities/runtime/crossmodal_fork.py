"""CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout — warm-fork fan-out over an engine cross-modal candidate set.

The agent-utilities side of the epistemic-graph cross-modal seam (was engine spec EG-397).
The warm-fork primitive (CONCEPT:AU-ORCH.sandbox.shared-host-helper-bridge /
:class:`~agent_utilities.rlm.sandboxes.base.ForkableSandbox` + the host
:class:`~agent_utilities.runtime.warm_registry.WarmParentRegistry`, ORCH-1.86..93) lives here,
not in the engine — so *consuming* the engine's cross-modal context is an agent-utilities job.

The capability, end to end:

1. **Retrieve once.** A cross-modal candidate set (vector + graph + text fusion over the
   epistemic-graph engine) is retrieved for a query ONE time, through a
   :class:`_RecomputeGuard` that fails loudly if anything tries to re-query. This is the
   expensive step the whole design exists to amortise.
2. **Warm one parent.** A single warm-fork parent is warmed (its heavy imports/deps paid once,
   copy-on-write resident) via the shared :class:`WarmParentRegistry`, and the candidate set is
   handed to it as forked-in context.
3. **Fork N copy-on-write branches.** Each branch reuses *that one* candidate set — no branch
   re-queries the engine — and runs its own divergent computation over it concurrently. Because
   each fork is a separate process that receives its own copy of the candidate set, a branch
   mutating its view can never leak into a sibling (structural isolation).

Reuse proof: :attr:`CrossModalForkResult.retrieval_calls` is ``1`` regardless of the branch
count (the guard would raise on a second retrieval). Isolation proof: divergent per-branch
mutations of the candidate set are observed only by the mutating branch.

Honest scope of the local rung: the ``forkserver`` backend shares *loaded modules* copy-on-write
through ``os.fork``; the candidate-set *data* is serialised into each child over the bridge
(``context.json``), so it is materialised once in the orchestrator but copied per child. A live
KV-cache-fork rung (LMCacheMPConnector snapshot → branch, the vLLM path in the KV-cache memory
note) additionally shares the candidate set's KV / embedding pages as copy-on-write memory,
making the data-sharing itself zero-copy — the same :class:`ForkableSandbox` seam, a stronger
backend. This module is backend-agnostic: it drives whichever warm-fork rung is available.

The zero-copy rung now EXISTS engine-side (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork): the
``epistemic-graph`` ``eg-kvcache`` crate implements the "snapshot → branch" primitive on its
content-addressed shared KV store — ``SharedKvIndex::snapshot(keys)`` pins the candidate-set
pages, ``fork(snapshot)`` fans out N branches that all read those SAME physical pages by ``Arc``
(one copy regardless of N), and ``branch_put`` is copy-on-write per branch — exposed over the
KV-cache HTTP surface as ``POST /kv/snapshot`` + ``POST /kv/snapshot/<id>/fork`` +
``GET|PUT /kv/branch/<bid>/<key>`` (``GET /kv/fork/stats`` proves resident bytes stay flat vs
branch count). This is the rung the ``max_concurrency>1`` path can now target to make fan-out
O(1) in copies instead of the forkserver's per-branch serialize-and-copy (18–43 ms/branch in the
phase-2 benchmark). Still external: the vLLM/LMCache connector that maps live attention KV pages
onto this store — the engine provides the zero-copy substrate, not the model-side page mapping.

:class:`CrossModalForkFanout` now *drives* that substrate as an OPT-IN rung: pass
:meth:`CrossModalForkFanout.fan_out`'s ``kv_page_keys`` (the engine KV-page keys of the retrieved
context) and it snapshots+forks them so the branches share those pages zero-copy. It is DEFAULT-OFF
— omit ``kv_page_keys`` and the fan-out is byte-for-byte the forkserver copy path. And it wires only
the SHARING/plumbing of pages that already live in the store; producing the pages (mapping live
attention KV onto the store) remains the external vLLM/LMCache job, so an absent/unreachable KV
backend degrades to the copy path rather than failing the cohort.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# A cross-modal retriever: query -> list of candidate records (each a JSON-able dict, e.g. the
# fused vector+graph+text hits :meth:`HybridRetriever.retrieve_hybrid` returns).
CrossModalRetriever = Callable[[str], list[dict[str, Any]]]


class RecomputeError(RuntimeError):
    """Raised when a single-shot cross-modal retrieval is invoked more than once.

    The whole point of warm-fork fan-out is that the candidate set is computed ONCE and reused
    across the cohort. A second retrieval inside one fan-out is a correctness bug (a branch
    re-querying the engine instead of reusing the parent context), so the guard refuses it rather
    than silently paying the cost N times.
    """


@dataclass
class _RecomputeGuard:
    """Wrap a :data:`CrossModalRetriever` so it can be invoked exactly once per fan-out.

    ``calls`` is the reuse counter the result surfaces: after fanning out over N branches it MUST
    still read ``1``. When ``single_shot`` (the default) a second call raises
    :class:`RecomputeError` — turning "a branch recomputed the candidate set" from a silent
    performance regression into a hard failure.
    """

    retriever: CrossModalRetriever
    single_shot: bool = True
    calls: int = 0
    last_result: list[dict[str, Any]] | None = None

    def __call__(self, query: str) -> list[dict[str, Any]]:
        if self.calls and self.single_shot:
            raise RecomputeError(
                f"cross-modal candidate set already retrieved once for this fan-out "
                f"(calls={self.calls}); branches must reuse the warm parent's context, "
                f"not re-query the engine"
            )
        self.calls += 1
        result = list(self.retriever(query))
        self.last_result = result
        return result


@dataclass
class CrossModalBranchResult:
    """Outcome of one forked branch's divergent computation over the shared candidate set."""

    index: int
    ok: bool
    stdout: str = ""
    error: str | None = None
    output: Any = None
    """Whatever the branch reported via the ``FINAL_VAR`` host helper (its divergent result)."""


@dataclass
class CrossModalForkResult:
    """The fan-out outcome + the proofs that make it a warm-fork win, not N cold recomputes."""

    query: str
    candidate_count: int
    retrieval_calls: int
    sandbox: str | None
    branches: list[CrossModalBranchResult] = field(default_factory=list)
    degraded: bool = False
    error: str | None = None
    # -- optional zero-copy KV-fork rung (default-off; populated only when the
    #    caller passes ``kv_page_keys`` AND the snapshot→fork plumbing succeeds).
    kv_snapshot_id: int | None = None
    """Engine snapshot id the branches' KV pages were pinned into, or ``None`` (rung off/unavailable)."""
    kv_branch_ids: list[int | None] = field(default_factory=list)
    """Per-branch copy-on-write KV branch id (index-aligned to ``branches``); ``None`` per entry on failure."""
    kv_fork_stats: dict[str, Any] = field(default_factory=dict)
    """``GET /kv/fork/stats`` snapshot taken after forking — ``shared_*`` flat vs branch count is the zero-copy proof."""

    @property
    def reused_without_recompute(self) -> bool:
        """True iff the cross-modal candidate set was retrieved exactly once for the whole cohort."""
        return self.retrieval_calls == 1

    @property
    def kv_fork_shared(self) -> bool:
        """True iff the KV-fork rung engaged and its branches share pages zero-copy.

        Reads the ``/kv/fork/stats`` counters: at least one forked branch id and a
        positive ``shared_bytes`` (pages read by ``Arc`` across branches, not copied).
        """
        return bool(
            self.kv_snapshot_id is not None
            and any(b is not None for b in self.kv_branch_ids)
            and int(self.kv_fork_stats.get("shared_bytes", 0)) > 0
        )


def _pick_warm_fork_sandbox(preferred: str = "") -> Any | None:
    """Cheapest available warm-fork rung from the ORCH-1.86 registry, or ``None``.

    Mirrors the MCP-side picker (``engine_surface_tools._pick_warm_fork_sandbox``) but lives in
    the runtime layer so the capability has no MCP dependency. ``preferred`` pins a rung by name.
    """
    try:
        from agent_utilities.rlm.sandboxes.registry import default_sandboxes
    except Exception:  # noqa: BLE001 — subsystem unimportable ⇒ no rung
        return None

    forkable = [
        sb
        for sb in default_sandboxes()
        if getattr(getattr(sb, "capabilities", None), "warm_fork", False)
    ]
    if preferred:
        forkable = [sb for sb in forkable if sb.name == preferred] or forkable
    for sb in forkable:
        try:
            if sb.is_available():
                return sb
        except Exception:  # noqa: BLE001 — an unprobeable rung is skipped
            continue
    return None


def engine_cross_modal_candidates(
    query: str,
    *,
    engine: Any | None = None,
    context_window: int = 10,
    multi_hop_depth: int = 2,
) -> list[dict[str, Any]]:
    """Default cross-modal retriever: the engine's vector+graph+text fusion for ``query``.

    Thin binding onto :meth:`HybridRetriever.retrieve_hybrid` — the epistemic-graph cross-modal
    fusion arm. Constructed lazily so importing this module never requires a live engine; callers
    without an engine inject their own retriever instead (tests do exactly this).
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    # ``HybridRetriever`` reads the full engine facade — ``.backend`` (Cypher
    # execute), ``._search_keyword``, ``.embed_model`` — which only
    # ``IntelligenceGraphEngine`` exposes. A bare ``GraphComputeEngine`` is the
    # low-level ``.graph`` compute layer and lacks ``.backend``, so passing it
    # raised "'GraphComputeEngine' object has no attribute 'backend'" and broke the
    # served ``graph_fork`` cross-modal fan-out. Prefer the live singleton (the same
    # engine graph-os serves), building one only when none is active.
    eng = (
        engine
        if engine is not None
        else (IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine())
    )
    retriever = HybridRetriever(eng)
    return retriever.retrieve_hybrid(
        query, context_window=context_window, multi_hop_depth=multi_hop_depth
    )


class CrossModalForkFanout:
    """Retrieve an engine cross-modal candidate set once, then warm-fork N branches over it.

    ``retriever`` is the engine seam (defaults to :func:`engine_cross_modal_candidates`, the
    vector+graph+text fusion); inject a callable to run without a live engine. ``sandbox`` pins a
    :class:`~agent_utilities.rlm.sandboxes.base.ForkableSandbox`; when unset the cheapest
    available warm-fork rung is auto-selected.

    ``kv_backend`` is the OPTIONAL zero-copy KV-fork rung
    (:class:`~agent_utilities.kvcache.EpistemicGraphKVBackend`, CONCEPT:EG-KG.memory.zero-copy-snapshot-fork).
    It is only engaged when a caller passes ``kv_page_keys`` to :meth:`fan_out`; leave it ``None``
    and the fan-out behaves EXACTLY as before (the local forkserver copy path). Inject one to reuse
    a pooled connector, or leave it ``None`` and one is lazily built from the KV-cache env when the
    rung is actually requested.
    """

    def __init__(
        self,
        retriever: CrossModalRetriever | None = None,
        sandbox: Any | None = None,
        kv_backend: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._sandbox = sandbox
        self._kv_backend = kv_backend

    def _resolve_retriever(self) -> CrossModalRetriever:
        if self._retriever is not None:
            return self._retriever
        return engine_cross_modal_candidates

    def _resolve_kv_backend(self) -> Any | None:
        """Return the injected KV-fork backend, else lazily build one from the KV-cache env.

        Lazy + guarded: importing this module must never pull ``httpx`` / the kvcache
        stack, and a host without a reachable engine must degrade to the copy path
        (``None``) rather than crash. Only called when the rung is actually requested.
        """
        if self._kv_backend is not None:
            return self._kv_backend
        try:
            from agent_utilities.kvcache import EpistemicGraphKVBackend

            self._kv_backend = EpistemicGraphKVBackend.from_env()
        except Exception as exc:  # noqa: BLE001 — no engine / no deps ⇒ copy path
            logger.debug("KV-fork rung unavailable, using copy path: %s", exc)
            self._kv_backend = None
        return self._kv_backend

    def _snapshot_and_fork(
        self, kv_page_keys: Sequence[str], branch_count: int
    ) -> tuple[int | None, list[int | None], dict[str, Any]]:
        """Pin ``kv_page_keys`` into one snapshot, then fork one CoW branch per cohort branch.

        Returns ``(snapshot_id, per_branch_ids, fork_stats)``. Every failure degrades to a
        no-op (``(None, [], {})`` or ``None`` branch entries) — the fan-out then simply falls
        back to the local forkserver copy path, never raising.

        HONEST SCOPE (read before relying on this): this shares KV pages that ALREADY EXIST
        in the engine's content-addressed store — it is the SHARING/plumbing rung, NOT page
        production. Mapping live vLLM/LMCache attention KV onto the store is external; if the
        retrieved context's pages were never offloaded, the snapshot pins nothing and the
        branches share nothing (a safe no-op, not an error).
        """
        backend = self._resolve_kv_backend()
        if backend is None:
            return None, [], {}
        snap_id = backend.snapshot(list(kv_page_keys))
        if snap_id is None:
            return None, [], {}
        branch_ids: list[int | None] = [
            backend.fork(snap_id) for _ in range(branch_count)
        ]
        stats = backend.fork_stats()
        return snap_id, branch_ids, stats

    async def fan_out(
        self,
        query: str,
        branches: Sequence[str],
        *,
        preferred: str = "",
        candidate_var: str = "candidates",
        extra_vars: dict[str, Any] | None = None,
        max_concurrency: int = 1,
        kv_page_keys: Sequence[str] | None = None,
    ) -> CrossModalForkResult:
        """Retrieve the candidate set ONCE and fork one CoW branch per snippet over it.

        Each branch snippet sees the shared candidate set bound to ``candidate_var`` in its
        namespace and may report a divergent result by calling ``FINAL_VAR('<name>', value)``
        (served host-side over the bridge). Every branch forks off the *one* warmed parent; the
        candidate set is retrieved exactly once for the whole cohort.

        ``max_concurrency`` bounds how many branches fork at once. It defaults to ``1`` because
        the local ``forkserver`` rung forks through the stdlib ``multiprocessing`` forkserver
        *process singleton*, whose control socket is not safe for concurrent ``os.fork`` requests
        — serialising the fork step is the correct semantics for it (the warm-fork win is the
        amortised parent, not wall-clock parallelism). A live KV-cache-fork rung
        (LMCacheMPConnector snapshot → branch) or the container-fork rung can fan out truly in
        parallel; raise ``max_concurrency`` when driving one of those.

        ``kv_page_keys`` is the OPTIONAL zero-copy KV-fork rung (default ``None`` ⇒ OFF, and the
        fan-out behaves EXACTLY as before). When supplied — the engine KV-page keys of the
        retrieved context — the candidate set's pages are pinned into one engine snapshot and one
        copy-on-write branch is forked per cohort branch (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork),
        so branches SHARE those pages by ``Arc`` (one physical copy regardless of branch count)
        instead of the forkserver's per-branch serialize-and-copy. That shared-immutable /
        per-branch-CoW topology is what makes ``max_concurrency>1`` safe to parallelise. Each
        branch's KV branch id is exposed to its snippet as ``kv_branch_id``; the fork ids +
        ``/kv/fork/stats`` land on the result (:attr:`CrossModalForkResult.kv_fork_shared`).

        HONEST SCOPE: this wires the SHARING/plumbing of KV pages only — it does NOT produce them.
        The vLLM/LMCache model-side mapping of live attention KV onto the engine store is external;
        if the context's pages were never offloaded there, snapshot/fork is a safe no-op and the
        fan-out transparently falls back to the copy path. A KV backend / reachable engine failure
        likewise degrades to the copy path — it never fails the cohort.
        """
        branch_list = list(branches)
        guard = _RecomputeGuard(self._resolve_retriever())

        # 1. Retrieve the cross-modal candidate set ONCE (the amortised expense).
        candidates = guard(query)

        sb = (
            self._sandbox
            if self._sandbox is not None
            else _pick_warm_fork_sandbox(preferred)
        )
        if sb is None:
            return CrossModalForkResult(
                query=query,
                candidate_count=len(candidates),
                retrieval_calls=guard.calls,
                sandbox=None,
                degraded=True,
                error=(
                    "no warm-fork rung available on this host (forkserver needs a "
                    "POSIX-fork-capable interpreter; container_fork needs docker; "
                    "firecracker needs forkd + KVM)"
                ),
            )

        # Optional zero-copy KV-fork rung (default-off): pin the retrieved context's KV pages
        # into ONE snapshot and fork one CoW branch per cohort branch so they share those pages
        # by Arc. A no-op (None ids) when the rung is off or the engine is unreachable — the
        # branch execution below is IDENTICAL either way; this only adds page-sharing + branch ids.
        kv_snapshot_id: int | None = None
        kv_branch_ids: list[int | None] = []
        kv_fork_stats: dict[str, Any] = {}
        if kv_page_keys:
            kv_snapshot_id, kv_branch_ids, kv_fork_stats = self._snapshot_and_fork(
                kv_page_keys, len(branch_list)
            )

        from agent_utilities.rlm.sandboxes.base import SandboxEnv

        async def _run_branch(idx: int, snippet: str) -> CrossModalBranchResult:
            # Per-branch FINAL_VAR closure: each fork reports into its own slot, so a branch's
            # divergent result can never be attributed to a sibling.
            captured: dict[str, Any] = {}

            def FINAL_VAR(
                name: str, value: Any
            ) -> None:  # served host-side over the UDS bridge
                captured[name] = value

            # 3. Fork a CoW child seeded with the SAME candidate set (deep-copied per child by the
            #    process/bridge boundary → siblings cannot observe each other's mutations).
            env_vars: dict[str, Any] = {candidate_var: copy.deepcopy(candidates)}
            if extra_vars:
                env_vars.update(copy.deepcopy(extra_vars))
            env_vars["branch_index"] = idx
            # Expose this branch's copy-on-write KV branch id (or None when the rung is off) so a
            # KV-aware snippet can address its branch-local pages via the /kv/branch/<bid> surface.
            if idx < len(kv_branch_ids):
                env_vars["kv_branch_id"] = kv_branch_ids[idx]
            try:
                res = await sb.execute(
                    snippet, SandboxEnv(vars=env_vars, helpers={"FINAL_VAR": FINAL_VAR})
                )
            except Exception as exc:  # noqa: BLE001 — one branch never fails the cohort
                return CrossModalBranchResult(index=idx, ok=False, error=str(exc))
            return CrossModalBranchResult(
                index=idx,
                ok=res.error is None,
                stdout=res.stdout,
                error=res.error,
                output=captured.get("out", captured or None),
            )

        # 2 + 3. Warm-or-reuse ONE parent (the ForkableSandbox.execute → WarmParentRegistry path)
        #        and fork every branch off it, bounded by ``max_concurrency``.
        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def _guarded(idx: int, snippet: str) -> CrossModalBranchResult:
            async with sem:
                return await _run_branch(idx, snippet)

        results = await asyncio.gather(
            *(_guarded(i, s) for i, s in enumerate(branch_list))
        )

        return CrossModalForkResult(
            query=query,
            candidate_count=len(candidates),
            retrieval_calls=guard.calls,
            sandbox=getattr(sb, "name", None),
            branches=list(results),
            kv_snapshot_id=kv_snapshot_id,
            kv_branch_ids=kv_branch_ids,
            kv_fork_stats=kv_fork_stats,
        )
