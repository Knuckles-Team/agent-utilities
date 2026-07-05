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

    @property
    def reused_without_recompute(self) -> bool:
        """True iff the cross-modal candidate set was retrieved exactly once for the whole cohort."""
        return self.retrieval_calls == 1


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
    """

    def __init__(
        self,
        retriever: CrossModalRetriever | None = None,
        sandbox: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._sandbox = sandbox

    def _resolve_retriever(self) -> CrossModalRetriever:
        if self._retriever is not None:
            return self._retriever
        return engine_cross_modal_candidates

    async def fan_out(
        self,
        query: str,
        branches: Sequence[str],
        *,
        preferred: str = "",
        candidate_var: str = "candidates",
        extra_vars: dict[str, Any] | None = None,
        max_concurrency: int = 1,
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
        )
