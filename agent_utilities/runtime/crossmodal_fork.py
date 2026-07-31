"""CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout — warm-fork fan-out over an engine cross-modal candidate set.

The agent-utilities side of the epistemic-graph cross-modal seam.
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
   re-queries the engine — and runs its own divergent computation over it concurrently. Each
   governed child receives an isolated view, so a branch mutating its view cannot leak into a
   sibling.

Reuse proof: :attr:`CrossModalForkResult.retrieval_calls` is ``1`` regardless of the branch
count (the guard would raise on a second retrieval). Isolation proof: divergent per-branch
mutations of the candidate set are observed only by the mutating branch.

The governed Firecracker rung provides the code-execution boundary. The engine KV-cache-fork
rung (LMCacheMPConnector snapshot → branch, the vLLM path in the KV-cache memory note) can also
share candidate-set KV / embedding pages as copy-on-write memory, making page sharing itself
zero-copy. This module stays backend-agnostic and drives the configured governed primitive.

The zero-copy rung now EXISTS engine-side (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork): the
``epistemic-graph`` ``eg-kvcache`` crate implements the "snapshot → branch" primitive on its
content-addressed shared KV store — ``SharedKvIndex::snapshot(keys)`` pins the candidate-set
pages, ``fork(snapshot)`` fans out N branches that all read those SAME physical pages by ``Arc``
(one copy regardless of N), and ``branch_put`` is copy-on-write per branch — exposed over the
KV-cache HTTP surface as ``POST /kv/snapshot`` + ``POST /kv/snapshot/<id>/fork`` +
``GET|PUT /kv/branch/<bid>/<key>`` (``GET /kv/fork/stats`` proves resident bytes stay flat vs
branch count). This is the rung the ``max_concurrency>1`` path targets to make page fan-out
O(1) in copies. Still external: the vLLM/LMCache connector that maps live attention KV pages
onto this store — the engine provides the zero-copy substrate, not the model-side page mapping.

:class:`CrossModalForkFanout` *drives* that substrate as a DEFAULT-ON rung: for a >1-branch
fan-out it derives page keys from the retrieved candidate set itself (content-hashed and stored
into the shared KV store) whenever the resolved backend advertises fork support, and snapshots+
forks them so the branches share those pages zero-copy — no caller wiring required. Pass
:meth:`CrossModalForkFanout.fan_out`'s ``kv_page_keys`` explicitly to use real engine KV-page keys
instead (e.g. an actual vLLM/LMCache prefix-hash mapping), or pass ``kv_page_keys=[]`` to opt out
entirely. The implementation wires the SHARING/plumbing of pages; producing RAW vLLM/LMCache
attention-KV tensors from nothing remains the external vLLM/LMCache job, so an absent/unreachable/
unsupported KV backend transparently degrades to the copy path rather than failing the cohort —
every engagement and every fallback (with its reason) is recorded via
``agent_utilities.observability.gateway_metrics.KVCACHE_FORK_EVENTS``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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


def _record_fork_outcome(
    result: str, *, reason: str = "", shared_bytes: int = 0
) -> None:
    """Best-effort KV-fork telemetry (WS-4 idiom — never breaks the fan-out on failure).

    ``result`` is ``"forked"`` (the CoW snapshot/fork rung engaged) or ``"fallback"``
    (it transparently degraded to the per-branch copy path, ``reason`` explaining why:
    ``backend_unavailable`` / ``fork_unsupported`` / ``snapshot_failed`` /
    ``no_derivable_pages``). Local import + broad except mirrors
    ``remote_backend._record_kvcache_client_outcome`` so this hot fan-out path never
    takes on a hard dependency on the ``metrics`` extra.
    """
    try:
        from agent_utilities.observability.gateway_metrics import (
            KVCACHE_FORK_EVENTS,
            KVCACHE_FORK_SHARED_BYTES,
        )

        KVCACHE_FORK_EVENTS.labels(result=result, reason=reason).inc()
        if shared_bytes:
            KVCACHE_FORK_SHARED_BYTES.set(shared_bytes)
    except Exception as exc:  # noqa: BLE001 — metrics must never break the fan-out
        logger.debug("KV-fork telemetry recording failed: %s", exc)


def _derive_kv_page_keys(backend: Any, candidates: list[dict[str, Any]]) -> list[str]:
    """Default-on derivation of KV-fork page keys from the retrieved candidate set.

    A caller with a genuine vLLM/LMCache prefix-hash mapping passes real
    ``kv_page_keys`` explicitly; absent that, the retrieved candidate set is ITSELF
    content-addressable, so each candidate is content-hashed and ``put`` into the
    SAME shared, content-addressed KV store the vLLM/LMCache connector uses (dedup
    applies identically — two branches, or two separate fan-outs, that retrieved the
    identical candidate share the one stored page). This is what makes the rung
    DEFAULT-ON rather than requiring a caller to already have engine-side KV pages:
    the candidate set becomes its own shareable page set.

    Best-effort and order-preserving: a candidate that fails to serialize or store is
    simply skipped (never raised) so one bad candidate never disables the whole rung.
    """
    keys: list[str] = []
    for candidate in candidates:
        try:
            payload = json.dumps(candidate, sort_keys=True, default=str).encode("utf-8")
        except (TypeError, ValueError) as exc:  # noqa: BLE001 — best-effort/order-preserving per the docstring above: a non-serializable candidate is simply skipped, never disabling the whole rung
            logger.debug("KV-fork candidate not serializable, skipped: %s", exc)
            continue
        key = hashlib.sha256(payload).hexdigest()
        try:
            if not backend.put(key, payload):
                continue
        except Exception as exc:  # noqa: BLE001 — a failed store just drops this page
            logger.debug("KV-fork candidate store failed for %s: %s", key, exc)
            continue
        keys.append(key)
    return keys


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
    eng = engine if engine is not None else IntelligenceGraphEngine.get_or_create()
    retriever = HybridRetriever(eng)
    return retriever.retrieve_hybrid(
        query, context_window=context_window, multi_hop_depth=multi_hop_depth
    )


#: Process-wide lazily-built KV-fork backend (see
#: :meth:`CrossModalForkFanout._resolve_kv_backend`). It owns a pooled, never-closed
#: ``httpx.Client``, and the fanout object itself is rebuilt per MCP call, so this must
#: NOT be per-instance state.
_SHARED_KV_BACKEND: Any | None = None
_SHARED_KV_BACKEND_RESOLVED = False


class CrossModalForkFanout:
    """Retrieve an engine cross-modal candidate set once, then warm-fork N branches over it.

    ``retriever`` is the engine seam (defaults to :func:`engine_cross_modal_candidates`, the
    vector+graph+text fusion); inject a callable to run without a live engine. ``sandbox`` pins a
    :class:`~agent_utilities.rlm.sandboxes.base.ForkableSandbox`; when unset the cheapest
    available warm-fork rung is auto-selected.

    ``kv_backend`` is the zero-copy KV-fork rung
    (:class:`~agent_utilities.kvcache.EpistemicGraphKVBackend`, CONCEPT:EG-KG.memory.zero-copy-snapshot-fork).
    **Default-on**: :meth:`fan_out` engages it automatically for a >1-branch cohort
    whenever the resolved backend advertises fork support (``supports_fork()``),
    deriving page keys from the retrieved candidate set — no caller wiring required.
    Transparent fallback when the backend is unreachable, unsupported, or a caller
    opts out with ``kv_page_keys=[]``: the fan-out runs on the ordinary copy path,
    never a hard failure. Inject ``kv_backend`` to reuse a pooled connector; leave it
    ``None`` and one is lazily built from the governed KV-cache configuration the
    first time the rung is attempted.
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
        """Return the injected KV-fork backend, else the PROCESS-WIDE lazy one.

        Lazy + guarded: importing this module must never pull ``httpx`` / the kvcache
        stack, and a host without a reachable engine must degrade to the copy path
        (``None``) rather than crash. Only called when the rung is actually requested.

        The lazily built backend is cached on the MODULE, not on ``self``. It owns a
        pooled ``httpx.Client`` (``max_connections=32``) that nothing ever closes, and
        ``CrossModalForkFanout()`` is constructed FRESH on every ``graph_fork`` MCP call
        (``mcp/tools/engine_surface_tools.py``). Caching per instance therefore built —
        and leaked — a brand-new keep-alive connection pool per call once this rung
        became default-on for every >1-branch cohort, exhausting sockets/file
        descriptors under sustained traffic. One process-wide backend also restores the
        point of ``EpistemicGraphKVBackend.supports_fork``'s own probe cache, which a
        per-call instance defeated (its docstring promises the probe costs at most one
        round trip, which only holds if the backend outlives the call).
        """
        if self._kv_backend is not None:
            return self._kv_backend
        global _SHARED_KV_BACKEND, _SHARED_KV_BACKEND_RESOLVED
        if not _SHARED_KV_BACKEND_RESOLVED:
            _SHARED_KV_BACKEND_RESOLVED = True
            try:
                from agent_utilities.kvcache import EpistemicGraphKVBackend

                _SHARED_KV_BACKEND = EpistemicGraphKVBackend.from_env()
            except Exception as exc:  # noqa: BLE001 — no engine / no deps ⇒ copy path
                logger.debug(
                    "KV-fork rung unavailable; using copy path (exception_type=%s)",
                    type(exc).__name__,
                )
                _SHARED_KV_BACKEND = None
        return _SHARED_KV_BACKEND

    def _auto_kv_page_keys(self, candidates: list[dict[str, Any]]) -> list[str]:
        """DEFAULT-ON derivation of ``kv_page_keys`` when a caller doesn't supply real ones.

        Only reached when :meth:`fan_out` was called with no explicit ``kv_page_keys``
        AND the cohort has more than one branch (a single branch has no sharing to
        gain). Resolves the KV backend, cheaply probes its fork-surface capability
        (:meth:`~agent_utilities.kvcache.EpistemicGraphKVBackend.supports_fork` when the
        backend exposes it — absent on a bare test double, which then just tries and
        degrades naturally), and derives pages from the candidate set
        (:func:`_derive_kv_page_keys`). Every dead end records a ``"fallback"``
        telemetry event with the specific reason and returns ``[]`` — never raises.
        """
        backend = self._resolve_kv_backend()
        if backend is None:
            _record_fork_outcome("fallback", reason="backend_unavailable")
            return []
        supports = getattr(backend, "supports_fork", None)
        if callable(supports):
            try:
                supported = supports()
            except Exception as exc:  # noqa: BLE001 — probe failure ⇒ unsupported
                logger.debug("KV-fork capability probe failed: %s", exc)
                supported = False
            if not supported:
                _record_fork_outcome("fallback", reason="fork_unsupported")
                return []
        keys = _derive_kv_page_keys(backend, candidates)
        if not keys:
            _record_fork_outcome("fallback", reason="no_derivable_pages")
        return keys

    def _snapshot_and_fork(
        self, kv_page_keys: Sequence[str], branch_count: int
    ) -> tuple[int | None, list[int | None], dict[str, Any]]:
        """Pin ``kv_page_keys`` into one snapshot, then fork one CoW branch per cohort branch.

        Returns ``(snapshot_id, per_branch_ids, fork_stats)``. Every failure degrades to a
        no-op (``(None, [], {})`` or ``None`` branch entries); branch execution continues without
        engine page sharing. Records a ``"forked"``/``"fallback"`` telemetry event either way
        (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork) so the rung's engagement rate is observable.

        HONEST SCOPE (read before relying on this): this shares KV pages that ALREADY EXIST
        in the engine's content-addressed store — it is the SHARING/plumbing rung, NOT page
        production. Mapping live vLLM/LMCache attention KV onto the store is external; if the
        retrieved context's pages were never offloaded, the snapshot pins nothing and the
        branches share nothing (a safe no-op, not an error).
        """
        backend = self._resolve_kv_backend()
        if backend is None:
            _record_fork_outcome("fallback", reason="backend_unavailable")
            return None, [], {}
        snap_id = backend.snapshot(list(kv_page_keys))
        if snap_id is None:
            _record_fork_outcome("fallback", reason="snapshot_failed")
            return None, [], {}
        branch_ids: list[int | None] = [
            backend.fork(snap_id) for _ in range(branch_count)
        ]
        stats = backend.fork_stats()
        _record_fork_outcome(
            "forked", shared_bytes=int(stats.get("shared_bytes", 0) or 0)
        )
        return snap_id, branch_ids, stats

    def _release_kv_fork_resources(
        self, kv_snapshot_id: int, kv_branch_ids: list[int | None]
    ) -> None:
        """Drop every forked branch, then release the parent snapshot (D-KVR-2).

        Engine-side ``release_snapshot`` refuses a snapshot with any still-live
        branch (``StillReferenced``, ``409``) — so branches must be dropped FIRST,
        in the same order they were forked, before the snapshot release can
        succeed. Called from :meth:`fan_out`'s ``finally`` once the cohort
        completes (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork), so a caller that
        never explicitly releases no longer leaks: the pinned pages free as soon
        as their fan-out is done, not only when/if something else calls
        ``release_snapshot``/``drop_branch`` directly. Every outcome is logged at
        debug (not warning) — a `"not_found"`/`"still_referenced"` here is
        unexpected but not actionable by this caller, and neither method ever
        raises (same graceful-degradation contract as the rest of the connector).
        """
        backend = self._resolve_kv_backend()
        if backend is None:
            return
        drop_branch = getattr(backend, "drop_branch", None)
        if callable(drop_branch):
            for branch_id in kv_branch_ids:
                if branch_id is None:
                    continue
                outcome = drop_branch(branch_id)
                if outcome not in ("released", "not_found"):
                    logger.debug(
                        "kv-fork cohort cleanup: drop_branch(%s) -> %s",
                        branch_id,
                        outcome,
                    )
        release_snapshot = getattr(backend, "release_snapshot", None)
        if callable(release_snapshot):
            outcome = release_snapshot(kv_snapshot_id)
            if outcome not in ("released", "not_found"):
                logger.debug(
                    "kv-fork cohort cleanup: release_snapshot(%s) -> %s",
                    kv_snapshot_id,
                    outcome,
                )

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

        ``max_concurrency`` bounds how many governed branches run at once. It defaults to ``1``
        to keep resource use conservative; raise it only when the selected backend and host
        capacity support concurrent branches.

        ``kv_page_keys`` is the zero-copy KV-fork rung's page-key override — the engine
        KV-page keys of the retrieved context, when the caller already has a genuine
        vLLM/LMCache prefix-hash mapping. **DEFAULT-ON**: when left ``None`` (the
        default) and the cohort has more than one branch, the rung is attempted
        automatically — the candidate set is content-hashed and stored into the
        engine's shared KV store (:func:`_derive_kv_page_keys`), so pages exist to
        share even with no external LLM-side KV mapping. Either way, once page keys
        are in hand, they are pinned into one engine snapshot and one copy-on-write
        branch is forked per cohort branch (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork),
        so branches SHARE those pages by ``Arc`` (one physical copy regardless of branch
        count) inside the ENGINE.

        **Scope, honestly.** This rung is an engine-side side-store plus its
        provenance/telemetry — it does NOT remove the per-branch
        ``copy.deepcopy(candidates)`` in ``_run_branch`` below, which still runs
        unconditionally for every branch. Nothing yet reads the candidate set back
        through ``backend.branch_get(...)``; each branch is handed its own in-process
        deep copy exactly as before and only ``kv_branch_id`` is additionally exposed.
        So the in-process fan-out memory footprint is unchanged by this rung. Making
        the fork load-bearing means branches must source their candidates from the
        engine-resident copy instead of an in-process dict — a redesign of the
        fan-out data path, recorded as D-W15-11 in
        ``reports/deferred/waves1-5-gate.md``.
        That shared-immutable / per-branch-CoW topology is what makes
        ``max_concurrency>1`` efficient. Each branch's KV branch id is exposed to its
        snippet as ``kv_branch_id``; the fork ids + ``/kv/fork/stats`` land on the
        result (:attr:`CrossModalForkResult.kv_fork_shared`). Pass an explicit empty
        sequence (``[]``) to opt fully out (no auto-derivation attempted either).

        **Transparent fallback, always.** Every dead end — an unreachable engine, a
        backend build that doesn't advertise the fork surface, a snapshot that pins
        no pages — degrades to the ordinary per-branch copy path and records a
        ``"fallback"`` telemetry event with the reason
        (:data:`~agent_utilities.observability.gateway_metrics.KVCACHE_FORK_EVENTS`);
        a successful engagement records ``"forked"`` plus the shared-byte count. This
        NEVER fails the cohort and never silently serves stale/wrong pages — a failed
        rung is simply absent from the result, not miscounted as engaged.

        HONEST SCOPE: this wires the SHARING/plumbing of KV pages only — it does NOT produce
        raw vLLM/LMCache attention-KV tensors from nothing; the model-side mapping of live
        attention KV onto the engine store remains external. What this rung DOES own end to
        end is the CANDIDATE SET itself: turning it into engine-resident, content-addressed,
        ref-counted pages that N branches share zero-copy instead of each holding its own
        ``deepcopy`` — a genuine, self-contained memory win with no external LLM cooperation
        required.
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
                    "no governed warm-fork rung is available on this host "
                    "(firecracker needs a reachable forkd controller + KVM)"
                ),
            )

        # Zero-copy KV-fork rung (DEFAULT-ON): pin the retrieved context's KV pages into
        # ONE snapshot and fork one CoW branch per cohort branch so they share those pages
        # by Arc. Explicit kv_page_keys wins; otherwise — when there is more than one
        # branch to share across — pages are auto-derived from the candidate set. A no-op
        # (None ids) when the rung can't engage (unreachable engine, unsupported build, an
        # explicit empty [] opt-out, or a single branch with nothing to share) — the branch
        # execution below is IDENTICAL either way; this only adds page-sharing + branch ids.
        kv_snapshot_id: int | None = None
        kv_branch_ids: list[int | None] = []
        kv_fork_stats: dict[str, Any] = {}
        effective_kv_page_keys = kv_page_keys
        if effective_kv_page_keys is None and len(branch_list) > 1:
            effective_kv_page_keys = self._auto_kv_page_keys(candidates)
        if effective_kv_page_keys:
            kv_snapshot_id, kv_branch_ids, kv_fork_stats = self._snapshot_and_fork(
                effective_kv_page_keys, len(branch_list)
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
                return CrossModalBranchResult(
                    index=idx,
                    ok=False,
                    error=f"sandbox execution failed ({type(exc).__name__})",
                )
            return CrossModalBranchResult(
                index=idx,
                ok=res.error is None,
                stdout=res.stdout,
                error="sandbox execution failed" if res.error is not None else None,
                output=captured.get("out", captured or None),
            )

        # 2 + 3. Warm-or-reuse ONE parent (the ForkableSandbox.execute → WarmParentRegistry path)
        #        and fork every branch off it, bounded by ``max_concurrency``.
        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def _guarded(idx: int, snippet: str) -> CrossModalBranchResult:
            async with sem:
                return await _run_branch(idx, snippet)

        try:
            results = await asyncio.gather(
                *(_guarded(i, s) for i, s in enumerate(branch_list))
            )
        finally:
            # D-KVR-2 (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork): the engine's
            # DELETE /kv/snapshot/<id> + DELETE /kv/branch/<bid> routes only close the
            # fork/snapshot leak if a caller actually issues them once the cohort is
            # done with the shared pages. Runs even if a branch's own execution raised
            # past `_run_branch`'s own catch-all (defense in depth) — never on the
            # engine's happy path only. Best-effort: `release_snapshot`/`drop_branch`
            # already never raise (same graceful-degradation contract as the rest of
            # this connector), so no try/except is needed around the calls themselves.
            if kv_snapshot_id is not None:
                self._release_kv_fork_resources(kv_snapshot_id, kv_branch_ids)

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
