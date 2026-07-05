"""Reserved-worker fair admission scheduler (CONCEPT:AU-ORCH.dispatch.worker-scheduling).

The KG ingest worker pool drains ONE shared queue partitioned into functional
*lanes* (see :mod:`.task_lanes`). The existing two-level rotation in
``_select_pending_task`` decides WHICH pending task is the next candidate, but it
cannot preempt *running* tasks: a handful of long-running ``codebase`` index jobs
can occupy every worker, so a freshly-enqueued ``content_url`` crawl (or an
interactive query) sits pending until a heavy job finishes — minutes later.

This module adds the missing *admission* layer. It maintains an in-process
registry of what each worker is currently processing (its lane/type) and, given
that live picture plus the pending-by-lane depths, decides whether THIS free
worker may claim a given lane/type *now* or should hold back. The policy:

1. **Hot spare** — keep at least ``reserved`` worker(s) free so a bursty task is
   picked up immediately, never queued behind a long codebase job. A worker is
   refused admission if claiming would drop the free count below ``reserved``…
2. **…unless refusing would starve a channel** — never hold a spare by leaving a
   lane that *has pending work* with zero coverage. If this worker is the only
   free worker AND some pending lane currently has no running worker, it is
   admitted to cover that lane even though that spends the spare (degrade to
   zero-spare rather than starve). Conversely a worker is steered toward an
   uncovered pending lane before piling onto an already-covered one.
3. **Heavy-type cap** — the heavy type (``codebase``) may occupy at most
   ``codebase_cap`` workers concurrently, where the default cap is
   ``workers − reserved − Σ(per-lane minimums for other pending lanes)`` so it can
   never monopolise the pool and there is always room for each pending lane's
   minimum coverage.

The decision is *advisory and composes with* the existing rotation + engine-CAS
claim: rotation proposes a candidate, the policy says admit/deny, the CAS still
arbitrates atomicity across hosts. Nothing here touches the queue or the backend;
it is pure in-process bookkeeping, so it is cheap and fully unit-testable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .task_lanes import (
    BEST_EFFORT_LANES,
    INTERACTIVE_LANES,
    LANE_NAMES,
    MEMORY_GEN_POOL,
    lane_for_task_type,
    pool_for,
)

__all__ = [
    "SchedulerConfig",
    "WorkerRegistry",
    "AdmissionPolicy",
    "scheduler_config_from_env",
    "durable_shard_writers",
    "resolve_engine_shard_writers",
    "set_engine_shard_writers",
]

# CONCEPT:AU-KG.ingest.floor-codebase-admission-cap - Floor the codebase admission cap at the engine's durable K-way redb shard-writer width so concurrent cross-graph ingests fan across all K shard writers instead of the derived one-to-three that leaves most writers idle

# The heavy task type that must never be allowed to occupy the whole pool.
HEAVY_TYPE = "codebase"

# CONCEPT:AU-KG.compute.resolve — the ENGINE's real shard-writer width, resolved once from the
# live engine and cached. In split-storage the engine is REMOTE (e.g. R510, K=4
# from its 8 cpus) while the scheduling host is a DIFFERENT box (e.g. RW710, 16
# cpus → the cpu-derived estimate would say 8, over-admitting against an engine
# that only has 4 shard writers). The floor must reflect the engine's ACTUAL K, so
# we ask the engine for it (it knows) and fall back to the cpu/env estimate only
# when the engine can't be reached.
_ENGINE_SHARD_K: int | None = None
_ENGINE_SHARD_LOCK = threading.Lock()


def set_engine_shard_writers(k: int | None) -> None:
    """Seed/override the cached engine shard-writer width (CONCEPT:AU-KG.compute.resolve).

    Callers that already know the engine's K (e.g. a daemon that just queried it)
    use this so :func:`durable_shard_writers` returns the authoritative value with
    no further round-trips. ``None``/non-positive clears the cache.
    """
    global _ENGINE_SHARD_K
    with _ENGINE_SHARD_LOCK:
        _ENGINE_SHARD_K = int(k) if k and int(k) > 0 else None


def resolve_engine_shard_writers(engine: Any) -> int | None:
    """Best-effort: ask the live ENGINE for its durable shard-writer width K.

    The engine's rebalance planner reports one entry per shard ("all K shards
    represented incl. empties"), so ``len(rebalance_plan()["shards"]) == K`` — the
    authoritative width straight from the process that owns the redb backend
    (CONCEPT:AU-KG.compute.resolve). Accepts a ``GraphComputeEngine``, a backend wrapping one
    (``._graph``/``.graph``), or a raw sync client. Cached on success; returns the
    cached value on later calls and ``None`` when the engine can't answer (non-redb
    build, unreachable, older engine) so the caller degrades to the cpu/env estimate.
    """
    global _ENGINE_SHARD_K
    if _ENGINE_SHARD_K is not None:
        return _ENGINE_SHARD_K
    client = _resolve_sync_client(engine)
    if client is None:
        return None
    try:
        plan = client.resharding.rebalance_plan()
    except Exception:  # noqa: BLE001 — non-redb / unreachable / old engine
        return None
    try:
        shards = plan.get("shards") if isinstance(plan, dict) else None
        k = len(shards) if shards is not None else 0
    except Exception:  # noqa: BLE001 — tolerate an unexpected shape
        return None
    if k <= 0:
        return None
    with _ENGINE_SHARD_LOCK:
        _ENGINE_SHARD_K = int(k)
    return _ENGINE_SHARD_K


def _resolve_sync_client(engine: Any) -> Any:
    """Dig the underlying sync epistemic-graph client out of various handles."""
    if engine is None:
        return None
    for obj in (
        engine,
        getattr(engine, "_graph", None),
        getattr(engine, "graph", None),
    ):
        if obj is None:
            continue
        client = getattr(obj, "_client", None)
        if client is not None and hasattr(client, "resharding"):
            return client
        if hasattr(obj, "resharding"):
            return obj
    return None


def durable_shard_writers() -> int:
    """The engine's durable K-way redb shard-writer width (CONCEPT:AU-KG.ingest.floor-codebase-admission-cap/2.281).

    Resolution order, most-authoritative first:

    1. The width resolved from the live ENGINE and cached (CONCEPT:AU-KG.compute.resolve) — the
       ground truth in split-storage, where the engine is a remote box with a
       different cpu count than this scheduling host. Seeded via
       :func:`set_engine_shard_writers` / :func:`resolve_engine_shard_writers`.
    2. An explicit ``EPISTEMIC_GRAPH_REDB_SHARDS`` override (clamped 1..=64) — valid
       only when the engine's env is shared with this host.
    3. The cpu-derived estimate ``clamp(cpu/2, 1, 8)`` mirroring the engine's EG-026
       auto-size — the safe fallback when the engine hasn't been queried yet.

    Each codebase ingest routes its structural writes to a per-repo graph
    (``code:<repo>``, CONCEPT:AU-KG.ingest.unified-query-routing) that hashes to ONE of these K shard writers;
    so K concurrent codebase ingests spread across K *independent* writer threads.
    The codebase admission cap uses this as a FLOOR so it never throttles
    durable-write concurrency below the substrate's own width (the submission-side
    limiter behind the profiled ``parallelism_factor``).
    """
    if _ENGINE_SHARD_K is not None:
        return _ENGINE_SHARD_K

    from agent_utilities.core._env import setting

    raw = setting("EPISTEMIC_GRAPH_REDB_SHARDS", None)
    if raw not in (None, ""):
        try:
            return max(1, min(int(raw), 64))
        except (TypeError, ValueError):
            pass
    try:
        import os

        cores = os.cpu_count() or 4
    except Exception:  # noqa: BLE001 — sizing is best-effort
        cores = 4
    return max(1, min(cores // 2, 8))


@dataclass(frozen=True)
class SchedulerConfig:
    """Tunable reservation/cap knobs (CONCEPT:AU-ORCH.dispatch.worker-scheduling).

    ``worker_count`` is the size of the pool the policy reasons about.
    ``reserved`` is the hot-spare count. ``per_lane_min`` is the minimum number of
    workers each lane *with pending work* is guaranteed. ``codebase_cap`` caps the
    heavy type; ``None`` means derive it as ``workers − reserved − Σ(other pending
    lane minimums)`` at decision time.

    ``memory_gen_cap`` / ``acquisition_floor`` size the TWO ingestion pools
    (CONCEPT:AU-ORCH.dispatch.two-pool). The memory-gen half back-pressures on the
    single per-graph write lock, so it may occupy at most ``memory_gen_cap``
    workers concurrently — the remaining ``acquisition_floor`` are held for the
    I/O-bound acquisition half so a memory-gen burst can never starve scraping.
    ``memory_gen_cap=None`` means derive it as ``worker_count − acquisition_floor``
    at decision time (the two knobs are the independent-sizing lever).
    """

    worker_count: int = 1
    reserved: int = 1
    per_lane_min: int = 1
    codebase_cap: int | None = None
    memory_gen_cap: int | None = None
    acquisition_floor: int = 1


def scheduler_config_from_env(worker_count: int) -> SchedulerConfig:
    """Build a :class:`SchedulerConfig` from env knobs (CONCEPT:AU-ORCH.dispatch.worker-scheduling).

    * ``KG_SCHED_RESERVED`` (default ``1``) — hot-spare worker count.
    * ``KG_SCHED_PER_LANE_MIN`` (default ``1``) — per-lane minimum coverage.
    * ``KG_SCHED_CODEBASE_CAP`` (default unset → derived) — hard cap on concurrent
      ``codebase`` workers.
    * ``KG_POOL_ACQUISITION_FLOOR`` (default ``max(1, worker_count // 4)``) —
      workers held for the acquisition pool that memory-gen may never consume
      (CONCEPT:AU-ORCH.dispatch.two-pool).
    * ``KG_POOL_MEMORY_GEN_CAP`` (default unset → derived
      ``worker_count − acquisition_floor``) — hard cap on concurrent memory-gen
      workers, so a write-lock-bound burst can't starve acquisition.

    All values are clamped to ``[0, worker_count]`` (cap to ``[1, worker_count]``)
    so a misconfiguration can never wedge the pool.
    """

    # Route every read through the shared config accessor (config.json + live env),
    # never bare os.environ — enforced by check_no_env_sprawl.py.
    from agent_utilities.core._env import setting

    def _int(name: str, default: int) -> int:
        try:
            return int(setting(name, default))
        except (TypeError, ValueError):
            return default

    wc = max(1, int(worker_count))
    reserved = max(0, min(_int("KG_SCHED_RESERVED", 1), wc))
    per_lane_min = max(0, min(_int("KG_SCHED_PER_LANE_MIN", 1), wc))
    cap_raw = setting("KG_SCHED_CODEBASE_CAP", None)
    codebase_cap: int | None
    if cap_raw is None or cap_raw == "":
        codebase_cap = None
    else:
        try:
            codebase_cap = max(1, min(int(cap_raw), wc))
        except ValueError:
            codebase_cap = None

    # Two-pool budgets (CONCEPT:AU-ORCH.dispatch.two-pool). The acquisition floor
    # defaults to a quarter of the pool (>=1); the memory-gen cap defaults to the
    # complement. Both clamp to [1, worker_count] so a misconfiguration can neither
    # zero out a pool nor exceed the pool.
    acq_floor = max(1, min(_int("KG_POOL_ACQUISITION_FLOOR", max(1, wc // 4)), wc))
    mg_raw = setting("KG_POOL_MEMORY_GEN_CAP", None)
    memory_gen_cap: int | None
    if mg_raw is None or mg_raw == "":
        memory_gen_cap = None
    else:
        try:
            memory_gen_cap = max(1, min(int(mg_raw), wc))
        except ValueError:
            memory_gen_cap = None

    return SchedulerConfig(
        worker_count=wc,
        reserved=reserved,
        per_lane_min=per_lane_min,
        codebase_cap=codebase_cap,
        memory_gen_cap=memory_gen_cap,
        acquisition_floor=acq_floor,
    )


@dataclass
class WorkerRegistry:
    """Live registry of what each worker is processing (CONCEPT:AU-ORCH.dispatch.worker-scheduling).

    Thread-safe. Workers ``start`` a claim (stamping their lane/type), and
    ``finish`` on ack/fail (clearing it). A worker not present, or mapped to
    ``None``, is *free*. The registry only ever holds in-process state for THIS
    host's pool — cross-host fairness is the queue's job; reservation/min-coverage
    are per-host concerns (each host keeps its own spare).
    """

    _busy: dict[str, tuple[str, str]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def start(self, worker_id: str, lane: str, task_type: str) -> None:
        """Record that ``worker_id`` began processing ``task_type`` (in ``lane``)."""
        with self._lock:
            self._busy[worker_id] = (lane, task_type)

    def finish(self, worker_id: str) -> None:
        """Clear ``worker_id`` back to free (idempotent)."""
        with self._lock:
            self._busy.pop(worker_id, None)

    def snapshot(self) -> dict[str, tuple[str, str]]:
        """A copy of the worker→(lane, type) map for busy workers."""
        with self._lock:
            return dict(self._busy)

    def busy_count(self) -> int:
        with self._lock:
            return len(self._busy)

    def free_count(self, worker_count: int) -> int:
        """Workers not currently processing anything (>= 0)."""
        with self._lock:
            return max(0, worker_count - len(self._busy))

    def running_by_lane(self) -> dict[str, int]:
        """How many workers are running in each lane right now."""
        out: dict[str, int] = {}
        with self._lock:
            for lane, _ in self._busy.values():
                out[lane] = out.get(lane, 0) + 1
        return out

    def running_by_type(self) -> dict[str, int]:
        """How many workers are running each task type right now."""
        out: dict[str, int] = {}
        with self._lock:
            for _, tk in self._busy.values():
                out[tk] = out.get(tk, 0) + 1
        return out

    def running_by_pool(self) -> dict[str, int]:
        """How many workers are running in each two-pool (CONCEPT:AU-ORCH.dispatch.two-pool).

        Resolves each busy worker's (lane, task_type) to its pool (honouring the
        per-type override), so the admission policy can enforce the memory-gen cap
        and acquisition floor. Un-pooled lanes (``queries`` / ``maint``) are not
        counted.
        """
        out: dict[str, int] = {}
        with self._lock:
            for lane, tk in self._busy.values():
                pool = pool_for(lane, tk)
                if pool is not None:
                    out[pool] = out.get(pool, 0) + 1
        return out


@dataclass(frozen=True)
class _Decision:
    admit: bool
    reason: str


class AdmissionPolicy:
    """Decide whether a free worker may claim a candidate now (CONCEPT:AU-ORCH.dispatch.worker-scheduling).

    Construct with the pool config + the live :class:`WorkerRegistry`. Call
    :meth:`admit` with the candidate's lane/type and a ``pending_by_lane`` snapshot
    (lane → count of *pending* tasks). The policy enforces, in order:

    * the heavy-type cap (``codebase`` concurrency ≤ cap), then
    * the best-effort lane cap (a ``BEST_EFFORT_LANES`` lane — maint interval ticks —
      may run at most its floor ``max(1, per_lane_min)`` workers, so a periodic-tick
      backlog never expands into the throughput lanes; CONCEPT:AU-ORCH.scheduling.low-value-high-volume), then
    * per-lane minimum coverage steering (don't pile onto a covered lane while an
      uncovered pending lane exists and this is the worker that could cover it),
      then
    * the hot-spare reservation (keep ``reserved`` free) — relaxed only when
      refusing would leave a pending lane with zero coverage.
    """

    def __init__(self, config: SchedulerConfig, registry: WorkerRegistry):
        self.config = config
        self.registry = registry

    # -- cap derivation ------------------------------------------------------
    def codebase_cap(self, pending_by_lane: dict[str, int]) -> int:
        """Effective concurrent-``codebase`` cap (CONCEPT:AU-ORCH.dispatch.worker-scheduling).

        Explicit ``config.codebase_cap`` wins; otherwise derive
        ``workers − reserved − Σ(per-lane min for OTHER pending lanes)`` so the
        heavy type always leaves room for the spare and every other pending
        lane's minimum coverage.

        The derived cap is then **floored at the engine's durable shard-writer
        width** (CONCEPT:AU-KG.ingest.floor-codebase-admission-cap). The original derivation collapses to ~1-3 on a
        busy box (many pending lanes), which throttles concurrent codebase ingests
        — and therefore the count of DISTINCT per-repo graphs written at once — far
        below the engine's K independent redb shard writers, leaving K-1 of them
        idle while one is hot (the profiled ``parallelism_factor`` ceiling). Because
        K concurrent codebase ingests route to K *different* shards (no two contend
        on one writer), admitting up to ``min(K, workers − reserved)`` of them
        saturates the durable tier without ever starving the hot-spare or another
        lane's minimum coverage. Floors at 1 (codebase must make *some* progress).
        """
        if self.config.codebase_cap is not None:
            return self.config.codebase_cap
        heavy_lane = lane_for_task_type(HEAVY_TYPE)
        other_pending = sum(
            1
            for lane in LANE_NAMES
            if lane != heavy_lane and pending_by_lane.get(lane, 0) > 0
        )
        derived = (
            self.config.worker_count
            - self.config.reserved
            - other_pending * self.config.per_lane_min
        )
        # KG-2.279: never throttle durable-write concurrency below the substrate's
        # own K-way shard-writer width — bounded by the pool minus the hot spare.
        shard_floor = min(
            durable_shard_writers(),
            max(1, self.config.worker_count - self.config.reserved),
        )
        return max(1, derived, shard_floor)

    # -- two-pool budget -----------------------------------------------------
    def memory_gen_cap(self) -> int:
        """Effective concurrent-``memory_gen`` worker cap (CONCEPT:AU-ORCH.dispatch.two-pool).

        Explicit ``config.memory_gen_cap`` wins; otherwise derive
        ``worker_count − acquisition_floor`` so the acquisition half always keeps
        its floor of workers no matter how much write-lock-bound memory-gen work
        is pending. Floors at 1 (memory-gen must make *some* progress) and is
        bounded by the pool minus the acquisition floor.
        """
        cfg = self.config
        floor = max(1, cfg.acquisition_floor)
        if cfg.memory_gen_cap is not None:
            return max(1, min(cfg.memory_gen_cap, cfg.worker_count))
        return max(1, cfg.worker_count - floor)

    # -- interactive reservation --------------------------------------------
    def interactive_floor(self) -> int:
        """Workers that NON-interactive lanes may never consume (CONCEPT:AU-KG.compute.interactive-lane-floor).

        The HARD guarantee that the host always has a free worker for interactive /
        MCP work even under a saturating bulk ingest. Auto-sized as
        ``max(1, reserved)`` so it is floored at 1 regardless of how ``reserved`` is
        configured, then clamped to ``worker_count − 1`` so a non-interactive task
        can always make *some* progress (a degenerate 1-worker pool reserves 0 —
        that single worker serves everything). Unlike the relaxable hot-spare
        (rule 3), this floor is NEVER spent to cover an uncovered ingestion lane:
        only an :data:`INTERACTIVE_LANES` task may claim into it.
        """
        cfg = self.config
        return min(max(1, cfg.reserved), max(0, cfg.worker_count - 1))

    # -- coverage helpers ----------------------------------------------------
    def _uncovered_pending_lanes(
        self, pending_by_lane: dict[str, int], running_by_lane: dict[str, int]
    ) -> set[str]:
        """Lanes that have pending work but fewer than ``per_lane_min`` workers."""
        floor = max(1, self.config.per_lane_min)
        return {
            lane
            for lane in LANE_NAMES
            if pending_by_lane.get(lane, 0) > 0 and running_by_lane.get(lane, 0) < floor
        }

    # -- the decision --------------------------------------------------------
    def decide(
        self,
        lane: str,
        task_type: str,
        pending_by_lane: dict[str, int],
    ) -> _Decision:
        """Admit/deny verdict with a reason (the testable core)."""
        cfg = self.config
        running_by_lane = self.registry.running_by_lane()
        running_by_type = self.registry.running_by_type()
        free = self.registry.free_count(cfg.worker_count)

        # 1) Heavy-type cap — never let codebase occupy more than its cap.
        if task_type == HEAVY_TYPE:
            cap = self.codebase_cap(pending_by_lane)
            if running_by_type.get(HEAVY_TYPE, 0) >= cap:
                return _Decision(False, f"codebase_cap reached ({cap})")

        # 1b) Best-effort lane cap (CONCEPT:AU-ORCH.scheduling.low-value-high-volume) — a low-value/high-volume
        #     lane (maint interval ticks) is guaranteed its floor coverage but never
        #     EXPANDS beyond it, so a backlog of cheap periodic ticks can't crowd out
        #     the throughput lanes. Below the floor it falls through to the normal
        #     steering/spare logic (so it still gets covered) — capped, not starved.
        if lane in BEST_EFFORT_LANES:
            floor = max(1, cfg.per_lane_min)
            if running_by_lane.get(lane, 0) >= floor:
                return _Decision(False, f"{lane} best-effort cap ({floor})")

        # 1b2) Two-pool budget (CONCEPT:AU-ORCH.dispatch.two-pool) — the memory-gen
        #      half (chunk→extract→embed→KG-write) BACK-PRESSURES on the single
        #      per-graph write lock, so it may occupy at most ``memory_gen_cap``
        #      workers concurrently. The complementary ``acquisition_floor`` workers
        #      are thereby always available to the I/O-bound acquisition half
        #      (connector syncs / feed sweeps / URL crawls), so a memory-gen burst
        #      can never drive scraping to zero. Capped, never starved: the cap is
        #      floored at 1 so memory-gen always makes progress.
        pool = pool_for(lane, task_type)
        if pool == MEMORY_GEN_POOL:
            cap = self.memory_gen_cap()
            if self.registry.running_by_pool().get(MEMORY_GEN_POOL, 0) >= cap:
                return _Decision(False, f"memory-gen pool cap ({cap})")

        # 1c) Interactive reservation (CONCEPT:AU-KG.compute.interactive-lane-floor) — the HARD floor that keeps
        #     the host responsive. A NON-interactive task is refused if claiming would
        #     drop the free-worker count below the interactive floor, and — unlike the
        #     hot-spare (rule 3) — this is NOT relaxed to cover an uncovered ingestion
        #     lane. So no amount of pending codebase/document/connector/maint work can
        #     drive interactive capacity to 0; an MCP/interactive call always lands.
        if lane not in INTERACTIVE_LANES:
            floor = self.interactive_floor()
            if floor > 0 and free - 1 < floor:
                return _Decision(False, f"reserve interactive ({floor})")

        uncovered = self._uncovered_pending_lanes(pending_by_lane, running_by_lane)
        this_lane_uncovered = lane in uncovered

        # 2) Coverage steering — if an uncovered pending lane exists and THIS
        #    candidate's lane is already covered, steer away so the rotation can
        #    offer the uncovered lane instead. (Never block the last-resort case
        #    where this candidate's own lane is the uncovered one.)
        if uncovered and not this_lane_uncovered:
            return _Decision(False, "steer to uncovered lane")

        # 3) Hot-spare reservation. Claiming consumes one free worker; refuse if
        #    that would drop below the reserved spare — UNLESS this claim is
        #    covering an otherwise-uncovered pending lane (min-coverage wins over
        #    the spare; degrade to zero-spare rather than starve a channel).
        if free - 1 < cfg.reserved and not this_lane_uncovered:
            return _Decision(False, "reserve hot spare")

        return _Decision(True, "admitted")

    def admit(
        self,
        lane: str,
        task_type: str,
        pending_by_lane: dict[str, int],
    ) -> bool:
        """Boolean convenience over :meth:`decide`."""
        return self.decide(lane, task_type, pending_by_lane).admit
