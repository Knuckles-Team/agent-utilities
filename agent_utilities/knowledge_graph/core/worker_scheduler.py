"""Reserved-worker fair admission scheduler (CONCEPT:ORCH-1.81).

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

from .task_lanes import BEST_EFFORT_LANES, LANE_NAMES, lane_for_task_type

__all__ = [
    "SchedulerConfig",
    "WorkerRegistry",
    "AdmissionPolicy",
    "scheduler_config_from_env",
]

# The heavy task type that must never be allowed to occupy the whole pool.
HEAVY_TYPE = "codebase"


@dataclass(frozen=True)
class SchedulerConfig:
    """Tunable reservation/cap knobs (CONCEPT:ORCH-1.81).

    ``worker_count`` is the size of the pool the policy reasons about.
    ``reserved`` is the hot-spare count. ``per_lane_min`` is the minimum number of
    workers each lane *with pending work* is guaranteed. ``codebase_cap`` caps the
    heavy type; ``None`` means derive it as ``workers − reserved − Σ(other pending
    lane minimums)`` at decision time.
    """

    worker_count: int = 1
    reserved: int = 1
    per_lane_min: int = 1
    codebase_cap: int | None = None


def scheduler_config_from_env(worker_count: int) -> SchedulerConfig:
    """Build a :class:`SchedulerConfig` from env knobs (CONCEPT:ORCH-1.81).

    * ``KG_SCHED_RESERVED`` (default ``1``) — hot-spare worker count.
    * ``KG_SCHED_PER_LANE_MIN`` (default ``1``) — per-lane minimum coverage.
    * ``KG_SCHED_CODEBASE_CAP`` (default unset → derived) — hard cap on concurrent
      ``codebase`` workers.

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
    return SchedulerConfig(
        worker_count=wc,
        reserved=reserved,
        per_lane_min=per_lane_min,
        codebase_cap=codebase_cap,
    )


@dataclass
class WorkerRegistry:
    """Live registry of what each worker is processing (CONCEPT:ORCH-1.81).

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


@dataclass(frozen=True)
class _Decision:
    admit: bool
    reason: str


class AdmissionPolicy:
    """Decide whether a free worker may claim a candidate now (CONCEPT:ORCH-1.81).

    Construct with the pool config + the live :class:`WorkerRegistry`. Call
    :meth:`admit` with the candidate's lane/type and a ``pending_by_lane`` snapshot
    (lane → count of *pending* tasks). The policy enforces, in order:

    * the heavy-type cap (``codebase`` concurrency ≤ cap), then
    * the best-effort lane cap (a ``BEST_EFFORT_LANES`` lane — maint interval ticks —
      may run at most its floor ``max(1, per_lane_min)`` workers, so a periodic-tick
      backlog never expands into the throughput lanes; CONCEPT:ORCH-1.82), then
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
        """Effective concurrent-``codebase`` cap (CONCEPT:ORCH-1.81).

        Explicit ``config.codebase_cap`` wins; otherwise derive
        ``workers − reserved − Σ(per-lane min for OTHER pending lanes)`` so the
        heavy type always leaves room for the spare and every other pending
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
        return max(1, derived)

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

        # 1b) Best-effort lane cap (CONCEPT:ORCH-1.82) — a low-value/high-volume
        #     lane (maint interval ticks) is guaranteed its floor coverage but never
        #     EXPANDS beyond it, so a backlog of cheap periodic ticks can't crowd out
        #     the throughput lanes. Below the floor it falls through to the normal
        #     steering/spare logic (so it still gets covered) — capped, not starved.
        if lane in BEST_EFFORT_LANES:
            floor = max(1, cfg.per_lane_min)
            if running_by_lane.get(lane, 0) >= floor:
                return _Decision(False, f"{lane} best-effort cap ({floor})")

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
