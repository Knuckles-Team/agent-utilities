"""Local scale proof for the agents-as-data activation layer (ADR-6 / W2.3 → feeds W5.3).

**The thesis:** a million dormant agents cost almost nothing to keep — each is one
durable statechart row + one graph node, no thread/connection/heartbeat — and you pay
ONLY for the ones you activate. So the load-bearing, in-process-provable property is:

    activating M of N dormant instances is O(M), independent of N.

These tests register N dormant instances and activate M of them through a stateless
worker POOL (the ADR-6 loop), then assert exactly M were touched (M ``:RunTrace`` nodes,
N-M instances still purely dormant) and log the RSS / throughput curve. The instance count
is a PARAMETER: the default is CI-fast; ``AGENT_ACTIVATION_SCALE_INSTANCES`` /
``AGENT_ACTIVATION_SCALE_ACTIVATIONS`` / ``AGENT_ACTIVATION_SCALE_WORKERS`` scale it up, and
``python -m tests.unit.orchestration.test_agent_activation_scale`` runs the full
100k-dormant / 1k-active proof standalone (see :func:`run_scale_proof`).

The in-memory double keeps every row in Python RAM, so its RSS curve is the FLOOR — a real
engine pages dormant instances out via the catalog/resident split (``EPISTEMIC_GRAPH_
MAX_RESIDENT_GRAPHS`` / cold-offload), doing strictly better on RSS. The
``activation-is-O(M)`` property holds on both.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_utilities.orchestration import agent_activation as aa
from agent_utilities.orchestration import work_item as wi

# Reuse the composed in-memory double (native WorkItem verbs + eg-statechart reference +
# the :AgentInstance/:AgentMessage reads), already made snapshot-safe for a worker pool.
from tests.unit.orchestration.test_agent_activation import ActivationEngine


class ScaleActivationEngine(ActivationEngine):
    """:class:`ActivationEngine` with an INDEXED claim, so the double faithfully models
    the real engine's ``ClaimWorkItem`` (which never scans the dormant-instance
    population to find a ready activation).

    The base :class:`NativeEngine._candidate` sorts EVERY node on each claim — O(N) in the
    dormant-instance count, a test-double artifact that would mask the very property under
    test. This override scans only the (few) WorkItem nodes via a maintained index, so a
    claim is O(#activations), independent of N — matching the engine's indexed selection.
    """

    def __init__(self) -> None:
        super().__init__()
        self._work_item_ids: set[str] = set()
        # instance_id -> [message node ids], so the mailbox read is O(depth) not O(N)
        # (the real engine indexes :AgentMessage by instance_id; the base double scans
        # every node, which would make the O(N)-artifact dominate at 100k instances).
        self._mailbox_index: dict[str, list[str]] = {}

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):  # type: ignore[override]
        super().add_node(node_id, node_type, properties, ephemeral)
        if node_type == "WorkItem":
            self._work_item_ids.add(node_id)
        elif node_type == aa._MESSAGE_LABEL:
            iid = (properties or {}).get("instance_id")
            if iid and node_id not in self._mailbox_index.get(iid, ()):
                self._mailbox_index.setdefault(iid, []).append(node_id)

    def query_cypher(self, cypher, params=None):  # type: ignore[override]
        q = " ".join(cypher.split())
        if "MATCH (m:AgentMessage)" in q:  # indexed mailbox read (O(depth))
            iid = (params or {}).get("iid")
            only_unconsumed = "m.consumed = false" in q
            out = []
            for mid in self._mailbox_index.get(iid, []):
                node = self.nodes.get(mid)
                if not node or (only_unconsumed and node.get("consumed")):
                    continue
                out.append(
                    {
                        "id": mid,
                        "message_ref": node.get("message_ref"),
                        "source": node.get("source"),
                        "correlation_id": node.get("correlation_id"),
                        "created_at": node.get("created_at"),
                    }
                )
            return out
        return super().query_cypher(cypher, params)

    def _candidate(self, request):
        if request.work_item_id:
            return super()._candidate(request)
        now = float(request.now_ms) / 1000.0
        best: tuple[str, dict] | None = None
        best_key: tuple[int, float] | None = None
        for wid in list(self._work_item_ids):
            node = self.nodes.get(wid)
            if not node:
                continue
            if request.queue_ref and node.get("queue") != request.queue_ref:
                continue
            if (
                request.resource_class
                and node.get("resource_class") != request.resource_class
            ):
                continue
            status = node.get("status")
            if status in {"leased", "running"}:
                if float(node.get("lease_expires_at") or 0) >= now:
                    continue
            elif status != "ready":
                continue
            if float(node.get("next_retry_at") or 0) > now:
                continue
            key = (
                int(node.get("prio_bucket") or 0),
                float(node.get("created_at") or 0),
            )
            if best_key is None or key < best_key:
                best_key, best = key, (wid, node)
        return best


def _rss_mb() -> float:
    """Current resident-set size in MiB (Linux /proc; falls back to peak rusage)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * (os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
    except (OSError, ValueError, IndexError):
        import resource

        # ru_maxrss is KiB on Linux, bytes on macOS; assume KiB (Linux CI).
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _count_terminal_activations(engine: ActivationEngine) -> int:
    terminal = wi.TERMINAL_WORK_ITEM_STATUSES
    # Scan only the WorkItem index (O(#activations)), never the whole node store — the
    # completion poll must not itself be O(N) in the dormant-instance count.
    ids = getattr(engine, "_work_item_ids", None)
    if ids is None:
        return sum(
            1
            for n in list(engine.nodes.values())
            if n.get("label") == "WorkItem" and n.get("status") in terminal
        )
    return sum(
        1
        for wid in list(ids)
        if (n := engine.nodes.get(wid)) is not None and n.get("status") in terminal
    )


@dataclass
class ScaleMetrics:
    instances: int
    activations: int
    workers: int
    register_seconds: float = 0.0
    activate_seconds: float = 0.0
    rss_start_mb: float = 0.0
    rss_after_register_mb: float = 0.0
    rss_after_activate_mb: float = 0.0
    register_curve: list[tuple[int, float]] = field(default_factory=list)
    touched_instances: int = 0
    dormant_after: int = 0
    completed_activations: int = 0

    @property
    def rss_per_dormant_kb(self) -> float:
        if self.instances == 0:
            return 0.0
        return (
            (self.rss_after_register_mb - self.rss_start_mb) * 1024.0
        ) / self.instances

    @property
    def activations_per_sec(self) -> float:
        return (
            self.completed_activations / self.activate_seconds
            if self.activate_seconds
            else 0.0
        )


def run_scale_proof(
    *,
    instances: int,
    activations: int,
    workers: int = 4,
    log: Any = None,
) -> tuple[ActivationEngine, ScaleMetrics]:
    """Register ``instances`` dormant agents, activate ``activations`` of them via a
    ``workers``-thread pool, and return ``(engine, metrics)``.

    Asserts the core thesis: exactly ``activations`` instances are touched (get a
    ``:RunTrace``); the other ``instances - activations`` remain purely dormant — so
    activation is O(activations), NOT O(instances).
    """
    log = log or (lambda *_a, **_k: None)
    engine = ScaleActivationEngine()
    m = ScaleMetrics(instances=instances, activations=activations, workers=workers)
    m.rss_start_mb = _rss_mb()

    # ── Phase 1: register N dormant instances (the "million rows" cost), logging the
    #    RSS curve at deciles so the per-instance footprint is visible.
    t0 = time.monotonic()
    instance_ids: list[str] = []
    checkpoint_every = max(1, instances // 10)
    for i in range(instances):
        iid = aa.register_agent_instance(
            engine,
            agent_name=f"agent-{i % 64}",  # 64 distinct agent kinds sharing the template
            tenant=f"tenant-{i % 8}",  # 8 tenants (per-tenant graphs, ADR-6 §4)
            instance_id=f"agent-instance:scale-{i:08d}",
            subscriptions=["data.new"],
            origin_principal=f"user:{i % 100}",
        )
        instance_ids.append(iid)
        if (i + 1) % checkpoint_every == 0:
            m.register_curve.append((i + 1, _rss_mb()))
    m.register_seconds = time.monotonic() - t0
    m.rss_after_register_mb = _rss_mb()
    log(
        f"  registered {instances} dormant instances in {m.register_seconds:.2f}s "
        f"(RSS {m.rss_start_mb:.0f}->{m.rss_after_register_mb:.0f} MiB, "
        f"{m.rss_per_dormant_kb:.2f} KiB/instance)"
    )

    # ── Phase 2: activate M of the N (evenly spread across the dormant population, so
    #    activation is NOT concentrated at one end of the catalog).
    stride = max(1, instances // activations)
    activated_ids = [instance_ids[(k * stride) % instances] for k in range(activations)]
    t1 = time.monotonic()
    for k, iid in enumerate(activated_ids):
        aa.deliver_activation(
            engine,
            iid,
            message_ref=f"msg:{k}",
            # Mix the QoS lanes across the batch (interactive preempts ingest).
            source=aa.ActivationSource.DIRECT
            if k % 3 == 0
            else aa.ActivationSource.CDC,
        )

    # ── Phase 3: the stateless worker POOL drains all M activations concurrently. The
    #    native ClaimWorkItem is tenant-scoped, so the pool serves the 8 registered tenants.
    tenants = [f"tenant-{i}" for i in range(8)]
    threads, stop = aa.start_activation_worker_pool(
        engine, worker_count=workers, tenants=tenants
    )
    # A generous CEILING (the loop breaks the instant all M are terminal). The in-process
    # pool is GIL-bound (~1 core); the production pool is one PROCESS per worker (W5.1) =
    # true N-core parallelism, so real throughput is higher than this floor.
    deadline = time.monotonic() + max(30.0, activations * 0.25)
    while time.monotonic() < deadline:
        if _count_terminal_activations(engine) >= activations:
            break
        time.sleep(0.02)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
    m.activate_seconds = time.monotonic() - t1
    m.rss_after_activate_mb = _rss_mb()
    m.completed_activations = _count_terminal_activations(engine)

    # ── The core-thesis assertions: activation touched ONLY the M activated instances.
    m.touched_instances = len(engine.by_label(aa._RUNTRACE_LABEL))
    m.dormant_after = len(
        aa.list_agent_instances(
            engine, lifecycle_state=aa.STATE_DORMANT, limit=instances + 1
        )
    )
    log(
        f"  activated {m.completed_activations}/{activations} via {workers} workers in "
        f"{m.activate_seconds:.2f}s ({m.activations_per_sec:.0f}/s); "
        f"touched={m.touched_instances} dormant_after={m.dormant_after} "
        f"RSS {m.rss_after_register_mb:.0f}->{m.rss_after_activate_mb:.0f} MiB"
    )

    assert m.completed_activations == activations, (
        "every activation must reach a terminal state"
    )
    assert m.touched_instances == activations, (
        "activation must touch ONLY the activated instances"
    )
    # After release, every activated instance is dormant again ⇒ all N are dormant.
    assert m.dormant_after == instances, "all instances dormant again after release"
    return engine, m


# ── the CI-fast test (parameterized; proves O(M)-not-O(N) at two instance counts) ────


def _cfg(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def test_activation_cost_is_independent_of_dormant_population() -> None:
    """Register a SMALL and a LARGER dormant population, activate the SAME M from each,
    and prove the activation phase does not scale with the dormant count N."""
    activations = _cfg("AGENT_ACTIVATION_SCALE_ACTIVATIONS", 100)
    workers = _cfg("AGENT_ACTIVATION_SCALE_WORKERS", 4)
    small = _cfg("AGENT_ACTIVATION_SCALE_INSTANCES", 1500)
    large = small * 4

    _e1, m_small = run_scale_proof(
        instances=small, activations=activations, workers=workers, log=print
    )
    _e2, m_large = run_scale_proof(
        instances=large, activations=activations, workers=workers, log=print
    )

    # Same M activated from both, regardless of the 4x larger dormant population.
    assert m_small.completed_activations == activations
    assert m_large.completed_activations == activations
    assert m_small.touched_instances == activations
    assert m_large.touched_instances == activations

    # The activation PHASE cost must not grow ~linearly with N (the thesis). The dormant
    # population grew 4x; the activation wall-time must stay well under 4x (generous 2.5x
    # ceiling absorbs scheduler/GC jitter — the point is sub-linear, not O(N)).
    if m_small.activate_seconds > 0.02:  # guard against divide-by-noise on a fast box
        ratio = m_large.activate_seconds / m_small.activate_seconds
        print(
            f"[scale] activate-time ratio (4x instances) = {ratio:.2f} "
            f"(small={m_small.activate_seconds:.3f}s large={m_large.activate_seconds:.3f}s)"
        )
        assert ratio < 2.5, (
            f"activation phase scaled with the dormant count (ratio {ratio:.2f}); "
            "it must be O(activations), not O(instances)"
        )


def test_dormant_registration_footprint_is_bounded_per_instance() -> None:
    """A dormant instance's in-process footprint is a small, roughly-constant per-instance
    cost — the FLOOR the real engine's paging improves on."""
    instances = _cfg("AGENT_ACTIVATION_SCALE_INSTANCES", 3000)
    _e, m = run_scale_proof(instances=instances, activations=50, workers=4, log=print)
    print(
        f"[scale] dormant footprint: {m.rss_per_dormant_kb:.2f} KiB/instance over "
        f"{instances} instances (RSS {m.rss_start_mb:.0f}->{m.rss_after_register_mb:.0f} MiB)"
    )
    # The two-passive-rows-per-dormant-agent footprint is bounded (well under the
    # engine's own paging threshold). Loose ceiling — the assertion is "small + bounded",
    # and the absolute number is logged for the W5.3 soak.
    assert m.rss_per_dormant_kb < 50.0, (
        "dormant per-instance footprint unexpectedly large"
    )


if __name__ == "__main__":  # pragma: no cover — the standalone high-scale proof
    import logging

    logging.disable(logging.INFO)  # silence per-activation logs at 100k/1k scale
    N = _cfg("AGENT_ACTIVATION_SCALE_INSTANCES", 100_000)
    M = _cfg("AGENT_ACTIVATION_SCALE_ACTIVATIONS", 1_000)
    W = _cfg("AGENT_ACTIVATION_SCALE_WORKERS", 8)
    print(
        f"[scale-proof] agents-as-data: {N} dormant instances, {M} concurrent "
        f"activations, {W} workers (in-memory floor)"
    )
    _engine, metrics = run_scale_proof(instances=N, activations=M, workers=W, log=print)
    print("[scale-proof] RSS registration curve (instances -> MiB):")
    for count, rss in metrics.register_curve:
        print(f"    {count:>8} -> {rss:8.1f} MiB")
    print(
        f"[scale-proof] SUMMARY: {N} dormant @ {metrics.rss_per_dormant_kb:.2f} KiB/inst; "
        f"{metrics.completed_activations} activations @ {metrics.activations_per_sec:.0f}/s; "
        f"peak RSS {metrics.rss_after_activate_mb:.0f} MiB; "
        f"touched {metrics.touched_instances} (== {M}), "
        f"{metrics.dormant_after} dormant after."
    )
