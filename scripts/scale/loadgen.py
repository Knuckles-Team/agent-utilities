#!/usr/bin/python
"""SCALE-P2-1 load generator: GENERATES the workload contract's traffic, measures SLOs.

Replaces the linear ``capacity_model.py`` arithmetic as the acceptance criterion for
"1,000,000 residents": instead of computing shard/worker counts from a formula, this
module actually DRIVES the contract's workload — submits :class:`WorkItem`s (turns),
publishes :class:`AgentBus` messages, at the contract's rates and tenant skew — and
measures the p50/p95/p99/p99.9 SLO percentiles the contract defines, against either:

* ``--engine mock`` (default): an in-memory :class:`FakeScaleEngine`
  (:mod:`scripts.scale.fake_engine`) — the CI-safe path, no live services required.
  This is what ``tests/scale/soak/`` runs at a small ``--scale``.
* ``--engine live``: the process-active epistemic-graph engine
  (:class:`agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine`) — the
  real-hardware soak path, run manually against a deployed fleet at ``--scale 1.0``.
  NOT exercised in CI; see ``docs/scaling/capacity_model.md`` for the honest
  measured-vs-modeled split.

``--scale`` (0, 1] shrinks the population/rate axes for a fast, deterministic CI run
(:class:`docs.scaling.workload_contract.ScaledWorkload`) while leaving the SLO percentile
targets untouched — an SLO is a per-operation contract, not a population-dependent one.

Reusable programmatic entry point: :func:`run_workload` (async). The CLI (:func:`main`)
is a thin wrapper that also supports optional scripted :class:`FaultEvent`\\ s so the
same driver backs both a plain load-generation run and the soak/chaos scenarios in
``tests/scale/soak/``.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import importlib.util
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from agent_utilities.messaging.bus import AgentBus
from agent_utilities.orchestration import work_item as wi

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path) -> ModuleType:
    """Dynamically load a sibling module by path (mirrors ``tests/scale/test_capacity_model.py``'s
    ``_load_model()`` — this repo's established convention for importing ``scripts/``/
    ``docs/scaling/`` files, neither of which is a packaged, installed module)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_fake_engine = _load_module(
    "agent_utilities_scale_fake_engine", Path(__file__).with_name("fake_engine.py")
)
_workload_contract = _load_module(
    "agent_utilities_scale_workload_contract",
    _REPO_ROOT / "docs" / "scaling" / "workload_contract.py",
)

FakeScaleEngine = _fake_engine.FakeScaleEngine
LatencyModel = _fake_engine.LatencyModel
WallClock = _fake_engine.WallClock
ScaledWorkload = _workload_contract.ScaledWorkload
WorkloadContract = _workload_contract.WorkloadContract
load_workload_contract = _workload_contract.load_workload_contract

# --------------------------------------------------------------------------- #
# Fault injection — scripted, deterministic, applied at a scheduled logical
# time inside the discrete-event loop (:func:`_run_mock_workload`). Only the
# two fault kinds :func:`run_workload` itself needs are implemented here
# (``degrade_latency`` — simulated broker/shard backpressure; ``pause_workers``
# / ``resume_workers`` — simulated worker/gateway loss during a steady/burst
# run). The richer chaos scenarios (duplicate delivery, tenant quota, cold
# restart, rolling upgrade, cancel/DLQ) are built directly against the engine
# primitives in ``tests/scale/soak/`` instead of through this generic hook —
# they need to construct a SPECIFIC state (e.g. "this exact item is leased by
# a stale token") that a rate-based fault schedule can't express precisely.
# --------------------------------------------------------------------------- #


@dataclass
class FaultEvent:
    """One scheduled fault, applied at ``at_s`` logical seconds into the run."""

    at_s: float
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultPlan:
    events: list[FaultEvent] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Percentile helpers
# --------------------------------------------------------------------------- #


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def _percentiles_ms(values_s: list[float]) -> dict[str, float]:
    ms = [v * 1000.0 for v in values_s]
    return {
        "p50": round(_pct(ms, 0.50), 3),
        "p95": round(_pct(ms, 0.95), 3),
        "p99": round(_pct(ms, 0.99), 3),
        "p99_9": round(_pct(ms, 0.999), 3),
    }


def _slo_pass(measured: dict[str, float], target: Any) -> dict[str, bool]:
    return {
        "p50": measured["p50"] <= target.p50,
        "p95": measured["p95"] <= target.p95,
        "p99": measured["p99"] <= target.p99,
        "p99_9": measured["p99_9"] <= target.p99_9,
    }


# --------------------------------------------------------------------------- #
# Tenant population (skew, incl. the elephant tenant)
# --------------------------------------------------------------------------- #


@dataclass
class TenantPlan:
    ids: list[str]
    turn_weights: list[float]  # normalized, sums to 1.0
    message_weights: list[float]  # normalized, sums to 1.0
    elephant_id: str

    def sample_turn_tenant(self, rng: random.Random) -> str:
        return rng.choices(self.ids, weights=self.turn_weights, k=1)[0]

    def sample_message_tenant(self, rng: random.Random) -> str:
        return rng.choices(self.ids, weights=self.message_weights, k=1)[0]


def build_tenant_plan(scaled: ScaledWorkload) -> TenantPlan:
    contract = scaled.contract
    n = scaled.tenant_count
    elephant_id = "tenant-elephant"
    ordinary_ids = [f"tenant-{i}" for i in range(n - 1)]
    ids = [elephant_id] + ordinary_ids

    elephant_active = contract.tenants.elephant.active_fraction
    elephant_messages = contract.tenants.elephant.messages_fraction
    remaining_active = max(0.0, 1.0 - elephant_active)
    remaining_messages = max(0.0, 1.0 - elephant_messages)

    ordinary_raw = [scaled.tenant_weight(i + 1) for i in range(len(ordinary_ids))]
    raw_sum = sum(ordinary_raw) or 1.0
    turn_weights = [elephant_active] + [
        remaining_active * (w / raw_sum) for w in ordinary_raw
    ]
    message_weights = [elephant_messages] + [
        remaining_messages * (w / raw_sum) for w in ordinary_raw
    ]
    return TenantPlan(
        ids=ids,
        turn_weights=turn_weights,
        message_weights=message_weights,
        elephant_id=elephant_id,
    )


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #


def build_mock_engine(latency: LatencyModel | None = None) -> FakeScaleEngine:
    # pace_mode="none": the discrete-event driver (_run_mock_workload) accounts
    # for synthetic op latency itself (sampled straight into its metrics), so the
    # engine must not ALSO consume real or simulated time per op — see
    # FakeScaleEngine's docstring / WallClock's docstring in fake_engine.py.
    return FakeScaleEngine(latency=latency, pace_mode="none")


def build_live_engine() -> Any:
    """Resolve the process-active epistemic-graph engine for a real hardware soak.

    Raises if no engine is active — a live run must be pointed at a real, configured
    deployment; it never silently falls back to the mock (that would defeat the
    entire measured-vs-modeled honesty point of this harness).
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine is None:
        raise RuntimeError(
            "--engine live requires an active IntelligenceGraphEngine (none is "
            "configured/reachable in this process) — start against a real graph-os "
            "deployment, or use --engine mock for the CI-safe simulated path."
        )
    return engine


def build_bus(engine: Any, *, force_graph_fallback: bool) -> AgentBus:
    """Construct an :class:`AgentBus` bound to ``engine``.

    ``force_graph_fallback=True`` (the default for mock-engine runs) pins the
    delivery/wakeup plane to the original ``:BusMessage`` graph-node model
    regardless of ambient ``AGENT_BUS_LOG_BACKEND``/``ENGINE_ENDPOINT`` env —
    :class:`FakeScaleEngine` has no real broker to bind to, and this dev
    workspace's ambient config can otherwise point ``resolve_bus_log_backend``
    at a REAL deployed hub (a hermetic-run hazard, not a hypothetical one — this
    was observed while building the harness). A live-engine run leaves the
    normal auto-resolution alone (a real deployment's actual delivery plane).
    """
    bus = AgentBus(engine=engine)
    if force_graph_fallback:
        bus._log_backend_cache = None  # noqa: SLF001 — deliberate hermetic pin, see above
    return bus


# --------------------------------------------------------------------------- #
# The workload driver
# --------------------------------------------------------------------------- #


@dataclass
class WorkloadReport:
    ok: bool
    scale: float
    duration_s: float
    real_duration_s: float
    turn_duration_s: float
    counts: dict[str, int]
    throughput: dict[str, float]
    latency_ms: dict[str, dict[str, float]]
    slo_target: dict[str, dict[str, float]]
    slo_pass: dict[str, dict[str, bool]]
    invariants: dict[str, Any]
    faults_applied: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "scale": self.scale,
            "duration_s": self.duration_s,
            "real_duration_s": self.real_duration_s,
            "turn_duration_s": self.turn_duration_s,
            "counts": self.counts,
            "throughput": self.throughput,
            "latency_ms": self.latency_ms,
            "slo_target": self.slo_target,
            "slo_pass": self.slo_pass,
            "invariants": self.invariants,
            "faults_applied": self.faults_applied,
        }


class _Metrics:
    def __init__(self) -> None:
        self.queue_latency_s: list[float] = []
        self.query_latency_s: list[float] = []
        self.write_latency_s: list[float] = []
        self.end_to_end_latency_s: list[float] = []
        self.turns_submitted = 0
        self.turns_succeeded = 0
        self.turns_failed = 0
        self.turns_dead_letter = 0
        self.turns_cancelled = 0
        self.messages_sent = 0
        self.messages_delivered = 0
        # Invariant bookkeeping: item_id -> list of (attempt, worker_id) side-effect executions.
        self.side_effects: dict[str, list[tuple[int, str]]] = {}
        self.submit_ts: dict[str, float] = {}
        self.submit_tenant: dict[str, str] = {}
        self.faults_applied: list[dict[str, Any]] = []


async def _turn_producer(
    engine: Any,
    scaled: ScaledWorkload,
    tenants: TenantPlan,
    metrics: _Metrics,
    clock: Any,
    stop_at: float,
    rng: random.Random,
) -> None:
    """Submits WorkItems (turns) at ``turns_per_sec`` with tenant skew (Poisson-ish)."""
    turns_per_sec = max(scaled.turns_per_sec, 0.001)
    n = 0
    while clock.now() < stop_at:
        tenant = tenants.sample_turn_tenant(rng)
        session_id = f"{tenant}:sess-{n}"
        n += 1
        item_id = wi.submit_work_item(
            engine,
            kind="agent_turn",
            payload_ref=session_id,
            tenant=tenant,
            priority=2,
        )
        metrics.turns_submitted += 1
        metrics.submit_ts[item_id] = clock.now()
        metrics.submit_tenant[item_id] = tenant
        # Poisson-ish inter-arrival: exponential with mean 1/rate.
        await clock.sleep(rng.expovariate(turns_per_sec))


async def _mutation_producer(
    engine: Any,
    scaled: ScaledWorkload,
    metrics: _Metrics,
    clock: Any,
    stop_at: float,
    rng: random.Random,
) -> None:
    """Independent axis: graph mutations at ``graph_mutations_per_sec`` (write-latency samples).

    Deliberately a SEPARATE paced task, not derived from per-turn counts — the
    contract declares turns/s, tool-calls/s, graph-mutations/s, and messages/s as
    INDEPENDENT rate axes (a real fleet's mutation traffic includes background
    writes with no 1:1 turn correspondence), and pacing each axis on its own
    keeps the simulated rate stable as ``--scale`` grows instead of exploding a
    per-turn multiplier on a single-threaded mock (observed while building this
    harness: deriving mutation count from graph_mutations_per_sec/turns_per_sec
    per turn made the simulated engine CPU-bound, throttling the very arrival
    rate scaling this generator exists to exercise).
    """
    rate = max(scaled.graph_mutations_per_sec, 0.001)
    n = 0
    while clock.now() < stop_at:
        t0 = clock.now()
        engine.add_node(
            f"scalegen:mutation:{n % _MUTATION_POOL_SIZE}",
            "ScaleGenMutation",
            properties={"i": n},
        )
        metrics.write_latency_s.append(clock.now() - t0)
        n += 1
        await clock.sleep(rng.expovariate(rate))


async def _tool_call_producer(
    engine: Any,
    scaled: ScaledWorkload,
    metrics: _Metrics,
    clock: Any,
    stop_at: float,
    rng: random.Random,
) -> None:
    """Independent axis: tool calls at ``tool_calls_per_sec``, modeled as graph reads
    (query-latency samples) — the read side of a tool invocation's KG lookup."""
    rate = max(scaled.tool_calls_per_sec, 0.001)
    while clock.now() < stop_at:
        t0 = clock.now()
        engine.query_cypher(
            "MATCH (w:WorkItem {status: $status, prio_bucket: $bucket}) "
            "RETURN w.id AS id, w.created_at AS created_at, w.next_retry_at AS next_retry_at, "
            "w.resource_class AS resource_class, w.tenant AS tenant, "
            "w.fairness_group AS fairness_group LIMIT 4",
            {"status": wi.WorkItemStatus.READY.value, "bucket": 2},
        )
        metrics.query_latency_s.append(clock.now() - t0)
        await clock.sleep(rng.expovariate(rate))


async def _worker(
    engine: Any,
    scaled: ScaledWorkload,
    metrics: _Metrics,
    turn_duration_s: float,
    worker_id: str,
    clock: Any,
    stop_at: float,
    drain_grace_s: float,
    rng: random.Random,
) -> None:
    """Claims + executes + commits turns until the hard deadline ``stop_at + drain_grace_s``.

    Keeps polling for the FULL grace window even once no more turns are arriving
    (rather than retiring the instant ``claim_and_start`` returns ``None`` past
    ``stop_at``) — a worker that retires early can strand a turn that was
    submitted a moment later than this worker's last check, which shows up as a
    spurious multi-second queue-latency outlier that is a test-harness boundary
    artifact, not a real system property (observed while building this harness).
    """
    poll_interval_s = 0.005
    while True:
        now = clock.now()
        if now >= stop_at + drain_grace_s:
            return
        t0 = clock.now()
        claim = wi.claim_and_start(engine, tenant=None)
        if claim is None:
            await clock.sleep(poll_interval_s)
            continue
        item_id = claim["work_item_id"]
        submit_ts = metrics.submit_ts.get(item_id)
        if submit_ts is not None:
            metrics.queue_latency_s.append(t0 - submit_ts)

        if turn_duration_s > 0:
            await clock.sleep(turn_duration_s)
        # Record the side effect BEFORE commit (mirrors real execution-then-writeback
        # ordering) — this is what the falsely-completed / duplicate-side-effect
        # invariant checks against after the run.
        metrics.side_effects.setdefault(item_id, []).append(
            (claim["attempt"], worker_id)
        )

        t_commit = clock.now()
        outcome = wi.commit_result(
            engine, item_id, claim, outcome="succeeded", result_ref="ok"
        )
        metrics.write_latency_s.append(clock.now() - t_commit)

        if outcome == "committed":
            metrics.turns_succeeded += 1
            if submit_ts is not None:
                metrics.end_to_end_latency_s.append(clock.now() - submit_ts)
        elif outcome == "dead_letter":
            metrics.turns_dead_letter += 1
        elif outcome == "retry_scheduled":
            pass  # will be reclaimed by a future poll
        # "noop"/"fenced"/"missing" indicate a redelivery race — never counted twice.


async def _message_producer(
    engine: Any,
    bus: AgentBus,
    scaled: ScaledWorkload,
    tenants: TenantPlan,
    metrics: _Metrics,
    clock: Any,
    stop_at: float,
    rng: random.Random,
) -> None:
    """Publishes AgentBus messages at ``messages_per_sec`` with tenant skew.

    Each tenant gets its OWN namespaced pair of bus participants
    (``{tenant}:sender`` / ``{tenant}:receiver``) so cross-tenant delivery leakage
    is directly observable: a receiver must never see a message whose payload
    tenant tag does not match its own namespace.
    """
    rate = max(scaled.messages_per_sec, 0.001)
    registered: set[str] = set()
    while clock.now() < stop_at:
        tenant = tenants.sample_message_tenant(rng)
        sender, receiver = f"{tenant}:sender", f"{tenant}:receiver"
        if tenant not in registered:
            bus.register(sender, provider="loadgen")
            bus.register(receiver, provider="loadgen")
            registered.add(tenant)
        result = bus.send(
            sender=sender, payload=json.dumps({"tenant": tenant}), to=receiver
        )
        metrics.messages_sent += 1
        if result.get("ok") and result.get("delivered"):
            metrics.messages_delivered += 1
        await clock.sleep(rng.expovariate(rate))


# --------------------------------------------------------------------------- #
# Mock-engine driver: a single-threaded discrete-event simulation (DES).
#
# NOT built from the async producer/worker functions above (those back
# ``--engine live``, where genuine OS wall-clock time correctly serializes
# truly-concurrent asyncio tasks). A synthetic "logical clock" shared by many
# concurrently-sleeping asyncio tasks does NOT have that property — see
# ``WallClock``'s docstring in ``fake_engine.py`` for the compounding-advance
# bug this replaced. A DES has exactly one thing happening at a time, in
# strict time order, so it is free of that race by construction, and it is
# also near-instant in real wall time (nothing ever really sleeps) and
# immune to host CPU contention — both wins for CI.
# --------------------------------------------------------------------------- #

_POLL_INTERVAL_S = 0.005
#: The mutation producer upserts a BOUNDED pool of node ids rather than a fresh
#: id per event — at the contract's graph_mutations_per_sec (up to 40,000/s
#: unscaled) a fresh-id-per-event design accumulates tens of thousands of nodes
#: within seconds, and FakeScaleEngine's query dispatch (a plain linear scan —
#: the SAME pattern the existing FakeEngine test doubles use, see
#: tests/unit/orchestration/test_work_item.py) then costs O(claims × nodes),
#: which measured minutes for a 5s run (observed while building this harness).
#: A small fixed pool still exercises a genuine write per event without the
#: unbounded node-count blowup — the SLO measurement cares about op latency,
#: not about accumulating a large corpus.
_MUTATION_POOL_SIZE = 200
#: Hard cap on simulated message-producer events in mock mode. FINDING (not a
#: SCALE-P2-1 fix — filed as a follow-up): ``AgentBus.send``'s governed path
#: (``ActionPolicy.decide`` -> notify) measured ~300ms/call in this sandbox —
#: two-plus orders of magnitude above the AddNode anchor — apparently from
#: per-call pydantic model construction somewhere on that path (profiled while
#: building this harness: repeated ``builtins.exec``/``compile``/pydantic
#: schema-generation frames per ``decide()`` call, not a one-time warmup cost).
#: Driving ``messages_per_sec`` at its full contract-scaled rate through that
#: path would make every CI run take minutes for no benefit to THIS harness's
#: purpose (cross-tenant delivery-isolation correctness, not benchmarking an
#: unrelated notifier). Capped to a small, still-meaningful correctness sample;
#: the full message RATE is exercised for real on ``--engine live``.
_MAX_MESSAGE_EVENTS = 8


def _run_mock_workload(
    engine: Any,
    bus: AgentBus,
    scaled: ScaledWorkload,
    tenants: TenantPlan,
    metrics: _Metrics,
    *,
    duration_s: float,
    turn_duration_s: float,
    num_workers: int,
    drain_grace_s: float,
    rng: random.Random,
    fault_plan: FaultPlan | None = None,
) -> float:
    """Run one workload against a :class:`FakeScaleEngine` via a time-ordered event heap.

    Returns the final logical time reached (~``duration_s + drain_grace_s``).
    """
    hard_stop = duration_s + drain_grace_s
    heap: list[tuple[float, int, str, Any]] = []
    seq = itertools.count()

    def push(at: float, kind: str, data: Any = None) -> None:
        heapq.heappush(heap, (at, next(seq), kind, data))

    turns_per_sec = max(scaled.turns_per_sec, 0.001)
    mutation_rate = max(scaled.graph_mutations_per_sec, 0.001)
    toolcall_rate = max(scaled.tool_calls_per_sec, 0.001)
    message_rate = max(scaled.messages_per_sec, 0.001)

    push(rng.expovariate(turns_per_sec), "arrival")
    push(rng.expovariate(mutation_rate), "mutation")
    push(rng.expovariate(toolcall_rate), "toolcall")
    push(rng.expovariate(message_rate), "message")
    for i in range(num_workers):
        push(0.0, "poll", i)

    events = sorted(fault_plan.events, key=lambda e: e.at_s) if fault_plan else []
    fault_idx = 0
    paused_workers: set[int] = set()
    applied_faults: list[dict[str, Any]] = []

    turn_n = 0
    message_n = 0
    registered_tenants: set[str] = set()
    t = 0.0

    while heap:
        at, _, kind, data = heapq.heappop(heap)
        if kind == "poll":
            if at > hard_stop:
                continue  # this worker has retired
        elif at > duration_s:
            continue  # producer streams stop scheduling past duration_s
        t = at

        # Apply any faults scheduled at or before this event's time (deterministic:
        # faults land strictly in ``at_s`` order, coalesced onto the next event).
        while fault_idx < len(events) and events[fault_idx].at_s <= t:
            ev = events[fault_idx]
            if ev.kind == "degrade_latency":
                engine.latency.degradation_multiplier = float(
                    ev.params.get("multiplier", 1.0)
                )
            elif ev.kind == "pause_workers":
                ids = ev.params.get("worker_ids")
                paused_workers.update(
                    range(num_workers) if ids is None else [int(i) for i in ids]
                )
            elif ev.kind == "resume_workers":
                ids = ev.params.get("worker_ids")
                if ids is None:
                    paused_workers.clear()
                else:
                    paused_workers.difference_update(int(i) for i in ids)
            applied_faults.append(
                {"at_s": ev.at_s, "kind": ev.kind, "params": ev.params}
            )
            fault_idx += 1

        if kind == "arrival":
            tenant = tenants.sample_turn_tenant(rng)
            session_id = f"{tenant}:sess-{turn_n}"
            turn_n += 1
            item_id = wi.submit_work_item(
                engine,
                kind="agent_turn",
                payload_ref=session_id,
                tenant=tenant,
                priority=2,
            )
            metrics.turns_submitted += 1
            metrics.submit_ts[item_id] = t
            metrics.submit_tenant[item_id] = tenant
            push(t + rng.expovariate(turns_per_sec), "arrival")

        elif kind == "mutation":
            delay = engine.latency.write_delay()
            engine.add_node(
                f"scalegen:mutation:{next(seq) % _MUTATION_POOL_SIZE}",
                "ScaleGenMutation",
                properties={},
            )
            metrics.write_latency_s.append(delay)
            push(t + rng.expovariate(mutation_rate), "mutation")

        elif kind == "toolcall":
            delay = engine.latency.query_delay()
            engine.query_cypher(
                "MATCH (w:WorkItem {status: $status, prio_bucket: $bucket}) "
                "RETURN w.id AS id, w.created_at AS created_at, w.next_retry_at AS next_retry_at, "
                "w.resource_class AS resource_class, w.tenant AS tenant, "
                "w.fairness_group AS fairness_group LIMIT 4",
                {"status": wi.WorkItemStatus.READY.value, "bucket": 2},
            )
            metrics.query_latency_s.append(delay)
            push(t + rng.expovariate(toolcall_rate), "toolcall")

        elif kind == "message":
            tenant = tenants.sample_message_tenant(rng)
            sender, receiver = f"{tenant}:sender", f"{tenant}:receiver"
            if tenant not in registered_tenants:
                bus.register(sender, provider="loadgen")
                bus.register(receiver, provider="loadgen")
                registered_tenants.add(tenant)
            result = bus.send(
                sender=sender, payload=json.dumps({"tenant": tenant}), to=receiver
            )
            metrics.messages_sent += 1
            message_n += 1
            if result.get("ok") and result.get("delivered"):
                metrics.messages_delivered += 1
            if message_n < _MAX_MESSAGE_EVENTS:
                push(t + rng.expovariate(message_rate), "message")

        elif kind == "poll":
            worker_idx = data
            if worker_idx in paused_workers:
                push(t + _POLL_INTERVAL_S, "poll", worker_idx)
                continue
            claim = wi.claim_and_start(engine, tenant=None)
            if claim is None:
                push(t + _POLL_INTERVAL_S, "poll", worker_idx)
                continue
            item_id = claim["work_item_id"]
            submit_ts = metrics.submit_ts.get(item_id)
            if submit_ts is not None:
                metrics.queue_latency_s.append(t - submit_ts)

            t_done = t + max(0.0, turn_duration_s)
            # Record the side effect BEFORE commit (mirrors real execution-then-
            # writeback ordering) — the falsely-completed / duplicate-side-effect
            # invariant checks read this after the run.
            metrics.side_effects.setdefault(item_id, []).append(
                (claim["attempt"], f"worker-{worker_idx}")
            )
            commit_delay = engine.latency.write_delay()
            outcome = wi.commit_result(
                engine, item_id, claim, outcome="succeeded", result_ref="ok"
            )
            metrics.write_latency_s.append(commit_delay)
            if outcome == "committed":
                metrics.turns_succeeded += 1
                if submit_ts is not None:
                    metrics.end_to_end_latency_s.append(t_done - submit_ts)
            elif outcome == "dead_letter":
                metrics.turns_dead_letter += 1
            # "noop"/"fenced"/"missing"/"retry_scheduled" — never double-counted.
            push(t_done, "poll", worker_idx)  # worker free again at t_done

    metrics.faults_applied = applied_faults
    return t


def _drain_messages_and_check_isolation(
    bus: AgentBus, tenants: TenantPlan
) -> dict[str, Any]:
    """Drain every tenant's receiver mailbox; assert every message's tenant tag matches."""
    cross_tenant_violations: list[dict[str, Any]] = []
    total_received = 0
    for tenant in tenants.ids:
        receiver = f"{tenant}:receiver"
        out = bus.receive(receiver)
        for m in out.get("messages", []):
            total_received += 1
            try:
                payload = json.loads(m.get("payload", "{}"))
            except (TypeError, ValueError):
                payload = {}
            if payload.get("tenant") != tenant:
                cross_tenant_violations.append(
                    {"receiver": receiver, "message": m, "expected_tenant": tenant}
                )
    return {
        "messages_received": total_received,
        "cross_tenant_violations": cross_tenant_violations,
    }


def _check_work_item_invariants(engine: Any, metrics: _Metrics) -> dict[str, Any]:
    """No lost / duplicate / falsely-completed WorkItems.

    * lost: submitted but never reaches a terminal or actively-progressing state
      by the end of the run (still ``ready``/``submitted`` with no claim ever
      recorded is fine — it just didn't get to run in the window — but a
      ``leased``/``running`` item whose lease has long expired with no
      resolution would indicate a genuinely stuck item).
    * duplicate: more than one side-effect execution recorded for one item id
      (the executor recorded the side effect twice — a real bug that a naive
      redelivery-unsafe worker would exhibit).
    * falsely-completed: an item is ``succeeded`` but has NO recorded side
      effect (committed without ever running the body), or a side effect ran
      but the item never reached ``succeeded`` (the inverse gap).
    """
    duplicate_side_effects = {
        item_id: execs
        for item_id, execs in metrics.side_effects.items()
        if len(execs) > 1
    }
    succeeded_ids = set()
    stuck_ids = []
    all_items = engine.work_items() if hasattr(engine, "work_items") else []
    for item in all_items:
        if item.get("label") != "WorkItem":
            continue
        status = item.get("status")
        if status == wi.WorkItemStatus.SUCCEEDED.value:
            succeeded_ids.add(item["id"])
        if status in (wi.WorkItemStatus.LEASED.value, wi.WorkItemStatus.RUNNING.value):
            expires = item.get("lease_expires_at") or 0.0
            if expires < time.time() - 3600:  # far stale, well past any reasonable TTL
                stuck_ids.append(item["id"])

    falsely_completed = [
        item_id for item_id in succeeded_ids if not metrics.side_effects.get(item_id)
    ]
    ran_but_not_succeeded = [
        item_id
        for item_id in metrics.side_effects
        if item_id not in succeeded_ids
        and item_id
        not in {
            i["id"]
            for i in all_items
            if i.get("status")
            in (
                wi.WorkItemStatus.FAILED.value,
                wi.WorkItemStatus.CANCELLED.value,
                wi.WorkItemStatus.DEAD_LETTER.value,
            )
        }
    ]

    return {
        "duplicate_side_effects": duplicate_side_effects,
        "falsely_completed": falsely_completed,
        "ran_but_not_terminal": ran_but_not_succeeded,
        "stuck_leases": stuck_ids,
    }


async def run_workload(
    contract: WorkloadContract,
    *,
    scale: float,
    duration_s: float,
    turn_duration_s: float | None = None,
    num_workers: int = 8,
    seed: int | None = None,
    engine: Any = None,
    drain_grace_s: float = 2.0,
    assert_invariants: bool = True,
    fault_plan: FaultPlan | None = None,
) -> WorkloadReport:
    """Drive the contract's workload against ``engine`` (a mock or live engine).

    ``engine=None`` (the default) drives the CI-safe path: a synchronous
    discrete-event simulation (:func:`_run_mock_workload`) against a fresh
    :class:`FakeScaleEngine` — near-instant in real time, immune to host CPU
    jitter, and free of the shared-clock race a naive concurrent-asyncio
    simulation would have (see ``fake_engine.WallClock``'s docstring). Passing
    a real engine (``build_live_engine()``) instead switches to genuine
    concurrent asyncio tasks against real wall-clock time — the real-hardware
    soak path, not exercised in CI.

    Returns a :class:`WorkloadReport` with measured percentiles, throughput, SLO
    pass/fail per axis, and invariant findings. Never raises on an SLO miss or an
    invariant violation — callers (CLI ``--assert-slo``, the soak/chaos pytest
    scenarios) decide what to do with ``report.ok``/``report.invariants``.

    ``fault_plan`` is only wired into the mock (DES) path today — a real
    hardware soak's faults are injected externally, by the operator, against
    the real infrastructure (see ``docs/scaling/capacity_model.md``'s
    hardware-pending scenario table), not scripted through this parameter.
    """
    rng = random.Random(seed)
    scaled = ScaledWorkload.for_scale(contract, scale)
    tenants = build_tenant_plan(scaled)
    metrics = _Metrics()
    effective_turn_duration = (
        turn_duration_s if turn_duration_s is not None else contract.avg_turn_duration_s
    )
    wall_clock_start = time.monotonic()

    if engine is None:
        resolved_engine = build_mock_engine()
        bus = build_bus(resolved_engine, force_graph_fallback=True)
        wall_s = _run_mock_workload(
            resolved_engine,
            bus,
            scaled,
            tenants,
            metrics,
            duration_s=duration_s,
            turn_duration_s=effective_turn_duration,
            num_workers=num_workers,
            drain_grace_s=drain_grace_s,
            rng=rng,
            fault_plan=fault_plan,
        )
    else:
        resolved_engine = engine
        bus = build_bus(resolved_engine, force_graph_fallback=False)
        clock = WallClock()
        start = clock.now()
        stop_at = start + duration_s
        workers = [
            asyncio.create_task(
                _worker(
                    resolved_engine,
                    scaled,
                    metrics,
                    effective_turn_duration,
                    f"worker-{i}",
                    clock,
                    stop_at,
                    drain_grace_s,
                    rng,
                )
            )
            for i in range(num_workers)
        ]
        producer = asyncio.create_task(
            _turn_producer(
                resolved_engine, scaled, tenants, metrics, clock, stop_at, rng
            )
        )
        msg_producer = asyncio.create_task(
            _message_producer(
                resolved_engine, bus, scaled, tenants, metrics, clock, stop_at, rng
            )
        )
        mutation_producer = asyncio.create_task(
            _mutation_producer(resolved_engine, scaled, metrics, clock, stop_at, rng)
        )
        tool_call_producer = asyncio.create_task(
            _tool_call_producer(resolved_engine, scaled, metrics, clock, stop_at, rng)
        )
        await asyncio.gather(
            producer, msg_producer, mutation_producer, tool_call_producer, *workers
        )
        wall_s = clock.now() - start

    real_wall_s = time.monotonic() - wall_clock_start

    bus_result = _drain_messages_and_check_isolation(bus, tenants)
    invariants = (
        _check_work_item_invariants(resolved_engine, metrics)
        if assert_invariants
        else {}
    )
    invariants.update(bus_result)

    latency_ms = {
        "queue_latency_ms": _percentiles_ms(metrics.queue_latency_s),
        "query_latency_ms": _percentiles_ms(metrics.query_latency_s),
        "write_latency_ms": _percentiles_ms(metrics.write_latency_s),
        "end_to_end_latency_ms": _percentiles_ms(metrics.end_to_end_latency_s),
    }
    slo_target = {axis: target.as_dict() for axis, target in contract.slo.items()}
    slo_pass = {
        axis: _slo_pass(latency_ms[axis], contract.slo[axis]) for axis in contract.slo
    }

    invariants_ok = (
        not invariants.get("duplicate_side_effects")
        and not invariants.get("falsely_completed")
        and not invariants.get("stuck_leases")
        and not invariants.get("cross_tenant_violations")
    )
    slos_ok = all(all(v.values()) for v in slo_pass.values())

    counts = {
        "turns_submitted": metrics.turns_submitted,
        "turns_succeeded": metrics.turns_succeeded,
        "turns_dead_letter": metrics.turns_dead_letter,
        "turns_failed": metrics.turns_failed,
        "turns_cancelled": metrics.turns_cancelled,
        "messages_sent": metrics.messages_sent,
        "messages_delivered": metrics.messages_delivered,
    }
    throughput = {
        "turns_per_sec_measured": round(metrics.turns_succeeded / wall_s, 3)
        if wall_s
        else 0.0,
        "messages_per_sec_measured": round(metrics.messages_sent / wall_s, 3)
        if wall_s
        else 0.0,
        "mutations_per_sec_measured": round(len(metrics.write_latency_s) / wall_s, 3)
        if wall_s
        else 0.0,
    }

    return WorkloadReport(
        ok=bool(slos_ok and invariants_ok),
        scale=scale,
        duration_s=wall_s,
        real_duration_s=round(real_wall_s, 3),
        turn_duration_s=effective_turn_duration,
        counts=counts,
        throughput=throughput,
        latency_ms=latency_ms,
        slo_target=slo_target,
        slo_pass=slo_pass,
        invariants=invariants,
        faults_applied=metrics.faults_applied,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "SCALE-P2-1 load generator: drives the workload_contract.yml traffic "
            "shape and measures its SLO percentiles (queue/query/write/end-to-end)."
        )
    )
    p.add_argument(
        "--contract",
        default=None,
        help="Path to workload_contract.yml (default: docs/scaling/workload_contract.yml)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Population/rate scale factor in (0, 1]. Default 0.001 (~1,000 residents) "
        "is CI-safe; use 1.0 for a real hardware soak against a live engine.",
    )
    p.add_argument("--duration-s", type=float, default=5.0)
    p.add_argument(
        "--turn-duration-s",
        type=float,
        default=None,
        help="Override the contract's avg_turn_duration_s for fast CI iteration "
        "(default: use the contract's value unscaled).",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--engine", choices=("mock", "live"), default="mock")
    p.add_argument("--assert-slo", action="store_true")
    p.add_argument("--report-json", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    contract = load_workload_contract(args.contract)
    engine = build_live_engine() if args.engine == "live" else None

    report = asyncio.run(
        run_workload(
            contract,
            scale=args.scale,
            duration_s=args.duration_s,
            turn_duration_s=args.turn_duration_s,
            num_workers=args.workers,
            seed=args.seed,
            engine=engine,
        )
    )
    payload = report.to_dict()
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.report_json:
        Path(args.report_json).write_text(text)

    if args.assert_slo and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
