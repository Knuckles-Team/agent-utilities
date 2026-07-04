#!/usr/bin/python
"""Capacity model for the agent platform (MODELED, not load-tested).

This module implements the arithmetic behind ``capacity_model.md``. It maps a
resident agent **population** and an **active fraction** onto the infrastructure
required along three independent axes:

1. **Active concurrency** -> worker pool size / node count.
2. **Resident population** -> Postgres shards and L0 (in-memory) shards.
3. **Event throughput** -> Kafka partitions.

The single *measured* anchor is the epistemic-graph transport benchmark
(``epistemic-graph/docs/benchmarks.md``): ``AddNode`` p50 = 0.187 ms on a single
connection, i.e. ~5,000 sequential ops/sec/connection. Everything else here is a
**linear extrapolation** from that anchor plus the per-unit capacity constants
declared below. None of the multi-host numbers (10k+) have been load-tested; they
are a planning model only.

All functions are pure and deterministic so they can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

# --------------------------------------------------------------------------- #
# Measured anchor (see epistemic-graph/docs/benchmarks.md)
# --------------------------------------------------------------------------- #

#: Measured AddNode p50 latency on a single UDS connection, milliseconds.
MEASURED_ADDNODE_P50_MS = 0.187

#: Derived from the anchor: sequential ops/sec on a single connection
#: (1000 / 0.187 ~= 5347, rounded to the documented ~5,000).
SINGLE_CONNECTION_OPS_PER_SEC = 5_000

# --------------------------------------------------------------------------- #
# Per-unit capacity constants (MODELED planning assumptions).
# These are deliberately conservative round numbers used for sizing.
# --------------------------------------------------------------------------- #

#: Resident agents whose persistent state fits comfortably in one Postgres shard.
#: Sized for headroom on a single well-provisioned primary + replicas.
RESIDENTS_PER_PG_SHARD = 250_000

#: Resident agents whose *hot* working set fits in one in-memory L0 shard.
#: L0 is the hot tier, so it holds fewer residents than a durable PG shard.
RESIDENTS_PER_L0_SHARD = 50_000

#: Active (concurrently executing) agents one worker can service.
#: An "active" agent is not pinned to a worker for its whole turn; workers
#: multiplex across I/O waits, so one worker covers several active agents.
ACTIVE_AGENTS_PER_WORKER = 25

#: Workers provisioned per node (mirrors AgentConfig.worker_pool_size default).
WORKERS_PER_NODE = 8

#: Sustained graph ops/sec a single Kafka partition's downstream consumer group
#: can absorb end-to-end. Anchored to the single-connection number: one consumer
#: connection drains ~5,000 ops/sec, so we size one partition per connection.
OPS_PER_SEC_PER_KAFKA_PARTITION = SINGLE_CONNECTION_OPS_PER_SEC

#: Average graph events emitted per active agent per second (turns, mutations,
#: reads driving downstream fan-out). MODELED.
EVENTS_PER_ACTIVE_AGENT_PER_SEC = 2.0

#: Minimum Kafka partitions (keeps a floor for ordering/parallelism headroom).
MIN_KAFKA_PARTITIONS = 3

# --------------------------------------------------------------------------- #
# Agent communication bus (CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent/ECO-4.87) — MODELED.
# The AgentBus is durable-store-first: one send writes one :BusMessage node + one
# :HAS_BUS_MESSAGE edge per recipient (~2 engine ops/recipient), so its throughput
# is bounded by the same single-connection anchor as everything else. A receive is
# a single cursor read. These constants size a single graph-os hub and tell you when
# to shard engines or add federated hubs.
# --------------------------------------------------------------------------- #

#: Engine write ops per delivered bus message per recipient (add_node + add_edge).
BUS_OPS_PER_MESSAGE = 2

#: Registered bus participants (presence + mailbox) one hub holds comfortably.
PARTICIPANTS_PER_HUB = 10_000


def bus_messages_per_sec_per_connection(
    ops_per_message: int = BUS_OPS_PER_MESSAGE,
) -> int:
    """Sustained direct (fanout=1) bus messages/sec on a single engine connection."""
    return SINGLE_CONNECTION_OPS_PER_SEC // max(1, ops_per_message)


def bus_engine_connections_for(
    messages_per_sec: float,
    avg_recipients: float = 1.0,
    ops_per_message: int = BUS_OPS_PER_MESSAGE,
) -> int:
    """Engine connections (≈ shards) needed for a bus send rate at a given fan-out."""
    if messages_per_sec <= 0:
        return 0
    ops_per_sec = messages_per_sec * max(1.0, avg_recipients) * ops_per_message
    return max(1, _ceil_div(ops_per_sec, SINGLE_CONNECTION_OPS_PER_SEC))


def bus_hubs_for(participants: int, per_hub: int = PARTICIPANTS_PER_HUB) -> int:
    """Federated hubs needed to hold ``participants`` (the federation/mesh axis)."""
    if participants <= 0:
        return 0
    return max(1, _ceil_div(participants, per_hub))


def _ceil_div(numerator: float, denominator: float) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return int(ceil(numerator / denominator))


def active_agents(residents: int, active_fraction: float) -> int:
    """Number of concurrently active agents.

    :param residents: Resident population.
    :param active_fraction: Fraction (0..1) of residents active at peak.
    """
    if residents < 0:
        raise ValueError("residents must be >= 0")
    if not 0.0 <= active_fraction <= 1.0:
        raise ValueError("active_fraction must be in [0, 1]")
    return int(ceil(residents * active_fraction))


def pg_shards_for(residents: int, per_shard: int = RESIDENTS_PER_PG_SHARD) -> int:
    """Postgres (durable) shards needed for ``residents`` (resident-population axis)."""
    if residents <= 0:
        return 0
    return max(1, _ceil_div(residents, per_shard))


def l0_shards_for(residents: int, per_shard: int = RESIDENTS_PER_L0_SHARD) -> int:
    """In-memory L0 (hot tier) shards needed for ``residents``."""
    if residents <= 0:
        return 0
    return max(1, _ceil_div(residents, per_shard))


def workers_for(
    residents: int,
    active_fraction: float,
    per_worker: int = ACTIVE_AGENTS_PER_WORKER,
) -> int:
    """Worker count needed for the active-concurrency axis."""
    active = active_agents(residents, active_fraction)
    if active <= 0:
        return 0
    return max(1, _ceil_div(active, per_worker))


def nodes_for(
    residents: int,
    active_fraction: float,
    workers_per_node: int = WORKERS_PER_NODE,
    per_worker: int = ACTIVE_AGENTS_PER_WORKER,
) -> int:
    """Node count needed to host the required worker pool."""
    workers = workers_for(residents, active_fraction, per_worker=per_worker)
    if workers <= 0:
        return 0
    return max(1, _ceil_div(workers, workers_per_node))


def kafka_partitions_for(
    residents: int,
    active_fraction: float,
    events_per_active_agent_per_sec: float = EVENTS_PER_ACTIVE_AGENT_PER_SEC,
    ops_per_partition: int = OPS_PER_SEC_PER_KAFKA_PARTITION,
) -> int:
    """Kafka partitions needed for the event-throughput axis."""
    active = active_agents(residents, active_fraction)
    if active <= 0:
        return MIN_KAFKA_PARTITIONS
    events_per_sec = active * events_per_active_agent_per_sec
    needed = _ceil_div(events_per_sec, ops_per_partition)
    return max(MIN_KAFKA_PARTITIONS, needed)


def event_throughput_per_sec(
    residents: int,
    active_fraction: float,
    events_per_active_agent_per_sec: float = EVENTS_PER_ACTIVE_AGENT_PER_SEC,
) -> float:
    """Modeled peak graph-event throughput (events/sec)."""
    return active_agents(residents, active_fraction) * events_per_active_agent_per_sec


@dataclass(frozen=True)
class CapacityPlan:
    """A full modeled capacity plan for one (residents, active_fraction) point."""

    residents: int
    active_fraction: float
    active_agents: int
    pg_shards: int
    l0_shards: int
    workers: int
    nodes: int
    kafka_partitions: int
    event_throughput_per_sec: float


def plan_for(residents: int, active_fraction: float = 0.02) -> CapacityPlan:
    """Compute the full modeled :class:`CapacityPlan` for a population.

    :param residents: Resident agent population.
    :param active_fraction: Fraction active at peak (default 2%).
    """
    return CapacityPlan(
        residents=residents,
        active_fraction=active_fraction,
        active_agents=active_agents(residents, active_fraction),
        pg_shards=pg_shards_for(residents),
        l0_shards=l0_shards_for(residents),
        workers=workers_for(residents, active_fraction),
        nodes=nodes_for(residents, active_fraction),
        kafka_partitions=kafka_partitions_for(residents, active_fraction),
        event_throughput_per_sec=event_throughput_per_sec(residents, active_fraction),
    )


@dataclass(frozen=True)
class BusCapacityPlan:
    """A modeled capacity plan for the agent bus at one (participants, rate) point."""

    participants: int
    messages_per_sec: float
    avg_recipients: float
    hubs: int
    engine_connections: int
    messages_per_sec_per_connection: int


def bus_plan_for(
    participants: int,
    messages_per_sec: float,
    avg_recipients: float = 1.0,
) -> BusCapacityPlan:
    """Compute the modeled :class:`BusCapacityPlan` for a bus workload."""
    return BusCapacityPlan(
        participants=participants,
        messages_per_sec=messages_per_sec,
        avg_recipients=avg_recipients,
        hubs=bus_hubs_for(participants),
        engine_connections=bus_engine_connections_for(messages_per_sec, avg_recipients),
        messages_per_sec_per_connection=bus_messages_per_sec_per_connection(),
    )


# Reference population points used by the docs and tests.
REFERENCE_POPULATIONS = (1_000, 100_000, 1_000_000, 100_000_000)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(
        f"{'residents':>12} {'active':>8} {'pg':>5} {'l0':>5} "
        f"{'workers':>8} {'nodes':>6} {'kafka':>6} {'events/s':>10}"
    )
    for pop in REFERENCE_POPULATIONS:
        p = plan_for(pop, 0.02)
        print(
            f"{p.residents:>12} {p.active_agents:>8} {p.pg_shards:>5} {p.l0_shards:>5} "
            f"{p.workers:>8} {p.nodes:>6} {p.kafka_partitions:>6} "
            f"{p.event_throughput_per_sec:>10.0f}"
        )
