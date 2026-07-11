"""Hardware-pending soak/chaos scenarios — SCALE-P2-1.

These scenarios cannot be honestly validated against an in-memory mock engine
on a single CI runner — they require REAL multi-node/multi-broker
infrastructure (a live Kafka cluster, multiple engine shards, multiple hosts/
zones, a real rolling deployment). Codex's guardrail for this workstream is
explicit: never call a MODELED capacity a DEMONSTRATED result. So rather than
faking these with the mock engine (which would silently misrepresent "ran
against 3 real brokers" as "ran in 40ms against a dict"), each is a real,
documented, currently-``skip``-marked test — the harness is READY (the body
is the actual command/assertion a real run would execute), but it is not
exercised in CI.

Run for real (manually, against real infrastructure) with:

    AGENT_UTILITIES_TESTING= PYTHONPATH=. python3 scripts/scale/loadgen.py \\
        --engine live --scale 1.0 --duration-s <hours-in-seconds> \\
        --workers 800 --report-json soak-report.json

pointed at a deployed fleet (``ENGINE_ENDPOINT`` set, real Kafka/Postgres
shards per ``docs/architecture/agent_dispatch.md`` / ``engine_sharding.md``),
with the specific fault (broker rebalance, node loss, etc.) injected by the
operator via the platform's own chaos tooling (``container-manager-mcp``
node/service kill, the swarm/k8s supervisory plane's pause/kill actions) at
the SAME time this generator is running, then asserting the JSON report's
``ok``/``invariants`` fields.

See ``docs/scaling/capacity_model.md``'s "What is CI-measured vs
hardware-pending" section for the authoritative status table.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]

_HARDWARE_PENDING = pytest.mark.skip(
    reason="hardware-pending: needs real multi-node/multi-broker infrastructure; "
    "see this module's docstring for the exact manual run recipe."
)


@_HARDWARE_PENDING
def test_24_72h_steady_and_burst_soak_at_full_1m_scale(loadgen):
    """The contract's real acceptance run: 24-72h steady-state + burst at
    ``--scale 1.0`` (the actual 1,000,000 residents) against a deployed fleet,
    asserting every SLO axis holds for the ENTIRE window (not a 3-second CI
    sample) and that resource use (queue depth, memory, shard count) stays
    bounded throughout, not just at the start.
    """
    import asyncio

    contract = loadgen.load_workload_contract()
    engine = loadgen.build_live_engine()
    report = asyncio.run(
        loadgen.run_workload(
            contract, scale=1.0, duration_s=72 * 3600, num_workers=800, engine=engine
        )
    )
    assert report.ok


@_HARDWARE_PENDING
def test_broker_rebalance_and_partition_expansion_under_load(loadgen):
    """Kafka consumer-group rebalance (a dispatch-worker joins/leaves) and a live
    ``AGENT_TURNS_PARTITIONS`` increase, both while turns are actively in
    flight — assert zero lost/duplicate turns across the rebalance boundary
    and that partition expansion does not stall in-flight session ordering.
    """
    pytest.skip(
        "needs a real Kafka cluster with a rebalance/partition-expansion trigger"
    )


@_HARDWARE_PENDING
def test_shard_split_or_move_under_concurrent_writes():
    """A live engine/L0 shard split or move (``AU-KG.sharding.tenant-partitioned-sharding-hrw``)
    while writes are landing against the affected tenant range — assert no
    write is lost or duplicated across the resharding boundary and that
    query/write SLOs recover within the RTO window afterward.
    """
    pytest.skip(
        "needs a real multi-shard engine deployment with a live resharding trigger"
    )


@_HARDWARE_PENDING
def test_worker_gateway_broker_leader_node_or_zone_loss():
    """Kill each tier in turn on REAL infrastructure (dispatch-worker process,
    gateway process, broker node, Postgres/consensus leader, a whole host, a
    whole availability zone) while load is running, and assert: the
    supervisory plane detects the loss within the contract's RTO (300s), the
    fleet recovers (workers reclaim orphaned leases via ``reap_expired_leases``,
    a new leader is elected, gateways route around the dead node), and no
    turn is lost or double-completed across the outage.
    """
    pytest.skip(
        "needs real multi-node/multi-zone infrastructure and a kill-injection tool"
    )


@_HARDWARE_PENDING
def test_rolling_upgrade_and_schema_migration_across_real_hosts():
    """A real rolling deploy (old-version + new-version dispatch workers/gateways
    coexisting mid-rollout) plus a live schema/ontology migration applied
    concurrently — assert continuity (no turn lost/duplicated across the
    version boundary) and that the migration itself completes without a
    write-availability gap exceeding the contract's RPO/RTO.
    """
    pytest.skip(
        "needs a real multi-host rolling-deploy pipeline + a live migration to run"
    )


@_HARDWARE_PENDING
def test_full_1m_resident_cold_activation_at_real_scale():
    """A cold start of the ACTUAL 1,000,000-resident population against real L0/
    PG shards (not the mock engine's snapshot/restore, which only proves the
    STATE-MACHINE semantics — see ``test_chaos_tenant_and_restart.py``'s
    ``test_full_restart_cold_activation_recovers_in_flight_work``): measure
    real hydrate-on-miss latency (``tenant_engine_pool.py``) at the full
    resident count and assert the contract's queue/query SLOs hold from the
    very first wave of activity, not just once caches are warm.
    """
    pytest.skip(
        "needs the real 1,000,000-resident population provisioned on real L0/PG shards"
    )
