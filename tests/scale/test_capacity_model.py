#!/usr/bin/python
"""Tests for the MODELED capacity model in docs/scaling/capacity_model.py.

Asserts internal consistency (monotonicity), the measured anchor, and that the
documented 1M reference numbers match the implementation so docs cannot drift.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _REPO_ROOT / "docs" / "scaling" / "capacity_model.py"


def _load_model():
    spec = importlib.util.spec_from_file_location(
        "agent_utilities_capacity_model", _MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module's namespace
    # (Python 3.14 dataclass processing looks the module up in sys.modules).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cm = _load_model()


def test_model_file_exists():
    assert _MODEL_PATH.is_file()


def test_measured_anchor_present():
    assert cm.MEASURED_ADDNODE_P50_MS == pytest.approx(0.187)
    assert cm.SINGLE_CONNECTION_OPS_PER_SEC == 5_000
    # Kafka partition sizing is anchored to the measured drain rate.
    assert cm.OPS_PER_SEC_PER_KAFKA_PARTITION == cm.SINGLE_CONNECTION_OPS_PER_SEC


def test_active_agents_arithmetic():
    assert cm.active_agents(1_000_000, 0.02) == 20_000
    assert cm.active_agents(0, 0.02) == 0
    with pytest.raises(ValueError):
        cm.active_agents(100, 1.5)
    with pytest.raises(ValueError):
        cm.active_agents(-1, 0.02)


@pytest.mark.parametrize("fn", ["pg_shards_for", "l0_shards_for"])
def test_shard_helpers_zero_and_floor(fn):
    f = getattr(cm, fn)
    assert f(0) == 0
    assert f(1) == 1  # any nonzero population needs at least one shard


def test_monotonic_in_residents():
    fns = [
        lambda r: cm.pg_shards_for(r),
        lambda r: cm.l0_shards_for(r),
        lambda r: cm.workers_for(r, 0.02),
        lambda r: cm.nodes_for(r, 0.02),
        lambda r: cm.kafka_partitions_for(r, 0.02),
        lambda r: cm.event_throughput_per_sec(r, 0.02),
    ]
    residents = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    for f in fns:
        vals = [f(r) for r in residents]
        assert vals == sorted(vals), vals


def test_monotonic_in_active_fraction():
    for frac in (0.0, 0.01, 0.02, 0.1, 0.5, 1.0):
        pass
    fracs = [0.0, 0.01, 0.02, 0.1, 0.5, 1.0]
    workers = [cm.workers_for(1_000_000, fr) for fr in fracs]
    assert workers == sorted(workers)
    parts = [cm.kafka_partitions_for(1_000_000, fr) for fr in fracs]
    assert parts == sorted(parts)


def test_one_thousand_reference_case():
    p = cm.plan_for(1_000, 0.02)
    assert p.active_agents == 20
    assert p.pg_shards == 1
    assert p.l0_shards == 1
    assert p.workers == 1
    assert p.nodes == 1
    assert p.kafka_partitions == 3  # floor
    assert p.event_throughput_per_sec == 40


def test_hundred_thousand_reference_case():
    p = cm.plan_for(100_000, 0.02)
    assert p.active_agents == 2_000
    assert p.pg_shards == 1
    assert p.l0_shards == 2
    assert p.workers == 80
    assert p.nodes == 10
    assert p.kafka_partitions == 3
    assert p.event_throughput_per_sec == 4_000


def test_one_million_matches_documented_numbers():
    """The documented 1M reference case (capacity_model.md summary table)."""
    p = cm.plan_for(1_000_000, 0.02)
    assert p.active_agents == 20_000
    assert p.pg_shards == 4
    assert p.l0_shards == 20
    assert p.workers == 800
    assert p.nodes == 100
    assert p.kafka_partitions == 8
    assert p.event_throughput_per_sec == 40_000


def test_hundred_million_linear_extrapolation():
    p = cm.plan_for(100_000_000, 0.02)
    assert p.active_agents == 2_000_000
    assert p.pg_shards == 400
    assert p.l0_shards == 2_000
    assert p.workers == 80_000
    assert p.nodes == 10_000
    assert p.kafka_partitions == 800
    assert p.event_throughput_per_sec == 4_000_000


def test_kafka_partition_floor():
    # Tiny / dormant populations never drop below the floor.
    assert cm.kafka_partitions_for(10, 0.0) == cm.MIN_KAFKA_PARTITIONS
    assert cm.kafka_partitions_for(0, 0.02) == cm.MIN_KAFKA_PARTITIONS


# --- Agent bus capacity (CONCEPT:ECO-4.84/ECO-4.87) ------------------------- #


def test_bus_message_rate_anchored_to_single_connection():
    # ~2 ops/message → half the single-connection op rate.
    assert cm.bus_messages_per_sec_per_connection() == 2_500
    assert cm.BUS_OPS_PER_MESSAGE == 2


def test_bus_engine_connections_scale_with_rate_and_fanout():
    assert cm.bus_engine_connections_for(0) == 0
    # 2,500 direct msg/s fits on one connection; double it → two.
    assert cm.bus_engine_connections_for(2_500, avg_recipients=1.0) == 1
    assert cm.bus_engine_connections_for(5_000, avg_recipients=1.0) == 2
    # Fan-out multiplies the write cost.
    assert cm.bus_engine_connections_for(2_500, avg_recipients=10.0) == 10


def test_bus_hubs_floor_and_sharding():
    assert cm.bus_hubs_for(0) == 0
    assert cm.bus_hubs_for(1) == 1
    assert cm.bus_hubs_for(cm.PARTICIPANTS_PER_HUB) == 1
    assert cm.bus_hubs_for(cm.PARTICIPANTS_PER_HUB + 1) == 2


def test_bus_plan_monotonic_and_consistent():
    small = cm.bus_plan_for(5_000, 1_000, avg_recipients=1.0)
    big = cm.bus_plan_for(50_000, 20_000, avg_recipients=4.0)
    assert small.hubs == 1 and small.engine_connections == 1
    assert big.hubs >= small.hubs
    assert big.engine_connections >= small.engine_connections
